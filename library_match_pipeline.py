"""
Ruffus pipeline to evaluate the library match (database lookup) performance
of various models.

The goal here is to enable us to:
- Quickly evaluate and get final, camera-ready metrics for certain output
  predictions of our models.

main_library:
  main database of molecules with observed reference spectra
query_library:
  database of molecules with replicate measured spectra
  we use mols+spectra from here as queries
pred_library:
  the augmenting database with a given model's predicted spectra
  should be molecules drawn from the augmenting library
  we use mols+spectra from here to augment the main library
pred_reference_library:
  because we only store (mol_id, spect, phase) in the pred_library (.spect files),
  we need a reference library to dereference the mol_ids against.
  this lets us get the actual MOLECULE objects from just having mol_ids.
"""
import pickle
from tqdm import tqdm
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Descriptors

from ruffus import * 

import sys

from rassp import util
from const import ANALYSIS_EXPERIMENTS
from const import LIBRARY_MATCH_EXPERIMENTS as EXPERIMENTS

MAX_MZ_BINS = 512
MASS_FILTER_DELTA = 15
DP_NAME = 'reg_dp'

WORKING_DIR = f"library_match_results.metrics.mass_filter_{MASS_FILTER_DELTA}.{DP_NAME}"
td = lambda x: os.path.join(WORKING_DIR, x)

# mass and intensity powers for dot product calculation
P_MAP = {
  'reg_dp': (1, 0.5),  # Adams paper and common database lookup procedures
  'stein_dp': (3, 0.6),  # Stein dot product
}
MASS_POW, INT_POW = P_MAP[DP_NAME]

def exp_params():
  for exp_name, exp_config in EXPERIMENTS.items():
    outfile = td(f"{exp_name}.results.all")
    infiles = [
      exp_config['main_library'],
      exp_config['query_library'],
    ]

    if 'pred_library' not in exp_config:
      # dereference the specified experiment
      assert 'exp_name' in exp_config
      exp_name = exp_config['exp_name']
      exp_config['pred_library'] = ANALYSIS_EXPERIMENTS[exp_name]['pred_spect']
    
      # special handling for NIST17 data
      # we use the mols parquet (no true spect) and combine it with the model predictions
      tspect_path = ANALYSIS_EXPERIMENTS[exp_name]['true_spect']
      if 'data.processed' in tspect_path:
        tspect_path = tspect_path.replace('.spect', '.mols')
      exp_config['pred_reference_library'] = tspect_path
    
    infiles += [
        exp_config['pred_library'],
        exp_config['pred_reference_library'],
    ]
    yield infiles, outfile

def get_max_mz(spect):
  max_mz = 0
  for m, i in spect:
    if m > max_mz:
      max_mz = m
  return max_mz

def bin_spect(spect_list, max_mz=MAX_MZ_BINS):
  """Recall that index 0 => mz 0, index 1 => mz 0... so index 511 => mz 511."""
  svec = np.zeros(max_mz)
  for m, i in spect_list:
    svec[int(m)] = i
  svec /= np.sum(svec)
  return svec

def rename_id_to_mol_id(df):
  if 'mol_id' not in df.columns:
    assert 'id' in df.columns
    return df.rename(columns={'id': 'mol_id'})
  else:
    return df

def filter_by_max_mz(df, max_mz=MAX_MZ_BINS):
  mask = df['spect'].apply(lambda s: get_max_mz(s) < max_mz)
  return df[mask]

def filter_dfs_by_ikey(df, ikey_set, name='df'):
  mask = df.inchi_key.apply(lambda ikey: ikey in ikey_set)
  print(f'filtering {name} from length {len(mask)} to {mask.sum()}')
  return df[mask]

@mkdir(WORKING_DIR)
@files(exp_params)
def run_analysis(infiles, outfile):
  print(f'Generating {outfile}')
  dfs = [util.read_molfile(f) for f in infiles]
  dfs = [rename_id_to_mol_id(df) for df in dfs]
  main_lib_df, query_lib_df, pred_lib_df, pred_ref_lib_df = dfs
  for df in [main_lib_df, query_lib_df, pred_lib_df]:
    # the pred_ref_lib doesn't need spect, since we're replacing the 
    # spect with the pred_lib spectra.
    assert 'spect' in df.columns, f'{df.columns} did not contain spect'
  for df in [main_lib_df, query_lib_df, pred_ref_lib_df]:
    assert 'inchi_key' in df.columns, f'{df.columns} did not contain inchi_key'
  for df in dfs:
    assert 'mol_id' in df.columns, f'{df.columns} did not contain mol_id'
  
  main_lib_df, query_lib_df, pred_lib_df = [
    filter_by_max_mz(df) for df in [main_lib_df, query_lib_df, pred_lib_df]
  ]
  
  if (
    'inchi_key' not in pred_lib_df.columns or
    len(set(pred_lib_df.mol_id)) == len(set(pred_lib_df.inchi_key))
  ):
    # each row in pred_ref_lib is a unique molecule
    # replace spectra in pred_ref_lib with spectra from pred_lib
    # it must have enough matching mol_ids with pred_lib!
    ref_replace_mol_ids = set(pred_ref_lib_df.mol_id).intersection(set(pred_lib_df.mol_id))
    print(f'replacing {len(ref_replace_mol_ids)} / {len(pred_ref_lib_df)} spectra in pred_ref_lib')

    rows = []
    for i, row in tqdm(pred_ref_lib_df.iterrows()):
      if row.mol_id not in ref_replace_mol_ids:
        continue
      candidates = pred_lib_df[pred_lib_df.mol_id == row.mol_id]
      if len(candidates) == 0:
        continue

      # take the first candidate, bc it has matching mol_id
      # and we're replacing it with prediction spectra
      p_row = candidates.iloc[0]
      p_dict = p_row.to_dict()
      assert p_dict['mol_id'] == row.mol_id
      rows.append({
        **row.to_dict(),
        'spect': p_dict['spect'],
      })
    pred_lib_df = pd.DataFrame(rows)
  else:
    # we are running on nist-ref bc
    # pred_lib has multiple replicates of the same molecule
    # (as measured by more unique mol_ids than inchi_keys)
    assert infiles[1] == infiles[2] == infiles[3]

    # keep only the first replicate of each and use that
    # as our pred_lib
    rows = []
    ikeys_we_have = set()
    for i, row in pred_lib_df.iterrows():
      if row.inchi_key in ikeys_we_have:
        continue
      else:
        rows.append(row)
        ikeys_we_have.add(row.inchi_key)
    pred_lib_df = pd.DataFrame(rows)
    
    # need to remove matching mol_ids from query
    # since they have exactly matching spectra...
    mol_ids_to_remove = set(pred_lib_df.mol_id)
    mask = ~query_lib_df.mol_id.isin(mol_ids_to_remove)
    query_lib_df = query_lib_df[mask]
    print(f'kept {mask.sum()} mols out of {len(mask)} total in query_lib')

  assert pd.isnull(pred_lib_df.spect).sum() == 0
  print(f'pred_lib has {len(pred_lib_df)} rows')

  # filter query + pred libs to have the same inchi keys?
  # keep only the inchi keys that pred and query libs have in common
  mutual_ikeys = set(query_lib_df.inchi_key).intersection(set(pred_lib_df.inchi_key))

  query_lib_df = filter_dfs_by_ikey(query_lib_df, mutual_ikeys, name='query_lib_df')
  pred_lib_df = filter_dfs_by_ikey(pred_lib_df, mutual_ikeys, name='pred_lib_df')

  # combine pred_lib with main_lib to get augmented_lib
  cols_we_need = ['mol_id', 'inchi_key', 'inchi', 'spect', 'rdmol']
  main_lib_df = main_lib_df[cols_we_need]
  pred_lib_df = pred_lib_df[cols_we_need]
  assert len(set(main_lib_df.inchi_key).intersection(set(pred_lib_df.inchi_key))) == 0

  augmented_lib_df = pd.concat([main_lib_df, pred_lib_df])
  augmented_lib_df['mw'] = augmented_lib_df.rdmol.apply(lambda m: Descriptors.ExactMolWt(Chem.Mol(m)))
  query_lib_df['mw'] = query_lib_df.rdmol.apply(lambda m: Descriptors.ExactMolWt(Chem.Mol(m)))
  query_lib_df['spect_binned'] = query_lib_df.spect.apply(lambda s: bin_spect(s))
  augmented_lib_df['spect_binned'] = augmented_lib_df.spect.apply(lambda s: bin_spect(s))

  # report stats prior to doing library match
  query_set = set(query_lib_df.inchi_key)
  lookup_set = set(augmented_lib_df.inchi_key)
  counts = [0, 0]  # in, out
  for ikey in query_set:
    if ikey in lookup_set:
      counts[0] += 1
    else:
      counts[1] += 1
  print(f'query set had {len(query_set)} total mols')
  print(f'of these, {counts[0]} ({round(counts[0] / len(query_set) * 100, 1)}%) '
         'have matching mols in the augmented lib')

  print(f'Query db had {len(set(query_lib_df.inchi_key))} unique inchi keys')
  print(f'Augmented library db had {len(set(augmented_lib_df.inchi_key))} unique inchi keys')
  ikey_list = sorted(mutual_ikeys)

  # do query lookup using each ikey from query_lib
  lookup_results = []
  for ikey in tqdm(ikey_list):
    # filter down only the query rows matching the ikey
    # TODO: add a check here that all the mws are close to matching...
    sub_query_df = query_lib_df[query_lib_df.inchi_key == ikey]
    masses, counts = np.unique([row.mw for _, row in sub_query_df.iterrows()], return_counts=True)
    mw_query = masses[np.argmax(counts)]

    query_row = sub_query_df.iloc[np.random.choice(len(sub_query_df))]

    # apply mass-filtering to augmented_lib
    # TODO: add ability to turn mass filter on/off
    mask = np.abs(augmented_lib_df.mw - mw_query) <= MASS_FILTER_DELTA
    sub_aug_lib_df = augmented_lib_df[mask]

    # compute SDPs against our query spect
    # slow_sdps = sub_aug_lib_df.spect_binned.apply(lambda svec: sdp(svec, query_row.spect_binned))
    ref_spect_mat = np.array(list(sub_aug_lib_df.spect_binned))  # (n_spect, mz_bins)
    ref_spect_mat /= np.expand_dims(np.linalg.norm(ref_spect_mat, axis=1), axis=1)
    query_spect = query_row.spect_binned  # (mz_bins, )
    query_spect /= np.linalg.norm(query_spect)
    mass_val_array = np.arange(len(query_row.spect_binned)).astype(np.float64)
    v1 = np.power(np.expand_dims(mass_val_array, axis=0), MASS_POW) * np.power(ref_spect_mat, INT_POW)
    v2 = np.power(mass_val_array, MASS_POW) * np.power(query_spect, INT_POW)
    sdps = (v1 @ v2) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2))

    assert query_row.inchi_key in set(augmented_lib_df.inchi_key)

    if query_row.inchi_key in set(sub_aug_lib_df.inchi_key):
      # we can get the actual rank of the matching rows in the augmented_lib
      # we can have more than one matching ikey, bc there are multiple replicates
      # in the event of multiple replicates, we report the mean rank.
      mask = sub_aug_lib_df.inchi_key == query_row.inchi_key

      # assert np.sum(mask) == 1
      # matches = sub_aug_lib_df[mask]
      # assert len(matches) == 1
      # matching_row = matches.iloc[0]

      ranks = []
      for match_ind in np.where(mask)[0]:
          rank = np.where(np.array(np.argsort(sdps)[::-1]) == match_ind)[0][0] + 1
          ranks.append(rank)
    else:
      # somehow the ikey was not found...
      # the rank is lower-bounded by the number of mols in post-mass-filtering df
      ranks = [len(sub_aug_lib_df) + 1]

    lookup_results.append({
      'ikey': ikey,
      'n_matching_query_rows': len(sub_query_df),
      'n_rows_post_mass_filter': len(sub_aug_lib_df),
      'ranks': ranks,
      'rank': np.mean(ranks),
    })
  results_df = pd.DataFrame(lookup_results)
  results_df.to_parquet(outfile)
  print(f'finished processing {len(results_df)} lookups')

@transform(run_analysis, suffix('.results.all'), '.results')
def summarize_results(infile, outfile):
  results_df = pd.read_parquet(infile)
  assert len(results_df) > 0

  cols_to_keep = [
      'ikey', 'n_matching_query_rows', 'n_rows_post_mass_filter',
      'rank',
  ]
  a = pd.melt(results_df[cols_to_keep], id_vars='ikey', var_name='metric')
  agg_results = a.groupby('metric').agg({
      'value': ['mean', 'min', 'max', 'std'] + [util.percentile(n) for n in [0.99, 0.95, 0.90, 0.5, 0.10, 0.05, 0.01]]
  })
  agg_results.columns = agg_results.columns.droplevel()
  agg_results.reset_index().to_parquet(outfile)

  with open(outfile + '.txt', 'w') as fp:
    fp.write(agg_results.to_string(float_format='%3.3f'))
    fp.write('\n')
    for n in [1, 3, 5, 10, 30, 50, 100]:
      fp.write(f'Recall @ {n}: {round(100 * (results_df["rank"] <= n).sum() / len(results_df), 2)}\n')
    fp.write('\n')

@collate(run_analysis, regex('.+\.results\.all'), td('recall_at_x.png'))
def generate_recall_at_x_plot(infiles, outfile):
  rmap = {}
  for infile in infiles:
    rmap[infile.replace('.results.all', '')] = infile

  plt.figure(figsize=(8, 6), dpi=150)
  for name, path in rmap.items():
      df = pd.read_parquet(path)
      assert len(df) > 0

      xs = np.arange(1, 101)
      ys = [(df['rank'] <= i).sum() / len(df) for i in xs]
      plt.plot(xs, ys, label=name)
  plt.xlim([1, 100])
  plt.ylim([0, 1])
  plt.xlabel('recall @ x')

  plt.axhline(y=0.95, label='0.95', lw=0.5)
  plt.axhline(y=0.99, label='0.99', lw=0.5)
  plt.legend(loc='best')
  plt.savefig(outfile)

if __name__ == "__main__":
  pipeline_run([run_analysis, summarize_results, generate_recall_at_x_plot], multiprocess=4)
