"""
Ruffus pipleine to evaluate the output of various models. The goal
is to enable :
1. Intermediate evaluation output that we can depend on for
reference values
2. A quick gut-check leaderboard to see how we're doing 
3. Easily add additonal metrics. 
4. Easily partition data by type 

All metrics evaluation code is in metrics.py

TODO: it should be possible to evaluate the replica set results from NIST17

TODO: we should use mol_file to only run the mol_ids for unique molecules
    due to replicates, there are unique mol_ids that correspond to the same molecule.

true_spect:
    Columns expected: mol_id, spect
pred_spect:
    Columns expected: mol_id, spect, phase
mol_file:
    Columns expected: mol_id, rdmol

"""
import numpy as np
import pandas as pd
import seaborn as sns
import os
import pickle
from rdkit import Chem
from tqdm import tqdm
from ruffus import * 

from rassp import util
from const import ANALYSIS_EXPERIMENTS as EXPERIMENTS
from const import DATA_DIR, WORKING_DIR, td

def exp_params():
    for exp_name, exp_config in EXPERIMENTS.items():

        outfile = td(f"{exp_name}.results.all")

        infiles = [exp_config['true_spect'],
                   exp_config['pred_spect']]
        if 'mol_file' in exp_config:
            infiles.append(exp_config['mol_file'])

        if '.msp' in infiles[1]:
            basedir, _ = os.path.split(infiles[1])
            infiles[1] = os.path.join(basedir, f'{exp_name}.spect')
        yield infiles, outfile

from rassp.metrics import sdp, dp, l1, top_k
from rassp.metrics import topk_precision, intensity_weighted_barcode_precision, intensity_weighted_barcode_false_positive_rate

DEFAULT_METRICS = {
    'sdp': (sdp, {}),
    'dp': (dp, {}),
    # 'l1': (l1, {}),

    'top1_precision': (topk_precision, {'k': 1}),
    'top5_precision': (topk_precision, {'k': 5}),
    'top10_precision': (topk_precision, {'k': 10}),
    'intensity_weighted_barcode_precision': (intensity_weighted_barcode_precision, {}),
    'intensity_weighted_false_positive_rate': (intensity_weighted_barcode_false_positive_rate, {}),
}

def verify_spect_df(spect_df):
    for colname in ['mol_id', 'spect']:
        assert colname in spect_df.columns, f'Missing {colname} from {spect_df.columns}'

@mkdir(WORKING_DIR)
@files(exp_params)
def compute_analysis(infiles, outfile):
    print(f'Generating {outfile}')
    mol_file = None
    
    if len(infiles) == 3:
        true_spect_file, pred_spect_file, mol_file = infiles
    else:
        true_spect_file, pred_spect_file = infiles

    # FIXME sanity check : does a m/z appear more than once in a file?
    #    if so we may have merged multiple experiments into a file

    true_spect_df = pd.read_parquet(true_spect_file)
    pred_spect_df = util.read_molfile(pred_spect_file)
    
    if mol_file:
        mol_df = pd.read_parquet(mol_file)

    verify_spect_df(true_spect_df)
    verify_spect_df(pred_spect_df)

    # sanity check of frac of spectra that have mass 0
    # (peaks at 1-amu are reported with mass 0 during pred)
    def frac_of_spect_with_mass_zero(spect):
        n_with_zero = spect.map(lambda slist: any([x[0] == 0.0 for x in slist])).sum()
        return n_with_zero / len(spect)
    
    print(f'True spect df had {100 * round(frac_of_spect_with_mass_zero(true_spect_df.spect), 2)}% with zero-mass peak')
    print(f'Pred spect df had {100 * round(frac_of_spect_with_mass_zero(pred_spect_df.spect), 2)}% with zero-mass peak')

    true_molid = true_spect_df.mol_id.unique()
    pred_molid = pred_spect_df.mol_id.unique()

    if mol_file is not None:
        mol_df = pd.read_parquet(mol_file)
        tgt_mol_ids = mol_df.mol_id.unique()
    else:
        tgt_mol_ids = pred_molid
        if not set(true_molid).issuperset(pred_molid):
            print("WARNING: pred mols are not a subset of true mols")
            print("there are", len(true_molid), "true mols and", len(pred_molid), "pred mols")

    eval_mol_id = set(true_molid).intersection(pred_molid).intersection(tgt_mol_ids)
    print("We are evaluating", len(eval_mol_id), "mols")

    # get all pred/true spects to compare
    pairs = []
    for mol_id in tqdm(eval_mol_id):
        # all pred rows should be identical since we start w the same mol as input
        p_row = pred_spect_df[pred_spect_df.mol_id == mol_id].iloc[0]

        # but ref/true rows may be different due to replicate experiments
        # (unless there was only one observed true experiment)
        ref_rows = true_spect_df[true_spect_df.mol_id == mol_id]
        for _, t_row in ref_rows.iterrows():
            pairs.append({
                'mol_id': mol_id,
                'p_row': p_row,
                't_row': t_row,
            })
    
    # compute metrics between the pred and true rows
    results_vals = []
    for pair in tqdm(pairs):
        mol_id = pair['mol_id']
        p_row  = pair['p_row']
        t_row  = pair['t_row']

        t_dict = {a[0]: a[1] for a in t_row.spect}
        p_dict = {a[0]: a[1] for a in p_row.spect}
        if len(p_dict) == 0:
            continue

        metric_vals = {}
        for metric_name, (metric_func, metric_kwargs) in\
            DEFAULT_METRICS.items():
            # ! metrics functions must take pred spect first, true spect second.
            metric_vals[metric_name] = metric_func(p_dict, t_dict, **metric_kwargs)

        metric_vals['mol_id'] = mol_id
        results_vals.append(metric_vals)
    results_df = pd.DataFrame(results_vals)

    # TODO: should we handle empty spectra here?
    print(f'Processed {len(results_df)} of {len(eval_mol_id)} mols')
    
    results_df.to_parquet(outfile)

@transform(compute_analysis, suffix(".results.all"), ".results")
def summarize_results(infile, outfile):
    results_df = pd.read_parquet(infile)
    
    if len(results_df) == 0:
        # if we had an error earlier in pipeline due to mismatched datasets...
        # skip it quietly so we can go back and fix it.
        results_df.to_parquet(outfile)
        with open(outfile + ".txt", 'w') as fid:
            fid.write('empty df\n')
        return

    # destructively do groupby over mol_ids (turning into index)
    # and then mean-agg (to deal with metrics from 
    # replicate experiments with repeated mol_ids)
    print(f'Results DF had {len(results_df)} rows prior to destructive mol_id mean-groupby')
    results_df = results_df.groupby('mol_id').mean()

    print(f'Results DF had {len(results_df)} rows after')
    a = pd.melt(results_df, var_name='metric')
    agg_results = a.groupby('metric').agg({'value' : ['mean', 'min', 'max', 'std',
                                                      util.percentile(0.9), util.percentile(0.5), util.percentile(0.1)]})
    agg_results.columns = agg_results.columns.droplevel()    

    agg_results.reset_index().to_parquet(outfile)

    # bonus file: aggregated metrics
    with open(outfile + ".txt", 'w') as fid:
        fid.write(agg_results.to_string(float_format="%3.3f"))
        fid.write("\n")

@transform(compute_analysis, suffix('.results.all'), '.scatter.sdp-v-atoms.png')
def generate_figures(infile, outfile):
    import matplotlib.pyplot as plt

    print(f'reading infile: {infile}')
    mol_df = pd.read_parquet(f'{DATA_DIR}/data.processed/nist17_hconfspcl_64a_512.mols')
    metrics_df = pd.read_parquet(infile)

    mol_df = mol_df[['mol_id', 'rdmol']]
    print(f'starting with {len(metrics_df)} rows in metrics df')
    joined_df = metrics_df.set_index('mol_id').join(mol_df.set_index('mol_id'), rsuffix='_')
    mask = pd.isnull(joined_df).sum(axis=1) == 0
    joined_df = joined_df[mask]
    print(f'ending with {len(joined_df)} rows after joining with mols df')
    if len(joined_df) == 0:
        import click
        # TODO: fix issue where we can't handle smallmols perf due to not knowing
        # which dataframe to join with
        click.secho('Halting w empty dataframe after joining', fg='red')
        return

    joined_df['n_atoms'] = joined_df['rdmol'].map(lambda r: Chem.Mol(r).GetNumAtoms())

    ys = joined_df[['n_atoms', 'sdp']].groupby(by='n_atoms').sdp.mean()
    dys = joined_df[['n_atoms', 'sdp']].groupby(by='n_atoms').sdp.std() * 1.5
    
    plt.figure(figsize=(8, 6), dpi=100)
    plt.scatter(joined_df['n_atoms'], joined_df['sdp'], s=0.2)
    plt.xlabel('n_atoms')
    plt.ylabel('SDP')
    plt.xlim([np.min(ys.index), np.max(ys.index)])
    plt.ylim([0, 1])
    plt.savefig(outfile)
    plt.clf()

    plt.plot(ys.index, ys)
    plt.fill_between(ys.index, ys - dys, ys + dys, alpha=0.2)
    plt.xlabel('n_atoms')
    plt.ylabel('SDP')
    plt.xlim([np.min(ys.index), np.max(ys.index)])
    plt.ylim([0, 1])
    plt.savefig(outfile.replace('.scatter.', '.smooth.'))

if __name__ == "__main__":
    # pipeline_run([compute_analysis, summarize_results, generate_figures], multiprocess=16)
    pipeline_run([compute_analysis, summarize_results, generate_figures])
    # pipeline_run([compute_analysis, summarize_results], multiprocess=16)
    # pipeline_run([compute_analysis, summarize_results])
