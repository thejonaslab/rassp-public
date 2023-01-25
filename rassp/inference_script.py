import gzip
import io
import json 
import numpy as np
import pandas as pd
import pickle
import sys
import time
import torch
import warnings
import os
import click

from datetime import datetime
from rdkit import Chem
from urllib.parse import urlparse
from tqdm import tqdm

import netutil
from util import num_unique_frag_formulae

from sqlalchemy import create_engine

warnings.filterwarnings("ignore")

MAX_N_FORMULA = 32768  # 30000
MAX_N_ATOMS = 64  # 56
MAX_VERT_SUBSETS = 49152  # 49152

DEFAULT_MODEL = {
    'meta': '/data/richardzhu/eimspred/simpleforward/models/formulanet_best_candidate_pcsim_pretrain.nist-fromscratch-3x9x128.35790555.meta',
    'checkpoint': '/data/richardzhu/eimspred/simpleforward/models/formulanet_best_candidate_pcsim_pretrain.nist-fromscratch-3x9x128.35790555.00000740.model',
}

"""
PubChem funcs

78888677 mols
max per job = 80000
987 jobs needed.
round up to 1000.
"""
def get_input_mols(input_filename, job_ix: int, jobs_needed: int=1000):
    engine = create_engine(f'sqlite:///{input_filename}')
    conn = engine.connect()

    # count = conn.execute('select count(*) from molecules').fetchall()[0][0]
    return conn.execute(f'select * from molecules where id % {jobs_needed} == {job_ix}').fetchall()

def predict_mols(
    raw_mols: list,
    predictor: netutil.PredModel,

    max_n_atoms: int=48,
    max_n_formula: int=4096,

    # standard settings for preprocessing mols
    add_h=True,
    sanitize=True, 

    batch_size=32,
    num_workers=0,
) -> list:
    """
    Given a list of RDKit.Mol objects, predict simpleforward model outputs.

    FUTURE: In the future we will want to pass other metadata associated 
    w each molecule.
    """

    t1 = time.time()
        
    # filter out mols we can't predict over
    # NOTE: we must restrict the max number of atoms...
    formula_limits = np.array([50, 46, 30, 30, 30, 30, 30, 30])
    atom_nums = [1, 6, 7, 8, 9, 15, 16, 17]
    assert len(formula_limits) == len(atom_nums)
    ind_map = {anum: i for i, anum in enumerate(atom_nums)}

    print('Filtering molecules...')
    to_skip = []
    for i, m in tqdm(enumerate(raw_mols), total=len(raw_mols)):
        try:
            if add_h:
                m = Chem.AddHs(m)
            else:
                m = Chem.Mol(Chem.RemoveHs(m))  # copy

            if sanitize:
                Chem.SanitizeMol(m)

            n_atoms = m.GetNumAtoms()
            n_formula = num_unique_frag_formulae(m)

            # get counts of each atomic num
            cvec = np.zeros(len(formula_limits)).astype(np.int16)
            for anum, count in zip(*np.unique(
                [m.GetAtomWithIdx(ai).GetAtomicNum() for ai in range(m.GetNumAtoms())],
                return_counts=True
            )):
                cvec[ind_map[anum]] = count

            if n_atoms > max_n_atoms:
                # print(f'molecule has {n_atoms} atoms > {max_n_atoms}, skipping')
                skip = True
            elif n_formula > max_n_formula:
                # print(f'molecule has {n_formula} formula > {max_n_formula}, skipping')
                skip = True
            elif np.any(cvec >= formula_limits):
                print(f'molecule exceeds formula limits: {cvec} > {formula_limits}, skipping')
                skip = True
            else:
                skip = False
        except:
            print(f'ran into error sanitizing molecule @ index {i}, skipping.')
            print(f'mol: {Chem.MolToSmiles(m)}')
            skip = True

        to_skip.append(skip)
    
    assert len(to_skip) == len(raw_mols)
    filtered_mols = []
    for skip, m in zip(to_skip, raw_mols):
        if not skip:
            filtered_mols.append(m)

    # TODO: REMOVE
    # cutoff_ind = 10000
    # to_take = 5000
    # print(f'skipping up to index {cutoff_ind}')
    # filtered_mols = filtered_mols[cutoff_ind:cutoff_ind + to_take]

    # print(f'Printing {cutoff_ind} to {cutoff_ind + to_take}')
    # for mol in filtered_mols:
    #     print(Chem.MolToSmiles(mol))

    print('Beginning inference...')
    pred_t1 = time.time()
    predictions = predictor.pred(
        filtered_mols,
        progress_bar=True,
        normalize_pred=True,
        output_hist_bins=True,
        batch_size=batch_size,
        dataloader_config={
            'pin_memory': False,
            'num_workers': num_workers,
            'persistent_workers': False,
        },
        benchmark_dataloader=False,
    )
    pred_t2 = time.time()
    print("Prediction over {} mols took {:3.2f} ms".format(
        len(filtered_mols),
        (pred_t2-pred_t1)*1000),
    )

    # sparsify preds
    pred_spects = [p.tolist() for p in predictions['pred_binned']]

    # interleave the spectra on predicted mols with the ones we ignored
    spects = [ [] for _ in range(len(to_skip)) ]
    pred_inds = np.where(~np.array(to_skip))[0]
    assert len(pred_inds) == len(pred_spects)
    for i, spect in zip(pred_inds, pred_spects):
        spects[i] = spect

    out_records = []
    for mol_i, (spect, mol) in enumerate(zip(spects, raw_mols)):
        try:
            smiles = Chem.MolToSmiles(mol)
            inchi = Chem.MolToInchi(mol)
        except Exception as e:
            print(str(e))
            print('Skipping bad molecule...')
            print(f'Mol ind: {mol_i}')
            print(f'Spect: {spect}')
            # raise e

            # TODO
            smiles = ''
            inchi = ''
            spect = []

        out = {
            'success': False if len(spect) == 0 else True,
            'smiles': smiles,
            'inchi': inchi,
            'spect': spect,
        }
        out_records.append(out)

    assert len(out_records) == len(raw_mols)
    return out_records


def s3_split(url):
    o = urlparse(url)
    bucket = o.netloc
    key = o.path.lstrip('/')
    return bucket, key

def default_predict(mols: list, cuda=True):
    """
    Prediction function that directly returns spectra.
    """
    # TODO: cleanup mols here.

    ts_start = time.time()

    # defaults
    model_meta_filename = DEFAULT_MODEL['meta']
    model_checkpoint_filename = DEFAULT_MODEL['checkpoint']

    meta = pickle.load(open(model_meta_filename, 'rb'))
    # max_n_atoms = meta['max_n']
    max_n_atoms = MAX_N_ATOMS
    max_n_formula = MAX_N_FORMULA

    cuda_attempted = cuda
    if cuda and not torch.cuda.is_available():
        warnings.warn("CUDA requested but not available, running with CPU")
        cuda = False
    predictor = netutil.PredModel(
        model_meta_filename, 
        model_checkpoint_filename,
        USE_CUDA=cuda,
        data_parallel=True,  # FUTURE put this flag into the YAML file
        featurize_config_update={
            'MAX_N': max_n_atoms,
            'vert_subset_samples_n': MAX_VERT_SUBSETS,
            'explicit_formulae_config': {
                'max_formulae': max_n_formula,
            }
        },
    )

    print('running inference...')
    if len(mols) > 0:
        all_results = predict_mols(
            mols, predictor,
            max_n_atoms, max_n_formula,
            add_h=False, sanitize=True,
            batch_size=1, num_workers=16,
        )
    else:
        all_results = []
    
    n_empty_results = len(list(filter(lambda x: not x['success'], all_results)))
    n_total_results = len(all_results)

    ts_end = time.time()
    output_dict = {
        'predictions': all_results, 
        'meta' : {
            'max_n_atoms': max_n_atoms,
            'max_n_formula': max_n_formula,
            'model_checkpoint_filename' : model_checkpoint_filename, 
            'model_meta_filename' : model_meta_filename, 
            'ts_start' : datetime.fromtimestamp(ts_start).isoformat(), 
            'ts_end': datetime.fromtimestamp(ts_end).isoformat(), 
            'runtime_sec' : ts_end - ts_start,
            'git_commit' : os.environ.get("GIT_COMMIT", ""),

            'rate_mol_sec' : (n_total_results - n_empty_results) / (ts_end - ts_start),
            'num_mol' : (n_total_results - n_empty_results), 
            'num_empty_mol': n_empty_results,
            'total_mol': n_total_results,

            'cuda_attempted' : cuda_attempted, 
            'use_cuda' : cuda,
        }
    }
    return output_dict

@click.command()
@click.option(
    '--input_filename', help='filename of pubchem sqlite to read',
    # default='/data/richardzhu/data/pubchem-64a-hconfspcl.db'
    default='/net/scratch/richardzhu/pubchem-64a-hconfspcl.db'
)
@click.option('--job_ix', help='job index (to reduce amount of mols considered)', type=int)
@click.option('--model_meta_filename')
@click.option('--model_checkpoint_filename')
@click.option('--addhs', help="Add Hs to the input molecules", default=False)
@click.option('--cuda/--no-cuda', default=False)
@click.option('--num_data_workers', default=0, type=click.INT)
@click.option('--output', default=None)
@click.option('--print_data', default=None, help='print the smiles/fingerprint of the data used for train or test') 
@click.option('--sanitize/--no-sanitize', help="sanitize the input molecules", default=True)
@click.option("--version", default=False, is_flag=True)
def predict_pubchem(
    input_filename,
    job_ix,
    model_meta_filename, 
    model_checkpoint_filename,
    addhs=False,
    cuda=False, 
    num_data_workers=0,
    output=None,
    print_data=None,
    sanitize=True,
    version=False,
):
    click.secho(f'STARTING JOB {job_ix}', fg='blue')
    assert output is not None, 'output cannot be none!'

    ts_start = time.time()
    if version:
        print(os.environ.get("GIT_COMMIT", ""))
        sys.exit(0)
        
    if model_meta_filename is None:
        # defaults
        model_meta_filename = DEFAULT_MODEL['meta']
        model_checkpoint_filename = DEFAULT_MODEL['checkpoint']

    if print_data is not None:
        data_info_filename = model_meta_filename.replace(".meta", "." + print_data + ".json")
        print(open(data_info_filename, 'r').read())
        sys.exit(0)

    meta = pickle.load(open(model_meta_filename, 'rb'))
    max_n_atoms = meta['max_n']
    max_n_formula = MAX_N_FORMULA

    cuda_attempted = True
    if cuda and not torch.cuda.is_available():
        warnings.warn("CUDA requested but not available, running with CPU")
        cuda = False
    predictor = netutil.PredModel(
        model_meta_filename, 
        model_checkpoint_filename,
        USE_CUDA=cuda,
        data_parallel=True,  # FUTURE put this flag into the YAML file
        featurize_config_update={
            'MAX_N': MAX_N_ATOMS,
            'vert_subset_samples_n': MAX_VERT_SUBSETS,
            'explicit_formulae_config': {
                'max_formulae': MAX_N_FORMULA,
            }
        },
    )

    # metadata of table in `pubchem-process/src/main.rs`
    print('getting input mols from pubchem db...')
    rows = get_input_mols(input_filename, job_ix)
    mols = [Chem.MolFromInchi(r[3]) for r in rows]
    mols = list(filter(lambda m: m is not None, mols))
    print(f'got {len(mols)} mols')

    print('running inference...')
    if len(mols) > 0:
        all_results = predict_mols(
            mols, predictor,
            max_n_atoms, max_n_formula,
            add_h=addhs, sanitize=sanitize,
            batch_size=1, num_workers=num_data_workers,
        )
    else:
        all_results = []
    
    n_empty_results = len(list(filter(lambda x: not x['success'], all_results)))
    n_total_results = len(all_results)

    ts_end = time.time()
    output_dict = {
        'predictions': all_results, 
        'meta' : {
            'max_n_atoms': max_n_atoms,
            'max_n_formula': max_n_formula,
            'model_checkpoint_filename' : model_checkpoint_filename, 
            'model_meta_filename' : model_meta_filename, 
            'ts_start' : datetime.fromtimestamp(ts_start).isoformat(), 
            'ts_end': datetime.fromtimestamp(ts_end).isoformat(), 
            'runtime_sec' : ts_end - ts_start,
            'git_commit' : os.environ.get("GIT_COMMIT", ""),

            'rate_mol_sec' : (n_total_results - n_empty_results) / (ts_end - ts_start),
            'num_mol' : (n_total_results - n_empty_results), 
            'num_empty_mol': n_empty_results,
            'total_mol': n_total_results,

            'cuda_attempted' : cuda_attempted, 
            'use_cuda' : cuda,
        }
    }

    json_str = json.dumps(output_dict, sort_keys=False, indent=4)
    if output is None:
        print(json_str)
    else:
        if output.startswith('s3://'):
            bucket, key = s3_split(output)
            s3 = boto3.client('s3')

            json_bytes = json_str.encode('utf-8')
            if key.endswith(".gz"):
                json_bytes = gzip.compress(json_bytes)
            
            output_fileobj = io.BytesIO(json_bytes)
            s3.upload_fileobj(output_fileobj, bucket, key)

        else:
            with open(output, 'w') as fid:
                fid.write(json_str)

if __name__ == "__main__":
    # predict()
    predict_pubchem()
