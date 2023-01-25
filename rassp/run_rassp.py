"""
Predictor class + script for running EI-MS pred on input files, inputs can 
be smiles files, sdf files, or pickled rdmol objects from RDKit. 

Setup (internal):
```
git clone git@github.com:thejonaslab/eimspred.git
```

Setup (public):
```
git clone git@github.com:thejonaslab/rassp_public.git
```

Demo usage
- Ensure you are running within the eimspred root directory.
- Ensure that `rassp` module has been installed, e.g. by running `pip install -e  .` from the eimspred root directory.
- Put molecule smiles/inchi strings (all mol strings must be the same) into `./rassp/sample_data/in.txt`
- Ensure that your `rassp/models` directory contains the .meta and .model files for the model you would like to use.

Integration test:
```
# use GPU 0
CUDA_VISIBLE_DEVICES=0 rassp \
     "./rassp/sample_data/in_plus_kds2010.txt" \
     "./rassp/sample_data/out_plus_kds2010.txt" \
    --file-type smiles \
    --model-name "FormulaNet" --gpu

# CPU only, slower
CUDA_VISIBLE_DEVICES='' rassp \
    "./rassp/sample_data/in_plus_kds2010.txt" \
    "./rassp/sample_data/out_plus_kds2010.txt" \
    --file-type smiles \
    --model-name "FormulaNet" --no-gpu
```
"""
from typing import List, Union, Tuple, Dict

import logging
# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
logger = logging.getLogger(__name__)

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
from pathlib import Path

from datetime import datetime
from rdkit import Chem
from tqdm import tqdm

from rassp import netutil
from rassp.util import num_unique_frag_formulae

model_dir = os.path.join(str(Path(__file__).resolve().parent), 'models')

MODELS = {
    'FormulaNet': {
        'checkpoint': os.path.join(model_dir, 'formulanet_best_candidate_pcsim_pretrain.nist-fromscratch-3x9x128.35790555.00000740.model'),
        'meta': os.path.join(model_dir, 'formulanet_best_candidate_pcsim_pretrain.nist-fromscratch-3x9x128.35790555.meta'),

        # NOTE(2023-01-24): FN currently has a reduced set of valid mol constraints bc this is what we trained with
        # and there is currently an outstanding bug where changing N_ATOMS here breaks FN predictions in subtle ways.
        'override_constraints': {
            'n_atom': 48,
            # 'n_formula': 4096,
            'n_formula': 32768,
            # 'n_subset': 12288,
            'n_subset': 49152,
        },
    },
    'SubsetNet': {
        'checkpoint': os.path.join(model_dir, 'subsetnet_best_candidate_nist_fromscratch.nist-fromscratch-test01-old-1x2048.36688199.00001000.model'),
        'meta': os.path.join(model_dir, 'subsetnet_best_candidate_nist_fromscratch.nist-fromscratch-test01-old-1x2048.36688199.meta'),

        # NOTE: this set of params works for SN GPU inference on a single RTX 2080 Ti (12GB VRAM)
        # despite SN being trained on <= 48 atoms like FN, it also scales well to mols up to <= 64 atoms (tested)
        'override_constraints': {
            'n_atom': 64,
            'n_formula': 32768,
            'n_subset': 49152,
        },
    },
}

FILE_TYPES = ['smiles', 'inchi', 'sdf', 'rdmol']

class Predictor:
    def __init__(
        self,
        use_gpu=False,
        batch_size=4,
        num_workers=0,
    ):
        """
        Defines a EI-MS Predictor.

        Default settings:
        - Use FormulaNet.
        - Use CPU only.
        """
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    @staticmethod
    def is_valid_mol(m: Chem.Mol, max_n_atoms=48, max_n_formula=4096) -> Tuple[Union[Chem.Mol, None], str]:
        """
        Check if Molecule is valid, i.e. we can run inference on.
        """
        formula_limits = np.array([50, 46, 30, 30, 30, 30, 30, 30])
        atom_nums = [1, 6, 7, 8, 9, 15, 16, 17]
        assert len(formula_limits) == len(atom_nums)
        ind_map = {anum: i for i, anum in enumerate(atom_nums)}
        try:
            n_atoms = m.GetNumAtoms()
            anum_set = set([m.GetAtomWithIdx(i).GetAtomicNum() for i in range(n_atoms)])
            if not anum_set.issubset(atom_nums):
                reason = f'molecule {Chem.MolToSmiles(m)} invalid; ATOM_TYPE constraint violated'
                return (None, reason)

            if m is None:
                raise RuntimeError('mol was None')
            
            # always add Hs and sanitize
            m = Chem.AddHs(m)
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

            is_bad = False
            if n_atoms > max_n_atoms:
                reason = f'molecule {Chem.MolToSmiles(m)} invalid bc {n_atoms} atoms > {max_n_atoms}; MAX_N_ATOMS constraint violated'
                is_bad = True
            elif n_formula > max_n_formula:
                reason = f'molecule {Chem.MolToSmiles(m)} invalid bc {n_formula} formula > {max_n_formula}; MAX_N_FORMULA constraint violated'
                is_bad = True
            elif np.any(cvec >= formula_limits):
                reason = f'molecule {Chem.MolToSmiles(m)} invalid bc exceeds formula limits: {cvec} > {formula_limits}; FORMULA_LIMITS constraint violated'
                is_bad = True
            elif len(Chem.GetMolFrags(m)) > 1:
                reason = f'molecule {Chem.MolToSmiles(m)} invalid bc multiple fragments; SINGLE_MOL constraint violated'
                is_bad = True
            if is_bad:
                return (None, reason)
            else:
                return (m, '')
        except Exception as e:
            logger.debug(f'Failed to sanitize and validate molecule with exception: {e}')
            if m is not None:
                try:
                    smiles = Chem.MolToSmiles(m)
                    logger.debug(f'mol was: {smiles}')
                except Exception as e2:
                    logger.debug(f'failed to convert mol to smiles with exception {e2}')
            return (None, f'Failed with exception {e}')
    
    @staticmethod
    def load_model(model_name, use_gpu=False) -> Tuple[netutil.PredModel, dict]:
        logger.info(f'Loading model {model_name}')
        meta_path = MODELS[model_name]['meta']
        ckpt_path = MODELS[model_name]['checkpoint']
        override_constraints = MODELS[model_name]['override_constraints']
        assert os.path.exists(meta_path), f'meta not found at {meta_path}'
        assert os.path.exists(ckpt_path), f'ckpt not found at {ckpt_path}'

        # pull featurize_config from model meta file
        # and use it to init model featurizer
        # override with our provided values
        meta = pickle.load(open(meta_path, 'rb'))
        feat_config = meta['featurize_config']
        for key, val in override_constraints.items():
            if key == 'n_atom':
                feat_config['MAX_N'] = val
            if key == 'n_formula':
                feat_config['explicit_formulae_config']['max_formulae'] = val
            if key == 'n_subset':
                feat_config['vert_subset_samples_n'] = val

        if use_gpu and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, running with CPU")
            use_gpu = False

        predictor = netutil.PredModel(
            meta_path, 
            ckpt_path,
            USE_CUDA=use_gpu,
            data_parallel=False,
            featurize_config_update=feat_config,
        )
        return predictor, meta
    
    def predict(self, mols: List[Chem.Mol], model_name='FormulaNet') -> Tuple[List[Dict], Dict]:
        metadata = {}
        t1 = time.time()

        # load model
        assert model_name in MODELS, f'{model_name} not found in {MODELS.keys()}'
        predictor, meta = Predictor.load_model(model_name, use_gpu=self.use_gpu)

        # check mols for validity based on model metadata (overriden here)
        max_n_atoms = MODELS[model_name]['override_constraints']['n_atom']
        max_n_formula = MODELS[model_name]['override_constraints']['n_formula']

        filtered_mols = []
        for i, m in enumerate(mols):
            o, reason = Predictor.is_valid_mol(m, max_n_atoms=max_n_atoms, max_n_formula=max_n_formula)
            filtered_mols.append(o)
        is_valid = [o is not None for o in filtered_mols]
        valid_mols = list(filter(lambda m: m is not None, filtered_mols))

        logger.info(f'Beginning inference on {len(valid_mols)} mols...')
        pred_t1 = time.time()
        predictions = predictor.pred(
            valid_mols,
            progress_bar=True,
            normalize_pred=True,
            output_hist_bins=True,
            batch_size=self.batch_size,
            dataloader_config={
                'pin_memory': False,
                'num_workers': self.num_workers,
                'persistent_workers': False,
            },
            benchmark_dataloader=False,
        )
        pred_t2 = time.time()
        logger.info("Prediction over {} mols took {:3.2f} ms".format(
            len(valid_mols),
            (pred_t2-pred_t1)*1000),
        )
        metadata['pred_time_secs'] = pred_t2 - pred_t1

        out_records = []

        pred_i = 0 
        for mol_i, mol in enumerate(mols):
            try:
                smiles = Chem.MolToSmiles(mol)
                inchi = Chem.MolToInchi(mol)
            except Exception as e:
                logger.info(str(e))
                smiles = ''
                inchi = ''
                spect = []
            if is_valid[mol_i]:
                p  = predictions['pred_binned'][pred_i]
                spect = p.tolist()
                pred_i += 1
            else:
                spect = []
                
            out = {
                'success': True if len(spect) != 0 else False,
                'smiles': smiles,
                'inchi': inchi,
                'spect': spect,
            }
            out_records.append(out)
        t2 = time.time()

        metadata['n_records'] = len(out_records)
        metadata['n_successful'] = sum([o['success'] for o in out_records])
        metadata['total_time_secs'] = t2 - t1
        assert len(out_records) == len(mols), f'len(out_records) != len(mols)'
        return out_records, metadata

def run_on_file(
    input_filename,
    output_filename,
    file_type,
    output_file_type, 
    model_name,
    gpu,
    batch_size,
    num_data_workers,
):
    click.secho(f'Loading mols from provided file {input_filename}', fg='blue')
    assert os.path.exists(input_filename), f'input file not found at {input_filename}'

    t1 = time.time()
    # with open(input_filename, 'r') as fp:
    #     lines = [line.rstrip('\n') for line in fp.readlines()]
    # moltype = 'inchi' if 'InChI' in lines[0] else 'smiles'
    # click.secho(f'Found {len(lines)} rows, assuming {moltype}')

    # mol_loader = {
    #     'inchi': Chem.MolFromInchi,
    #     'smiles': Chem.MolFromSmiles,
    # }[moltype]
    # mols = [mol_loader(line) for line in lines]
    if file_type == 'smiles':
        mols = [Chem.AddHs(m) for m in Chem.SmilesMolSupplier(input_filename, titleLine=0)]
    elif file_type == 'inchi':
        mols = [Chem.AddHs(Chem.MolFromInchi(s.strip())) for s in open(input_filename, 'r').readlines()]
    elif file_type == 'sdf':
        mols = [Chem.AddHs(m) for m in Chem.SDMolSupplier(input_filename)]
    elif file_type == 'rdmol':
        mols = [Chem.Mol(m) for m in pickle.load(open(input_filename, 'rb'))]
    else:
        raise ValueError("unknown input type")
                
    click.secho(f'Loaded {len(mols)} mols', fg='blue')
    
    click.secho(f'Initializing Predictor', fg='blue')
    predictor = Predictor(
        use_gpu=gpu,
        batch_size=batch_size,
        num_workers=num_data_workers,
    )

    click.secho(f'Running RASSP forward model', fg='blue')
    all_results, pred_metadata = predictor.predict(mols, model_name=model_name)

    click.secho(f'Writing outputs to disk', fg='blue')
    n_empty_results = len(list(filter(lambda x: not x['success'], all_results)))
    n_total_results = len(all_results)

    t2 = time.time()
    if output_file_type == 'txt':
        # format for a mol spect is (separated by new line)
        # <smiles>
        # <inchi>
        # <mass_0> <intensity_0>
        # ...
        # <mass_n> <intensity_n>
        with open(output_filename, 'w') as fp:
            for result in all_results:
                if not result['success']:
                    continue

                smiles = result['smiles']
                fp.write(f'{smiles}\n')
                inchi = result['inchi']
                fp.write(f'{inchi}\n')

                # normalize spectrum to 1.0 sum
                tot_z = 0.0
                for m, z in result['spect']:
                    tot_z += z

                # write <mass> <intensity> rows
                for m, z in result['spect']:
                    fp.write(f'{int(m)} {float(z)}\n')
                fp.write('\n')
    elif output_file_type == 'json':
        json.dump({'predictions' : all_results,
                   'runtime' : t2-t1},
                  open(output_filename, 'w'))
        

    click.secho('Processing complete.', fg='green')

@click.command()
@click.argument('infile', default=None, required=True)
@click.argument('outfile',  default=None,  required=True)
@click.option('--file-type', help='Input file type', type=click.Choice(FILE_TYPES))
@click.option('--output-file-type', help='Output file type', default='txt', type=click.Choice(['txt', 'json']))
@click.option('--model-name', help='Model name', type=click.Choice(list(MODELS.keys())), default='FormulaNet')
@click.option('--gpu/--no-gpu', help='Use GPU for inference', default=False)
@click.option('--batch-size', help="batch size", default=4)
@click.option('--num-data-workers', help="number of data workers to use", default=4)
def run_predictions(
        infile,
        outfile,
        file_type,
        output_file_type, 
        model_name,
        gpu,
        batch_size,
        num_data_workers,
):
    """
    Run RASSP on input mols and produce output

    run_rassp input_filename output_filename args

    """
    run_on_file(
        infile,
        outfile,
        file_type,
        output_file_type, 
        model_name,
        gpu,
        batch_size,
        num_data_workers,
    )

if __name__ == '__main__':
    run_predictions()
