"""
Pipeline for generating predictions for simpleforward models
"""
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import time
import pickle
import os
from tqdm import tqdm
from glob import glob
import copy
from ruffus import * 
import zlib
from rdkit import Chem

import logging
logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy import sql
from sqlalchemy import Integer, String, Column, LargeBinary

import netutil
from const import FORWARD_EVAL_EXPERIMENTS as EXPERIMENTS
from util import spect_list_to_string

PRED_DIR = "forward.preds"

td = lambda x : os.path.join(PRED_DIR, x)

CV_METHOD_DEFAULT =   {'how': 'morgan_fingerprint_mod', 
                       'mod' : 10, 
                       'test' : [0, 1]}

def params():
    for exp_name, ec in EXPERIMENTS.items():
        assert ec.get('streaming_save', False), \
            'we only support inference with streaming save to sqlite now'
        
        outfiles = [
            os.path.join(PRED_DIR, f"{exp_name}.spect.sqlite"),
            os.path.join(PRED_DIR, f"{exp_name}.done"),
        ]
        yield None, outfiles, ec, exp_name

@mkdir(PRED_DIR)
@files(params)
def run_exp(infile, outfiles, ec, exp_name):
    outfile, done_file = outfiles

    logger.info(f'Generating {outfile}')
    logger.info(f'Running against checkpoint {ec.get("checkpoint")}')

    checkpoint_base = ec['checkpoint']
    meta_filename = checkpoint_base + ".meta"
    checkpoint_filename = f"{checkpoint_base}.{ec['epoch']:08d}.model"
    assert os.path.exists(checkpoint_filename), f'{checkpoint_filename} does not exist'
    
    # create outfile sqlite and metadata
    engine = create_engine(f'sqlite:///{outfile}', echo=False, connect_args={'timeout': 120})
    metadata = MetaData()

    mol_id_ptype = ec.get('mol_id_type', int)
    mol_id_stype = String if mol_id_ptype == str else Integer
    logger.info(f'mol_id column type will be: {mol_id_stype}')

    table = Table(
        'spect', metadata,
        Column('mol_id', mol_id_stype, primary_key=True),
        Column('spect', LargeBinary),
        Column('phase', String),
    )
    metadata.create_all(engine)

    # load data from provided parquet
    df = pd.read_parquet(ec['dataset'])
    if 'max_data_n' in ec:
        df = df.sample(ec['max_data_n'], random_state=0)
    df['rdmol'] = df.rdmol.apply(Chem.Mol)

    assert np.all(df['mol_id'].map(lambda x: isinstance(x, mol_id_ptype))), \
        f'mol_ids must be {mol_id_ptype}'

    # make sure mol_ids are unique
    unique_df = df.drop_duplicates(subset=['mol_id'])
    logger.info(f'Removed {len(df) - len(unique_df)} duplicates, now have {len(unique_df)} mols remaining')
    df = unique_df

    # filter out existing mol_ids already in table
    conn = engine.connect()
    mol_ids_we_have = [x[0] for x in conn.execute('select mol_id from spect').fetchall()]
    conn.close()
    mask = ~df.mol_id.isin(set(mol_ids_we_have))
    df = df[mask]
    logger.info(f'Ignoring {len(mask) - sum(mask)} mols already in sqlite')

    cv_splitter = netutil.CVSplit(**ec['cv_method'])
    
    # do additional filter by phase (throw out null)
    df['phase'] = df.apply(lambda row : cv_splitter.get_phase(row.rdmol, 
                                                        row.morgan4_crc32), 
                      axis=1)
    mask = df.phase.isin(['train', 'test'])
    df = df[mask]
    logger.info(f'Filtering for phase reduced from {len(df)} to {sum(mask)}')
    
    use_cuda = int(os.environ.get("USE_CUDA", "1") ) > 0
    pm = netutil.PredModel(meta_filename, checkpoint_filename,
                           USE_CUDA=use_cuda, data_parallel=ec.get('data_parallel', False))

    N = len(df)
    BATCH_SIZE = 500
    slices = [slice(i * BATCH_SIZE, (i + 1) * BATCH_SIZE, None) for i in range(N // BATCH_SIZE + 1)]

    conn = engine.connect()
    for s in tqdm(slices):
        df_slice = df.iloc[s]

        # do inference in batches
        predictions = pm.pred(
            df_slice.rdmol.tolist(),
            batch_size=ec.get('batch_size', 32),
            progress_bar=True,
            normalize_pred=ec.get('normalize_pred', False),
            output_hist_bins=True,
            dataloader_config={
                'pin_memory': False,
                'num_workers': 32,
                'persistent_workers': True,
            },
        )

        # sparsify preds
        pred_spects = [p.tolist() for p in predictions['pred_binned']]

        out_records = []
        for pred_spect, record in zip(pred_spects, df_slice.to_records()):
            out = {
                'mol_id': mol_id_ptype(record['mol_id']),
                'spect': spect_list_to_string(pred_spect),
                'phase': record['phase'],
            }
            out_records.append(out)
        
        # write this batch to sqlite, overwriting any previous result
        # if this line errors with type issues, you may need to delete the .sqlite file and start over
        conn.execute(table.insert().prefix_with('OR REPLACE'), out_records)

        # n_records = conn.execute('select count(*) from spect').fetchone()[0]
        # logger.info(f'table now has {n_records} records')

    conn.close()

    # create 'done' file which tells us to complete this job
    from pathlib import Path
    Path(done_file).touch()

if __name__ == "__main__":
    pipeline_run([run_exp])
