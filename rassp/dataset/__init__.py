import numpy as np
import pandas as pd

import torch.utils.data
import pickle
from rdkit import Chem
from sqlalchemy import create_engine, MetaData, select, Table
import os

from rassp.featurize import create_mol_featurizer, create_pred_featurizer

class DBDataset:
    def __init__(
        self,
        db_filename,
        db_id_field,
        db_ids,
        spect_bin_config,
        featurizer_config,
        pred_featurizer_config,
    ):
        self.db_filename = db_filename
        self.db_id_field = db_id_field
        self.db_ids = db_ids

        self.featurizer = create_mol_featurizer(spect_bin_config, featurizer_config)
        self.pred_featurizer = create_pred_featurizer(spect_bin_config, pred_featurizer_config)

    def create_db_engine(self, db_filename):
        assert os.path.exists(db_filename)
        engine = create_engine(f"sqlite+pysqlite:///{db_filename}", future=True)
        return engine
    
    def __len__(self):
        return len(self.db_ids)
    
    def __getitem__(self, idx):
        
        engine = self.create_db_engine(self.db_filename)
        metadata_obj = MetaData()

        mol_table = Table("molecules", metadata_obj, autoload_with=engine)

        tgt_id = self.db_ids[idx]

        # load from database
        with engine.connect() as conn:
            table = mol_table
            stmt = select([table.c.bmol, table.c.num_spectrum_peaks, table.c.binary_spectrum]).where(table.c[self.db_id_field] == tgt_id)
            
            db_record = conn.execute(stmt).one()
            
        mol = Chem.Mol(db_record['bmol'])
        num_peaks = db_record['num_spectrum_peaks']
        sparse_spect_shape = (num_peaks, 2)

        spect_sparse = np.frombuffer(db_record['binary_spectrum'],
                                     dtype=np.float32).reshape(sparse_spect_shape)

        features_dict = self.featurizer(mol)
        preds_dict = self.pred_featurizer(mol, spect_sparse)
        out_dict = {
            **features_dict,
            **preds_dict,
        }
        out_dict['input_idx'] = idx

        return out_dict

class ParquetDataset:
    def __init__(
        self,
        filename,
        spect_bin_config,
        featurizer_config,
        pred_featurizer_config,
    ):
        self.df = pd.read_parquet(filename)

        required_cols = ['rdmol', 'spect']
        for col in required_cols:
            assert col in self.df.columns, f'{col} must be in df'

        self.featurizer = create_mol_featurizer(spect_bin_config, featurizer_config)
        self.pred_featurizer = create_pred_featurizer(spect_bin_config, pred_featurizer_config)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mol = Chem.Mol(row['rdmol'])
        spect = np.array(row['spect'])

        features_dict = self.featurizer(mol)
        preds_dict = self.pred_featurizer(mol, spect)
        out_dict = {
            **features_dict,
            **preds_dict,
        }
        out_dict['input_idx'] = idx
        return out_dict

class WrapperDataset:
    """
    Simple dataset to process mols
    """
    def __init__(
        self,
        mols,
        spect_bin_config,
        featurizer_config,
    ):
        self.featurizer = create_mol_featurizer(spect_bin_config, featurizer_config)
        self.mols = mols
        
    def __len__(self):
        return len(self.mols)
    
    def __getitem__(self, idx):
        mol = self.mols[idx]

        features_dict = self.featurizer(mol)
        out_dict = {
            **features_dict,
        }
        out_dict['input_idx'] = idx
        return out_dict

def filter_db_records(dataset_config, cv_splitter):
    """
    Filter out db records, filtering trhough the records as necessary. 
    """

    db_filename = dataset_config['db_filename']
    phase = dataset_config.get('phase', 'train')
    
    assert os.path.exists(db_filename)
    engine = create_engine(f"sqlite+pysqlite:///{db_filename}", future=True)
    metadata_obj = MetaData()
    print("reading from", db_filename)

    molecules = Table("molecules", metadata_obj, autoload_with=engine)

    cv_fp_field = dataset_config.get('cv_fp_field', 'morgan4_crc32')
    
    sql_stmt = select([molecules.c.id, molecules.c[cv_fp_field]])

    if 'filter_max_n' in dataset_config:
        sql_stmt = sql_stmt.where(molecules.c.atom_n <= dataset_config['filter_max_n'])

    if 'filter_max_mass' in dataset_config:
        sql_stmt = sql_stmt.where(molecules.c.mol_wt <= dataset_config['filter_max_mass'])

    if 'filter_max_unique_formulae' in dataset_config:
        sql_stmt = sql_stmt.where(molecules.c.unique_formulae <= dataset_config['filter_max_unique_formulae'])

    target_records = []
    with engine.connect() as conn:
        for row in conn.execute(sql_stmt):
            if cv_splitter.get_phase(None, row[cv_fp_field]) == phase:
                target_records.append(row['id'])

    return target_records

def make_db_dataset(dataset_config,
                    spect_bin_config, 
                    featurizer_config,
                    pred_config,
                    cv_splitter):
    db_ids = filter_db_records(dataset_config, cv_splitter)
    db_filename = dataset_config['db_filename']
    db_dataset = DBDataset(db_filename, 'id',
                           db_ids,
                           spect_bin_config,
                           featurizer_config,
                           pred_config)
    return db_dataset

def load_pq_dataset(dataset_config,
                    spect_bin_config, 
                    featurizer_config,
                    pred_config,
                    ):
    db_filename = dataset_config['db_filename']
    assert '.parquet' in db_filename or '.pq' in db_filename

    return ParquetDataset(
        db_filename,
        spect_bin_config,
        featurizer_config,
        pred_config,
    )
