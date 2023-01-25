import contextlib
import os
import numpy as np
import tempfile

from rdkit import Chem
import rdkit
import math
import scipy.optimize
import pandas as pd
import re 
import itertools
import time
import numba
import torch
import io
import zlib
import pickle

import collections
import scipy.optimize
import scipy.special
import scipy.spatial.distance
from tqdm import tqdm

Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

@contextlib.contextmanager
def cd(path):
   old_path = os.getcwd()
   os.chdir(path)
   try:
       yield
   finally:
       os.chdir(old_path)

def array_to_conf(mat):
    """
    Take in a (N, 3) matrix of 3d positions and create
    a conformer for those positions. 
    
    ASSUMES atom_i = row i so make sure the 
    atoms in the molecule are the right order!
    
    """
    N = mat.shape[0]
    conf = Chem.Conformer(N)
    
    for ri in range(N):
        p = rdkit.Geometry.rdGeometry.Point3D(*mat[ri])                                      
        conf.SetAtomPosition(ri, p)
    return conf

def add_empty_conf(mol):
   N = mol.GetNumAtoms()
   pos = np.zeros((N, 3))

   conf = array_to_conf(pos)
   mol.AddConformer(conf)

def numpy(x):
   """
   pytorch convenience method just to get a damn 
   numpy array back from a tensor or variable
   wherever the hell it lives
   """
   if isinstance(x, np.ndarray):
      return x
   if isinstance(x, list):
      return np.array(x)

   if isinstance(x, torch.Tensor):
      if x.is_cuda:
         return x.cpu().numpy()
      else:
         return x.numpy()
   raise NotImplementedError(str(type(x)))


def move(tensor, cuda=False):
    from torch import nn
    if cuda:
        if isinstance(tensor, nn.Module):
            return tensor.cuda()
        else:
            return tensor.cuda(non_blocking=True)
    else:
        return tensor.cpu()

def recursive_update(d, u):
   ### Dict recursive update 
   ### https://stackoverflow.com/a/3233356/1073963
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def morgan4_crc32(m):
   mf = Chem.rdMolDescriptors.GetHashedMorganFingerprint(m, 4)
   crc = zlib.crc32(mf.ToBinary())
   return crc
 
def get_formula(mol):
    """
    Return a dictionary of atomicno:num 
    """
    out = {}
    for a in mol.GetAtoms():
        an = a.GetAtomicNum()
        out[an] = out.get(an, 0) + 1
    return out

@numba.jit(nopython=True)
def fast_multi_onehot(x, oh_offsets, out_array, accum=False):
    """
    Fast structured one-hot encoder. X is a N x C array of values, oh_offsets[i] is
    the col offset into the out_array to begin encoding for x[:, i]
    accum controls if the oh is cumulative [1111000] or not. 
    """
    for row_i, row in enumerate(x):
        for i, v in enumerate(row):
            if accum:
                out_array[row_i, oh_offsets[i] : oh_offsets[i] + v] =1 
            else:
                out_array[row_i, oh_offsets[i] + v] =1 
                     
def num_unique_frag_formulae(m):
    f = get_formula(Chem.AddHs(Chem.Mol(m)))
    return np.prod([v + 1 for v in f.values()])

def spect_to_bins(s, bins):
    """
    Convert each spectrum to a histogram with unit l1 norm. 
    """
    s = np.stack(s)
    h, _ =np.histogram(s[:, 0], bins, weights=s[:, 1])
    h = h / np.sum(h)
    return h

def check_if_mol_is_valid(mol, allowed_atoms=[]):
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    from molmass import Formula

    mol_form = CalcMolFormula(mol)
    comp = Formula(mol_form).composition()

    for c in comp:
        asymbol = c[0]
        if asymbol not in allowed_atoms:
            return False
    return True

def percentile(n):
    """
    Generate a percentile for pandas agg function. 
    From https://stackoverflow.com/a/54593214
    """

    def percentile_(x):
        return x.quantile(n)
    percentile_.__name__ = 'percentile_{:2.0f}'.format(n*100)
    return percentile_

def spect_list_to_string(spect_list):
    assert isinstance(spect_list, list)
    return pickle.dumps(spect_list)

def spect_list_from_string(spect_string: str):
    assert isinstance(spect_string, str)
    return pickle.loads(spect_string)

def read_molfile(path):
  if '.spect.sqlite' in path:
    # TODO: fix dupe code from analysis_pipeline
    from sqlalchemy import create_engine, MetaData, Table
    from sqlalchemy import sql

    # load sqlite spectra
    engine = create_engine(f'sqlite:///{path}', echo=False)
    metadata = MetaData()
    metadata.reflect(bind=engine)

    table = metadata.tables['spect']

    pred_spect_df = pd.read_sql(sql.select([table]), engine)
    pred_spect_df['spect'] = pred_spect_df.spect.map(lambda s: pickle.loads(s))
    return pred_spect_df
  else:
    return pd.read_parquet(path)
