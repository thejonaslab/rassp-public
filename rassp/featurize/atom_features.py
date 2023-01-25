"""
Per-atom features : Featurizations that return one per atom
[in contrast to whole-molecule featurizations]
"""

import pandas as pd
import numpy as np
import sklearn.metrics
import torch
from numba import jit
import scipy.spatial
from rdkit import Chem

import logging
logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from .util import get_nos_coords, get_nos


def atom_adj_mat(mol, conformer_i, **kwargs):
    """
    OUTPUT IS ATOM_N x (adj_mat, tgt_atom, atomic_nos, dists )
    
    This is really inefficient given that we explicitly return the same adj
    matrix for each atom, and index into it
    
    Adj mat is valence number * 2
    
    
    """
    
    MAX_ATOM_N = kwargs.get('MAX_ATOM_N', 64)
    atomic_nos, coords = get_nos_coords(mol, conformer_i)
    ATOM_N = len(atomic_nos)

    atomic_nos_pad, adj = mol_to_nums_adj(mol, MAX_ATOM_N)
    

    features = np.zeros((ATOM_N,), 
                    dtype=[('adj', np.uint8, (MAX_ATOM_N, MAX_ATOM_N)), 
                           ('my_idx', np.int), 
                           ('atomicno', np.uint8, MAX_ATOM_N), 
                           ('pos', np.float32, (MAX_ATOM_N, 3,))])
    
    for atom_i in range(ATOM_N):
        vects = coords - coords[atom_i]
        features[atom_i]['adj'] = adj*2
        features[atom_i]['my_idx'] =  atom_i
        features[atom_i]['atomicno'] = atomic_nos_pad
        features[atom_i]['pos'][:ATOM_N] = vects
    return features

def advanced_atom_props(mol, conformer_i, **kwargs):
    import rdkit.Chem.rdPartialCharges
    pt = Chem.GetPeriodicTable()
    atomic_nos, coords = get_nos_coords(mol, conformer_i)
    mol = Chem.Mol(mol)
    Chem.SanitizeMol(mol, Chem.rdmolops.SanitizeFlags.SANITIZE_ALL, 
                     catchErrors=True)
    Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
    ATOM_N = len(atomic_nos)
    out = np.zeros(ATOM_N, 
                   dtype=[('total_valence', np.int),  
                          ('aromatic', np.bool), 
                          ('hybridization', np.int), 
                          ('partial_charge', np.float32),
                          ('formal_charge', np.float32), 
                          ('atomicno', np.int), 
                          ('r_covalent', np.float32),
                          ('r_vanderwals', np.float32),
                          ('default_valence', np.int),
                          ('rings', np.bool, 5), 
                          ('pos', np.float32, 3)])
    
      
    for i in range(mol.GetNumAtoms()):
        a = mol.GetAtomWithIdx(i)
        atomic_num = int(atomic_nos[i])
        out[i]['total_valence'] = a.GetTotalValence()
        out[i]['aromatic'] = a.GetIsAromatic()
        out[i]['hybridization'] = a.GetHybridization()
        out[i]['partial_charge'] = a.GetProp('_GasteigerCharge')
        out[i]['formal_charge'] = a.GetFormalCharge()
        out[i]['atomicno'] = atomic_nos[i]
        out[i]['r_covalent'] =pt.GetRcovalent(atomic_num)
        out[i]['r_vanderwals'] =  pt.GetRvdw(atomic_num)
        out[i]['default_valence'] = pt.GetDefaultValence(atomic_num)
        out[i]['rings'] = [a.IsInRingSize(r) for r in range(3, 8)]
        out[i]['pos'] = coords[i]
                          
    return out

HYBRIDIZATIONS = [Chem.HybridizationType.UNSPECIFIED,
                  Chem.HybridizationType.S, 
                  Chem.HybridizationType.SP, 
                  Chem.HybridizationType.SP2, 
                  Chem.HybridizationType.SP3, 
                  Chem.HybridizationType.SP3D, 
                  Chem.HybridizationType.SP3D2,
                  Chem.HybridizationType.OTHER
                  ]

CHI_TYPES = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED, 
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]


def to_onehot(x, vals):
    return [x == v for v in vals]
        
MMFF94_ATOM_TYPES = [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 
                      11, 12, 15, 16, 17, 18, 20,
                      21, 22, 23, 24, 25, 26, 27, 28, 29, 
                      30, 31, 32, 33, 37, 38, 39, 40,
                      42, 43, 44, 46, 48, 59, 62, 63, 64, 
                      65, 66, 70, 71, 72, 74, 75, 78]


ELECTRONEGATIVITIES = {1: 2.20,
                       6: 2.26,
                       7: 3.04,
                       8: 3.44,
                       9: 3.98,
                       15: 2.19,
                       16: 2.58,
                       17: 3.16}
                       
                       
                  
def feat_tensor_atom(
    mol, 
    feat_pos=True, 
    feat_atomicno=True,
    feat_atomicno_onehot=[1, 6, 7, 8, 9], 
    feat_valence=True,
    total_valence_onehot=False, 
    aromatic=True, 
    hybridization=True, 
    partial_charge=False,
    formal_charge=True, 
    r_covalent=True,
    r_vanderwals=True,
    default_valence=True, 
    rings=False, 
    mmff_atom_types_onehot=False,
    max_ring_size=8,
    rad_electrons=False,
    chirality=False,
    assign_stereo=False,
    electronegativity=False,
    DEBUG_fchl=False,
    total_h=False,
    total_h_oh=True, 
    conf_idx=0
):
    """
    Featurize a molecule on a per-atom basis
    feat_atomicno_onehot : list of atomic numbers

    Always assume using conf_idx unless otherwise passed

    Returns an (ATOM_N x feature) float32 tensor

    NOTE: Performs NO santization or cleanup of molecule, 
    assumes all molecules have sanitization calculated ahead
    of time. 

    """

    pt = Chem.GetPeriodicTable()
    mol = Chem.Mol(mol) # copy molecule

    if feat_pos:
        atomic_nos, coords = get_nos_coords(mol, conf_idx)
    else:
        atomic_nos = get_nos(mol)

    ATOM_N = len(atomic_nos)    

    if partial_charge:
        Chem.rdPartialCharges.ComputeGasteigerCharges(mol)

    atom_features = []
    if mmff_atom_types_onehot:
        mmff_p = Chem.rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)

    if assign_stereo:
        Chem.rdmolops.AssignStereochemistryFrom3D(mol)

    if DEBUG_fchl:
        import qml
        assert feat_pos
    
        fchl_rep = qml.fchl.generate_representation(coords, atomic_nos, max_size=64)
        fchl_rep = fchl_rep.reshape(fchl_rep.shape[0], -1)
        
    for i in range(mol.GetNumAtoms()):
        a = mol.GetAtomWithIdx(i)
        atomic_num = int(atomic_nos[i])
        atom_feature = []

        if feat_atomicno:
            to_append = [atomic_num]
            logger.debug(f'adding {len(to_append)} feats for feat_atomicno')
            atom_feature += to_append

        if feat_pos:
            to_append = coords[i].tolist()
            logger.debug(f'adding {len(to_append)} feats for feat_pos')
            atom_feature += to_append

        if feat_atomicno_onehot is not None:
            to_append = to_onehot(atomic_num, feat_atomicno_onehot)
            logger.debug(f'adding {len(to_append)} feats for feat_atomicno_onehot')
            atom_feature += to_append

        if feat_valence:
            to_append = [a.GetTotalValence()]
            logger.debug(f'adding {len(to_append)} feats for feat_valence')
            atom_feature += to_append

        if total_valence_onehot:
            to_append = to_onehot(a.GetTotalValence(), range(1, 7))
            logger.debug(f'adding {len(to_append)} feats for total_valence_onehot')
            atom_feature += to_append

        if aromatic:
            to_append = [a.GetIsAromatic()]
            logger.debug(f'adding {len(to_append)} feats for aromatic')
            atom_feature += to_append

        if hybridization:
            to_append = to_onehot(a.GetHybridization(), HYBRIDIZATIONS)
            logger.debug(f'adding {len(to_append)} feats for hybridization')
            atom_feature += to_append

        if partial_charge:
            gc = float(a.GetProp('_GasteigerCharge'))
            if not np.isfinite(gc):
                gc = 0.0
            
            to_append = [gc]
            logger.debug(f'adding {len(to_append)} feats for partial_charge')
            atom_feature += to_append

        if formal_charge:
            to_append = to_onehot(a.GetFormalCharge(), [-1, 0, 1])
            logger.debug(f'adding {len(to_append)} feats for formal_charge')
            atom_feature += to_append

        if r_covalent:
            to_append = [pt.GetRcovalent(atomic_num)]
            logger.debug(f'adding {len(to_append)} feats for r_covalent')
            atom_feature += to_append

        if r_vanderwals:
            to_append = [pt.GetRvdw(atomic_num)]
            logger.debug(f'adding {len(to_append)} feats for r_vanderwals')
            atom_feature += to_append

        if default_valence:
            to_append = to_onehot(pt.GetDefaultValence(atomic_num), range(1, 7))
            logger.debug(f'adding {len(to_append)} feats for default_valence')
            atom_feature += to_append

        if rings:
            to_append = [a.IsInRingSize(r) for r in range(3, max_ring_size)]
            logger.debug(f'adding {len(to_append)} feats for rings')
            atom_feature += to_append

        if rad_electrons:
            if a.GetNumRadicalElectrons() > 0:
                raise ValueError("RADICAL")

        if chirality:
            to_append = to_onehot(a.GetChiralTag(), CHI_TYPES)
            logger.debug(f'adding {len(to_append)} feats for chirality')
            atom_feature += to_append

        if electronegativity:
            to_append = [ELECTRONEGATIVITIES[atomic_num]]
            logger.debug(f'adding {len(to_append)} feats for electronegativity')
            atom_feature += to_append

        if mmff_atom_types_onehot:
            if mmff_p is None:
                to_append = [0] * len(MMFF94_ATOM_TYPES)
            else:
                to_append = to_onehot(mmff_p.GetMMFFAtomType(i), 
                                          MMFF94_ATOM_TYPES)

            logger.debug(f'adding {len(to_append)} feats for mmff_atom_types_onehot')
            atom_feature += to_append

        if DEBUG_fchl:
            fchl_val = fchl_rep[i]
            fchl_val[fchl_val > 1e5] = 0
            
            to_append = fchl_val.tolist()
            logger.debug(f'adding {len(to_append)} feats for fchl')
            atom_feature += to_append

        if total_h or total_h_oh:
            attached_h = np.sum([an.GetAtomicNum() == 1 for an in a.GetNeighbors()])
            explicit = a.GetNumExplicitHs()
            implicit = a.GetNumImplicitHs()
            h_num = attached_h + explicit + implicit
            if total_h:
                to_append = [h_num]
            if total_h_oh:
                to_append = to_onehot(h_num, [0, 1, 2, 3, 4, 5])
            logger.debug(f'adding {len(to_append)} feats for total_h')
            atom_feature += to_append
            
        atom_features.append(atom_feature)

    return torch.Tensor(atom_features)

