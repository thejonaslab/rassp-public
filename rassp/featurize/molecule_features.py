import pandas as pd
import numpy as np
import sklearn.metrics
import torch
from numba import jit
import scipy.spatial
from rdkit import Chem
from rdkit.Chem import AllChem
import networkx as nx
import rdkit.Chem.Descriptors
import sys

from .util import get_nos_coords, get_nos
from rassp.msutil import masscompute

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') 

def feat_tensor_mol(mol, feat_distances=False,
                    feat_r_pow = None,
                    feat_r_max = None,
                    feat_r_onehot_tholds = [],
                    feat_r_gaussian_filters = [],
                    conf_embed_mol = False,
                    conf_opt_mmff = False,
                    conf_opt_uff = False,
                    
                    is_in_ring=False,
                    is_in_ring_size = None, 
                    MAX_POW_M = 2.0,
                    conf_idx = 0,
                    add_identity=False,
                    edge_type_tuples = [],
                    adj_pow_bin = [],
                    adj_pow_scale = [],
                    graph_props_config = {},
                    frag_props_config = {}, 
                    columb_mat = False,
                    dihedral_mat = False, 
                    dihedral_sincos_mat = False, 
                    norm_mat=False, mat_power=1):
    """
    Return matrix features for molecule
    
    """
    res_mats = []
    mol_init = mol
    if conf_embed_mol:
        mol_change = Chem.Mol(mol)
        try:
            Chem.AllChem.EmbedMolecule(mol_change)
            if conf_opt_mmff:
                Chem.AllChem.MMFFOptimizeMolecule(mol_change)
            elif conf_opt_uff:
                Chem.AllChem.UFFOptimizeMolecule(mol_change)
            if mol_change.GetNumConformers() > 0:
                mol = mol_change
        except Exception as e:
            print('error generating conformer', e)

        
    assert mol.GetNumConformers() > 0
    if feat_distances or feat_r_pow:
        atomic_nos, coords = get_nos_coords(mol, conf_idx)
    else:
        atomic_nos = get_nos(mol)
    ATOM_N = len(atomic_nos)

    if feat_distances:
        pos = coords
        a = pos.T.reshape(1, 3, -1)
        b = np.abs((a - a.T))
        c = np.swapaxes(b, 2, 1)
        res_mats.append(c)
    if feat_r_pow is not None:
        pos = coords
        a = pos.T.reshape(1, 3, -1)
        b = (a - a.T)**2
        c = np.swapaxes(b, 2, 1)
        d = np.sqrt(np.sum(c, axis=2))
        e = (np.eye(d.shape[0]) + d)[:, :, np.newaxis]
        if feat_r_max is not None:
            d[d >= feat_r_max] = 0.0
                       
        for p in feat_r_pow:
            e_pow = e**p
            if (e_pow > MAX_POW_M).any():
                e_pow = np.minimum(e_pow, MAX_POW_M)

            res_mats.append(e_pow)
        for th in feat_r_onehot_tholds:
            e_oh = (e <= th).astype(np.float32)
            res_mats.append(e_oh)

        for mu, sigma in feat_r_gaussian_filters:
            
            e_val = np.exp(-(e - mu)**2/(2*sigma**2))
            res_mats.append(e_val)
            
    if len(edge_type_tuples) > 0:
        a = np.zeros((ATOM_N, ATOM_N, len(edge_type_tuples)))
        for et_i, et in enumerate(edge_type_tuples):
            for b in mol.GetBonds():
                a_i = b.GetBeginAtomIdx()
                a_j = b.GetEndAtomIdx()
                if set(et) == set([atomic_nos[a_i], atomic_nos[a_j]]):
                    a[a_i, a_j, et_i] = 1
                    a[a_j, a_i, et_i] = 1
        res_mats.append(a)
        
    if is_in_ring:
        a = np.zeros((ATOM_N, ATOM_N, 1), dtype=np.float32)
        for b in mol.GetBonds():
            a[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] = 1
            a[b.GetEndAtomIdx(), b.GetBeginAtomIdx()] = 1
        res_mats.append(a)
        
    if is_in_ring_size is not None:
        for rs in is_in_ring_size:
            a = np.zeros((ATOM_N, ATOM_N, 1), dtype=np.float32)
            for b in mol.GetBonds():
                if b.IsInRingSize(rs):
                    a[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] = 1
                    a[b.GetEndAtomIdx(), b.GetBeginAtomIdx()] = 1
            res_mats.append(a)
            
    if columb_mat:
        res_mats.append(np.expand_dims(get_columb_mat(mol, conf_idx), -1))

    if dihedral_mat:
        res_mats.append(np.expand_dims(get_dihedral_angles(mol, conf_idx), -1))
        
    if dihedral_sincos_mat:
        res_mats.append(get_dihedral_sincos(mol, conf_idx))
        
    if len(graph_props_config) > 0:
        res_mats.append(get_graph_props(mol, **graph_props_config))

        
    if len(frag_props_config) > 0:
        res_mats.append(get_single_bond_fragment_masses(mol, **frag_props_config))

    if len(adj_pow_bin) > 0:
        _, A = mol_to_nums_adj(mol)
        A = torch.Tensor((A > 0).astype(int))
        
        for p in adj_pow_bin:

            adj_i_pow = torch.clamp(torch.matrix_power(A, p),
                                    max=1)

            res_mats.append(adj_i_pow.unsqueeze(-1))

    if len(adj_pow_scale) > 0:
        _, A = mol_to_nums_adj(mol)
        A = torch.Tensor((A > 0).astype(int))
        
        for p in adj_pow_scale:

            adj_i_pow = torch.matrix_power(A, p) / 2**p

            res_mats.append(adj_i_pow.unsqueeze(-1))
            
    if len(res_mats) > 0:
        M = np.concatenate(res_mats, 2)
    else: # Empty matrix
        M = np.zeros((ATOM_N, ATOM_N, 0), dtype=np.float32)


    M = torch.Tensor(M).permute(2, 0, 1)
    
    if add_identity:
        M = M + torch.eye(ATOM_N).unsqueeze(0)

    if norm_mat:
        res = []
        for i in range(M.shape[0]):
            a = M[i]
            D_12 = 1.0 / torch.sqrt(torch.sum(a, dim=0))
            assert np.min(D_12.numpy()) > 0
            s1 = D_12.reshape(ATOM_N, 1)
            s2 = D_12.reshape(1, ATOM_N)
            adj_i = s1 * a * s2 

            if isinstance(mat_power, list):
                for p in mat_power:
                    adj_i_pow = torch.matrix_power(adj_i, p)

                    res.append(adj_i_pow)

            else:
                if mat_power > 1: 
                    adj_i = torch.matrix_power(adj_i, mat_power)

                res.append(adj_i)
        M = torch.stack(res, 0)

    assert np.isfinite(M).all()
    return M.permute(1, 2, 0) 

def mol_to_nums_adj(m, MAX_ATOM_N=None):
    """
    molecule to symmetric adjacency matrix
    """

    m = Chem.Mol(m)

    ATOM_N = m.GetNumAtoms()
    if MAX_ATOM_N is None:
        MAX_ATOM_N = ATOM_N

    adj = np.zeros((MAX_ATOM_N, MAX_ATOM_N))
    atomic_nums = np.zeros(MAX_ATOM_N)

    assert ATOM_N <= MAX_ATOM_N

    for i in range(ATOM_N):
        a = m.GetAtomWithIdx(i)
        atomic_nums[i] = a.GetAtomicNum()

    for b in m.GetBonds():
        head = b.GetBeginAtomIdx()
        tail = b.GetEndAtomIdx()
        order = b.GetBondTypeAsDouble()
        adj[head, tail] = order
        adj[tail, head] = order
    return atomic_nums, adj

def feat_mol_adj(
    mol, 
    edge_weighted=False, 
    edge_bin = False,
    add_identity=False,
    norm_adj=False,
    split_weights=None,
    mat_power=1,
):
    """
    Compute the adjacency matrix for this molecule

    If split-weights == [1, 2, 3] then we create separate adj matrices for those
    edge weights

    NOTE: We do not kekulize the molecule, we assume that has already been done

    """
    
    atomic_nos, adj = mol_to_nums_adj(mol)
    ADJ_N = adj.shape[0]
    input_adj = torch.Tensor(adj)
    
    adj_outs = []

    if edge_weighted:
        adj_weighted = input_adj.unsqueeze(0)
        adj_outs.append(adj_weighted)

    if edge_bin:
        adj_bin = input_adj.unsqueeze(0).clone()
        adj_bin[adj_bin > 0] = 1.0
        adj_outs.append(adj_bin)

    if split_weights is not None:
        split_adj = torch.zeros((len(split_weights), ADJ_N, ADJ_N ))
        for i in range(len(split_weights)):
            split_adj[i] = (input_adj == split_weights[i])
        adj_outs.append(split_adj)
    adj = torch.cat(adj_outs,0)

    if norm_adj and not add_identity:
        raise ValueError()
        
    if add_identity:
        adj = adj + torch.eye(ADJ_N)

    if norm_adj:
        res = []
        for i in range(adj.shape[0]):
            a = adj[i]
            D_12 = 1.0 / torch.sqrt(torch.sum(a, dim=0))

            s1 = D_12.reshape(ADJ_N, 1)
            s2 = D_12.reshape(1, ADJ_N)
            adj_i = s1 * a * s2 

            if isinstance(mat_power, list):
                for p in mat_power:
                    adj_i_pow = torch.matrix_power(adj_i, p)

                    res.append(adj_i_pow)
            else:
                if mat_power > 1: 
                    adj_i = torch.matrix_power(adj_i, mat_power)

                res.append(adj_i)
        adj = torch.stack(res)
    return adj


def whole_molecule_features(full_record, atom_type_counts=False):
    """
    return a vector of features for the full molecule 
    """
    out_feat = []
    if atom_type_counts:
        atom_counts = np.zeros(32, dtype=np.float32)
        for atom in full_record['rdmol'].GetAtoms():
            atom_counts[atom.GetAtomicNum()] +=1

        out_feat.append(atom_counts)

    if len(out_feat) == 0:
        return torch.Tensor([])
    return torch.Tensor(np.concatenate(out_feat).astype(np.float32))


def get_columb_mat(mol, conf_idx = 0):
    """
    from 
    https://github.com/cameronus/coulomb-matrix/blob/master/generate.py

    """

    n_atoms = mol.GetNumAtoms()
    m = np.zeros((n_atoms, n_atoms), dtype=np.float32)
    z, xyz = get_nos_coords(mol, conf_idx)
    
    for r in range(n_atoms):
      for c in range(n_atoms):
          if r == c:
              m[r][c] = 0.5 * z[r] ** 2.4
          elif r < c:
              v = z[r] * z[c] / np.linalg.norm(np.array(xyz[r]) - np.array(xyz[c])) * 0.52917721092
              m[r][c] = v
              m[c][r] = v
    return m

def dist_mat(mol,
             conf_idx = 0,
             feat_distance_pow = [{'pow' : 1,
                                   'max' : 10,
                                   'min' : 0,
                                   'offset' : 0.1}],
             mmff_opt_conf = False, 
             ):
    """
    Return matrix features for molecule
    
    """
    res_mats = []
    if mmff_opt_conf:
        Chem.AllChem.EmbedMolecule(mol)
        Chem.AllChem.MMFFOptimizeMolecule(mol)
    atomic_nos, coords = get_nos_coords(mol, conf_idx)
    ATOM_N = len(atomic_nos)

    pos = coords
    a = pos.T.reshape(1, 3, -1)
    b = np.abs((a - a.T))
    c = np.swapaxes(b, 2, 1)
    c = np.sqrt((c**2).sum(axis=-1))
    dist_mat = torch.Tensor(c).unsqueeze(-1).numpy() # ugh i am sorry
    for d in feat_distance_pow:
        power = d.get('pow', 1)
        max_val = d.get('max', 10000)
        min_val = d.get('min', 0)
        offset = d.get('offset', 0)

        v = (dist_mat + offset) ** power
        v = np.clip(v, a_min = min_val,
                    a_max = max_val)
        res_mats.append(v)

    if len(res_mats) > 0:
        M = np.concatenate(res_mats, 2)

    assert np.isfinite(M).all()
    return M

def mol_to_nx(mol):
    g = nx.Graph()
    g.add_nodes_from(range(mol.GetNumAtoms()))
    g.add_edges_from([(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), 
     {'weight' : b.GetBondTypeAsDouble()}) for b in mol.GetBonds()])
    return g

w_lut = {1.0 : 0, 1.5 : 1, 2.0: 2, 3.0: 3}

def get_min_path_length(g):
    N = len(g.nodes)
    out = np.zeros((N, N), dtype=np.int32)
    sp = nx.shortest_path(g)
    for i, j in sp.items():
        for jj, path in j.items():
            out[i, jj] = len(path)
    return out

def get_bond_path_counts(g):
    N = len(g.nodes)
    out = np.zeros((N, N, 4), dtype=np.int32)
    sp = nx.shortest_path(g)   
    
    for i, j in sp.items():
        for jj, path in j.items():
            for a, b in zip(path[:-1], path[1:]):
                w = g.edges[a, b]['weight']
                
                out[i, jj, w_lut[w]] +=1
                
    return out

def get_cycle_counts(g, cycle_size_max = 10):
    cb = nx.cycle_basis(g)
    N = len(g.nodes)
    M = cycle_size_max - 2
    cycle_mat = np.zeros((N, N, M), dtype=np.float32)
    for c in nx.cycle_basis(g):
        x = np.zeros(N)
        x[c] = 1
        if len(c) <= cycle_size_max:
            
            cycle_mat[:, :, len(c)-3] += np.outer(x, x)
    return cycle_mat

def get_dihedral_angles(mol, conf_idx=0):
    c = mol.GetConformers()[conf_idx]

    atom_n = mol.GetNumAtoms()

    out = np.zeros((atom_n, atom_n), dtype=np.float32)
    for i in range(atom_n):
        for j in range(i+1, atom_n):
            
            sp = Chem.rdmolops.GetShortestPath(mol, i, j)
            if len(sp) < 4:
                dh = 0
            else:
                try:
                    dh = Chem.rdMolTransforms.GetDihedralDeg(c, sp[0], sp[1], sp[-2], sp[-1])
                except ValueError:
                    dh = 0

            if not np.isfinite(dh):
                print(f"WARNING {dh} is not finite between {sp}")
                dh = 0
            
            out[i, j] = dh
            out[j, i] = dh

    return out

def get_dihedral_sincos(mol, conf_idx=0):
    c = mol.GetConformers()[conf_idx]

    atom_n = mol.GetNumAtoms()

    out = np.zeros((atom_n, atom_n, 2), dtype=np.float32)
    for i in range(atom_n):
        for j in range(i+1, atom_n):
            
            sp = Chem.rdmolops.GetShortestPath(mol, i, j)
            if len(sp) < 4:
                dh = 0
            else:
                try:
                    dh = Chem.rdMolTransforms.GetDihedralRad(c, sp[0], sp[1], sp[-2], sp[-1])
                except ValueError:
                    dh = 0

            if not np.isfinite(dh):
                print(f"WARNING {dh} is not finite between {sp}")
                dh = 0
            
            dh_sin = np.sin(dh)
            dh_cos = np.cos(dh)
            out[i, j, 0] = dh_sin
            out[j, i, 0] = dh_sin
            out[i, j, 1] = dh_cos
            out[j, i, 1] = dh_cos

    return out

def get_graph_props(mol, min_path_length=False,
                    bond_path_counts=False,
                    cycle_counts = False,
                    cycle_size_max=9, 
                    
                    ):
    g = mol_to_nx(mol)

    out = []
    if min_path_length:
        out.append(np.expand_dims(get_min_path_length(g), -1))

    if bond_path_counts:
        out.append(get_bond_path_counts(g))

    if cycle_counts:
        out.append(get_cycle_counts(g, cycle_size_max=cycle_size_max))

    if len(out) == 0:
        return None
    out = np.concatenate(out, axis=-1)

    assert np.isfinite(out).all()
    return out

def pad(M, MAX_N):
    """
    Pad M with shape N x N x C  to MAX_N x MAX_N x C
    """
    N, _, C = M.shape
    X = np.zeros((MAX_N, MAX_N, C),
                 dtype=M.dtype)

    for c in range(C):
        X[:N, :N, c] = M[:, :, c]
    return X
        
def get_geom_props(mol,
              dist_mat_mean = False,
              dist_mat_std = False):
    """
    returns geometry features for mol
    
    """
    res_mats = []

    Ds = np.stack([Chem.rdmolops.Get3DDistanceMatrix(mol, c.GetId()) for c in mol.GetConformers()], -1)

    M = None

    if dist_mat_mean:
        D_mean = np.mean(Ds, -1)

        res_mats.append(np.expand_dims(D_mean.astype(np.float32), -1))
        
    if dist_mat_std:
        D_std = np.std(Ds, -1)

        res_mats.append(np.expand_dims(D_std.astype(np.float32), -1))
        
    if len(res_mats) > 0:
        M = np.concatenate(res_mats, 2)


    return M

def recon_features_edge(mol,
                        graph_recon_config = {}, 
                        geom_recon_config = {},
                        
                        ):

    p = []
    p.append(get_graph_props(mol, **graph_recon_config))
    p.append(get_geom_props(mol, **geom_recon_config))

    a_sub = [a for a in p if a is not None]
    if len(a_sub) == 0:
        return np.zeros((mol.GetNumAtoms(),
                         mol.GetNumAtoms(),
                         0), dtype=np.float32)
    return np.concatenate(a_sub, -1)

def get_single_bond_fragment_masses(mol, **config):
    """
    Return a N x N x 2 matrix of the weights of the fragments 
    if that bond were cut
    """

    N = mol.GetNumAtoms()
    out_masses = np.zeros((N, N, 2), dtype=np.float32)
    for bi, b in enumerate(mol.GetBonds()):

        mol_f = Chem.FragmentOnBonds(mol, (bi,))
        atomic_masses = np.array([int(a.GetMass()) for a in mol_f.GetAtoms()])

        out_frags = []
        frags = Chem.GetMolFrags(mol_f, sanitizeFrags=False, 
                                 fragsMolAtomMapping=out_frags)
        frag_masses = [np.sum(atomic_masses[np.array(f)]) for f in frags][:2]
        if len(frag_masses) == 1:
            frag_masses.append(frag_masses[0])
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()            
        out_masses[i, j, :] = frag_masses
        out_masses[j, i, :] = frag_masses[::-1]
    return out_masses    
    
class PossibleFormulaFeaturizer:

    def __init__(
        self,
        formula_possible_atomicno=[1, 6, 7, 8, 9, 15, 16, 17],
        featurize_mode='numerical',
        enumerator_type='multipeak',
        num_peaks_per_formula=6, 
        max_formulae=4096,
        clip_mass=511, 
        use_highres=False, 
        formula_max_num_atomicno=20,
    ):

        self.enumerator_type = enumerator_type
        if enumerator_type == 'naive':
            self.ffe = masscompute.FragmentFormulaEnumerator(formula_possible_atomicno)
        elif enumerator_type == 'multipeak':
            self.ffe = masscompute.FragmentFormulaPeakEnumerator(formula_possible_atomicno, use_highres=use_highres, max_peak_num=num_peaks_per_formula)
            
        self.formula_possible_atomicno = formula_possible_atomicno
        self.featurize_mode = featurize_mode
        self.formula_max_atomicno =formula_max_num_atomicno
        self.max_formulae = max_formulae
        self.num_peaks_per_formula = num_peaks_per_formula

        if isinstance(formula_max_num_atomicno, int):
            # constant int
            formula_max_num_atomicno = [formula_max_num_atomicno] * len(formula_possible_atomicno)
        # create one-hot luts
        running_pos = 0
        self.oh_an_lut = {}
        self.oh_an_max = {}
        for an, m in zip(formula_possible_atomicno, formula_max_num_atomicno):
            self.oh_an_lut[an] = running_pos
            self.oh_an_max[an] = m + 1
            running_pos += m + 1
        self.oh_max = running_pos
        self.oh_offsets = np.array([self.oh_an_lut[a] for a in self.formula_possible_atomicno])

        self.clip_mass = clip_mass
        
    def num_unique_f(self, m):
        f = util.get_formula(Chem.AddHs(Chem.Mol(m)))
        return np.prod([v + 1 for v in f.values()])
        
    def __call__(self, mol, formulae=None, masses = None):

        if formulae is None:
            formulae, masses = self.ffe.get_frag_formulae(mol)
            
        if self.enumerator_type == 'multipeak':
            formulae, masses = self.ffe.get_frag_formulae(mol)
            
            masses = masses[:, :self.num_peaks_per_formula]
            
            # normalize masses? # FIXME DEBUG
            # The first formula is always [0, 0, 0, ...0]. High res 
            # a peak intensity of 0 for this, low res returns 1. Thus the clip
            masses[:, :, 1] = masses[:, :, 1]/ np.clip(np.sum(masses[:, :, 1], axis=1).reshape(-1, 1),
                                                       a_min = 0.01, a_max=1.0)
            #pickle.dump({'masses' : masses}, open('masses.pickle', 'wb'))
            
        if len(formulae) > self.max_formulae:
            naive_number = self.num_unique_f(mol)
            
            raise ValueError(f"molecule {Chem.MolToSmiles(mol)} has {len(formulae)}, more than the limit of {self.max_formulae} (naive count={self.num_unique_f(mol)})")
        
        src_masses_n = masses.shape[0]
        masses = np.pad(masses, ((0, self.max_formulae - src_masses_n),
                                 (0, 0),
                                 (0, 0)))

        # set first intensities =1 for padded region
        masses[src_masses_n:, 0, 1] = 1.0
                        
        if np.max(masses) > 511:
            s = f"DEBUG mass too large, mol={Chem.MolToSmiles(mol)} weight={ Chem.Descriptors.ExactMolWt(mol)}, peak_mass={np.max(masses)}"
            if self.clip_mass > 0:
                #print(s)
                masses = np.clip(masses, 0, self.clip_mass)
            else:
                raise ValueError(s)

        if self.featurize_mode == 'numerical':
            formulae_feat = formulae.astype(np.float32)
            formulae_feat = np.pad(formulae_feat, 
                                   [(0, self.max_formulae - formulae_feat.shape[0]), 
                                    (0, 0)])

            return formulae_feat, masses 
            
        elif self.featurize_mode == 'onehot' :
            formulae_feat = np.zeros((self.max_formulae, 
                                      self.oh_max), dtype=np.float32)

            util.fast_multi_onehot(formulae, self.oh_offsets, formulae_feat, accum=False)
            return formulae_feat, masses
        
        elif self.featurize_mode == 'onehot_accum' :
            formulae_feat = np.zeros((self.max_formulae, 
                                      self.oh_max), dtype=np.float32)

            util.fast_multi_onehot(formulae, self.oh_offsets, formulae_feat, accum=True)
            return formulae_feat, masses

        else:
            raise ValueError(f"Unknown featurize mode {self.featurize_mode}")
