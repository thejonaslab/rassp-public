import numpy as np
from rdkit import Chem
import torch
import torch.nn.functional as F

import tinygraph as tg

from itertools import combinations_with_replacement

from util import get_nos_coords
from featurize.atom_features import to_onehot
from featurize.molecule_features import mol_to_nums_adj

def feat_edges(mol, MAX_ATOM_N=None, MAX_EDGE_N=None):
    """
    Create features for edges, edge connectivity
    matrix, and edge/vert matrix

    Note: We really really should parameterize this somehow. 
    """
    
    atom_n = mol.GetNumAtoms()
    atomicno, vert_adj = mol_to_nums_adj(mol)
    
    double_edge_n = np.sum(vert_adj > 0)
    assert double_edge_n % 2 == 0
    edge_n = double_edge_n // 2
    
    edge_adj = np.zeros((edge_n, edge_n))
    edge_vert_adj = np.zeros((edge_n, atom_n))
    
    edge_list = []
    for i in range(atom_n):
        for j in range(i +1, atom_n):
            if vert_adj[i, j] > 0:
                edge_list.append((i, j))
                e_idx = len(edge_list) - 1
                edge_vert_adj[e_idx, i] = 1
                edge_vert_adj[e_idx, j] = 1

    # now which edges are connected
    edge_adj = edge_vert_adj @ edge_vert_adj.T - 2 * np.eye(edge_n)
    assert edge_adj.shape == (edge_n, edge_n)
    
    # now create edge features
    edge_features = []
    
    for edge_idx, (i, j) in enumerate(edge_list):
        f = []
        f += to_onehot(vert_adj[i, j], [1.0, 1.5, 2.0, 3.0])
        
        edge_features.append(f)              
        
    edge_features = np.array(edge_features)
    
    return {'edge_edge': np.expand_dims(edge_adj, 0), 
            'edge_feat': edge_features, 
            'edge_vert': np.expand_dims(edge_vert_adj, 0)}

def feat_edges_for_bipartite_model(mol, adj, MAX_ATOM_N=48, MAX_EDGE_N=64):
    """

    Edge feature types and counts
    -----------------------------
    Bond order (4): one-hot for 1, 1.5, 2, 3
    Neighboring atoms (36): one-hot for all combinations of HCONFSPCl atoms
    Atom count on paths (16): counts of atoms on paths of distance 1 and 2 from bond sites
    """
    n_atoms = mol.GetNumAtoms()
    n_bonds = mol.GetNumBonds()

    # get bipartite adjacency matrix
    orders, ii, jj = torch.where(torch.triu(adj))
    bb = torch.arange(len(ii))

    bipartite_adj_nopad = torch.zeros((n_atoms, n_bonds))
    bipartite_adj_nopad[ii, bb] += 1
    bipartite_adj_nopad[jj, bb] += 1
    assert np.isclose(bipartite_adj_nopad.sum() / 2, n_bonds)

    # create the features for each edge

    # one-hot encoding of the bond order
    bond_orders = F.one_hot(orders, num_classes=4)
    assert np.isclose(bond_orders.sum(), n_bonds)

    # one-hot encoding of which atoms were at either end
    # ALLOWED_ATOMS in preprocess_wishart_dataset = HCONFSPCl
    ALLOWED_ATOM_NUMS = [1, 6, 8, 7, 9, 16, 15, 17]
    bond_map = {}
    for i, a_pair in enumerate(combinations_with_replacement(ALLOWED_ATOM_NUMS, 2)):
        a_pair = tuple(sorted(a_pair))
        bond_map[a_pair] = i
    n_total_feats = len(bond_map)

    neighboring_atoms = []
    for bi in range(n_bonds):
        bond = mol.GetBondWithIdx(bi)
        ai = bond.GetBeginAtomIdx()
        aj = bond.GetEndAtomIdx()

        # check bond order
        assert adj[:, ai, aj].sum() != 0.0

        a_pair = tuple(sorted(
            (
                mol.GetAtomWithIdx(ai).GetAtomicNum(),
                mol.GetAtomWithIdx(aj).GetAtomicNum(),
            )
        ))
        ind = bond_map[a_pair]
        neighboring_atoms.append(ind)
    neighboring_atoms = F.one_hot(
        torch.Tensor(neighboring_atoms).to(torch.int64),
        num_classes=n_total_feats)

    # count-up encoding of atoms on paths walking away from the bond
    # for now, we only have atoms at distance +1 from the bond
    g = tg.io.rdkit.from_rdkit_mol(mol)
    shortest_paths = tg.algorithms.get_shortest_paths(g, False)
    a_map = {an: i for i, an in enumerate(ALLOWED_ATOM_NUMS)}

    neighboring_atoms_1 = []
    for bi in range(n_bonds):
        bond = mol.GetBondWithIdx(bi)
        ai = bond.GetBeginAtomIdx()
        aj = bond.GetEndAtomIdx()

        neighbor_inds = list(np.where(shortest_paths[ai] == 1)[0]) + list(np.where(shortest_paths[aj] == 1)[0])
        neighbor_atom_nums = [mol.GetAtomWithIdx(int(i)).GetAtomicNum() for i in neighbor_inds]

        vec = np.zeros(len(ALLOWED_ATOM_NUMS))
        for a_num in neighbor_atom_nums:
            vec[a_map[a_num]] += 1
        neighboring_atoms_1.append(vec)
        
    neighboring_atoms_2 = []
    for bi in range(n_bonds):
        bond = mol.GetBondWithIdx(bi)
        ai = bond.GetBeginAtomIdx()
        aj = bond.GetEndAtomIdx()

        neighbor_inds = list(np.where(shortest_paths[ai] == 2)[0]) + list(np.where(shortest_paths[aj] == 1)[0])
        neighbor_atom_nums = [mol.GetAtomWithIdx(int(i)).GetAtomicNum() for i in neighbor_inds]

        vec = np.zeros(len(ALLOWED_ATOM_NUMS))
        for a_num in neighbor_atom_nums:
            vec[a_map[a_num]] += 1
        neighboring_atoms_2.append(vec)
        
    atom_count = torch.cat(
        [
            torch.Tensor(neighboring_atoms_1).to(torch.int64),
            torch.Tensor(neighboring_atoms_2).to(torch.int64),
        ], dim=-1)

    edge_feat_nopad = torch.cat([
        bond_orders,
        neighboring_atoms,
        atom_count,
    ], dim=-1)

    # pad up to the maximum number of edges
    bipartite_adj = torch.zeros((MAX_ATOM_N, MAX_EDGE_N))
    bipartite_adj[:n_atoms, :n_bonds] = bipartite_adj_nopad
    edge_feat = torch.zeros((MAX_EDGE_N, edge_feat_nopad.shape[1]))
    edge_feat[:n_bonds, :] = edge_feat_nopad

    return {
        'bipartite_adj': bipartite_adj,
        'edge_feat': edge_feat,
    }
