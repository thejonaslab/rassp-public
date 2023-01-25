import numpy as np
import pandas as pd
import pickle
from rdkit import Chem
import rdkit.Chem.Descriptors

import os
import zlib


import itertools
import tinygraph as tg
import tinygraph.io
import tinygraph.io.rdkit
from tqdm import tqdm

from rassp import vertsubsetgen_fast as fast

class SamplerUniform:

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, mol, num_samples):

        M = mol.GetNumAtoms()

        out = np.random.rand(num_samples, M) <= self.p

        return out.astype(np.uint8)

class SamplerUniformNumber:

    def __init__(self, min_size = 4):
        self.min_size = min_size

    def __call__(self, mol, num_samples):

        M = mol.GetNumAtoms()
        out = np.zeros((num_samples, M))

        min_size = max(min(self.min_size, self.min_size-4), 0)

        
        for i in range(num_samples):
            try:
                j = np.random.randint(min_size, M)
            except ValueError as e:
                print(self.min_size, M)
                raise
                
            out[i][np.random.permutation(M)[:j]] = 1

        return out.astype(np.uint8)

class EnumerateBreaks:
    def __init__(self, num_breaks=1, min_size=1):
        """
        Enumerate breaks
        """
        self.num_breaks = num_breaks
        self.min_size = min_size

    def __call__(self, mol, num_samples=None, return_subsets=False):

        M = mol.GetNumAtoms()

        g = tg.io.rdkit.from_rdkit_mol(mol)
        
        src_edges = list(g.edges())

        # start with whole mol
        subsets = set([frozenset(range(M))])

        for broken_bond_n in range(1, self.num_breaks+1):
            for ss in itertools.combinations(src_edges, broken_bond_n):
                g2 = g.copy()
                for e in ss:
                    g2[e[0], e[1]] = 0
                cc = tg.algorithms.get_connected_components(g2)

                for c in cc:
                    if len(c) >= self.min_size:
                        subsets.add(frozenset(c))
        
        if return_subsets:
            return subsets

        out = np.zeros((len(subsets),  M), dtype=np.uint8)
        for si, s in enumerate(subsets):
            out[si][list(s)] = 1

        if num_samples is not None and num_samples < len(out):
            return out[np.random.permutation(len(out))[:num_samples]]
        return out

    
class BreakAndRearrange:
    def __init__(self, num_breaks=3, min_size=1, all_h=True):
        """
        Enumerate all breaks and do exhaustive hydrogen rearrangement,
        ignoring radicals.
        """
        self.num_breaks = num_breaks
        self.min_size = min_size
        self.all_h = all_h

    def from_rdkit_mol(self, mol):
        """Need a custom rdkit mol to tinygraph func."""
        g = tg.io.rdkit.from_rdkit_mol(mol, use_implicit_h=False, use_explicit_h=True)
        g.add_vert_prop('formal_charge', np.int8)
        g.add_vert_prop('n_radical_electrons', np.uint8)
        for a in mol.GetAtoms():
            i = a.GetIdx()
            g.v['formal_charge'][i] = a.GetFormalCharge()
            g.v['n_radical_electrons'][i] = a.GetNumRadicalElectrons()
        return g

    def  __call__(self, mol, num_samples=None, max_samples=None, seed=0, return_subsets=False):
        np.random.seed(seed)

        # clean and sanitize mol
        Chem.SanitizeMol(
            mol, Chem.rdmolops.SanitizeFlags.SANITIZE_ALL, catchErrors=True)
        mol.UpdatePropertyCache()
        Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
        Chem.SetAromaticity(mol)
        mol = Chem.AddHs(mol)
        
        g = self.from_rdkit_mol(mol)
        orig_mol = mol

        # break only the non-hydrogen bonds in graph
        n_total_atoms = len(g.v['atomicno'])
        h_atom_inds = set(filter(lambda x: x is not None, [i if e == 1 else None for i, e in enumerate(g.v['atomicno'])]))
        heavy_atom_inds = set(range(n_total_atoms)).difference(h_atom_inds)
        src_edges = list(g.edges())
        non_h_edges = list(filter(lambda e: (e[0] not in h_atom_inds) and (e[1] not in h_atom_inds), src_edges))

        heavy_atom_h_map = {i: [] for i in heavy_atom_inds}
        for e in src_edges:
            if e in non_h_edges:
                continue

            if e[0] in heavy_atom_inds:
                heavy_ind, h_ind = e
            elif e[1] in heavy_atom_inds:
                h_ind, heavy_ind = e
            else:
                # both atoms of edge are hydrogens...
                continue

            assert heavy_ind in heavy_atom_inds, f'Failed on {Chem.MolToSmiles(mol)}'
            assert h_ind in h_atom_inds
            g.v['explicit_h'][heavy_ind] += 1
            heavy_atom_h_map[heavy_ind].append(h_ind)

        # get all heavy atoms with hydrogens to donate
        heavy_atom_inds_with_hs = []
        for i in heavy_atom_inds:
            if len(heavy_atom_h_map[i]) > 0:
                heavy_atom_inds_with_hs.append(i)

        # generate all valid permutations of hydrogen donations
        valid_h_donations = list(itertools.product(heavy_atom_inds_with_hs, heavy_atom_inds))

        # generate fragments from breaking non-hydrogen bonds
        g2_list = []
        for broken_bond_n in range(1, self.num_breaks + 1):
            for ss in itertools.combinations(non_h_edges, broken_bond_n):
                g2 = g.copy()
                for e in ss:
                    g2[e[0], e[1]] = 0
                g2_list.append(g2)
        subsets = set([frozenset(range(n_total_atoms))])
        for g2 in g2_list:
            rearranged_graphs = []
            for perm in valid_h_donations:
                i, j = perm
                i_hs, j_hs = [g2.v['explicit_h'][k] for k in perm]
                assert i_hs != 0, 'invalid h donor w zero hs'

                g3 = g2.copy()

                # update explicit H count
                g3.v['explicit_h'][i] -= 1
                g3.v['explicit_h'][j] += 1

                # pick a random hydrogen to move, since they're all equivalent?
                # we have to limit the branching factor somehow

                if self.all_h:
                    heavy_hs_to_use = heavy_atom_h_map[i]
                else:
                    heavy_hs_to_use = [np.random.choice(heavy_atom_h_map[i])]
                    
                for h_ind in heavy_hs_to_use:
                    g3_h = g3.copy()
                    assert g3_h.adjacency[i, h_ind] == 1, 'h donor must be connected via single-bond'
                    g3_h.adjacency[i, h_ind] = 0
                    g3_h.adjacency[h_ind, i] = 0

                    assert g3_h.adjacency[j, h_ind] == 0, 'h recipient must not be connected'
                    g3_h.adjacency[j, h_ind] = 1
                    g3_h.adjacency[h_ind, j] = 1

                    rearranged_graphs.append({
                        'tg': g3_h,
                    })
                


                assert g3.adjacency[i, h_ind] == 1, 'h donor must be connected via single-bond'
                g3.adjacency[i, h_ind] = 0
                g3.adjacency[h_ind, i] = 0

                assert g3.adjacency[j, h_ind] == 0, 'h recipient must not be connected'
                g3.adjacency[j, h_ind] = 1
                g3.adjacency[h_ind, j] = 1

                rearranged_graphs.append({
                    'tg': g3,
                })
            
            for rearrangement in rearranged_graphs:
                # fracture graph into the connected components
                gmol = rearrangement['tg']
                cc = tg.algorithms.get_connected_components(gmol)
                for c in cc:
                    subsets.add(frozenset(c))
                
            if max_samples is not None and len(subsets) > max_samples:
                # terminate early
                return None

        # orig_mol = to_rdkit_mol(g, **DEFAULT_PROP_NAMES)
        breaker = EnumerateBreaks(num_breaks=self.num_breaks)
        for c in breaker(orig_mol, num_samples, return_subsets=True):
            subsets.add(frozenset(c))
           
        if return_subsets:
            return subsets

        out = np.zeros((len(subsets), n_total_atoms), dtype=np.uint8)
        for si, s in enumerate(subsets):
            out[si][list(s)] = 1

        if num_samples is not None and num_samples < len(out):
            return out[np.random.permutation(len(out))[:num_samples]]
        return out


class BreakAndRearrangeAllH:
    """
    Like BreakAndRearrange but we move all the hydrogens
    """
    def __init__(self, num_breaks=3, min_size=1):
        """
        Enumerate all breaks and do exhaustive hydrogen rearrangement,
        ignoring radicals.
        """
        self.num_breaks = num_breaks
        self.min_size = min_size

    def from_rdkit_mol(self, mol):
        """Need a custom rdkit mol to tinygraph func."""
        g = tg.io.rdkit.from_rdkit_mol(mol, use_implicit_h=False, use_explicit_h=True)
        g.add_vert_prop('formal_charge', np.int8)
        g.add_vert_prop('n_radical_electrons', np.uint8)
        for a in mol.GetAtoms():
            i = a.GetIdx()
            g.v['formal_charge'][i] = a.GetFormalCharge()
            g.v['n_radical_electrons'][i] = a.GetNumRadicalElectrons()
        return g

    def  __call__(self, mol, num_samples=None, max_samples=None, seed=0, return_subsets=False):
        np.random.seed(seed)

        # clean and sanitize mol
        Chem.SanitizeMol(
            mol, Chem.rdmolops.SanitizeFlags.SANITIZE_ALL, catchErrors=True)
        mol.UpdatePropertyCache()
        Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
        Chem.SetAromaticity(mol)
        mol = Chem.AddHs(mol)
        
        g = self.from_rdkit_mol(mol)
        orig_mol = mol

        # break only the non-hydrogen bonds in graph
        n_total_atoms = len(g.v['atomicno'])
        h_atom_inds = set(filter(lambda x: x is not None, [i if e == 1 else None for i, e in enumerate(g.v['atomicno'])]))
        heavy_atom_inds = set(range(n_total_atoms)).difference(h_atom_inds)
        src_edges = list(g.edges())
        non_h_edges = list(filter(lambda e: (e[0] not in h_atom_inds) and (e[1] not in h_atom_inds), src_edges))

        heavy_atom_h_map = {i: [] for i in heavy_atom_inds}
        for e in src_edges:
            if e in non_h_edges:
                continue

            if e[0] in heavy_atom_inds:
                heavy_ind, h_ind = e
            elif e[1] in heavy_atom_inds:
                h_ind, heavy_ind = e
            else:
                # both atoms of edge are hydrogens...
                continue

            assert heavy_ind in heavy_atom_inds, f'Failed on {Chem.MolToSmiles(mol)}'
            assert h_ind in h_atom_inds
            g.v['explicit_h'][heavy_ind] += 1
            heavy_atom_h_map[heavy_ind].append(h_ind)

        # get all heavy atoms with hydrogens to donate
        heavy_atom_inds_with_hs = []
        for i in heavy_atom_inds:
            if len(heavy_atom_h_map[i]) > 0:
                heavy_atom_inds_with_hs.append(i)

        # generate all valid permutations of hydrogen donations
        valid_h_donations = list(itertools.product(heavy_atom_inds_with_hs, heavy_atom_inds))
        #print("BreakAndRearrangeAll: valid_h_donations=", valid_h_donations)
        # generate fragments from breaking non-hydrogen bonds
        g2_list = []
        for broken_bond_n in range(1, self.num_breaks + 1):
            for ss in itertools.combinations(non_h_edges, broken_bond_n):
                g2 = g.copy()
                for e in ss:
                    g2[e[0], e[1]] = 0
                g2_list.append(g2)
        subsets = set([frozenset(range(n_total_atoms))])
        for g2 in g2_list:
            rearranged_graphs = []
            for perm in valid_h_donations:
                i, j = perm
                i_hs, j_hs = [g2.v['explicit_h'][k] for k in perm]
                assert i_hs != 0, 'invalid h donor w zero hs'

                g3 = g2.copy()

                # update explicit H count
                g3.v['explicit_h'][i] -= 1
                g3.v['explicit_h'][j] += 1

                # pick a random hydrogen to move, since they're all equivalent?
                # we have to limit the branching factor somehow
                for h_ind in heavy_atom_h_map[i]:
                    g3_h = g3.copy()
                    assert g3_h.adjacency[i, h_ind] == 1, 'h donor must be connected via single-bond'
                    g3_h.adjacency[i, h_ind] = 0
                    g3_h.adjacency[h_ind, i] = 0

                    assert g3_h.adjacency[j, h_ind] == 0, 'h recipient must not be connected'
                    g3_h.adjacency[j, h_ind] = 1
                    g3_h.adjacency[h_ind, j] = 1

                    rearranged_graphs.append({
                        'tg': g3_h,
                    })
            
            for rearrangement in rearranged_graphs:
                # fracture graph into the connected components
                gmol = rearrangement['tg']
                cc = tg.algorithms.get_connected_components(gmol)
                for c in cc:
                    subsets.add(frozenset(c))
                
            if max_samples is not None and len(subsets) > max_samples:
                # terminate early
                return None

        # orig_mol = to_rdkit_mol(g, **DEFAULT_PROP_NAMES)
        breaker = EnumerateBreaks(num_breaks=self.num_breaks)
        for c in breaker(orig_mol, num_samples, return_subsets=True):
            subsets.add(frozenset(c))
           
        if return_subsets:
            return subsets

        out = np.zeros((len(subsets), n_total_atoms), dtype=np.uint8)
        for si, s in enumerate(subsets):
            out[si][list(s)] = 1

        if num_samples is not None and num_samples < len(out):
            return out[np.random.permutation(len(out))[:num_samples]]
        return out


class BreakAndRearrangeFast:
    """
    Break and rearrange using cython helper functions
    """
    
    def __init__(self, num_breaks=3, min_size=1):
        """
        Enumerate all breaks and do exhaustive hydrogen rearrangement,
        ignoring radicals.
        """
        self.num_breaks = num_breaks
        self.min_size = min_size

    def from_rdkit_mol(self, mol):
        """Need a custom rdkit mol to tinygraph func."""
        g = tg.io.rdkit.from_rdkit_mol(mol, use_implicit_h=False, use_explicit_h=True)
        g.add_vert_prop('formal_charge', np.int8)
        g.add_vert_prop('n_radical_electrons', np.uint8)
        for a in mol.GetAtoms():
            i = a.GetIdx()
            g.v['formal_charge'][i] = a.GetFormalCharge()
            g.v['n_radical_electrons'][i] = a.GetNumRadicalElectrons()
        return g

    def  __call__(self, mol, num_samples=None, max_samples=None,
                  seed=0, return_subsets=False):
        np.random.seed(seed)
        if mol.GetNumAtoms()> 63:
            raise ValueError(f" MOlecule has {mol.GetNumAtoms()} atoms, currently we only support up to 63")

        # clean and sanitize mol
        Chem.SanitizeMol(
            mol, Chem.rdmolops.SanitizeFlags.SANITIZE_ALL, catchErrors=True)
        mol.UpdatePropertyCache()
        Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
        Chem.SetAromaticity(mol)
        #mol = Chem.AddHs(mol)
        
        g = self.from_rdkit_mol(mol)
        orig_mol = mol

        # break only the non-hydrogen bonds in graph
        n_total_atoms = len(g.v['atomicno'])
        h_atom_inds = set(filter(lambda x: x is not None, [i if e == 1 else None for i, e in enumerate(g.v['atomicno'])]))
        heavy_atom_inds = set(range(n_total_atoms)).difference(h_atom_inds)
        src_edges = list(g.edges())
        non_h_edges = list(filter(lambda e: (e[0] not in h_atom_inds) and (e[1] not in h_atom_inds), src_edges))
        h_edges = list(filter(lambda e: (e[0] in h_atom_inds) or (e[1]  in h_atom_inds), src_edges))

        heavy_atom_h_map = {i: [] for i in heavy_atom_inds}
        heavy_atom_h_lut = np.zeros((n_total_atoms, 4), dtype=np.int32)

        explicit_h = np.zeros(n_total_atoms, dtype=np.int32)
        
        for e in src_edges:
            if e in non_h_edges:
                continue

            if e[0] in heavy_atom_inds:
                heavy_ind, h_ind = e
            elif e[1] in heavy_atom_inds:
                h_ind, heavy_ind = e
            else:
                # both atoms of edge are hydrogens...
                continue

            assert heavy_ind in heavy_atom_inds, f'Failed on {Chem.MolToSmiles(mol)}'
            assert h_ind in h_atom_inds
            g.v['explicit_h'][heavy_ind] += 1

            heavy_atom_h_map[heavy_ind].append(h_ind)
            # construct LUT
            heavy_atom_h_lut[heavy_ind][explicit_h[heavy_ind]] = h_ind
            explicit_h[heavy_ind] += 1


        # get all heavy atoms with hydrogens to donate
        heavy_atom_inds_with_hs = []
        for i in heavy_atom_inds:
            if len(heavy_atom_h_map[i]) > 0:
                heavy_atom_inds_with_hs.append(i)

        # generate all valid permutations of hydrogen donations
        valid_h_donations = np.array(list(itertools.product(heavy_atom_inds_with_hs, heavy_atom_inds)))
        if len(valid_h_donations) == 0:
            # array needs to have 2-dims and we explicitly do_rearrange
            # later, so this needs to be a rearrange no-op
            # we check for -1, -1 in `do_rearrangements_lut_bits`
            valid_h_donations = np.array([[-1, -1]])

        subsets = [np.array([fast.bitset_create(range(n_total_atoms))], dtype=np.int64)]
        
        # generate fragments from breaking non-hydrogen bonds

        g_adj_uint8 = g.adjacency.astype(np.uint8)

        # first set of bonds
        non_h_remove_edges = enumerate_break_edges(non_h_edges, self.num_breaks)

        non_h_remove_subsets = \
            fast.subsets_from_edge_removals(non_h_remove_edges,
                                       g_adj_uint8,
                                       1,
                                       valid_h_donations,
                                       heavy_atom_h_lut,
                                       explicit_h)
        

        subsets.append(non_h_remove_subsets)
        
        all_remove_edges = enumerate_break_edges(src_edges, self.num_breaks)
        all_remove_subsets = \
            fast.subsets_from_edge_removals(all_remove_edges, 
                                       g_adj_uint8,
                                       0,
                                       valid_h_donations,
                                       heavy_atom_h_lut,
                                       explicit_h)

        subsets.append(all_remove_subsets)
        subsets = np.unique(np.concatenate(subsets))
        
        if return_subsets:
            return subsets

        out = fast.array_bitsets_to_array(subsets,
                                              n_total_atoms)
        return out
    
    
def create_sampler(name, config):

    if name == 'uniform':
        return SamplerUniform(**config)
    elif name == 'uniform_number':
        return SamplerUniformNumber(**config)
    elif name == 'enumerate_breaks':
        return EnumerateBreaks(**config)
    elif name == 'break_and_rearrange':
        return BreakAndRearrange(**config)
    raise NotImplementedError(name)



def enumerate_break_edges(edge_list, num_breaks):
    M = len(edge_list)
    edgesets_to_break = np.ones((len(edge_list) ** num_breaks+1,
                                 num_breaks, 2), 
                                dtype=np.int32) * -1

    to_break_pos = 1
    for broken_bond_n in range(1, num_breaks+1):
        for ss in itertools.combinations(edge_list, broken_bond_n):

            for ei, e in enumerate(ss):
                edgesets_to_break[to_break_pos, ei] = e
            to_break_pos +=1

    edgesets_to_break = edgesets_to_break[:to_break_pos]
    return edgesets_to_break
