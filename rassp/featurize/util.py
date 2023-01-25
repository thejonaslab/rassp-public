import numpy as np

def get_nos_coords(mol, conf_i):
    conformer = mol.GetConformers()[conf_i]
    coord_objs = [conformer.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
    coords = np.array([(c.x, c.y, c.z) for c in coord_objs])
    atomic_nos = np.array([a.GetAtomicNum() for a in mol.GetAtoms()]).astype(int)
    return atomic_nos, coords

def get_nos(mol):
    return np.array([a.GetAtomicNum() for a in mol.GetAtoms()]).astype(int)

def get_subset_peaks_from_formulae(all_formulae, all_formulae_peaks,
                                   vert_element_oh,
                                   atom_subsets):
    """
    Get the mass peaks associated with each atom subset by looking them up
    in all_formulae and associated peaks. 
     
     all_formulae: N x ELEMENT_N all possible formuale
     all_formuale_peaks: N x peak_sizes
     vert_element_oh: one-hot-encoded matrix of atom types, ATOM_N x ELEMENT_N
     atom_subsets : atom subsets, M x ATOM_N
    """
     
    FORMULAE_N, ELEMENT_N = all_formulae.shape
    ATOM_N, _ = vert_element_oh.shape
    assert vert_element_oh.shape[1] == ELEMENT_N
    SUBSET_N, _ = atom_subsets.shape
    assert atom_subsets.shape[1] == ATOM_N
    
    formula_lut_idx_all = []
    formulae_pos_lut = {tuple(e) : i for i, e in enumerate(all_formulae)}
    
    for s, atom_subset in enumerate(atom_subsets):
       formula = (vert_element_oh * atom_subset.astype(np.float).reshape(-1, 1)).sum(axis=0)
       formula_int = tuple(formula.astype(int))
       formula_lut_idx = formulae_pos_lut[formula_int]
       formula_lut_idx_all.append(formula_lut_idx)
    formula_lut_idx_all = np.array(formula_lut_idx_all)
    atom_subsets_peaks = all_formulae_peaks[formula_lut_idx_all]

    return atom_subsets_peaks

def get_subset_peaks_idx_from_formulae_fast(all_formulae, 
                                            vert_element_oh,
                                            atom_subsets):
    """
    Get the mass peaks indices associated with each atom subset by looking them up
    in all_formulae and associated peaks. 

    This is like get_subset_peaks_from_formulae, except we use more numpy
    operations and compute an integer for the hash instead of 
    using the formula as a tuple. 

     
     all_formulae: N x ELEMENT_N all possible formuale
     all_formuale_peaks: N x peak_sizes
     vert_element_oh: one-hot-encoded matrix of atom types, ATOM_N x ELEMENT_N
     atom_subsets : atom subsets, M x ATOM_N
    """
     
    FORMULAE_N, ELEMENT_N = all_formulae.shape
    ATOM_N, _ = vert_element_oh.shape
    assert vert_element_oh.shape[1] == ELEMENT_N
    SUBSET_N, _ = atom_subsets.shape
    assert atom_subsets.shape[1] == ATOM_N

    vert_element_oh = vert_element_oh.astype(np.int32)
    # how many of each element: What is the max of that value for the formula
    max_formula = vert_element_oh.astype(np.int64).sum(axis=0)

    # positionally-encode each element type as a larger and larger
    # integer to compute a hash. So our hash is
    #     num_h * 1  +  num_C * MAX_NUM_H  +  num_N * (MAX_NUM_H * MAX_NUM_C) + ....
    # 
    vert_element_accum =  np.cumprod(max_formula + 1)
    vert_element_accum[1:] = vert_element_accum[:-1]
    vert_element_accum[0] = 1
    
    f_int_key = all_formulae.astype(np.int64) @ vert_element_accum
    
    formulae_pos_lut = {e : i for i, e in enumerate(f_int_key)}

    formula_lut_idx_all = np.zeros(atom_subsets.shape[0], dtype=np.uint64)

    # compute the hashes
    formulae_ints = atom_subsets @ (vert_element_oh @ vert_element_accum)

    # do the lookups
    for formula_i, formula_int in enumerate(formulae_ints):
        formula_lut_idx_all[formula_i] = formulae_pos_lut[formula_int]

    return formula_lut_idx_all
