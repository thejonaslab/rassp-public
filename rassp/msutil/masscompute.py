import numpy as np
from rdkit import Chem
import diskcache as dc

from rassp import masseval

USE_DC = False

def count_seq(sequences):
    if len(sequences) == 1:
        return [[i] for i in range(sequences[0]+1)]
    
    subseq = count_seq(sequences[1:])
    
    j = sequences[0]
    out = []
    for i in range(j+1):
        out += [[i] + s for s in subseq]
    return out

def get_formula(mol):
    """
    Return a dictionary of atomicno:num 
    """
    out = {}
    for a in mol.GetAtoms():
        an = a.GetAtomicNum()
        out[an] = out.get(an, 0) + 1
    return out
    
class FragmentFormulaEnumerator:
    """
    Enumerate all possible formulae and their naive weights

    """
    
    def __init__(self, formula_possible_atomicnos):

        """
        formula_possible_atomicnos: possible atomicnos in the formula, in order
        #formula_max_atoms: maximum number of atoms in the formula
        """
       
        
        self.formula_possible_atomicnos = formula_possible_atomicnos
        #self.formula_max_atoms = formula_max_atoms
        pt = Chem.GetPeriodicTable()
        self.atomic_weights = np.array([pt.GetMostCommonIsotopeMass(a) \
                                      for a in self.formula_possible_atomicnos])
        
    def get_frag_formulae(self, mol, weights='naive'):
        f = get_formula(mol)
        
        seq = [ f.get(a, 0) for a in self.formula_possible_atomicnos  ]
        frag_formulae = count_seq(seq)
        frag_formulae = np.array(frag_formulae, dtype=np.int32)

        if weights == 'naive':
           frag_formulae_masses = frag_formulae @ self.atomic_weights
        else:
           raise NotImplementedError("Have not yet implemented proper isotopes")
        
        return frag_formulae, frag_formulae_masses 

class FragmentFormulaPeakEnumerator:
    """
    Enumerate all possible formulae and their peak weights. 

    """
    
    def __init__(self, formula_possible_atomicnos,
                 use_highres=False, max_peak_num=6):

        """
        formula_possible_atomicnos: possible atomicnos in the formula, in order
        """
        self.formula_possible_atomicnos = formula_possible_atomicnos

        if USE_DC:
            self.dc = dc.Cache("FragFormulaPeakEnumerator")

        self.use_highres = use_highres
        self.max_peak_num = max_peak_num

    def get_frag_formulae(self, mol):
    
        formula_dict = get_formula(mol)
        if USE_DC:
            key = str(formula_dict)
            if key in self.dc:
                return self.dc[key]

        if self.use_highres:
            np_out = masseval.py_get_all_frag_spect_highres(formula_dict ,
                                                            MAX_PEAK_NUM=self.max_peak_num)
        else:
            np_out = masseval.py_get_all_frag_spect_np_nostruct(formula_dict ,
                                                                MAX_PEAK_NUM=self.max_peak_num)
        # np_out = masseval.py_get_all_frag_spect_np(formula_dict)

        formulae = np_out['formula'][:, self.formula_possible_atomicnos]
        peaks = np_out['peaks']
        mass = peaks['mass']
            
        # if self.use_highres:
        #     mass = np.round(mass) # DEBUG DEBUG FIX THIS
            
        intensity = peaks['intensity']
        mp = np.stack([mass, intensity], -1)
        out= formulae, mp

        if USE_DC:
            self.dc[key] = out
        
        return out
