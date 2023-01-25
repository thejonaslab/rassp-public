import numpy as np
from rdkit import Chem
import itertools
from rassp import masseval
import scipy.stats

def enumerate_multinomial(N, p, cutoff_thold=0.0001):
    """
    Enumerate the state space of rolling a die with |p| faces
    N times
    
    returns: [(prob_1, (tuple_of_counts)), 
               (prob_2, (tuple_of_counts)), 
               ...]
    
    cutoff_thold: Do not return points in the state space 
    with prob less than cutoff_thold
    """
    
    K = len(p)
    s = list()
    counts = []
    for s in itertools.combinations_with_replacement(range(K), N):
        count = np.zeros(K, dtype=np.int32)
        for i in s:
            count[i] +=1
        counts.append(count)
    counts = np.array(counts)


    p = p / np.sum(p)
    rv = scipy.stats.multinomial(N, p)

    probs = rv.pmf(counts)
    sort_idx = np.argsort(probs)[::-1]

    probs_sorted = probs[sort_idx]
    thold_idx = np.argwhere(np.cumsum(probs_sorted) >= 1-cutoff_thold).flatten()[0] 
    thold_idx += 1
    
    
    return list(zip(probs_sorted[:thold_idx], 
                    [tuple(a) for a in counts[sort_idx][:thold_idx]]))


def get_isotopes(atomicno, pct_thold = 0.01):
    """
    Get all isotopes and their masses for atomic
    number atomicno
    
    returns [(int_mass, prob, exact_mass),...]
    for each isotope
    
    """
    mass_probs = []
    pt = Chem.GetPeriodicTable()
    for m in range(atomicno, 4*atomicno):
        
        pct = pt.GetAbundanceForIsotope(atomicno, m)
        if pct >= pct_thold:
            exact_mass = pt.GetMassForIsotope(atomicno, m)
            mass_probs.append((m, pct/100, exact_mass))
    return sorted(mass_probs, key=lambda r: -r[1])


def get_peaks_for_elt_num(atomicno, num_atoms):
    """
    Get the peaks for a formula with num_atoms of atomicno
    
    returns list of [(prob, mass)]
    
    """
    isotopes = get_isotopes(atomicno)
    probs = [i[1] for i in isotopes]

    comb_probs = enumerate_multinomial(num_atoms, probs, 0.001)

    exact_masses = [np.sum([isotopes[a_i][2]*a for a_i, a in enumerate(m)]) for p, m in comb_probs]

    peaks_masses_probs = [(comb_probs[i][0], m) for i, m in enumerate(exact_masses)] 
    return peaks_masses_probs



def create_cpp_lut(filename):
    fid = open(filename, 'w')
    fid.write("static const\n std::vector<std::vector<std::vector<std::pair<float, float>>>> PEAK_LUT = {\n")

    # generate the table

    ATOMICNOS = [1, 6, 7, 8, 9, 15, 16, 17]
    FORMULA_SIZES = {1: 60,
                     6: 60,
                     7: 60,
                     8: 60,
                     9: 60,
                     15: 60,
                     16: 60,
                     17: 60}
                     
    formula_element_peaks = {}
    for an in ATOMICNOS:
        formula_element_peaks[an] = {num: get_peaks_for_elt_num(an, num)
                                     for num in range(1, FORMULA_SIZES[an])}
        
    for an_key in range(np.max(ATOMICNOS)+1):
        fid.write(f"/* an_key={an_key}*/ ")
        if an_key in formula_element_peaks:

            fid.write("{")
            for num in range(np.max(list(formula_element_peaks[an_key].keys()))+1):
                if num not in formula_element_peaks[an_key]:
                    fid.write("   {},\n")
                else:
                    fid.write(f"  /* num={num}*/ " + "{")
                    for p, m in formula_element_peaks[an_key][num]:
                        fid.write(f"{{ {p}, {m} }},")
                        #fid.write(f"{p},")
                    fid.write("},\n")


            fid.write("},\n")
        else:
            fid.write("{},\n")
    fid.write("};\n")
    fid.close()    

def peaks_via_sampling(formula, SAMPLE_N = 100000):
    """
    Roll a weighted die sample_n times and compute the resulting masses. 
    Returns a distribution of masses

    """
    
    masses = np.zeros(SAMPLE_N)
    for atomicno, num in formula.items():
        isos = get_isotopes(atomicno)
        a = np.random.choice([i[2] for i in isos], p=[i[1] for i in isos], size=(SAMPLE_N, num))
        masses += a.sum(axis=1)
    return masses


if __name__ == "__main__":
    # generate the mass element LUT
    create_cpp_lut('floatmasseval_per_elt_peaks.h')
    
