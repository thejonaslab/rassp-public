import numpy as np
from typing import Mapping

EPS = 1e-10

# ----------------------------------------------------------------
# utils for computing metrics on EI-MS spectra using bins
# to aggregate and match peaks
def spect_to_bins(spect_dict, bins):
    """
    Compute the bins of a spectrum dict. Bins is a monotonically-increasing
    array of bin-edges, with each bin being [, ). 
    """
    mz = list(spect_dict.keys())
    v = [spect_dict[k] for k in mz]

    h, _ = np.histogram(mz, bins=bins, weights=v)
    return h

def norm_vect(x, ord=2):
    """
    Return a normed version of the vector
    """
    return x / np.linalg.norm(x, ord=ord)

def bin_spectra(true, pred, bins, max_bin):
    """True and pred are both dictionaries of {mz: intensity}.
    
    For EI-MS, the default binning should be rounding mass values
    to the nearest integer. This is accomplished by setting the 
    bin edges at intervals of 1, starting from -0.5.
    """
    if max_bin is None:
        max_bin = np.max(list(true.keys()) + list(pred.keys()))
    if bins is None:
        # assume integer bins
        bins = np.arange(int(np.ceil(max_bin))) - 0.5

    true_binned = norm_vect(spect_to_bins(true, bins))
    pred_binned = norm_vect(spect_to_bins(pred, bins))
    return true_binned, pred_binned

def convert_list_to_sdict(alist):
    d = {}
    for a, b in alist:
        d[float(a)] = float(b)
    return d

def convert_svect_to_sdict(svect, cutoff=0.0):
    # normalize svect first...
    svect = svect / np.sum(svect)

    d = {}
    tot_mass = 0.0
    for m, i in enumerate(svect):
        intensity = float(i)
        if intensity >= cutoff:
            d[float(m) + 1] = intensity
            tot_mass += intensity
    
    for key in d.keys():
        d[key] /= tot_mass
    return d

def convert_list_to_svect(alist):
    # round the masses to the nearest integer...
    svect = np.zeros(512)
    for m, i in alist:
        m = round(m)
        svect[int(m - 1)] = float(i)
    return svect / np.sum(svect)

def get_tup_rep(d):
    # represent a sdict as tuple of (sorted masses, intensities)
    masses = np.array(sorted(d.keys()))
    intensities = np.array([d[m] for m in masses])
    return masses, intensities

def resort_by_intensity_decreasing(t, k=5):
    masses, intensities = t
    sort_inds = np.argsort(intensities)[::-1][:k]
    return (
        masses[sort_inds],
        intensities[sort_inds],
    )

def round_sdict_masses(sdict):
    new_sdict = {}
    for m, i in sdict.items():
        mp = round(m)
        if mp in new_sdict:
            new_sdict[mp] += float(i)
        else:
            new_sdict[mp] = float(i)
    return new_sdict

# ----------------------------------------------------------------
# metrics that are computed against pairs of true/pred spectra represented as dictionaries
def sdp(true: Mapping[float, float], pred: Mapping[float, float], bins=None, max_bin=None):
    """
    Stein dot product. True and pred are dictionaries of {mz: intensity }
    """
    true_binned, pred_binned = bin_spectra(true, pred, bins, max_bin)
    return dot_product(true_binned, pred_binned)

def dp(true: Mapping[float, float], pred: Mapping[float, float], bins=None, max_bin=None):
    """
    Regular dot product 
    """
    true_binned, pred_binned = bin_spectra(true, pred, bins, max_bin)
    return dot_product(true_binned, pred_binned, 0.5, 0.5)

def dot_product(spec1, spec2, mass_pow=3, intensity_pow=0.6, mass_val_array=None):
    """
    Weighted dot product per Stein and Scott, used for
    searching against databases
    
    spec1 and spec2 are vectors of intensities. The masses
    are in mass_val_array; if mass_val_array is None we assume integer-valued
    masses with spec1[0] = spec2[0] == 0 and increasing by integer values
    """
    assert len(spec1) == len(spec2)
    N = len(spec1)
    if mass_val_array is None:
        mass_val_array = np.arange(N).astype(np.float64)
        
    wl = mass_val_array ** mass_pow * spec1**intensity_pow
    wu = mass_val_array ** mass_pow * spec2**intensity_pow
    return np.sum(wl*wu) / (np.linalg.norm(wl) * np.linalg.norm(wu))

def topk_precision(
    d_pred: Mapping[float, float],
    d_true: Mapping[float, float],
    k=5,
    round_pred_masses=True
):
    """How many of the top-K peaks we predict are also in the top-K peaks in the true spectrum."""
    if round_pred_masses:
        d_pred = round_sdict_masses(d_pred)

    t_pred_topk = resort_by_intensity_decreasing(get_tup_rep(d_pred), k=k)
    t_true_topk = resort_by_intensity_decreasing(get_tup_rep(d_true), k=k)    
    n_matched = len(set(t_pred_topk[0]).intersection(set(t_true_topk[0])))
    return n_matched / k

def intensity_weighted_barcode_precision(
    d_pred: Mapping[float, float],
    d_true: Mapping[float, float],
    frac_true_cutoff=1e-3,
    round_pred_masses=True
):
    """Fraction of total predicted spectral mass contained in bins SEEN in the true spectrum.

    ~ SumIntensity(Predicted Bin in True Spectrum) / SumIntensity(Predicted Bin)
    
    If support matches the true spectrum, value is 1.0
    
    Intensity-weighted precision = true positives
    """
    if round_pred_masses:
        d_pred = round_sdict_masses(d_pred)
    t_true = get_tup_rep(d_true)

    # normalize
    mm, ii = t_true
    ii = ii / np.sum(ii)

    # filter out small peaks
    zero_mask = ii <= frac_true_cutoff
    ii[zero_mask] = 0.0
    nonzero_mask = ~zero_mask
    mm = mm[nonzero_mask]
    ii = ii[nonzero_mask]
    ii = ii / np.sum(ii)
    t_true = mm, ii
    # print(f'filtered from {len(nonzero_mask)} to {sum(nonzero_mask)}')

    true_masses = set(mm)
    total_true_weight = 0.0
    total_weight = 0.0
    for m, i in d_pred.items():
        total_weight += i
        if m in true_masses:
            total_true_weight += i
    return total_true_weight / (total_weight + EPS)

def intensity_weighted_barcode_false_positive_rate(
    d_pred: Mapping[float, float],
    d_true: Mapping[float, float],
    frac_true_cutoff=1e-3,
    round_pred_masses=True,
):
    """Fraction of total predicted spectral mass contained in bins NOT SEEN in the true spectrum.

    ~ SumIntensity(Predicted Bin NOT IN True Spectrum) / SumIntensity(Predicted Bin)

    If support matches the true spectrum, value is 0.0
    
    Intensity-weighted false-positive rate = false positives
    """
    return 1 - intensity_weighted_barcode_precision(d_pred, d_true, frac_true_cutoff, round_pred_masses)

# ----------------------------------------------------------------
# summary statistics associated with individual spectra
def topk_frac(d: Mapping[float, float], k=5):
    """Fraction of total spectrum intensity contained in the top_k peaks."""
    t = get_tup_rep(d)
    t_topk = resort_by_intensity_decreasing(t, k=k)

    all_mass = np.sum(t[1])
    topk_mass = np.sum(t_topk[1])
    return topk_mass / all_mass

def spect_entropy(d: Mapping[float, float]):
    """Entropy of spectral distribution.

    We need to normalize by the max entropy for the # of bins, since the entropy
    scales with the number of bins!
    """
    from scipy.stats import entropy

    mm, ii = get_tup_rep(d)
    ii = ii / np.sum(ii)
    
    max_ent = np.log(len(ii))
    ent = entropy(ii)
    return ent / max_ent

# ----------------------------------------------------------------
# old implementation of SDP and DP
def sdp_old(x, y):
    x = norm_vect(x)
    y = norm_vect(y)
    return dot_product(x, y)

def dp_old(x, y):
    x = norm_vect(x)
    y = norm_vect(y)
    return dot_product(x, y, 0.5, 0.5)
    
# ----------------------------------------------------------------
# other metrics, like recall
def unweighted_recall(sdict, fragment_weights, frac_intensity_cutoff=1e-3):
    """Get unweighted recall (fraction of NIST spectral peaks with int masses accounted for in fraggraph output)"""
    int_fraggraph_weights = []
    for w in fragment_weights:
        int_fraggraph_weights += [int(w), np.ceil(w)]
    int_fraggraph_weights = set(int_fraggraph_weights)
    
    int_nist_weights = []
    total_mass = sum([w for m, w in sdict.items()])
    for m, w in sdict.items():
        m = int(m)
        if w / total_mass >= frac_intensity_cutoff:
            int_nist_weights.append(m)
    
    n_in_fraggraph = sum([w in int_fraggraph_weights for w in int_nist_weights])
    return n_in_fraggraph / len(int_nist_weights)

def intensity_weighted_recall(sdict, fragment_weights, frac_intensity_cutoff=1e-3):
    """Get intensity-weighted recall.
    
    Fraction of total intensity (L1 norm) that is accounted for by 
    fraggraph fragments.
    """
    int_fraggraph_weights = []
    for w in fragment_weights:
        int_fraggraph_weights += [int(w), np.ceil(w)]
    int_fraggraph_weights = set(int_fraggraph_weights)
    
    int_nist_weights = []
    total_mass = sum([w for m, w in sdict.items()])
    for m, w in sdict.items():
        m = int(m)
        if w / total_mass >= frac_intensity_cutoff:
            int_nist_weights.append((m, w))
    
    mass_accounted_for = 0.0
    total_mass = 0.0
    for m, w in int_nist_weights:
        if m in int_fraggraph_weights:
            mass_accounted_for += w
        total_mass += w
    
    return mass_accounted_for / total_mass

# ----------------------------------------------------------------
# reproduced metrics from wishart code
def wishart_dot_product(pred_masses, pred_intensities, true_masses, true_intensities, mass_pow=0.5, intensity_pow=0.5):
    """Reproduction of the discrete peak-matching dot product calculation from Wishart code.
    
    Original Stein dot product is computed with masspow 3 and intensity pow 0.6.

    NOTE: this code is not optimized!
    """
    # dynamic programming calculation to match up peak pairs between
    # the pred and true spectra (symmetric in inputs)
    # dp_val[i, j] stores the path that matches up 
    # the first i peaks in pred to first j peaks in true
    dp_vals = np.zeros((len(pred_masses) + 1, len(true_masses) + 1))
    for i, mi, wi in zip(range(1, len(pred_masses) + 1), pred_masses, pred_intensities):
        for j, mj, wj in zip(range(1, len(true_masses) + 1), true_masses, true_intensities):
            best_val = 0.0
            mass_tol = (mi / 1000000.0) * 10
            if np.abs(mj - mi) <= mass_tol:
                best_val = dp_vals[i - 1, j - 1] + 0.5 * (wi + wj)

            if dp_vals[i - 1, j] > best_val:
                best_val = dp_vals[i - 1, j]
            if dp_vals[i, j - 1] > best_val:
                best_val = dp_vals[i, j - 1]

            dp_vals[i, j] = best_val

    peak_pairs = []

    i = len(pred_masses)
    j = len(true_masses)
    eps = 1e-12
    while i > 0 and j > 0:
        if np.abs(dp_vals[i, j] - dp_vals[i, j - 1]) < eps:
            j -= 1
        elif np.abs(dp_vals[i, j] - dp_vals[i - 1, j]) < eps:
            i -= 1
        elif dp_vals[i, j] > dp_vals[i - 1, j - 1] + eps:
            peak_pairs.append(
                (
                    (pred_masses[i - 1], pred_intensities[i - 1]),
                    (true_masses[j - 1], true_intensities[j - 1]),
                )
            )
            i -= 1
            j -= 1

    def get_adjusted_intensity(peak, mass_pow=mass_pow, intensity_pow=intensity_pow):
        mass, weight = peak
        return np.power(weight, intensity_pow) * np.power(mass, mass_pow)
    
    def get_total_intensity(masses, intensities):
        val = 0.0
        for mass, weight in zip(masses, intensities):
            i = get_adjusted_intensity((mass, weight))
            val += i * i
        return val

    if len(peak_pairs) == 0:
        return 0.0

    num = 0.0
    for pred, true in peak_pairs:
        num += get_adjusted_intensity(pred) * get_adjusted_intensity(true)
    denom_p = get_total_intensity(pred_masses, pred_intensities)
    denom_q = get_total_intensity(true_masses, true_intensities)
    return (num * num) / (denom_p * denom_q)


def l1(true, pred, bins=None, max_bin=None):
    """
    Compute l1 norm of delta
    assume true and pred have unit l1 norm 
    """
    true_binned, pred_binned = bin_spectra(true, pred, bins, max_bin)
    return np.linalg.norm(true_binned - pred_binned, ord=1)

    
def invalid_mass(true, pred, bins=None, max_bin=None, thold=1e-3):
    """
    How much of the mass in pred is in bins where true has < threshold mass? 
    
    (Effectively how much mass are we assigning to invalid bins)
    assume true and pred have unit l1 norm 

    """
    true_binned, pred_binned = bin_spectra(true, pred, bins, max_bin)

    mask = true_binned < thold
    assert np.sum(pred_binned) <= 1.0001, f"sum was {np.sum(pred_binned)}"
    assert np.sum(true_binned) <= 1.0001, f"sum was {np.sum(true_binned)}"
    
    return np.sum(pred_binned[mask])


def top_k(true, pred, bins=None, max_bin=None, k=1):
    """
    what fraction of the top-k in true are in the top-k in pred
    """
    true_binned, pred_binned = bin_spectra(true, pred, bins, max_bin)

    top_k_true = set(np.argsort(true_binned)[::-1][:k])
    top_pred_true = set(np.argsort(pred_binned)[::-1][:k])

    return len(top_k_true.intersection(top_pred_true))
