#cython: language_level=3

import numpy as np
cimport numpy as np

cimport cython
import math
import time
from libc.math cimport erff, sqrt, abs, floor, fmax
from libcpp.vector cimport vector
from libcpp.utility cimport pair as pair
from libcpp.algorithm cimport sort as stdsort
from libcpp cimport bool


### extra features that need to be fast, not related to massval
@cython.cdivision(True) 
cdef size_t bin_round(float val, float first_bin_center,
          float bin_width, int bin_n):
     """
     helper function for rounding floating point values
     similar to how np.histogram does it
     """

     cdef float s = (val - first_bin_center + bin_width/2.0) / bin_width
     cdef size_t idx = int(floor(s))
     return idx

cpdef py_bin_round(float val, float first_bin_center,
                   float bin_width, int bin_n):
     return bin_round(val, first_bin_center, bin_width, bin_n)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True) 
cpdef mass_bin_peaks_fast(np.float32_t[:, :, :]  peaks_and_bins,
                          float first_bin_center,             
                          float bin_width, int bin_n):
    BATCH_N = peaks_and_bins.shape[0]
    MAX_PEAK = peaks_and_bins.shape[1]
    assert peaks_and_bins.shape[2] == 2
    cdef float mass, intensity
    cdef size_t mass_idx
    cdef size_t num_peaks_in_use
    cdef float total_intensity
    cdef bool was_used_before
    cdef size_t fi = 0
    cdef size_t j = 0 
    cdef np.float32_t[:, :] f
    
    cdef size_t used_bin = 0
    
    cdef vector[float] scratch_hist = vector[float](bin_n) #  = np.zeros(bin_n, dtype=np.float32)
    
    #used_bins = np.ones(bin_n, dtype=np.int64) * -1
    cdef vector[size_t] used_bins = vector[size_t](bin_n)
    output_peak_idx_np = np.zeros((BATCH_N, MAX_PEAK), dtype=np.uint64)
    cdef np.uint64_t[:, :] output_peak_idx = output_peak_idx_np
    output_peak_val_np = np.zeros((BATCH_N, MAX_PEAK), dtype=np.float32)
    cdef np.float32_t[:, :] output_peak_val = output_peak_val_np
    
    #for fi, f in enumerate(peaks_and_bins):
    for fi in range(BATCH_N):
        num_peaks_in_use = 0 
        total_intensity = 0.0
        for j in range(MAX_PEAK):
            mass = peaks_and_bins[fi, j, 0]
            intensity = peaks_and_bins[fi, j, 1]
        #for mass, intensity in f:
            mass_idx = bin_round(mass, first_bin_center, bin_width, bin_n)
            if mass_idx >= 0 and mass_idx < bin_n:
                was_used_before = scratch_hist[mass_idx] > 0
                scratch_hist[mass_idx] += intensity
                total_intensity += intensity
                if not was_used_before:
                    used_bins[num_peaks_in_use] =  mass_idx
                    num_peaks_in_use += 1
        total_intensity = fmax(total_intensity, 1.0e-6)
        # no guarantee on ordering
        for i in range(num_peaks_in_use):
            output_peak_idx[fi, i] = used_bins[i]
            used_bin = used_bins[i]
            output_peak_val[fi, i] = scratch_hist[used_bin] / total_intensity
            scratch_hist[used_bin] = 0.0
            used_bins[i] = -1


    return output_peak_idx_np, output_peak_val_np
                        
     
