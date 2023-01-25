"""
Generic utilities for binning spectra with exact precision. 

"""
import numpy as np
import rassp.binutils_fast as fast

class SpectrumBins:
    """
    A class that contains configuration about how we are binning
    our spectrum. This should be the canonical source of binning
    information, and all functions dependent on mapping from 
    continuous peaks ->binned spectra should use this. 

    """

    def __init__(self, first_bin_center,
                 bin_width, bin_number):
        self.first_bin_center = first_bin_center
        self.bin_width = bin_width
        self.bin_number = bin_number

        self.bin_centers = np.arange(self.bin_number) * self.bin_width + self.first_bin_center
        
        # we only support partitions right now, that is
        # spectral bins with no gaps
        self._is_partition = True

    def config(self):
        return {'first_bin_center' : self.first_bin_center,
                'bin_width' : self.bin_width,
                'bin_number' : self.bin_number}
    
    def get_value_range(self):
        """
        Returns the smallest value that could go in the first bin and
        the outer edge of the largest bin
        
        that is [min, max)
        """
        return (self.first_bin_center - self.bin_width/2.0, 
                self.first_bin_center + self.bin_number * self.bin_width + self.bin_width/2.0)

    def get_bin_width(self):
        return self.bin_width
    
    def get_num_bins(self):
        return self.bin_number
    
    def get_bin_centers(self):
        """
        Get the centers of the bins
        """
        return np.arange(self.bin_number)*self.bin_width + self.first_bin_center
        
    def is_partition(self):
        """
        Returns true if this is a partitioning of a range (that is, no gaps)

        Right now always returns true. 
        """
        return self._is_partition

    def __getitem__(self, value):
        """
        SLOW returns the bin associated with a value, or -1 
        if none
        """
        
        return self.to_bins([value])[0]

    def to_bins(self, values):
        """
        Returns the integer bin that a particular mass value maps to, or
        -1 for outside of the range. 
        """
        values = np.array(values)
        
        binned_value_min, binned_value_max = self.get_value_range()
        value_range = self.bin_width * self.bin_number
        
        value_bins = (values -self.first_bin_center + self.bin_width/2) / self.bin_width
        value_bins_int = np.floor(value_bins).astype(np.int32)
        value_bins_int[(value_bins_int < 0) | (value_bins_int >= self.bin_number)] = -1
        return value_bins_int

    def histogram(self, masses, intensities):
        """
        Bin "masses" into the appropriate bins weighted with intensities, 
        and then normalize the entire resulting histogrammed spectrum
        to have unit mass. 

        Returns :
         (locations of non-zero bins, 
          values at those non-zero bins, 
          dense histogram of values)
        """
        target_bins = self.to_bins(masses)

        target_bins_valid = target_bins >= 0
        h, _ = np.histogram(target_bins[target_bins_valid],
                            bins=np.arange(self.bin_number+1),
                            weights=intensities[target_bins_valid])
        idx = np.argwhere(h).flatten()
        
        total_p = max(np.sum(h[idx]), 1e-6)
        p = h[idx] / total_p 
        dense_out = h / total_p
        return idx, p, dense_out

        

def create_spectrum_bins(**config):
    """
    Create a spectrum config. Right now just a wrapper around
    instantiating the class directly. 
    
    """
    
    return SpectrumBins(**config)



class MassPeaksToBins:
    """
    Take in an array of N spectra of real-valued mass/intensity pairs
    and discretize them into bins, returning index/intensity pairs. 

    
    
    """
    def __init__(self, first_bin_center, bin_width, bin_number):
        self.sb = SpectrumBins(first_bin_center, bin_width, bin_number)
        
    def __call__(self, peaks_and_bins):
        """
        peaks_and_bins are dimensions BATCH_N x MAX_PEAK x 2 

        BATCH_N: total number of peak sets
        MAX_PEAKS : maximum number of peaks in a spectrum
        x2 : (mass, intensity)

        """
        BATCH_N, MAX_PEAK, _ = peaks_and_bins.shape
        assert peaks_and_bins.shape[2] == 2
        
        output_peak_idx = np.zeros((BATCH_N, MAX_PEAK), dtype=np.int64)
        output_val = np.zeros((BATCH_N, MAX_PEAK), dtype=np.float32)
        for fi, f in enumerate(peaks_and_bins):
            idx, p, _ = self.sb.histogram(f[:, 0], f[:, 1])
            output_peak_idx[fi, :len(idx)] = idx
            output_val[fi, :len(idx)] = p
        return output_peak_idx, output_val



class MassPeaksToBinsFast:
    """
    Take in an array of N spectra of real-valued mass/intensity pairs
    and discretize them into bins, returning index/intensity pairs. 


    This is the fast cython version of MassPeaksToBins
    
    """
    def __init__(self, first_bin_center, bin_width, bin_number):
        self.first_bin_center = first_bin_center
        self.bin_width = bin_width
        self.bin_number = bin_number
        
    
    def __call__(self, peaks_and_bins):
        
        output_peak_idx, output_peak_val = fast.mass_bin_peaks_fast(peaks_and_bins, 
                                                               self.first_bin_center,
                                                               self.bin_width,
                                                               self.bin_number)

            
        return output_peak_idx, output_peak_val     
        

def create_peaks_to_bins(spectrum_bins):
    """
    Factory for creating the mass-peaks-to-bins object, 
    for future expansion/configuraiton
    """

    mptbf = MassPeaksToBinsFast(spectrum_bins.get_bin_centers()[0],
                                spectrum_bins.get_bin_width(),
                                spectrum_bins.get_num_bins())
    return mptbf
