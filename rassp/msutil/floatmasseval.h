#pragma once

#include <vector>
#include <tuple>
#include <list>
#include <array>
#include <iostream>
#include <assert.h>
#include <algorithm>
#include "shared.h"

namespace floatmasseval {

// peak is float
typedef float mass_t;
typedef float intensity_t; 
typedef std::pair<mass_t, intensity_t> fpeak_t;
typedef std::vector<fpeak_t> fpeaklist_t; 

// for sorting the peaks by intensity
bool inline sort_peaks_desc(const fpeak_t &a,
                           const fpeak_t &b)
{
    return (a.second > b.second);
}


struct fspectrum_t {
    formula_t formula;
    fpeaklist_t peaks;

    inline void sort_peaks() {
        // sort peaks in descending mass
        
        // this lives here due to weird segfaulting in cython
        std::sort(peaks.begin(), peaks.end(), sort_peaks_desc);
        
    }
};

fpeaklist_t get_formula_peaklist(const formula_t & formula); 

std::vector<fspectrum_t> get_all_frag_fspect(const formula_t & formula);

std::vector<formula_t> generate_sub_formulae(const formula_t & formula);


}; 
