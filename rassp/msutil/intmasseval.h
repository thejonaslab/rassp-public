#pragma once

#include <vector>
#include <tuple>
#include <list>
#include <array>
#include <iostream>
#include <algorithm>
#include "shared.h"

namespace intmasseval {

typedef std::pair<int, float> peak_t;
typedef std::vector<peak_t> peaklist_t; 

// for sorting the peaks
bool inline sort_peaks_desc(const peak_t &a,
                           const peak_t &b)
{
    return (a.second > b.second);
}


struct spectrum_t {
    formula_t formula;
    peaklist_t peaks;

    inline void sort_peaks() {
        // sort peaks in descending mass
        
        // this lives here due to weird segfaulting in cython
        std::sort(peaks.begin(), peaks.end(), sort_peaks_desc);
        
    }
};

/*
TODO: figure out if this is too small.
This is causing errors on molecule: `COSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS`.
eg
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
<ipython-input-15-1b9a5b9e4210> in <module>
----> 1 formulae, masses = self.ffe.get_frag_formulae(mol)

/tank/data/richardzhu/eimspred/simpleforward/msutil/masscompute.py in get_frag_formulae(self, mol)
     75         else:
     76             np_out = masseval.py_get_all_frag_spect_np_nostruct(formula_dict ,
---> 77                                                                 MAX_PEAK_NUM=self.max_peak_num)
     78         # np_out = masseval.py_get_all_frag_spect_np(formula_dict)
     79 

masseval.pyx in msutil.masseval.py_get_all_frag_spect_np_nostruct()

masseval.pyx in msutil.masseval.py_get_all_frag_spect_np_nostruct()

RuntimeError: Number of polynomial coefficients exceeded MAX_POLY_SIZE
*/
#define MAX_POLY_SIZE 64

class poly_t {
public:
    inline poly_t(size_t size) :
        size_(size) {
        if(size > MAX_POLY_SIZE) {
            std::cout << "size=" << size << std::endl;
            throw std::runtime_error("Number of polynomial coefficients exceeded MAX_POLY_SIZE"); 
        }
        for(size_t x = 0; x < size; ++x) {
            poly_[x] = 0.0; 
        }

    }
    inline poly_t(const std::vector<float> & p) 
    {
        size_ = p.size();
        for(int i = 0; i < size_; ++i) {
            poly_[i] = p[i]; 
        }

    }

    inline poly_t(std::initializer_list<float> f) {
        size_ = f.size();
        int i = 0;
        for(auto v : f) { 
            poly_[i] = v;
            i++; 
        }
    }
    
    inline poly_t()    
    {

    }

    inline size_t size() const {
        return size_; 
    }

    inline float & operator[](int idx) {
        return poly_[idx];

    }

    inline float operator[](int idx) const {
        return poly_[idx];

    }


private:
    //std::vector<float> poly_; 
    std::array<float, MAX_POLY_SIZE> poly_;
    size_t size_; 
    

}; 
        
//typedef std::vector<float> poly_t;


class offset_poly_t {
public:
    int offset; 
    poly_t poly;
    inline offset_poly_t() :
        offset(0)
    {

    }
    // inline offset_poly_t(int os, std::vector<float> p) :
    //     offset(os),
    //     poly(p)
    // {

    // }
    inline offset_poly_t(int os, const poly_t & p) :
        offset(os),
        poly(p)
    {

    }
};

typedef std::vector<std::pair<formula_t, offset_poly_t>> frag_poly_t ;



poly_t poly_mul(const poly_t & a, const poly_t & b);

offset_poly_t poly_mul(const offset_poly_t & a,
                       const offset_poly_t & b); 

offset_poly_t poly_coalesce(const offset_poly_t & a, float threshold); 

frag_poly_t get_all_frag_poly(const formula_t & formula); 
std::vector<spectrum_t> get_all_frag_spect(const formula_t & formula);

peaklist_t get_mass_peaks(const formula_t & formula); 
peaklist_t poly_to_peaks(const offset_poly_t & p); 


}; 
