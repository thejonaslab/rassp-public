#include <iostream>
#include <cmath>
#include <vector>
#include <utility>

#include "floatmasseval.h"



namespace floatmasseval {

#include "floatmasseval_per_elt_peaks.h"



const size_t NUM_FORMULA_ELEMENTS = 8; 
const int FORMULA_ORDER[NUM_FORMULA_ELEMENTS] = {1, 6, 7, 8, 9, 15, 16, 17};


std::ostream& operator<<(std::ostream& os, const formula_t & f)
{
    os << " Formula[";
    for(int i = 0; i < NUM_FORMULA_ELEMENTS; ++i)
        {
            if (i > 0) {
                os << " "; 
            }
            os << FORMULA_ORDER[i] << ":" << f.counts[i];

        }
    os << "]"; 
    return os;
}

void recursive_peaklist_prods(const std::vector<const fpeaklist_t*> & input_pl,
                              //std::vector<const fpeaklist_t*> input_pl,
                              int current_pl_pos,
                              float p_in,
                              float m_in,
                              fpeaklist_t &  output_pl) {
    if (current_pl_pos == input_pl.size()) {
        return;
    }

    if (current_pl_pos == (input_pl.size() -1)) {
        for(auto const & pm : *input_pl[current_pl_pos]) {
            output_pl.push_back({m_in + pm.second,
                                 p_in * pm.first,}); 

        }
        
    } else {
        
        for(auto const & pm : *input_pl[current_pl_pos]) {
            float p = pm.first;
            float m = pm.second; 
            if (p_in * p > 0.001) {
                recursive_peaklist_prods(input_pl, current_pl_pos +1,
                                         p_in * p,
                                         m_in + m,
                                         output_pl); 
                
            }
            
        }
    }

}


fpeaklist_t get_formula_peaklist(const formula_t & formula) {
    fpeaklist_t output_list;
    output_list.reserve(10000); 

    std::vector<const fpeaklist_t*> input_pl; // speed up using pointers?
    for (int i = 0; i < NUM_FORMULA_ELEMENTS; ++i) {
        int an = FORMULA_ORDER[i];
        
        assert(an < PEAK_LUT.size());
        assert(formula.counts[i] < PEAK_LUT[an].size());
        if(formula.counts[i] > 0) {
            if (formula.counts[i] >= PEAK_LUT[an].size()){ 
                std::cout << " we don't have an entry for " << formula.counts[i]
                          << " of element " << an << std::endl; 
                throw std::runtime_error("too many atoms of element");
            }
            input_pl.push_back(&(PEAK_LUT[an][formula.counts[i]]));
        }
    }


    recursive_peaklist_prods(input_pl,
                             0,
                             1.0, 0.0,
                             output_list);

    return output_list; 
    
}



std::vector<fspectrum_t> get_all_frag_fspect(const formula_t & formula)
{

    std::vector<formula_t> sub_formulae = generate_sub_formulae(formula);

    std::vector<fspectrum_t> output_peaklist ;
    output_peaklist.reserve(sub_formulae.size());
    
    for(auto f: sub_formulae) {

        fspectrum_t spectrum;
        spectrum.formula = f;
        spectrum.peaks = get_formula_peaklist(f);
        output_peaklist.push_back(spectrum);
    }
    
    return output_peaklist; 

}

std::vector<formula_t> generate_sub_formulae(const formula_t & formula) {
    if(formula.atom_count() == 0) {
        return {formula}; 
    }

    size_t first_nz_formula_idx = 0;
    for(int i = 0; i < NUM_FORMULA_ELEMENTS; ++i) {
        if (formula.counts[i] > 0) {
            first_nz_formula_idx = i; 
            break; 
        }
    }

    std::vector<formula_t> result;
    result.reserve(100);
    for (int i = 0; i <= formula.counts[first_nz_formula_idx]; ++i) {
        formula_t f_new(formula);
        f_new.counts[first_nz_formula_idx] = 0;
        for(auto f_sub: generate_sub_formulae(f_new)) {
            f_sub.counts[first_nz_formula_idx] = i;
            result.push_back(f_sub); 

        }

    }
    return result; 

}



    
}
