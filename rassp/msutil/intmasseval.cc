#include <iostream>
#include <cmath>
#include "intmasseval.h"

namespace intmasseval {

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


const float COALESCE_THRESHOLD = 0.00001;

offset_poly_t get_element_poly(atomicno_t atomicno) {
    switch(atomicno) {
    case 6:
        return offset_poly_t(12, {0.9893f, 0.0107f}); 
    case 1:
        return offset_poly_t(1, {0.999885f, 0.000115f}); 
    case 7:
        return offset_poly_t(14, {0.9963f, 0.00368f}); 
    case 8:
        return offset_poly_t(16, {0.99975f, 0.00038f,0.00205f});
    case 9:
        return offset_poly_t(19, {1.00f});
    case 15:
        return offset_poly_t(31, {1.0f}); 
    case 16:
        return offset_poly_t(32, {0.9493f, 0.0076f, 0.0429f, 0.0f, 0.0002f});
    case 17:
        return offset_poly_t(35, {0.7578f, 0, 0.2422f}); 
        
    default:
        throw std::runtime_error("unknown atomicno"); 

    }; 

}

const offset_poly_t ONEPOLY(0, {1.0f});
    
offset_poly_t get_formula_poly(const formula_t & formula){

    offset_poly_t outpoly = ONEPOLY;

    for (size_t i = 0; i < NUM_FORMULA_ELEMENTS; ++i) {
        for (size_t j = 0; j < formula.counts[i]; ++j) {
            outpoly = poly_mul(outpoly, get_element_poly(FORMULA_ORDER[i]));
        }
    }
    return outpoly; 
}

peaklist_t get_mass_peaks(const formula_t & formula){

    offset_poly_t outpoly  = get_formula_poly(formula);
    return poly_to_peaks(outpoly);
}

peaklist_t poly_to_peaks(const offset_poly_t & p) {
    peaklist_t out;
    out.reserve(p.poly.size());
    for(size_t i = 0; i < p.poly.size(); ++i) {
        out.push_back(peak_t(i + p.offset, p.poly[i])); 
    }
    return out; 
}

poly_t poly_mul(const poly_t & a, const poly_t & b) {
    size_t M = a.size();
    size_t N = b.size();
    
    poly_t output(M + N - 1);
    for(size_t i = 0; i < M + N-1; ++i) {
        for (size_t m = 0; m < M; ++m) { 
            size_t bpos = i -m ;
            if ((bpos >= 0) & (bpos <N)) {
                output[i] += a[m] * b[bpos]; 
            }
        }
            
    }
    return output;
}

                                                    
offset_poly_t poly_mul(const offset_poly_t & a,
                       const offset_poly_t & b) {

    // FIXME should we also do trimming here? 
    return offset_poly_t(a.offset + b.offset,
                         poly_mul(a.poly, b.poly)); 
        
}


frag_poly_t get_elemental_frag_poly(int formula_idx, int num) {
    /*
     */

    frag_poly_t out;
    for (int i = 0; i <= num; ++i) {
        formula_t new_formula;
        new_formula.counts[formula_idx] = i;
        offset_poly_t p = get_formula_poly(new_formula);
        if (p.poly.size() > 5) {
            p = poly_coalesce(p, COALESCE_THRESHOLD); 

        }
        out.push_back(std::make_pair(new_formula, p)); 
    }

    return out; 
}
std::vector<spectrum_t> get_all_frag_spect(const formula_t & formula){
    std::vector<spectrum_t> out;

    for(auto fp: get_all_frag_poly(formula)) {
        spectrum_t s;
        s.formula = fp.first;
        s.peaks = poly_to_peaks(fp.second);
        out.push_back(s); 
    }
    return out;
    
}

frag_poly_t get_all_frag_poly(const formula_t & formula){
    /*    
          for the first non-zero element in the formula
          create all possible spectra for those atomic nos

          and then set that one to zero and get the result 
          
    */

    frag_poly_t out;
    // find first nonzer
    int tgt_idx = 0;
    for(size_t  i = 0; i < NUM_FORMULA_ELEMENTS; i++) {
        if(formula.counts[i] > 0) {
            tgt_idx = i; 
            break; 
        }
    }
    frag_poly_t single_elt_spects = get_elemental_frag_poly(tgt_idx, formula.counts[tgt_idx]);     
    // check if this is the last set one
    if (formula.counts[tgt_idx] == formula.atom_count() ) {
        // base case
        return single_elt_spects; 
    }
    
    formula_t sub_formula = formula;
    // zero out the current one 
    sub_formula.counts[tgt_idx] = 0;


    frag_poly_t intermediate_spects = get_all_frag_poly(sub_formula);


    // all possible combinations
    for(auto single_spect : single_elt_spects){
        for (auto intermediate_spect : intermediate_spects) {
            // combine them
            formula_t new_formula = single_spect.first + intermediate_spect.first;

            offset_poly_t new_p  = poly_mul(single_spect.second, intermediate_spect.second);
            if (new_p.poly.size() > 5) { 
                new_p = poly_coalesce(new_p, COALESCE_THRESHOLD); 
            }
            out.push_back(std::make_pair(new_formula, new_p)); 
            
        }
    }

    
    return out; 
    
}

offset_poly_t poly_coalesce(const offset_poly_t & a, float threshold) {
    
    // for an offset polynomial, remove leading and trailing values < threshold.
    // and update the offset

    size_t start_pos = 0;
    size_t end_pos = 0;
    
    for(size_t i = 0; i < a.poly.size(); ++i) {
        if(std::fabs(a.poly[i]) < threshold) {
            if (start_pos  == i) {
                start_pos++; 
            }
        } else {
            end_pos = i; 
        }
    }

    size_t new_poly_size = (1+ end_pos) - start_pos; 
    poly_t new_poly(new_poly_size);
    for(size_t i = 0; i < new_poly_size; ++i) {
        new_poly[i] = a.poly[start_pos + i]; 
    }

    return offset_poly_t(a.offset + start_pos,
                         new_poly); 
            
        
}



    
}
