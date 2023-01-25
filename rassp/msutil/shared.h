#pragma once

#include <vector>
#include <tuple>
#include <list>
#include <array>
#include <iostream>
#include <algorithm>

typedef char atomicno_t;

const size_t NUM_FORMULA_ELEMENTS = 8; 
const int FORMULA_ORDER[NUM_FORMULA_ELEMENTS] = {1, 6, 7, 8, 9, 15, 16, 17};

class formula_t {
public:
    size_t counts[NUM_FORMULA_ELEMENTS];
    //uint16_t counts[NUM_FORMULA_ELEMENTS];
    inline formula_t() {
        for(size_t i = 0; i < NUM_FORMULA_ELEMENTS; ++i) {
            counts[i] = 0; 
        }

    }
    inline size_t atom_count() const{
        size_t sum = 0; 
        for(size_t i = 0; i < NUM_FORMULA_ELEMENTS; ++i) {
            sum += counts[i]; 
        }
        return sum; 
    }

    inline formula_t operator+(const formula_t & of) {
        formula_t new_f;
        for(size_t i = 0; i < NUM_FORMULA_ELEMENTS; ++i) {
            new_f.counts[i] = counts[i] + of.counts[i];  
        }
        
        return new_f; 
      }
          
}; 

