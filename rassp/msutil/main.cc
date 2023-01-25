#include <iostream>
#include "floatmasseval.h"
#include <chrono>

using namespace     floatmasseval;

int main(int argc, char ** argv)
{
    int ITERS = 100;
    
    formula_t f;
    f.counts[0] = 20;
    f.counts[1] = 20;
    f.counts[2] = 5;
    f.counts[3] = 5;
    f.counts[7] = 5;
    auto start = std::chrono::steady_clock::now();    
    for(int i = 0; i < ITERS; ++i){ 
        auto res = get_all_frag_fspect(f);
    }
    auto end =  std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;

    std::cout << "elapsed time: " << elapsed_seconds.count()*1000/ITERS << "ms per iter\n";



}
