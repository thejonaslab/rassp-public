Masseval high res is about 4x slower than the int-based one

25% of that time is `parse_frag_spect_f`, the vast majority of which 
is taken up in the sorting of peaks. 



TODO:
- [ ] test `generate_sub_formulae`
- [ ] Make `generate_sub_formulae` faster
- [ ] pass constant reference vectors
- [ ] reserve capacity everywhere you can 

