#cython: language_level=3

import numpy as np
cimport numpy as np

cimport cython
import math
import time
from libc.math cimport erff, sqrt, abs, floor
from libcpp.vector cimport vector
from libcpp.utility cimport pair as pair
from libcpp.algorithm cimport sort as stdsort
from libcpp cimport bool

cdef extern from "shared.h":
     cdef size_t NUM_FORMULA_ELEMENTS
     cdef int * FORMULA_ORDER
     
     cdef cppclass formula_t:
          formula_t() except +
          size_t * counts

cdef extern from "intmasseval.h" namespace "intmasseval":
     cdef cppclass poly_t:
          poly_t(int size) except +
          poly_t(vector[float] p) except +
          size_t size()
          float operator[](int idx)
          
     cdef cppclass offset_poly_t:
          offset_poly_t() except +
          offset_poly_t(int offset, const poly_t & poly) except +
          int offset
          poly_t poly
          
     cdef cppclass spectrum_t:
          formula_t formula
          vector[pair[int, float]] peaks
          void sort_peaks()
          
     cdef offset_poly_t poly_mul(const offset_poly_t &, const offset_poly_t &) except +

     cdef vector[pair[int, float] ] get_mass_peaks(const formula_t &) except +

     cdef vector[spectrum_t] get_all_frag_spect(const formula_t &) except +


     cdef offset_poly_t poly_coalesce(const offset_poly_t & a, float threshold)

     cdef bool sort_peak_desc(const pair[int, float] & a, const pair[int, float] & b)

cdef extern from "floatmasseval.h" namespace "floatmasseval":
     cdef vector[formula_t] generate_sub_formulae(const formula_t &)

     cdef cppclass fspectrum_t:
          formula_t formula
          vector[pair[float, float]] peaks
          void sort_peaks()

     cdef vector[fspectrum_t] get_all_frag_fspect (const formula_t &) except +
          
cdef offset_poly_t  vect_to_offset_poly(int os, float[:] a):
     cdef vector[float] a_v;
     for v in a:
         a_v.push_back(v)

     cdef offset_poly_t o = offset_poly_t(os, poly_t(a_v))
     return o

 
cpdef py_poly_mul(float[:] a, float[:] b):

     # cdef vector[float] a_v;
     # for v in a:
     #     a_v.push_back(v)

     # cdef vector[float] b_v
     # for v in b:
     #     b_v.push_back(v)

         
     cpdef offset_poly_t p1 = vect_to_offset_poly(0, a)
     cpdef offset_poly_t p2 = vect_to_offset_poly(0, b)
     cpdef offset_poly_t out = poly_mul(p1, p2)

     s = []
     for p in range(out.offset):
         s.append(0)
     for i in range(out.poly.size()):
         s.append(out.poly[i])
     return s


cdef class OffsetPoly:
     cdef offset_poly_t offsetpoly;
     def __cinit__(self, int offset, float[:] a):
         self.offsetpoly = vect_to_offset_poly(offset, a)
         
     # def __cinit__(self):

     #     pass
     
     def dense(self):
         out = []
         for i in range(self.offsetpoly.offset):
             out.append(0)
         for j in range(self.offsetpoly.poly.size()):
             out.append(self.offsetpoly.poly[j])
         return out

     def __mul__(OffsetPoly self, OffsetPoly other):
         cdef offset_poly_t result = poly_mul(self.offsetpoly, other.offsetpoly)
         cpdef OffsetPoly outp = OffsetPoly(0, np.array([], dtype=np.float32))
         outp.offsetpoly = result
         
         return outp

     def coalesce(OffsetPoly self, float thold):

         cdef offset_poly_t  out = poly_coalesce(self.offsetpoly, thold)
         cpdef OffsetPoly outp = OffsetPoly(0, np.array([], dtype=np.float32))
         outp.offsetpoly = out
         
         return outp


     @property
     def offset(OffsetPoly self):
         return self.offsetpoly.offset
    
     @property
     def coeff(OffsetPoly self):
         out = []
         
         for i in range(self.offsetpoly.poly.size()):
             out.append(self.offsetpoly.poly[i])
         return out

         
atomicno_formula_pos = {}
for i in range(NUM_FORMULA_ELEMENTS):
    atomicno_formula_pos[FORMULA_ORDER[i]] = i

cdef class Formula:
    cdef formula_t formula;
    def __cinit__(self, atomicno_num):
        self.formula = formula_t()
        for k, v in atomicno_num.items():
            self.formula.counts[atomicno_formula_pos[k]] = v

cpdef py_get_mass_peaks(atomicno_num):
     f = Formula(atomicno_num)

     cdef vector[pair[int, float] ] pl = get_mass_peaks(f.formula)
     out = {}
     for k in pl:
         out[k.first] = k.second

     return out

cdef formula_to_dict(formula_t f):
     out = {}
     for i in range(NUM_FORMULA_ELEMENTS):
         if f.counts[i] > 0:
             out[FORMULA_ORDER[i]] = f.counts[i]
     return out
     
cdef peaks_to_list(vector[pair[int, float]] peaks):
     out = []
     for p in peaks:
         out.append((p.first, p.second))
     return out

 
def py_get_all_frag_spect(atomicno_num):
     f = Formula(atomicno_num)

     t1 = time.time()
     cdef vector[spectrum_t] out = get_all_frag_spect(f.formula)
     t2 =time.time()

     pyout = []
     for i in range(out.size()):
         pyout.append((formula_to_dict(out[i].formula), peaks_to_list(out[i].peaks)))
         
     return pyout

cpdef second(x):
      return x[1]



cpdef py_get_all_frag_spect_np(atomicno_num, MAX_ATOMICNO=20, MAX_PEAK_NUM=6):
     f = Formula(atomicno_num)
     t1 = time.time()
     cdef vector[spectrum_t] out = get_all_frag_spect(f.formula)
     t2 = time.time()

     cdef pair[int, float] p
     cdef formula_t frag_f

     np_peak_dt = np.dtype([('mass' , np.int16), ('intensity', np.float16)])
     
     np_spect_dt = [('formula', 'u8', (MAX_ATOMICNO, )), ('peaks', np_peak_dt, MAX_PEAK_NUM)]

     npout = np.zeros(out.size(), dtype=np_spect_dt)

     pyout = []
     for i in range(out.size()):
         for fi in range(NUM_FORMULA_ELEMENTS):
             frag_f = out[i].formula
             if frag_f.counts[fi] > 0:
                 npout[i]['formula'][FORMULA_ORDER[fi]] = frag_f.counts[fi]
         mps = []
         for p in out[i].peaks:
             mps.append([p.first, p.second])
         mps = sorted(mps, key=second)[::-1]
         
         for pi, (mass, intensity) in zip(range(MAX_PEAK_NUM), mps):
             npout[i]['peaks'][pi]['mass'] = mass
             npout[i]['peaks'][pi]['intensity'] = intensity
     t3 = time.time()
     #print(f"get frags : {(t2-t1)*1000:3.3f}ms clean frags : {(t3-t2)*1000:3.3f} ms ")
     
     return npout
 

cpdef py_get_all_frag_spect_np_nostruct(atomicno_num, MAX_ATOMICNO=20, MAX_PEAK_NUM=6):
     # faster, non-struct version? 
     f = Formula(atomicno_num)
     #t1 = time.time()
     cdef vector[spectrum_t] frag_spect = get_all_frag_spect(f.formula)
     #t2 = time.time()

     cdef pair[int, float] p
     cdef formula_t frag_f

     np_peak_dt = np.dtype([('mass' , np.int16), ('intensity', np.float16)])
     
     np_spect_dt = [('formula', 'u8', (MAX_ATOMICNO, )), ('peaks', np_peak_dt, MAX_PEAK_NUM)]

     formula_out = np.zeros((frag_spect.size(), MAX_ATOMICNO),  dtype=np.uint8)
     mass_out = np.zeros((frag_spect.size(), MAX_PEAK_NUM), dtype=np.int16)
     intensity_out = np.zeros((frag_spect.size(), MAX_PEAK_NUM), dtype=np.float32)

     parse_frag_spect(frag_spect, formula_out, mass_out, intensity_out)
     
     npout = np.zeros(frag_spect.size(), dtype=np_spect_dt)

     npout[:]['formula'] = formula_out
     npout['peaks']['mass'] = mass_out
     npout['peaks']['intensity'] = intensity_out
     #t3 = time.time()

     #print(f"new get frags : {(t2-t1)*1000:3.3f}ms clean frags : {(t3-t2)*1000:3.3f} ms ")

     return npout
 
@cython.boundscheck(False)
@cython.wraparound(False)
cdef parse_frag_spect(vector[spectrum_t] input_peaks,
                      np.uint8_t [:, :] formulae,
                      np.int16_t[:, :] mass,
                      np.float32_t[:, :] intensity):

     cdef size_t i = 0
     cdef size_t fi = 0
     cdef size_t j = 0
     cdef float largest_intensity = -1
     
     for i in range(input_peaks.size()):
        for fi in range(NUM_FORMULA_ELEMENTS):
             if input_peaks[i].formula.counts[fi] > 0:
                formulae[i, FORMULA_ORDER[fi]] = input_peaks[i].formula.counts[fi]

        input_peaks[i].sort_peaks()
        for j in range(min(mass.shape[1], input_peaks[i].peaks.size())):
            mass[i, j] = input_peaks[i].peaks[j].first
            intensity[i, j] = input_peaks[i].peaks[j].second
             


@cython.boundscheck(False)
@cython.wraparound(False)
cdef parse_frag_fspect(vector[fspectrum_t] input_peaks,
                      np.uint8_t [:, :] formulae,
                      np.float32_t[:, :] mass,
                      np.float32_t[:, :] intensity):

     cdef size_t i = 0
     cdef size_t fi = 0
     cdef size_t j = 0
     cdef float largest_intensity = -1
     
     for i in range(input_peaks.size()):
        for fi in range(NUM_FORMULA_ELEMENTS):
             if input_peaks[i].formula.counts[fi] > 0:
                formulae[i, FORMULA_ORDER[fi]] = input_peaks[i].formula.counts[fi]

        input_peaks[i].sort_peaks()
        for j in range(min(mass.shape[1], input_peaks[i].peaks.size())):
            mass[i, j] = input_peaks[i].peaks[j].first
            intensity[i, j] = input_peaks[i].peaks[j].second
             



def foo():
    return "Hello world"

cpdef py_generate_sub_formulae(atomicno_num_dict):

    cdef Formula f = Formula(atomicno_num_dict)

    cdef vector[formula_t] formulae = generate_sub_formulae(f.formula)
    cdef formula_t fl
    
    out = []
    for fl in formulae:
        out.append(formula_to_dict(fl))

    return out


cpdef py_get_all_frag_spect_highres(atomicno_num, MAX_ATOMICNO=20, MAX_PEAK_NUM=6):

     f = Formula(atomicno_num)

     cdef vector[fspectrum_t] frag_spect = get_all_frag_fspect(f.formula)

     cdef pair[float, float] p
     cdef formula_t frag_f

     np_peak_dt = np.dtype([('mass' , np.float32), ('intensity', np.float16)])
     
     np_spect_dt = [('formula', 'u8', (MAX_ATOMICNO, )), ('peaks', np_peak_dt, MAX_PEAK_NUM)]

     formula_out = np.zeros((frag_spect.size(), MAX_ATOMICNO),  dtype=np.uint8)
     mass_out = np.zeros((frag_spect.size(), MAX_PEAK_NUM), dtype=np.float32)
     intensity_out = np.zeros((frag_spect.size(), MAX_PEAK_NUM), dtype=np.float32)
     
     parse_frag_fspect(frag_spect, formula_out, mass_out, intensity_out)
     npout = np.zeros(frag_spect.size(), dtype=np_spect_dt)


     npout[:]['formula'] = formula_out
     npout['peaks']['mass'] = mass_out
     npout['peaks']['intensity'] = intensity_out

     return npout


