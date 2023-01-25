from __future__ import print_function

# #cython: linetrace=True
# #cython: profile=True
# #distutils: define_macros=CYTHON_TRACE_NOGIL=1

import numpy as np
cimport numpy as np
from cython.view cimport array as cvarray
from libc.stdlib cimport rand, RAND_MAX

import tinygraph
from libc.stdlib cimport calloc, free, malloc
from cython cimport view
import time

cimport cython


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef void _get_all_neighbors(np.uint8_t[:,:] adj, np.int32_t[:, :] neighbors_out) nogil:  
    """
    For a TG with N vertices returns a NxN numpy array of
    neighbors, where the ith row lists the vertex IDs of the 
    neighbors (and -1 otherwise)
    """
    
    cdef int N = adj.shape[0]
    cdef int * current_pos = <int *> calloc(N,sizeof(int))
    
    for i in range(N):
        for j in range(N):
            if adj[i, j]:
                neighbors_out[i][current_pos[i]] = j
                current_pos[i] += 1

    free(current_pos)

def get_all_neighbors(tg):  
    """
    For a TG with N vertices returns a NxN numpy array of
    neighbors, where the ith row lists the vertex IDs of the 
    neighbors (and -1 otherwise)
    """

    neighbors_out = np.ones((tg.vert_N, tg.vert_N), dtype=np.int32) * -1
    
    
    _get_all_neighbors(tg.adjacency != tinygraph.default_zero(tg.adjacency.dtype),
                       neighbors_out)

    return neighbors_out

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef int _any_vert_unassigned(np.int32_t[:] components_out) nogil:
    cdef int N = components_out.shape[0]
    cdef int i = 0

    for i in range(N):
        if components_out[i] == -1:
            return i
    return -1
            
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef int _dfs(np.int32_t[:, :] neighbors,
          int node_idx,
          np.int32_t[:] components,
          int current_color) nogil:
    #visit_order.append(node_idx)
    components[node_idx] = current_color
    cdef int cur_neighbor_idx = 0
    cdef int cur_neighbor = -1
    while neighbors[node_idx, cur_neighbor_idx] != -1:
        cur_neighbor = neighbors[node_idx, cur_neighbor_idx]
        
        if components[cur_neighbor] == -1:
            _dfs(neighbors, cur_neighbor, components,
                 current_color)
        cur_neighbor_idx += 1

    return 0 
        
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef _get_connected_components_dfs(np.uint8_t[:, :] adj,
                                np.int32_t[:] components_out):

    cdef int N = adj.shape[0]

    # precompute neighbor list
    cdef np.int32_t[:, ::1] neighbors = view.array(shape=(N, N), itemsize=sizeof(int), format="i")

    
    neighbors[:] = -1
    _get_all_neighbors(adj, neighbors)

    components_out[:] = -1
    cdef int current_color = 0
    cdef int start_node_idx =  _any_vert_unassigned(components_out)

    while start_node_idx > -1:
        
        _dfs(neighbors, start_node_idx, components_out, current_color)
        
        current_color += 1
        start_node_idx = _any_vert_unassigned(components_out)

        


cpdef get_connected_components_dfs(g):
   N = g.vert_N
   components_out = np.ones(N, dtype=np.int32)*-1
   
   _get_connected_components_dfs(g.adjacency, 
                             components_out)

   d = {}
   for i,c in enumerate(components_out):
       if c not in d:
           d[c] = []
       d[c].append(i)
   return [frozenset(v) for v in d.values()]




cpdef unsigned long bitset_remove_item(unsigned long input_set, int item):
    return input_set & (((1ul << 64) - 1) ^ (1ul << int(item)))

cpdef unsigned long bitset_add_item(unsigned long input_set, int item):
    cdef unsigned long x = 1
    
    return input_set | (x << int(item))

cpdef unsigned long bitset_create(items):
    cdef unsigned long bs = 0
    cdef int i = 0
    
    for i in items:
        bs = bitset_add_item(bs, i)
    return bs

cpdef bitset_to_list(unsigned long input_set):
    out = []
    for i in range(64):
        if (input_set & 0x1) > 0:
            out.append(i)
        input_set = input_set >> 1
    return out 


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef int do_rearrangements_lut_bits(np.int64_t[:, :] valid_h_donations,
                                     np.int32_t[:, :] heavy_atom_h_lut,
                                     np.int32_t[:] explicit_h, 
                                     np.int32_t[:] g2_src_cc_ids,
                                     np.int64_t[:] g2_src_cc_comps,
                                     np.int64_t[:] output_buffer):

    cdef unsigned long i, j, h_len, h_ind, rnd_idx
    cdef unsigned long comp_minus_h, comp_plus_h, source_cc, dest_cc
    cdef int donation_i = 0

    #print("the input ccs are")
    #for i, q in enumerate(g2_src_cc_comps):
    #    print(i, ":", bitset_to_list(q))

    cdef int output_i = 0
    for donation_i in range(valid_h_donations.shape[0]):
        i = valid_h_donations[donation_i][0]
        j = valid_h_donations[donation_i][1]        
        if i == -1 and j == -1:
            # this was placeholder empty... return
            # print('placeholder empty, returning...')
            return output_i

        h_len = explicit_h[i]
        # rnd_idx = 0 # np.random.randint(h_len)
        #rnd_idx = 1 + int(rand()/(RAND_MAX*h_len))
        for rnd_idx in range(h_len):
            h_ind = heavy_atom_h_lut[i][rnd_idx]

            if g2_src_cc_ids[h_ind] == g2_src_cc_ids[j]:
                continue # same component
            else:
                # add the component that h is in, minus h
                assert output_i < (output_buffer.shape[0]  -2) 
                source_cc = g2_src_cc_comps[g2_src_cc_ids[h_ind]]
                dest_cc = g2_src_cc_comps[g2_src_cc_ids[j]]
                #print("i=", i, "j=", j, "rnd_idx=", rnd_idx, "h_ind=", h_ind, g2_src_cc_comps[h_ind], g2_src_cc_comps[j])
                #print("removing ", h_ind, "from", bitset_to_list(source_cc))
                
                comp_minus_h = bitset_remove_item(source_cc, 
                                                  h_ind)
                output_buffer[output_i] = (comp_minus_h)
                output_i +=1
                # add h to destination component
                #print("adding ", h_ind, "to", bitset_to_list(dest_cc))
                comp_plus_h = bitset_add_item(dest_cc, 
                                            h_ind)
                output_buffer[output_i] = comp_plus_h
                output_i += 1
            
    return output_i
                
cpdef array_bitsets_to_array(np.int64_t[:] sets,
                             int max_elements):

    cdef int N = len(sets)
    cdef int i = 0
    
    cdef unsigned long val = 0
    cdef unsigned long val_i = 0 
    out = np.zeros((N, max_elements), dtype=np.uint8)
    for val_i, val in enumerate(sets):
        mask = 0

        for i in range(max_elements):
            if (val & 1 ):
                out[val_i, i] = 1
            val = val >> 1
    return out

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef graph_enumerate_breaks(np.int32_t[:, :, :] edges_to_break,
                             np.uint8_t[:, :] graph):

    cdef int N = graph.shape[0]

    cdef np.int32_t[:] components = np.zeros(N, dtype=np.int32)

    cdef np.int64_t[:] possible_bitsets = np.zeros(N, dtype=np.int64)
    cdef np.int64_t[:] size = np.zeros(N, dtype=np.int64)
    

    cdef int i, j,k, e1, e2 = 0
    cdef int last_component = 0
    cdef int c = 0

    output_bitsets = []

    for i in range(edges_to_break.shape[0]):
        # remove edge
        for j in range(edges_to_break.shape[1]):
            e1 = edges_to_break[i, j, 0]
            e2 = edges_to_break[i, j, 1]

            if e1 >= 0:
                #assert graph[e1, e2] > 0
                graph[e1, e2] = 0
                graph[e2, e1] = 0


        _get_connected_components_dfs(graph, components)

        possible_bitsets[:] = 0
        last_component = 0
        # output bitsets, assumes componets increase sequentially
        for k in range(N):
            c = components[k]
            if c > last_component:
                last_component = c
            possible_bitsets[c] = bitset_add_item(possible_bitsets[c],
                                                  k)
        # valid new bitsets are in 
        for k in range(last_component+1):
            output_bitsets.append(possible_bitsets[k])

        # restore
        for j in range(edges_to_break.shape[1]):
            e1 = edges_to_break[i, j, 0]
            e2 = edges_to_break[i, j, 1]

            if e1 >= 0:
                graph[e1, e2] = 1
                graph[e2, e1] = 1
    return output_bitsets

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef int connected_comps_for_graph(np.uint8_t[:, :] graph,
                                    np.int32_t[:] components,
                                    np.int64_t[:] component_bitsets, 
                                    np.int64_t[:] vert_cc_bitset):

     """
     For the input graph g compute the connected components and a mapping
     from vertex to its cc membership
     """

     cdef int N = graph.shape[0]
     cdef int last_component = 0
     cdef int c 
      
                              
     _get_connected_components_dfs(graph, components)
     component_bitsets[:] = 0 
     vert_cc_bitset[:] = 0

     
     for k in range(N):
        c = components[k]
        if c > last_component:
            last_component = c
        component_bitsets[c] = bitset_add_item(component_bitsets[c], k)

     for i in range(N):
         vert_cc_bitset[i] = component_bitsets[components[i]]

     
     return last_component +1 

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef  subsets_from_edge_removals(np.int32_t[:, :, :] edges_to_remove,
                                  np.uint8_t[:, :] graph,
                                  # rearrangement args
                                  int do_rearrange, 
                                  np.int64_t[:, :] valid_h_donations,
                                  np.int32_t[:, :] heavy_atom_h_lut,
                                  np.int32_t[:] explicit_h):
    cdef int N = graph.shape[0]
    cdef int e1, e2
    cdef int k = 0

    cdef np.int32_t[:] g2_src_cc_ids = np.zeros(N, dtype=np.int32)
    cdef np.int64_t[:] g2_src_cc_comp_bitsets = np.zeros(N,
                                      dtype=np.int64)
    cdef np.int64_t[:] g2_src_vert_cc_bitsets = np.zeros(N, dtype=np.int64)

    cdef int OUTPUT_BUFFER_SCALE_FACTOR  = 20 # this is a fudge factor
    cdef np.int64_t[:] output_buffer = np.zeros(valid_h_donations.shape[0]*OUTPUT_BUFFER_SCALE_FACTOR,
                             dtype=np.int64)

    ### FIXME these are waaaay too large sometimes, and we should be able to analytically calculate what the
    ### padding should be. Sometimes they allocate multiple GB of ram! 
    cdef size_t MAX_SUBSETS = max(edges_to_remove.shape[0]**2,  edges_to_remove.shape[0] * 1000) # FIXME how large should this be?
    cdef np.int64_t[:] subsets = np.zeros(MAX_SUBSETS, dtype=np.int64)
    cdef int subset_i = 0;
    
    for i in range(edges_to_remove.shape[0]):
        # remove edge from graph
        for j in range(edges_to_remove.shape[1]):
            e1 = edges_to_remove[i, j, 0]
            e2 = edges_to_remove[i, j, 1]

            if e1 >= 0:
                #assert graph[e1, e2] > 0
                #print("edge delete i, removing edge", e1, e2)
                
                graph[e1, e2] = 0
                graph[e2, e1] = 0



        num_cc = connected_comps_for_graph(graph, 
                                           g2_src_cc_ids,
                                           g2_src_cc_comp_bitsets,
                                           g2_src_vert_cc_bitsets)

        assert num_cc + subset_i < subsets.shape[0], "too many subsets for preallocation"

        for cc_i in range(num_cc):
            subsets[subset_i] = g2_src_cc_comp_bitsets[cc_i]
            subset_i += 1
            
        if do_rearrange > 0:
            #print('doing rearrangements')
            output_n = do_rearrangements_lut_bits(valid_h_donations,
                                                  heavy_atom_h_lut,
                                                  explicit_h,
                                                  g2_src_cc_ids,
                                                  g2_src_cc_comp_bitsets,
                                                  output_buffer)
            assert output_n + subset_i < subsets.shape[0], "too many subsets for preallocation"
            
            for k in range(output_n):
                #print('subset_i', subset_i, len(edges_to_remove))
                subsets[subset_i] = output_buffer[k]
                subset_i += 1
            #subsets += output_buffer[:output_n].flatten().tolist()
            

        # restore edge to graph
        for j in range(edges_to_remove.shape[1]):
            e1 = edges_to_remove[i, j, 0]
            e2 = edges_to_remove[i, j, 1]

            if e1 >= 0:
                graph[e1, e2] = 1
                graph[e2, e1] = 1
    assert subset_i < subsets.shape[0]
    return subsets[:subset_i]



