"""
Nets that use vertex subsets

"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import scipy.stats
import time

from .nets import *
from .formulaenets import StructuredOneHot, create_mass_matrix_sparse, mat_matrix_sparse_mm

class GraphVertSpect(nn.Module):
    def __init__(self, g_feature_n, spect_bin,
                 g_feature_out_n=None, 
                 int_d = None, layer_n = None, 
                 resnet=True,
                 spect_out_config = {}, 
                 gml_class = 'GraphMatLayers',
                 gml_config = {}, 
                 init_noise=1e-5,
                 init_bias = 0.0,
                 agg_func=None,
                 GS=1,
                 spect_out_class='',
                 spect_mode = 'dense', 
                 input_norm='batch',
                 
                 inner_norm=None,
                 default_render_width = 0.1,

        ):
        
        """

        """
        super( GraphVertSpect, self).__init__()

        if layer_n is not None:
            g_feature_out_n = [int_d] * layer_n

        self.gml = eval(gml_class)(g_feature_n, g_feature_out_n, 
                                   resnet=resnet, noise=init_noise,
                                   agg_func=parse_agg_func(agg_func), 
                                   norm=inner_norm, 
                                   GS=GS,
                                   **gml_config)

        if input_norm == 'batch':
            self.input_norm = MaskedBatchNorm1d(g_feature_n)
        elif input_norm == 'layer':
            self.input_norm = MaskedLayerNorm1d(g_feature_n)
        else:
            self.input_norm = None


        self.spect_out = eval(spect_out_class)(g_feat_in = g_feature_out_n[-1],
                                               spect_bin = spect_bin, 
                                               **spect_out_config)


        self.spect_mode = spect_mode
        self.default_render_width = default_render_width
        
        self.pos = 0

        
    def forward(self, adj, vect_feat, input_mask, input_idx, adj_oh,
                return_g_features = False, 
                mol_feat=None,
                vert_element_oh = None,
                formula_frag_count = None,
                atom_subsets = None,
                atom_subsets_peaks = None, 
                atom_subsets_peaks_mass_idx = None, 
                atom_subsets_peaks_intensity = None, 
                sparse_mass_matrix=None,
                **kwargs):

        G = adj
        
        BATCH_N, MAX_N, F_N = vect_feat.shape

        if self.input_norm is not None:
            vect_feat = apply_masked_1d_norm(self.input_norm, 
                                             vect_feat, 
                                             input_mask)
        
        G_features = self.gml(G, vect_feat, input_mask)
        if return_g_features:
            return G_features

        g_squeeze = G_features.squeeze(1)

        pred_dense_spect_dict = self.spect_out(
            g_squeeze, input_mask,
            vert_element_oh, adj_oh,
            atom_subsets, atom_subsets_peaks,
            sparse_mass_matrix,
            atom_subsets_peaks_mass_idx = atom_subsets_peaks_mass_idx, 
            atom_subsets_peaks_intensity = atom_subsets_peaks_intensity, 
            
        )
        pred_dense_spect = pred_dense_spect_dict['spect_out']
        subset_probs =  pred_dense_spect_dict['subset_probs']

        # pred_masses, pred_probs = dense_to_sparse(pred_dense_spect)
        pred_masses = None
        pred_probs = None
            
        out = {'spect' : pred_dense_spect,
               'masses' : pred_masses,
               'probs' : pred_probs}
        for k, v in pred_dense_spect_dict.items():
            if k != 'spect_out':
                out[k] = v

        self.pos += 1
        return out


class SubsetsSampleWeighted(nn.Module):
    """
    Per-vertex features to sparse points of peak, mass

    Input:
    BATCH_N x ATOM_N x F

    Output:
    BATCH_N x SPECT_BIN

    """
    def __init__(self, g_feat_in,
                 internal_d = 512,
                 prob_softmax = True,
                 subset_norm = True, 
                 linear_layer_n = 1,
                 spect_bin_n = 512):

        
        super( SubsetsSampleWeighted, self).__init__()

            
        self.subset_norm = subset_norm
        if self.subset_norm:
            self.subset_weighted_norm = nn.LayerNorm(g_feat_in)

        self.f_combine_l1 = nn.Linear(g_feat_in, internal_d)
        
        self.f_combine_l2 = nn.Sequential(*[nn.Sequential(nn.Linear(internal_d, internal_d),
                                                          nn.ReLU()) for _ in range(linear_layer_n)])

        self.norm_pre_score = nn.LayerNorm(internal_d)
        self.f_combine_score = nn.Linear(internal_d, 1)

        
        self.spect_bin_n = spect_bin_n
        self.prob_softmax = prob_softmax


    def forward(self, vert_feat_in, vert_mask_in,
                vert_element_oh, adj_oh,
                atom_subsets, atom_subsets_peaks):
        
                
        """
        vert_feat_in : BATCH_N x ATOM_N x G_F
        vert_mask_in: BATCH_N x ATOM_N  
        atom_subsets : BATCH_N x SUBSET_SAMPLES_N x ATOM_N
        atom_subsets_peaks : BATCH_N x SUBSET_SAMPLES_N x NUM_MASSES x 2  (mass, peak intensity) 
        """

        BATCH_N, ATOM_N, G_F  = vert_feat_in.shape
        _, SUBSET_SAMPLES_N, _ = atom_subsets.shape
        
        
        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1)


        atom_subsets_masked = atom_subsets * vert_mask_in.unsqueeze(1)
        atom_subsets_masked_float = atom_subsets_masked.float()
        sample_weighted_vert_sum = torch.einsum("ijk,ilj->ilk", masked_vert_feat, atom_subsets_masked_float)

        assert sample_weighted_vert_sum.shape == (BATCH_N, SUBSET_SAMPLES_N, G_F)
        subset_size = atom_subsets_masked_float.sum(dim=2) + 1e-4
        sample_weighted_vert_mean = sample_weighted_vert_sum / subset_size.unsqueeze(-1)
        
        if self.subset_norm:
            subset_normed = self.subset_weighted_norm(sample_weighted_vert_mean)
        else:
            subset_normed = sample_weighted_vert_mean
        
        x = F.relu(self.f_combine_l1(subset_normed))
        x = F.relu(self.f_combine_l2(x))
        x = self.norm_pre_score(x)  # BATCH_N x SUBSET_SAMPLES_N x F
        formulae_scores = self.f_combine_score(x).squeeze(2)
        
        if self.prob_softmax:
            formulae_probs = torch.softmax(formulae_scores, dim=-1)
        else:
            formulae_probs = torch.sigmoid(formulae_scores)

        #t1 = time.time()
        formulae_masses = atom_subsets_peaks
        sparse_mass_matrices = [create_mass_matrix_sparse(formulae_masses[:, :, i, 0].round().long(),
                                                   self.spect_bin_n,
                                                   formulae_masses[:, :, i, 1]
        )\
                         .to(vert_feat_in.device) for i in range(formulae_masses.shape[2])]
        
        sparse_mass_matrix = sparse_mass_matrices[0]
        for m in sparse_mass_matrices[1:]:
            sparse_mass_matrix += m

        spect_out = mat_matrix_sparse_mm(sparse_mass_matrix, formulae_probs)

        # torch.cuda.synchronize()
        # t2 = time.time()
        # print(f"end of net took {(t2-t1)*1000:3.2f}ms ")

        return {'spect_out': spect_out,
                'formulae_probs' : formulae_probs}


class SubsetsSampleWeightedFormula(nn.Module):
    """
    Vertex-subset-based weighting of output fragment mass spectra

    Vertex features reduced by vertex subset, combined with molecular formuula

    Input:
    BATCH_N x ATOM_N x F

    Output:
    BATCH_N x SPECT_BIN

    """
    def __init__(self, g_feat_in,
                 internal_d = 512,
                 prob_softmax = True,
                 subset_norm = True,
                 formula_oh_sizes = [20, 20, 20, 20, 20],
                 formula_oh_accum = True,
                 linear_layer_n = 2,
                 spect_bin_n = 512):

        
        super( SubsetsSampleWeightedFormula, self).__init__()

        formula_encoding_n = np.sum(formula_oh_sizes)

        self.formula_to_oh = StructuredOneHot(formula_oh_sizes, formula_oh_accum)
        
        self.subset_norm = subset_norm
        if self.subset_norm:
            self.subset_weighted_norm = nn.LayerNorm(g_feat_in)

        self.f_combine_l1 = nn.Linear(g_feat_in + formula_encoding_n, internal_d)
        
        self.f_combine_l2 = nn.Sequential(*[nn.Sequential(nn.Linear(internal_d, internal_d),
                                                          nn.ReLU()) for _ in range(linear_layer_n)])

        self.norm_pre_score = nn.LayerNorm(internal_d)
        self.f_combine_score = nn.Linear(internal_d, 1)

        
        self.spect_bin_n = spect_bin_n
        self.prob_softmax = prob_softmax


    def forward(self, vert_feat_in, vert_mask_in,
                vert_element_oh, adj_oh,
                atom_subsets, atom_subsets_peaks):
        
                
        """
        vert_feat_in : BATCH_N x ATOM_N x G_F
        vert_mask_in: BATCH_N x ATOM_N  
        atom_subsets : BATCH_N x SUBSET_SAMPLES_N x ATOM_N
        atom_subsets_peaks : BATCH_N x SUBSET_SAMPLES_N x NUM_MASSES x 2  (mass, peak intensity) 
        """

        BATCH_N, ATOM_N, G_F  = vert_feat_in.shape
        _, SUBSET_SAMPLES_N, _ = atom_subsets.shape
        
        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1)

        atom_subsets_masked = atom_subsets * vert_mask_in.unsqueeze(1)
        atom_subsets_masked_float = atom_subsets_masked.float()
        sample_weighted_vert_sum = torch.einsum("ijk,ilj->ilk", masked_vert_feat.float(), atom_subsets_masked_float)

        possible_formulae = torch.einsum("bae,bfa->bfe", vert_element_oh.float(), atom_subsets.float()).byte()
        #print("possible_formulae.shape=", possible_formulae.shape)
        
        possible_formulae_flat = possible_formulae.reshape(-1, possible_formulae.shape[-1])
        possible_formulae_flat_oh = self.formula_to_oh(possible_formulae_flat)
        possible_formulae = possible_formulae_flat_oh.reshape(possible_formulae.shape[0],
                                                              possible_formulae.shape[1], -1)
        

        assert sample_weighted_vert_sum.shape == (BATCH_N, SUBSET_SAMPLES_N, G_F)
        subset_size = atom_subsets_masked_float.sum(dim=2) + 1e-4
        sample_weighted_vert_mean = sample_weighted_vert_sum / subset_size.unsqueeze(-1)
        
        if self.subset_norm:
            subset_normed = self.subset_weighted_norm(sample_weighted_vert_mean)
        else:
            subset_normed = sample_weighted_vert_mean

        x = F.relu(self.f_combine_l1(torch.cat([subset_normed, possible_formulae], -1)))
        x = F.relu(self.f_combine_l2(x))
        x = self.norm_pre_score(x)
        formulae_scores = self.f_combine_score(x).squeeze(2)
        
        if self.prob_softmax:
            formulae_probs = torch.softmax(formulae_scores, dim=-1)
        else:
            formulae_probs = torch.sigmoid(formulae_scores)

        #t1 = time.time()
        formulae_masses = atom_subsets_peaks
        sparse_mass_matrices = [create_mass_matrix_sparse(formulae_masses[:, :, i, 0].round().long(),
                                                   self.spect_bin_n,
                                                   formulae_masses[:, :, i, 1]
        )\
                         .to(vert_feat_in.device) for i in range(formulae_masses.shape[2])]

        
        sparse_mass_matrix = sparse_mass_matrices[0]
        for m in sparse_mass_matrices[1:]:
            sparse_mass_matrix += m

        spect_out = mat_matrix_sparse_mm(sparse_mass_matrix, formulae_probs)
        # torch.cuda.synchronize()
        # t2 = time.time()
        # print(f"end of net took {(t2-t1)*1000:3.2f}ms ")

        return {'spect_out': spect_out,
                'formulae_probs' : formulae_probs}




class SubsetsSampleWeightedFormulaGRU(nn.Module):
    """
    Vertex-subset-based weighting of output fragment mass spectra

    Vertex features reduced by vertex subset, combined with molecular formuula
    
    Using GRU

    Input:
    BATCH_N x ATOM_N x F

    Output:
    BATCH_N x SPECT_BIN

    """
    def __init__(
        self, g_feat_in,
            spect_bin, 
            internal_d = 512,
            prob_softmax = True,
            subset_norm = True,
            formula_oh_sizes = [20, 20, 20, 20, 20],
            formula_oh_accum = True,
            linear_layer_n = 2,
            possible_formulae_norm = False,
            normalize_1_output = False, 
            
            concat_feats = False,
            
            norm_post_combine = True,
    ):
        
        super( SubsetsSampleWeightedFormulaGRU, self).__init__()

        formula_encoding_n = np.sum(formula_oh_sizes)

        self.spect_bin = spect_bin
        
        self.formula_to_oh = StructuredOneHot(formula_oh_sizes, formula_oh_accum)
        
        self.subset_norm = subset_norm
        if self.subset_norm:
            self.subset_weighted_norm = nn.LayerNorm(g_feat_in)

        self.possible_formulae_norm = possible_formulae_norm
        if self.possible_formulae_norm:
            self.pf_norm = nn.BatchNorm1d(formula_encoding_n)
            
        self.combine_layers = nn.GRUCell(formula_encoding_n, g_feat_in)
            
        self.concat_feats = concat_feats
        if self.concat_feats:
            self.f_combine_l1 = nn.Linear(3 * g_feat_in, internal_d)
        else:
            self.f_combine_l1 = nn.Linear(g_feat_in, internal_d)
        self.f_combine_l2 = nn.Sequential(*[nn.Sequential(nn.Linear(internal_d, internal_d),
                                                          nn.ReLU()) for _ in range(linear_layer_n)])

        if norm_post_combine:
            self.do_norm_post_combine = True
            self.norm_post_combine = nn.LayerNorm(g_feat_in)
           
        self.norm_pre_score = nn.LayerNorm(internal_d)
        self.f_combine_score = nn.Linear(internal_d, 1)
        
        self.prob_softmax = prob_softmax
        self.normalize_1_output = normalize_1_output

    def forward(
            self, vert_feat_in, vert_mask_in,
            vert_element_oh, adj_oh,
            atom_subsets, atom_subsets_peaks,
            sparse_mass_matrix,
            return_formulae_scores=False,
            **kwargs,
        ):
        """
        vert_feat_in : BATCH_N x ATOM_N x G_F
        vert_mask_in: BATCH_N x ATOM_N  
        atom_subsets : BATCH_N x SUBSET_SAMPLES_N x ATOM_N
        atom_subsets_peaks : BATCH_N x SUBSET_SAMPLES_N x NUM_MASSES x 2  (mass, peak intensity) 
        """

        BATCH_N, ATOM_N, G_F  = vert_feat_in.shape
        _, SUBSET_SAMPLES_N, _ = atom_subsets.shape
        
        # apply input mask to keep only features for atoms that actually exist
        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1)
        atom_subsets_masked = atom_subsets * vert_mask_in.unsqueeze(1)
        atom_subsets_masked_float = atom_subsets_masked.float()

        # each subset embedding is obtained by summing the vertex features
        # for only the atoms contained in that subset
        sample_weighted_vert_sum = torch.einsum("ijk,ilj->ilk", masked_vert_feat.float(), atom_subsets_masked_float)

        # generate the formulae (how many of element E did we have?) for each vertex subset
        # BATCH_N x SUBSET_SAMPLES_N x 8
        possible_formulae = torch.einsum("bae,bfa->bfe", vert_element_oh.float(), atom_subsets.float()).byte()
        
        # convert to "cumulative one-hot" encoding
        # BATCH_N x SUBSET_SAMPLES_N x sum(formula_oh_sizes)
        possible_formulae_flat = possible_formulae.reshape(-1, possible_formulae.shape[-1])
        possible_formulae_flat_oh = self.formula_to_oh(possible_formulae_flat)
        possible_formulae = possible_formulae_flat_oh.reshape(possible_formulae.shape[0],
                                                              possible_formulae.shape[1], -1)

        if self.possible_formulae_norm:
            # do batch-norm over formulae, averaging over all formulae w unique batch_index x formula_index
            possible_formulae = self.pf_norm(possible_formulae.reshape(-1, possible_formulae.shape[-1])).reshape(possible_formulae.shape)

        # print("vert_element_oh.shape=", vert_element_oh.shape,
        #       "atom_subset.shape=", atom_subsets.shape,
        #       "possible_formulae.shape=", possible_formulae.shape)
        assert sample_weighted_vert_sum.shape == (BATCH_N, SUBSET_SAMPLES_N, G_F)
        
        # compute vertsubset embedding as mean over all present vertex embeddings
        eps = 1e-4
        subset_size = atom_subsets_masked_float.sum(dim=2) + eps  # BATCH_N x SUBSET_SAMPLES_N
        sample_weighted_vert_mean = sample_weighted_vert_sum / subset_size.unsqueeze(-1)
        
        if self.subset_norm:
            # do layer-norm over all subset embeddings
            subset_normed = self.subset_weighted_norm(sample_weighted_vert_mean)
        else:
            subset_normed = sample_weighted_vert_mean

        # combine the "cumulative one-hot" formulae (n_feats = sum(formula_oh_sizes))
        # as well as the subset embeddings obtained from summing the global embeddings (n_feats=512)
        # here, we use a GRU with input: formulae features, hidden: subset embeddings
        # essentially it's a particular form of a MLP layer...
        # import ipdb; ipdb.set_trace()
        combined = self.combine_layers(
            possible_formulae.reshape(-1, possible_formulae.shape[-1]),
            subset_normed.reshape(-1, subset_normed.shape[-1])
        ).reshape(subset_normed.shape)  # BATCH_N x SUBSET_SAMPLES_N x FEAT_DIM=512

        if self.do_norm_post_combine:
            x = self.norm_post_combine(combined)
        else:
            x = combined

        # we compute mean/std-dev across the subset sample dim
        if self.concat_feats:
            subset_mask = (subset_size > eps).float()  # BATCH_N x SUBSET_SAMPLES_N
            n_subsets = (subset_size > eps).sum(dim=1)
            x_sum_over_subsets = x.sum(dim=1).unsqueeze(1)  # BATCH_N x 1 x FEAT_DIM
            x_mean_over_subsets = x_sum_over_subsets / n_subsets.unsqueeze(dim=-1).unsqueeze(dim=-1)  # BATCH_N x 1 x FEAT_DIM

            x_mean_minus = (x - x_mean_over_subsets) * subset_mask.unsqueeze(-1)  # BATCH_N x SUBSET_SAMPLES_N x FEAT_DIM
            x_mean_minus_sq_sum = (x_mean_minus ** 2).sum(dim=1)  # BATCH_N x FEAT_DIM
            x_std = x_mean_minus_sq_sum.unsqueeze(1) / n_subsets.unsqueeze(dim=-1).unsqueeze(dim=-1)  # BATCH_N x 1 x FEAT_DIM

            # # the features we want to concat...
            x = torch.cat([
                x,
                x_mean_over_subsets.repeat(1, SUBSET_SAMPLES_N, 1),
                x_std.repeat(1, SUBSET_SAMPLES_N, 1),
            ], dim=-1)

        # TODO: add batch/layer norms to this part?
        x = F.relu(self.f_combine_l1(x))
        x = F.relu(self.f_combine_l2(x))

        # this LayerNorm normalizes s.t. x.sum(dim=-1) is zero-mean
        x = self.norm_pre_score(x)  # BATCH_N x SUBSET_SAMPLES_N x INTERNAL_D

        formulae_scores = self.f_combine_score(x).squeeze(2)   # BATCH_N x SUBSET_SAMPLES_N

        if return_formulae_scores:
            return formulae_scores
        
        if self.prob_softmax:
            formulae_probs = torch.softmax(formulae_scores, dim=-1)
        else:
            formulae_probs = torch.sigmoid(formulae_scores)

        #t1 = time.time()
        if sparse_mass_matrix is not None:
            with torch.cuda.amp.autocast(enabled=False):
                spect_out = mat_matrix_sparse_mm(sparse_mass_matrix, formulae_probs)  # BATCH_N x SPECT_BIN_N
        else:
            # generate sparse mass matrix on the fly
            with torch.cuda.amp.autocast(enabled=False):
                formulae_masses = atom_subsets_peaks
                sparse_mass_matrices = [create_mass_matrix_sparse(formulae_masses[:, :, i, 0].round().long(),
                                                        self.spect_bin.get_num_bins(),
                                                        formulae_masses[:, :, i, 1]
                )\
                                .to(vert_feat_in.device) for i in range(formulae_masses.shape[2])]
                sparse_mass_matrix = sparse_mass_matrices[0]
                for m in sparse_mass_matrices[1:]:
                    sparse_mass_matrix += m

                spect_out = mat_matrix_sparse_mm(sparse_mass_matrix, formulae_probs)  # BATCH_N x SPECT_BIN_N

        # torch.cuda.synchronize()
        # t2 = time.time()
        # print(f"end of net took {(t2-t1)*1000:3.2f}ms ")

        if self.normalize_1_output:
            spect_out = spect_out / (torch.sum(spect_out, dim=1) + 1e-4).unsqueeze(-1)
            
        return {'spect_out': spect_out,
                'formulae_probs' : formulae_probs}


class SubsetsSampleWeightedFormulaGRUHighway(nn.Module):
    """
    Vertex-subset-based weighting of output fragment mass spectra

    Vertex features reduced by vertex subset, combined with molecular formuula
    
    Using GRU

    Input:
    BATCH_N x ATOM_N x F

    Output:
    BATCH_N x SPECT_BIN

    """
    def __init__(self, g_feat_in,
                 internal_d = 512,
                 highway_steps_n = 4, 
                 prob_softmax = True,
                 subset_norm = True,
                 formula_oh_sizes = [20, 20, 20, 20, 20],
                 formula_oh_accum = True,
                 linear_layer_n = 2,
                 possible_formulae_norm = False,
                 normalize_1_output = False, 
                 spect_bin_n = 512):

        
        super( SubsetsSampleWeightedFormulaGRUHighway, self).__init__()

        g_feat_in = g_feat_in * highway_steps_n

        formula_encoding_n = np.sum(formula_oh_sizes)

        self.formula_to_oh = StructuredOneHot(formula_oh_sizes, formula_oh_accum)
        
        self.subset_norm = subset_norm
        if self.subset_norm:
            self.subset_weighted_norm = nn.LayerNorm(g_feat_in)

        self.possible_formulae_norm = possible_formulae_norm
        if self.possible_formulae_norm:
            self.pf_norm = nn.BatchNorm1d(formula_encoding_n)
            
        self.combine_layers = nn.GRUCell(formula_encoding_n, g_feat_in)
        
            
        self.f_combine_l1 = nn.Linear(g_feat_in, internal_d)
        
        self.f_combine_l2 = nn.Sequential(*[nn.Sequential(nn.Linear(internal_d, internal_d),
                                                          nn.ReLU()) for _ in range(linear_layer_n)])

        self.norm_pre_score = nn.LayerNorm(internal_d)
        self.f_combine_score = nn.Linear(internal_d, 1)
        
        self.spect_bin_n = spect_bin_n
        self.prob_softmax = prob_softmax
        self.normalize_1_output = normalize_1_output


    def forward(self, vert_feat_in, vert_mask_in,
                vert_element_oh, adj_oh,
                atom_subsets, atom_subsets_peaks):
        
                
        """
        vert_feat_in : BATCH_N x ATOM_N x G_F
        vert_mask_in: BATCH_N x ATOM_N  
        atom_subsets : BATCH_N x SUBSET_SAMPLES_N x ATOM_N
        atom_subsets_peaks : BATCH_N x SUBSET_SAMPLES_N x NUM_MASSES x 2  (mass, peak intensity) 
        """

        BATCH_N, ATOM_N, G_F, HW  = vert_feat_in.shape
        vert_feat_in = vert_feat_in.reshape(BATCH_N, ATOM_N, -1)
        G_F = G_F * HW
        
        _, SUBSET_SAMPLES_N, _ = atom_subsets.shape
        
        
        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1)


        atom_subsets_masked = atom_subsets * vert_mask_in.unsqueeze(1)
        atom_subsets_masked_float = atom_subsets_masked.float()
        sample_weighted_vert_sum = torch.einsum("ijk,ilj->ilk", masked_vert_feat.float(), atom_subsets_masked_float)

        possible_formulae = torch.einsum("bae,bfa->bfe", vert_element_oh.float(), atom_subsets.float()).byte()
        #print("possible_formulae.shape=", possible_formulae.shape)
        
        possible_formulae_flat = possible_formulae.reshape(-1, possible_formulae.shape[-1])
        possible_formulae_flat_oh = self.formula_to_oh(possible_formulae_flat)
        possible_formulae = possible_formulae_flat_oh.reshape(possible_formulae.shape[0],
                                                              possible_formulae.shape[1], -1)
        
        if self.possible_formulae_norm:
            possible_formulae = self.pf_norm(possible_formulae.reshape(-1, possible_formulae.shape[-1])).reshape(possible_formulae.shape)


        # print("vert_element_oh.shape=", vert_element_oh.shape,
        #       "atom_subset.shape=", atom_subsets.shape,
        #       "possible_formulae.shape=", possible_formulae.shape)

        
        assert sample_weighted_vert_sum.shape == (BATCH_N, SUBSET_SAMPLES_N, G_F)
        subset_size = atom_subsets_masked_float.sum(dim=2) + 1e-4
        sample_weighted_vert_mean = sample_weighted_vert_sum / subset_size.unsqueeze(-1)
        
        if self.subset_norm:
            subset_normed = self.subset_weighted_norm(sample_weighted_vert_mean)
        else:
            subset_normed = sample_weighted_vert_mean


        combined = self.combine_layers(possible_formulae.reshape(-1, possible_formulae.shape[-1]),
                                       subset_normed.reshape(-1, subset_normed.shape[-1]))\
                       .reshape(subset_normed.shape)
        x = F.relu(self.f_combine_l1(combined))
        x = F.relu(self.f_combine_l2(x))
        x = self.norm_pre_score(x)
        formulae_scores = self.f_combine_score(x).squeeze(2)
        
        if self.prob_softmax:
            formulae_probs = torch.softmax(formulae_scores, dim=-1)
        else:
            formulae_probs = torch.sigmoid(formulae_scores)

        #t1 = time.time()
        formulae_masses = atom_subsets_peaks
        sparse_mass_matrices = [create_mass_matrix_sparse(formulae_masses[:, :, i, 0].round().long(),
                                                   self.spect_bin_n,
                                                   formulae_masses[:, :, i, 1]
        )\
                         .to(vert_feat_in.device) for i in range(formulae_masses.shape[2])]

        
        sparse_mass_matrix = sparse_mass_matrices[0]
        for m in sparse_mass_matrices[1:]:
            sparse_mass_matrix += m

        spect_out = mat_matrix_sparse_mm(sparse_mass_matrix, formulae_probs)
        # torch.cuda.synchronize()
        # t2 = time.time()
        # print(f"end of net took {(t2-t1)*1000:3.2f}ms ")

        if self.normalize_1_output:
            spect_out = spect_out / (torch.sum(spect_out, dim=1) + 1e-4).unsqueeze(-1)
            
        return {'spect_out': spect_out,
                'formulae_probs' : formulae_probs}





class SubsetsSampleWeightedFormulaGRUNewSparse(nn.Module):
    """
    Vertex-subset-based weighting of output fragment mass spectra

    Vertex features reduced by vertex subset, combined with molecular formuula
    
    Using GRU

    Input:
    BATCH_N x ATOM_N x F

    Output:
    BATCH_N x SPECT_BIN

    """
    def __init__(
        self, g_feat_in,
            spect_bin, 
            internal_d = 512,
            prob_softmax = True,
            subset_norm = True,
            formula_oh_sizes = [20, 20, 20, 20, 20],
            formula_oh_accum = True,
            linear_layer_n = 2,
            possible_formulae_norm = False,
            normalize_1_output = False, 
            
            mask_zero_subsets=False, 
            concat_feats = False,
            
            norm_post_combine = True,
    ):
        
        super( SubsetsSampleWeightedFormulaGRUNewSparse, self).__init__()

        formula_encoding_n = np.sum(formula_oh_sizes)

        self.spect_bin = spect_bin
        
        self.formula_to_oh = StructuredOneHot(formula_oh_sizes, formula_oh_accum)
        
        self.subset_norm = subset_norm
        if self.subset_norm:
            self.subset_weighted_norm = nn.LayerNorm(g_feat_in)

        self.possible_formulae_norm = possible_formulae_norm
        if self.possible_formulae_norm:
            self.pf_norm = nn.BatchNorm1d(formula_encoding_n)
            
        self.combine_layers = nn.GRUCell(formula_encoding_n, g_feat_in)
            
        self.concat_feats = concat_feats
        if self.concat_feats:
            self.f_combine_l1 = nn.Linear(3 * g_feat_in, internal_d)
        else:
            self.f_combine_l1 = nn.Linear(g_feat_in, internal_d)
        self.f_combine_l2 = nn.Sequential(*[nn.Sequential(nn.Linear(internal_d, internal_d),
                                                          nn.ReLU()) for _ in range(linear_layer_n)])

        if norm_post_combine:
            self.do_norm_post_combine = True
            self.norm_post_combine = nn.LayerNorm(g_feat_in)
           
        self.norm_pre_score = nn.LayerNorm(internal_d)
        self.f_combine_score = nn.Linear(internal_d, 1)
        
        self.prob_softmax = prob_softmax
        self.normalize_1_output = normalize_1_output

        self.mask_zero_subsets = mask_zero_subsets

    def forward(
            self, vert_feat_in, vert_mask_in,
            vert_element_oh, adj_oh,
            atom_subsets, atom_subsets_peaks,
            sparse_mass_matrix,
            return_formulae_scores=False,
            atom_subsets_peaks_mass_idx=None, 
            atom_subsets_peaks_intensity=None,
            **kwargs

        ):
        """
        vert_feat_in : BATCH_N x ATOM_N x G_F
        vert_mask_in: BATCH_N x ATOM_N  
        atom_subsets : BATCH_N x SUBSET_SAMPLES_N x ATOM_N
        #atom_subsets_peaks : BATCH_N x SUBSET_SAMPLES_N x NUM_MASSES x 2  (mass, peak intensity) 
        """

        BATCH_N, ATOM_N, G_F  = vert_feat_in.shape
        _, SUBSET_SAMPLES_N, _ = atom_subsets.shape

        assert atom_subsets_peaks_mass_idx is not None
        assert atom_subsets_peaks_intensity is not None
        
        # apply input mask to keep only features for atoms that actually exist
        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1)
        atom_subsets_masked = atom_subsets * vert_mask_in.unsqueeze(1)
        atom_subsets_masked_float = atom_subsets_masked.float()

        nonzero_atom_subsets = (atom_subsets.sum(dim=2) != 0).float()
        
        # each subset embedding is obtained by summing the vertex features
        # for only the atoms contained in that subset
        sample_weighted_vert_sum = torch.einsum("ijk,ilj->ilk", masked_vert_feat.float(), atom_subsets_masked_float)

        # generate the formulae (how many of element E did we have?) for each vertex subset
        # BATCH_N x SUBSET_SAMPLES_N x 8
        possible_formulae = torch.einsum("bae,bfa->bfe", vert_element_oh.float(), atom_subsets.float()).byte()
        
        # convert to "cumulative one-hot" encoding
        # BATCH_N x SUBSET_SAMPLES_N x sum(formula_oh_sizes)
        possible_formulae_flat = possible_formulae.reshape(-1, possible_formulae.shape[-1])
        possible_formulae_flat_oh = self.formula_to_oh(possible_formulae_flat)
        possible_formulae = possible_formulae_flat_oh.reshape(possible_formulae.shape[0],
                                                              possible_formulae.shape[1], -1)

        if self.possible_formulae_norm:
            # do batch-norm over formulae, averaging over all formulae w unique batch_index x formula_index
            possible_formulae = self.pf_norm(possible_formulae.reshape(-1, possible_formulae.shape[-1])).reshape(possible_formulae.shape)

        # print("vert_element_oh.shape=", vert_element_oh.shape,
        #       "atom_subset.shape=", atom_subsets.shape,
        #       "possible_formulae.shape=", possible_formulae.shape)
        assert sample_weighted_vert_sum.shape == (BATCH_N, SUBSET_SAMPLES_N, G_F)
        
        # compute vertsubset embedding as mean over all present vertex embeddings
        eps = 1e-4
        subset_size = atom_subsets_masked_float.sum(dim=2) + eps  # BATCH_N x SUBSET_SAMPLES_N
        sample_weighted_vert_mean = sample_weighted_vert_sum / subset_size.unsqueeze(-1)
        
        if self.subset_norm:
            # do layer-norm over all subset embeddings
            subset_normed = self.subset_weighted_norm(sample_weighted_vert_mean)
        else:
            subset_normed = sample_weighted_vert_mean

        # combine the "cumulative one-hot" formulae (n_feats = sum(formula_oh_sizes))
        # as well as the subset embeddings obtained from summing the global embeddings (n_feats=512)
        # here, we use a GRU with input: formulae features, hidden: subset embeddings
        # essentially it's a particular form of a MLP layer...
        # import ipdb; ipdb.set_trace()
        combined = self.combine_layers(
            possible_formulae.reshape(-1, possible_formulae.shape[-1]),
            subset_normed.reshape(-1, subset_normed.shape[-1])
        ).reshape(subset_normed.shape)  # BATCH_N x SUBSET_SAMPLES_N x FEAT_DIM=512

        if self.do_norm_post_combine:
            x = self.norm_post_combine(combined)
        else:
            x = combined

        # we compute mean/std-dev across the subset sample dim
        if self.concat_feats:
            subset_mask = (subset_size > eps).float()  # BATCH_N x SUBSET_SAMPLES_N
            n_subsets = (subset_size > eps).sum(dim=1)
            x_sum_over_subsets = x.sum(dim=1).unsqueeze(1)  # BATCH_N x 1 x FEAT_DIM
            x_mean_over_subsets = x_sum_over_subsets / n_subsets.unsqueeze(dim=-1).unsqueeze(dim=-1)  # BATCH_N x 1 x FEAT_DIM

            x_mean_minus = (x - x_mean_over_subsets) * subset_mask.unsqueeze(-1)  # BATCH_N x SUBSET_SAMPLES_N x FEAT_DIM
            x_mean_minus_sq_sum = (x_mean_minus ** 2).sum(dim=1)  # BATCH_N x FEAT_DIM
            x_std = x_mean_minus_sq_sum.unsqueeze(1) / n_subsets.unsqueeze(dim=-1).unsqueeze(dim=-1)  # BATCH_N x 1 x FEAT_DIM

            # # the features we want to concat...
            x = torch.cat([
                x,
                x_mean_over_subsets.repeat(1, SUBSET_SAMPLES_N, 1),
                x_std.repeat(1, SUBSET_SAMPLES_N, 1),
            ], dim=-1)

        # TODO: add batch/layer norms to this part?
        x = F.relu(self.f_combine_l1(x))
        x = F.relu(self.f_combine_l2(x))

        # this LayerNorm normalizes s.t. x.sum(dim=-1) is zero-mean
        x = self.norm_pre_score(x)  # BATCH_N x SUBSET_SAMPLES_N x INTERNAL_D

        formulae_scores = self.f_combine_score(x).squeeze(2)   # BATCH_N x SUBSET_SAMPLES_N

        if return_formulae_scores:
            return formulae_scores

        if self.mask_zero_subsets:
            zero_atom_subsets = 1- nonzero_atom_subsets
            formulae_scores = formulae_scores + (-1000 * zero_atom_subsets)
        if self.prob_softmax:
            formulae_probs = torch.softmax(formulae_scores, dim=-1)
        else:
            formulae_probs = torch.sigmoid(formulae_scores)

        #t1 = time.time()
        if sparse_mass_matrix is not None:
            with torch.cuda.amp.autocast(enabled=False):
                spect_out = mat_matrix_sparse_mm(sparse_mass_matrix, formulae_probs)  # BATCH_N x SPECT_BIN_N
        else:
            # generate sparse mass matrix on the fly
            with torch.cuda.amp.autocast(enabled=False):

                # per batch
                out_y = []
                for batch_i in range(BATCH_N):
                    sparse_mat_matrix = peak_indices_intensities_to_sparse_matrix(atom_subsets_peaks_mass_idx[batch_i],
                                                                                       atom_subsets_peaks_intensity[batch_i], 
                                                                                       self.spect_bin.get_num_bins())
                    y = sparse_mat_matrix @ formulae_probs[batch_i]
                    out_y.append(y)
                spect_out = torch.stack(out_y)
                
        if self.normalize_1_output:
            spect_out = spect_out / (torch.sum(spect_out, dim=1) + 1e-9).unsqueeze(-1)

        subset_probs = formulae_probs
        return {'spect_out': spect_out,
                'subset_probs' : subset_probs}


    


class DebugNet(nn.Module):
    """
    Vertex-subset-based weighting of output fragment mass spectra

    Vertex features reduced by vertex subset, combined with molecular formuula
    
    Using GRU

    Input:
    BATCH_N x ATOM_N x F

    Output:
    BATCH_N x SPECT_BIN

    """
    def __init__(
        self, g_feat_in,
            spect_bin, 
            internal_d = 512,
            prob_softmax = True,
            subset_norm = True,
            formula_oh_sizes = [20, 20, 20, 20, 20],
            formula_oh_accum = True,
            linear_layer_n = 2,
            possible_formulae_norm = False,
            normalize_1_output = False, 
            
            mask_zero_subsets=False, 
            concat_feats = False,
            
            norm_post_combine = True,
    ):
        
        super( DebugNet, self).__init__()

        formula_encoding_n = np.sum(formula_oh_sizes)

        self.spect_bin = spect_bin
        
        self.formula_to_oh = StructuredOneHot(formula_oh_sizes, formula_oh_accum)
        
        self.subset_norm = subset_norm
        if self.subset_norm:
            self.subset_weighted_norm = nn.InstanceNorm1d(g_feat_in)

        self.possible_formulae_norm = possible_formulae_norm
        if self.possible_formulae_norm:
            self.pf_norm = nn.BatchNorm1d(formula_encoding_n)
            
        #self.combine_layers = nn.GRUCell(formula_encoding_n, g_feat_in)
        self.combine_layers = nn.Linear(formula_encoding_n + g_feat_in, g_feat_in)
        #self.combine_layers = nn.Bilinear(formula_encoding_n, g_feat_in, g_feat_in)
            
        self.concat_feats = concat_feats
        if self.concat_feats:
            self.f_combine_l1 = nn.Linear(3 * g_feat_in, internal_d)
        else:
            self.f_combine_l1 = nn.Linear(g_feat_in, internal_d)
        self.f_combine_l2 = nn.Sequential(*[nn.Sequential(nn.Linear(internal_d, internal_d),
                                                          nn.LeakyReLU()) for _ in range(linear_layer_n)])

        if norm_post_combine:
            self.do_norm_post_combine = True
            self.norm_post_combine = nn.InstanceNorm1d(g_feat_in)
           
        self.norm_pre_score = nn.InstanceNorm1d(internal_d)
        self.f_combine_score = nn.Linear(internal_d, 1)
        
        self.prob_softmax = prob_softmax
        self.normalize_1_output = normalize_1_output


        self.mask_zero_subsets = mask_zero_subsets

    def forward(
            self, vert_feat_in, vert_mask_in,
            vert_element_oh, adj_oh,
            atom_subsets, atom_subsets_peaks,
            sparse_mass_matrix,
            return_formulae_scores=False,
            atom_subsets_peaks_mass_idx=None, 
            atom_subsets_peaks_intensity=None,
            **kwargs

        ):
        """
        vert_feat_in : BATCH_N x ATOM_N x G_F
        vert_mask_in: BATCH_N x ATOM_N  
        atom_subsets : BATCH_N x SUBSET_SAMPLES_N x ATOM_N
        #atom_subsets_peaks : BATCH_N x SUBSET_SAMPLES_N x NUM_MASSES x 2  (mass, peak intensity) 
        """

        BATCH_N, ATOM_N, G_F  = vert_feat_in.shape
        _, SUBSET_SAMPLES_N, _ = atom_subsets.shape

        assert atom_subsets_peaks_mass_idx is not None
        assert atom_subsets_peaks_intensity is not None
        
        # apply input mask to keep only features for atoms that actually exist
        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1)
        atom_subsets_masked = atom_subsets * vert_mask_in.unsqueeze(1)
        atom_subsets_masked_float = atom_subsets_masked.float()

        nonzero_atom_subsets = (atom_subsets.sum(dim=2) != 0).float()
        
        # each subset embedding is obtained by summing the vertex features
        # for only the atoms contained in that subset
        sample_weighted_vert_sum = torch.einsum("ijk,ilj->ilk", masked_vert_feat.float(), atom_subsets_masked_float)

        # generate the formulae (how many of element E did we have?) for each vertex subset
        # BATCH_N x SUBSET_SAMPLES_N x 8
        possible_formulae = torch.einsum("bae,bfa->bfe", vert_element_oh.float(), atom_subsets.float()).byte()
        
        # convert to "cumulative one-hot" encoding
        # BATCH_N x SUBSET_SAMPLES_N x sum(formula_oh_sizes)
        possible_formulae_flat = possible_formulae.reshape(-1, possible_formulae.shape[-1])
        possible_formulae_flat_oh = self.formula_to_oh(possible_formulae_flat)
        possible_formulae = possible_formulae_flat_oh.reshape(possible_formulae.shape[0],
                                                              possible_formulae.shape[1], -1)

        if self.possible_formulae_norm:
            # do batch-norm over formulae, averaging over all formulae w unique batch_index x formula_index
            possible_formulae = self.pf_norm(possible_formulae.reshape(-1, possible_formulae.shape[-1])).reshape(possible_formulae.shape)

        # print("vert_element_oh.shape=", vert_element_oh.shape,
        #       "atom_subset.shape=", atom_subsets.shape,
        #       "possible_formulae.shape=", possible_formulae.shape)
        assert sample_weighted_vert_sum.shape == (BATCH_N, SUBSET_SAMPLES_N, G_F)
        
        # compute vertsubset embedding as mean over all present vertex embeddings
        eps = 1e-4
        subset_size = atom_subsets_masked_float.sum(dim=2) + eps  # BATCH_N x SUBSET_SAMPLES_N
        sample_weighted_vert_mean = sample_weighted_vert_sum / subset_size.unsqueeze(-1)
        
        if self.subset_norm:
            # do layer-norm over all subset embeddings
            subset_normed = self.subset_weighted_norm(sample_weighted_vert_mean)
        else:
            subset_normed = sample_weighted_vert_mean

        combined = F.relu(self.combine_layers(
            torch.cat([possible_formulae, subset_normed], -1)))

        if self.do_norm_post_combine:
            x = self.norm_post_combine(combined)
        else:
            x = combined


        # TODO: add batch/layer norms to this part?
        x = F.relu(self.f_combine_l1(x))
        x = F.relu(self.f_combine_l2(x))

        # this LayerNorm normalizes s.t. x.sum(dim=-1) is zero-mean
        x = self.norm_pre_score(x)  # BATCH_N x SUBSET_SAMPLES_N x INTERNAL_D

        formulae_scores = self.f_combine_score(x).squeeze(2)   # BATCH_N x SUBSET_SAMPLES_N

        if return_formulae_scores:
            return formulae_scores

        if self.mask_zero_subsets:
            zero_atom_subsets = 1- nonzero_atom_subsets
            formulae_scores = formulae_scores + (-1000 * zero_atom_subsets)
        if self.prob_softmax:
            formulae_probs = torch.softmax(formulae_scores, dim=-1)
        else:
            formulae_probs = torch.sigmoid(formulae_scores)

        #t1 = time.time()
        if sparse_mass_matrix is not None:
            with torch.cuda.amp.autocast(enabled=False):
                spect_out = mat_matrix_sparse_mm(sparse_mass_matrix, formulae_probs)  # BATCH_N x SPECT_BIN_N
        else:
            # generate sparse mass matrix on the fly
            with torch.cuda.amp.autocast(enabled=False):

                # per batch
                out_y = []
                for batch_i in range(BATCH_N):
                    sparse_mat_matrix = peak_indices_intensities_to_sparse_matrix(atom_subsets_peaks_mass_idx[batch_i],
                                                                                       atom_subsets_peaks_intensity[batch_i], 
                                                                                       self.spect_bin.get_num_bins())
                    y = sparse_mat_matrix @ formulae_probs[batch_i]
                    out_y.append(y)
                spect_out = torch.stack(out_y)
                
        if self.normalize_1_output:
            spect_out = spect_out / (torch.sum(spect_out, dim=1) + 1e-9).unsqueeze(-1)

        subset_probs = formulae_probs
        return {'spect_out': spect_out,
                'subset_probs' : subset_probs}

    
