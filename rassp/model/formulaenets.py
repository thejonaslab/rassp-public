"""
Nets that have an explicit representation of the formulae
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import scipy.stats
import time

from .nets import *

def create_mass_matrix_oh(masses, SPECT_BIN_N, mass_intensities=None):
    """
    Create a one-hot mass matrix from a list of formula masses

    input: 
    masses = BATCH_N x POSSIBLE_FORMULAE_N (integers)
    spect_bin_N : max number of spectral bins
    
    mass_intensities: array with same shape as masses, if None then we just use all 1s
    
    
    output:
    BATCH_N x POSSIBLE_FORMAULE_N x SPECT_BIN_N : 1-hot encoding of masses 
    """
    BATCH_N, POSSIBLE_FORMULAE_N = masses.shape

    #print(masses.shape, masses.dtype, mass_intensities.shape, mass_intensities.dtype)
    out = torch.zeros((BATCH_N, POSSIBLE_FORMULAE_N, SPECT_BIN_N)).to(masses.device)
    a = out.reshape(-1, SPECT_BIN_N)

    if mass_intensities is None:
        mass_intensities = torch.ones_like(masses).float().to(masses.device)
    
    b = masses.reshape(-1, 1)
    c = mass_intensities.reshape(-1, 1)
    a.scatter_(1, b,  c)

    return a.reshape(out.shape)

class StructuredOneHot(nn.Module):
    def __init__(self, oh_sizes, cumulative=False):
        """
      
        """
        super(StructuredOneHot, self).__init__()
        self.oh_sizes = oh_sizes
        self.cumulative = cumulative 
        
        if self.cumulative:
            n = np.sum(oh_sizes)
            accum_mat = np.zeros((n, n), dtype=np.int64)
            offset = 0
            for i, s in enumerate(oh_sizes):
                e = np.tril(np.ones((s, s)))
                accum_mat[offset:offset+s, offset:offset+s, ] = e
                offset += s

            a = torch.Tensor(accum_mat).long()
            self.register_buffer('accum_mat', a)
            
    def forward(self, data):
        BATCH_N, OH_N = data.shape
        data = data.long()
        try:
            
            oh_list = [F.one_hot(data[:, i], gs) for i, gs in enumerate(self.oh_sizes)]
        except:
            print([(torch.max(data[:, i]).item(), gs)  for i, gs in enumerate(self.oh_sizes)])
            raise
        oh = torch.cat(oh_list, -1)
        
        if self.cumulative:
            return oh.float() @ self.accum_mat.float()
        else:
            return oh.float()

class MolDotFormulaNet(nn.Module):
    """
    Per-vertex features to sparse points of peak, mass

    Input:
    BATCH_N x ATOM_N x F

    Output:
    BATCH_N x SPECT_BIN

    """
    def __init__(self, g_feat_in,
                 mol_agg = 'sum',
                 formula_encoding_n = 128, 
                 spect_bin_n = 512):

        
        super( MolDotFormulaNet, self).__init__()

        self.mol_agg = mol_agg

        self.norm = nn.LayerNorm(g_feat_in)

        self.f_embed_l = nn.Linear(formula_encoding_n, g_feat_in)
        self.spect_bin_n = spect_bin_n
        
    def forward(self, vert_feat_in, vert_mask_in,
                possible_formulae, formulae_masses):
        """
        vert_feat_in : BATCH_N x ATOM_N x F
        vert_feat_mask : BATCH_N x ATOM_N  
        possible_formulae : BATCH_X x MAX_FORMULAE_N x FORMULA_ENCODING_N
        formula_masses : BATCH_N x MAX_FORMULAE_N  integer masses

        """

        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1)
        
        if self.mol_agg == 'sum':
            mol_agg = torch.sum(masked_vert_feat, dim=1)
        elif self.mol_agg == 'max':
            mol_agg = goodmax(masked_vert_feat, dim=1)
        mol_agg = self.norm(mol_agg)

        embedded_formulae = self.f_embed_l(possible_formulae)

        formulae_scores = torch.einsum("ij,ikj->ik", mol_agg, embedded_formulae)
        formulae_probs = torch.softmax(formulae_scores, dim=-1)

        mass_matrix = create_mass_matrix_oh(formulae_masses, self.spect_bin_n).to(vert_feat_in.device)
        spect_out = torch.einsum("ij,ijk->ik", formulae_probs, mass_matrix)
        return spect_out


class MolLinearFormulaNet(nn.Module):
    """
    Per-vertex features to sparse points of peak, mass

    Input:
    BATCH_N x ATOM_N x F

    Output:
    BATCH_N x SPECT_BIN

    """
    def __init__(self, g_feat_in,
                 mol_agg = 'sum',
                 formula_encoding_n = 128, 
                 spect_bin_n = 512):

        
        super( MolLinearFormulaNet, self).__init__()

        self.mol_agg = mol_agg

        self.norm = nn.LayerNorm(g_feat_in)

        self.combine_norm = nn.LayerNorm(formula_encoding_n +  g_feat_in)
        self.f_combine_l1 = nn.Linear(formula_encoding_n +  g_feat_in, 512)
        self.f_combine_l2 = nn.Linear(512, 512)
        self.f_combine_score = nn.Linear(512, 1)
        
        self.spect_bin_n = spect_bin_n
        
    def forward(self, vert_feat_in, vert_mask_in,
                possible_formulae, formulae_masses):
        """
        vert_feat_in : BATCH_N x ATOM_N x F
        vert_feat_mask : BATCH_N x ATOM_N  
        possible_formulae : BATCH_X x MAX_FORMULAE_N x FORMULA_ENCODING_N
        formula_masses : BATCH_N x MAX_FORMULAE_N  integer masses

        """

        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1)
        
        if self.mol_agg == 'sum':
            mol_agg = torch.sum(masked_vert_feat, dim=1)
        elif self.mol_agg == 'mean':
            mol_agg = torch.mean(masked_vert_feat, dim=1)
        elif self.mol_agg == 'max':
            mol_agg = goodmax(masked_vert_feat, dim=1)
        mol_agg = self.norm(mol_agg)

        mol_agg_expand =  mol_agg.unsqueeze(1).expand(-1, possible_formulae.shape[1],-1)
        combined = torch.cat([possible_formulae, mol_agg_expand], -1)
        combined = self.combine_norm(combined)
        x = F.relu(self.f_combine_l1(combined))
        x = F.relu(self.f_combine_l2(x))
        formulae_scores = self.f_combine_score(x).squeeze(2)
        
        
        formulae_probs = torch.softmax(formulae_scores, dim=-1)

        mass_matrix = create_mass_matrix_oh(formulae_masses, self.spect_bin_n).to(vert_feat_in.device)
        spect_out = torch.einsum("ij,ijk->ik", formulae_probs, mass_matrix)
        return spect_out


class GraphVertSpect(nn.Module):
    """
    g_feature_n: starting number of input features
    g_feature_out_n: (override)the number of intermediate features
    int_d: number of intermediate features
    layer_n: number of layers to apply


    """
    def __init__(self,
                 g_feature_n, spect_bin,
                 g_feature_out_n=None, 
                 int_d = None, layer_n = None, 

                resnet=True, 
                gml_class = 'GraphMatLayers',
                gml_config = {}, 
                init_noise=1e-5,
                init_bias=0.0,
                agg_func=None,
                GS=1,

                spect_out_class='',
                spect_out_config = {},
                spect_mode = 'dense', 
                input_norm='batch',
                
                inner_norm=None,
                default_render_width = 0.1,
                default_mass_max = 512, 
        ):
        
        """

        """
        super( GraphVertSpect, self).__init__()

        if layer_n is not None:
            g_feature_out_n = [int_d] * layer_n

        self.gml = eval(gml_class)(g_feature_n, g_feature_out_n, 
                                   resnet=resnet, noise=init_noise,
                                   agg_func=parse_agg_func(agg_func), 
                                   norm = inner_norm, 
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
        self.default_mass_max = default_mass_max
        
        self.pos = 0
        
    def forward(self, adj, vect_feat, input_mask, input_idx, adj_oh,
                return_g_features = False, 
                mol_feat=None,
                formulae_features = None,
                formulae_peaks_mass_idx = None,
                formulae_peaks_intensity = None, 
                vert_element_oh = None,
                formula_frag_count = None,
                **kwargs):

        G = adj
        
        BATCH_N, MAX_N, F_N = vect_feat.shape

        if self.input_norm is not None:
            vect_feat = apply_masked_1d_norm(self.input_norm, 
                                             vect_feat, 
                                             input_mask)
        
        # we compute a global embedding of the parent molecule
        G_features = self.gml(G, vect_feat, input_mask)
        if return_g_features:
            return G_features

        g_squeeze = G_features.squeeze(1)

        
        pred_dense_spect_dict = self.spect_out(g_squeeze, input_mask,
                                               formulae_features,
                                               formulae_peaks_mass_idx,
                                               formulae_peaks_intensity, 
                                               vert_element_oh, adj_oh, formula_frag_count)
        pred_dense_spect = pred_dense_spect_dict['spect_out']

        #pred_masses, pred_probs = dense_to_sparse(pred_dense_spect)
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


    

class MolBilinearFormulaNet(nn.Module):
    """
    ERROR TOO MUCH MEMORY 
    Per-vertex features to sparse points of peak, mass

    Input:
    BATCH_N x ATOM_N x F

    Output:
    BATCH_N x SPECT_BIN

    """
    def __init__(self, g_feat_in,
                 mol_agg = 'sum',
                 formula_encoding_n = 128, 
                 spect_bin_n = 512):

        
        super( MolBilinearFormulaNet, self).__init__()

        self.mol_agg = mol_agg

        self.norm = nn.LayerNorm(g_feat_in)


        self.combine_bilinear = nn.Bilinear(g_feat_in, formula_encoding_n, 512)
        self.combine_norm = nn.LayerNorm(512)
        self.f_combine_l1 = nn.Linear(512, 512)
        self.f_combine_l2 = nn.Linear(512, 512)
        self.f_combine_score = nn.Linear(512, 1)
        
        self.spect_bin_n = spect_bin_n
        
    def forward(self, vert_feat_in, vert_mask_in,
                possible_formulae, formulae_masses):
        """
        vert_feat_in : BATCH_N x ATOM_N x F
        vert_feat_mask : BATCH_N x ATOM_N  
        possible_formulae : BATCH_X x MAX_FORMULAE_N x FORMULA_ENCODING_N
        formula_masses : BATCH_N x MAX_FORMULAE_N  integer masses

        """

        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1)
        
        if self.mol_agg == 'sum':
            mol_agg = torch.sum(masked_vert_feat, dim=1)
        elif self.mol_agg == 'mean':
            mol_agg = torch.mean(masked_vert_feat, dim=1)
        elif self.mol_agg == 'max':
            mol_agg = goodmax(masked_vert_feat, dim=1)
        mol_agg = self.norm(mol_agg)

        mol_agg_expand =  mol_agg.unsqueeze(1).expand(-1, possible_formulae.shape[1],-1).clone()

        combined = self.combine_bilinear(mol_agg_expand, possible_formulae)
        combined = self.combine_norm(combined)
        x = F.relu(self.f_combine_l1(combined))
        x = F.relu(self.f_combine_l2(x))
        formulae_scores = self.f_combine_score(x).squeeze(2)
        
        
        formulae_probs = torch.softmax(formulae_scores, dim=-1)

        mass_matrix = create_mass_matrix_oh(formulae_masses, self.spect_bin_n).to(vert_feat_in.device)
        spect_out = torch.einsum("ij,ijk->ik", formulae_probs, mass_matrix)
        return spect_out

    
class MolLinearFormulaNet2(nn.Module):
    """
    Per-vertex features to sparse points of peak, mass

    Input:
    BATCH_N x ATOM_N x F

    Output:
    BATCH_N x SPECT_BIN

    """
    def __init__(self, g_feat_in,
                 mol_agg = 'sum',
                 formula_encoding_n = 128, 
                 spect_bin_n = 512):

        
        super( MolLinearFormulaNet2, self).__init__()

        self.mol_agg = mol_agg

        self.norm = nn.LayerNorm(g_feat_in)
        self.norm2 = nn.LayerNorm(512)

        self.combine_norm = nn.LayerNorm(formula_encoding_n +  g_feat_in)
        self.f_combine_l1 = nn.Linear(formula_encoding_n +  g_feat_in, 512)
        self.f_combine_l2 = nn.Linear(512, 512)
        self.f_combine_score = nn.Linear(512, 1)
        
        self.spect_bin_n = spect_bin_n
        
    def forward(self, vert_feat_in, vert_mask_in,
                possible_formulae, formulae_masses):
        """
        vert_feat_in : BATCH_N x ATOM_N x F
        vert_feat_mask : BATCH_N x ATOM_N  
        possible_formulae : BATCH_X x MAX_FORMULAE_N x FORMULA_ENCODING_N
        formula_masses : BATCH_N x MAX_FORMULAE_N  integer masses

        """

        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1)
        
        if self.mol_agg == 'sum':
            mol_agg = torch.sum(masked_vert_feat, dim=1)
        elif self.mol_agg == 'mean':
            mol_agg = torch.mean(masked_vert_feat, dim=1)
        elif self.mol_agg == 'max':
            mol_agg = goodmax(masked_vert_feat, dim=1)
        mol_agg = self.norm(mol_agg)

        mol_agg_expand =  mol_agg.unsqueeze(1).expand(-1, possible_formulae.shape[1],-1)
        combined = torch.cat([possible_formulae, mol_agg_expand], -1)
        combined = self.combine_norm(combined)
        x = F.relu(self.f_combine_l1(combined))
        x = F.relu(self.f_combine_l2(x))
        x = self.norm2(x)
        formulae_scores = self.f_combine_score(x).squeeze(2)
        
        
        formulae_probs = torch.softmax(formulae_scores, dim=-1)

        mass_matrix = create_mass_matrix_oh(formulae_masses, self.spect_bin_n).to(vert_feat_in.device)
        spect_out = torch.einsum("ij,ijk->ik", formulae_probs, mass_matrix)
        return spect_out


class MolAttentionNet(nn.Module):
    """
    Per-vertex features to sparse points of peak, mass

    Input:
    BATCH_N x ATOM_N x F

    Output:
    BATCH_N x SPECT_BIN

    """
    def __init__(self, g_feat_in,
                 mol_agg = 'sum',
                 formula_encoding_n = 128,
                 embedding_key_size = 16,
                 spect_bin_n = 512):

        
        super( MolAttentionNet, self).__init__()

        self.mol_agg = mol_agg

        self.norm = nn.LayerNorm(g_feat_in)

        self.embed_g_feat = nn.Linear(g_feat_in, embedding_key_size)
        self.embed_formulae_feat = nn.Linear(formula_encoding_n, embedding_key_size)
        
        self.norm2 = nn.LayerNorm(512)

        self.combine_norm = nn.LayerNorm(formula_encoding_n +  g_feat_in)
        self.f_combine_l1 = nn.Linear(formula_encoding_n +  g_feat_in, 512)
        self.f_combine_l2 = nn.Linear(512, 512)
        self.f_combine_score = nn.Linear(512, 1)
        
        self.spect_bin_n = spect_bin_n
        
    def forward(self, vert_feat_in, vert_mask_in,
                possible_formulae, formulae_masses):
        """
        vert_feat_in : BATCH_N x ATOM_N x F
        vert_feat_mask : BATCH_N x ATOM_N  
        possible_formulae : BATCH_N x MAX_FORMULAE_N x FORMULA_ENCODING_N
        formula_masses : BATCH_N x MAX_FORMULAE_N  integer masses

        """

        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1)

        # encoded inputs
        vert_encoded = self.embed_g_feat(vert_feat_in)
        formulae_encoded = self.embed_formulae_feat(possible_formulae)

        dot_prod = (formulae_encoded.unsqueeze(1) * vert_encoded.unsqueeze(2)).sum(dim=-1)
        weighting = torch.softmax(dot_prod, dim=1)
        
        vert_att = masked_vert_feat.unsqueeze(3) * weighting.unsqueeze(2)
        vert_att_reduce = vert_att.sum(dim=1).permute(0, 2, 1) # (BATCH_N, FORMULAE_N, F)
        combined = torch.cat([possible_formulae, vert_att_reduce], -1)
        combined = self.combine_norm(combined)
        x = F.relu(self.f_combine_l1(combined))
        x = F.relu(self.f_combine_l2(x))
        x = self.norm2(x)
        formulae_scores = self.f_combine_score(x).squeeze(2)
        
        
        formulae_probs = torch.softmax(formulae_scores, dim=-1)

        mass_matrix = create_mass_matrix_oh(formulae_masses, self.spect_bin_n).to(vert_feat_in.device)
        spect_out = torch.einsum("ij,ijk->ik", formulae_probs, mass_matrix)
        return spect_out


class MolAttentionNetOH(nn.Module):
    """
    Per-vertex features to sparse points of peak, mass

    Input:
    BATCH_N x ATOM_N x F

    Output:
    BATCH_N x SPECT_BIN

    """
    def __init__(self, g_feat_in,
                 embedding_key_size = 16,
                 formula_oh_sizes = [20, 20, 20, 20, 20],
                 formula_oh_accum = True,
                 internal_d = 512,
                 prob_softmax = True,
                 spect_bin_n = 512):

        
        super( MolAttentionNetOH, self).__init__()

        self.norm = nn.LayerNorm(g_feat_in)

        self.embed_g_feat = nn.Linear(g_feat_in, embedding_key_size)
        formula_encoding_n = np.sum(formula_oh_sizes)

        self.formula_to_oh = StructuredOneHot(formula_oh_sizes, formula_oh_accum)
        
        self.embed_formulae_feat = nn.Linear(formula_encoding_n, embedding_key_size)
        
        self.norm2 = nn.LayerNorm(internal_d)

        self.combine_norm = nn.LayerNorm(formula_encoding_n +  g_feat_in)
        self.f_combine_l1 = nn.Linear(formula_encoding_n +  g_feat_in, internal_d)
        self.f_combine_l2 = nn.Linear(internal_d, internal_d)
        self.f_combine_score = nn.Linear(internal_d, 1)

        
        self.spect_bin_n = spect_bin_n
        self.prob_softmax = prob_softmax
        
    def forward(self, vert_feat_in, vert_mask_in,
                possible_formulae, formulae_masses, vert_element_oh):
        """
        vert_feat_in : BATCH_N x ATOM_N x F
        vert_feat_mask : BATCH_N x ATOM_N  
        possible_formulae : BATCH_N x MAX_FORMULAE_N x FORMULA_ENCODING_N
        formula_masses : BATCH_N x MAX_FORMULAE_N x NUM_MASSES x 2  (mass, peak intensity) 
        """

        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1)

        # one-hot encode possible formulae
        possible_formulae_flat = possible_formulae.reshape(-1, possible_formulae.shape[-1])
        possible_formulae_flat_oh = self.formula_to_oh(possible_formulae_flat)
        possible_formulae = possible_formulae_flat_oh.reshape(possible_formulae.shape[0],
                                                              possible_formulae.shape[1], -1)
        
        
        # encoded inputs
        vert_encoded = self.embed_g_feat(vert_feat_in)
        formulae_encoded = self.embed_formulae_feat(possible_formulae)

        dot_prod = (formulae_encoded.unsqueeze(1) * vert_encoded.unsqueeze(2)).sum(dim=-1)
        weighting = torch.softmax(dot_prod, dim=1)
        
        vert_att = masked_vert_feat.unsqueeze(3) * weighting.unsqueeze(2)
        vert_att_reduce = vert_att.sum(dim=1).permute(0, 2, 1) # (BATCH_N, FORMULAE_N, F)
        combined = torch.cat([possible_formulae, vert_att_reduce], -1)
        combined = self.combine_norm(combined)
        x = F.relu(self.f_combine_l1(combined))
        x = F.relu(self.f_combine_l2(x))
        x = self.norm2(x)
        formulae_scores = self.f_combine_score(x).squeeze(2)
        
        if self.prob_softmax:
            formulae_probs = torch.softmax(formulae_scores, dim=-1)
        else:
            formulae_probs = torch.sigmoid(formulae_scores)

        mass_matrices = [create_mass_matrix_oh(formulae_masses[:, :, i, 0].round().long(),
                                               self.spect_bin_n,
                                               formulae_masses[:, :, i, 1]
        )\
                         .to(vert_feat_in.device) for i in range(formulae_masses.shape[2])]
        mass_matrix = torch.sum(torch.stack(mass_matrices, -1), -1)


        spect_out = torch.einsum("ij,ijk->ik", formulae_probs, mass_matrix)
        return spect_out


class MolAttentionNetOHMultiHead(nn.Module):
    """
    Per-vertex features to sparse points of peak, mass

    Input:
    BATCH_N x ATOM_N x F

    Output:
    BATCH_N x SPECT_BIN

    """
    def __init__(self, g_feat_in,
                 embedding_key_size = 16,
                 formula_oh_sizes = [20, 20, 20, 20, 20],
                 formula_oh_accum = True,
                 head_n = 1, 
                 internal_d = 512, 
                 spect_bin_n = 512):

        
        super( MolAttentionNetOHMultiHead, self).__init__()


        self.embed_g_feat = nn.Linear(g_feat_in, embedding_key_size)
        formula_encoding_n = np.sum(formula_oh_sizes)

        self.formula_to_oh = StructuredOneHot(formula_oh_sizes, formula_oh_accum)
        
        self.embed_formulae_feat = nn.Linear(formula_encoding_n, embedding_key_size)

        self.vert_formula_attn = nn.ModuleList([VertFormulaAttn(g_feat_in, formula_encoding_n,
                                                                embedding_key_size) for _ in range(head_n)])
        

        self.combine_norm = nn.LayerNorm(formula_encoding_n +  g_feat_in * head_n)
        self.f_combine_l1 = nn.Linear(formula_encoding_n +  g_feat_in*head_n, internal_d)
        self.f_combine_l2 = nn.Linear(internal_d, internal_d)
        self.norm2 = nn.LayerNorm(internal_d)
        
        self.f_combine_score = nn.Linear(internal_d, 1)

        
        self.spect_bin_n = spect_bin_n
        
    def forward(self, vert_feat_in, vert_mask_in,
                possible_formulae, formulae_masses):
        """
        vert_feat_in : BATCH_N x ATOM_N x F
        vert_feat_mask : BATCH_N x ATOM_N  
        possible_formulae : BATCH_N x MAX_FORMULAE_N x FORMULA_ENCODING_N
        formula_masses : BATCH_N x MAX_FORMULAE_N x NUM_MASSES x 2  (mass, peak intensity) 
        """

        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1)

        # one-hot encode possible formulae
        possible_formulae_flat = possible_formulae.reshape(-1, possible_formulae.shape[-1])
        possible_formulae_flat_oh = self.formula_to_oh(possible_formulae_flat)
        possible_formulae = possible_formulae_flat_oh.reshape(possible_formulae.shape[0],
                                                              possible_formulae.shape[1], -1)
        
        # vert_attn_reduce = self.vert_formula_attn(masked_vert_feat,
        #                                           possible_formulae)
        vert_attn_reduce_list = [h(masked_vert_feat,possible_formulae) for h in self.vert_formula_attn]
        
        
        combined = torch.cat([possible_formulae] + vert_attn_reduce_list, -1)
        combined = self.combine_norm(combined)
        x = F.relu(self.f_combine_l1(combined))
        x = F.relu(self.f_combine_l2(x))
        x = self.norm2(x)
        formulae_scores = self.f_combine_score(x).squeeze(2)
        
        
        formulae_probs = torch.softmax(formulae_scores, dim=-1)

        mass_matrices = [create_mass_matrix_oh(formulae_masses[:, :, i, 0].round().long(),
                                               self.spect_bin_n,
                                               formulae_masses[:, :, i, 1]
        )\
                         .to(vert_feat_in.device) for i in range(formulae_masses.shape[2])]
        mass_matrix = torch.sum(torch.stack(mass_matrices, -1), -1)


        spect_out = torch.einsum("ij,ijk->ik", formulae_probs, mass_matrix)
        return spect_out




class VertFormulaAttn(nn.Module):
    """
    """
    def __init__(self, g_feat_in,
                 formula_encoding_n, 
                 embedding_key_size = 16):

        
        super( VertFormulaAttn, self).__init__()


        self.embed_g_feat = nn.Linear(g_feat_in, embedding_key_size)
        
        self.embed_formulae_feat = nn.Linear(formula_encoding_n, embedding_key_size)
        
        
    def forward(self, masked_vert_feat, 
                possible_formulae):
        
        # encoded inputs
        vert_encoded = self.embed_g_feat(masked_vert_feat)
        formulae_encoded = self.embed_formulae_feat(possible_formulae)

        dot_prod = (formulae_encoded.unsqueeze(1) * vert_encoded.unsqueeze(2)).sum(dim=-1)
        weighting = torch.softmax(dot_prod, dim=1)
        
        vert_att = masked_vert_feat.unsqueeze(3) * weighting.unsqueeze(2)
        vert_att_reduce = vert_att.sum(dim=1).permute(0, 2, 1) # (BATCH_N, FORMULAE_N, F)
        return vert_att_reduce


class MolAttentionNetElementOH(nn.Module):
    """
    Per-vertex features to sparse points of peak, mass

    Input:
    BATCH_N x ATOM_N x F

    Output:
    BATCH_N x SPECT_BIN

    """
    def __init__(self, g_feat_in,
                 embedding_key_size = 16,
                 formula_oh_sizes = [20, 20, 20, 20, 20],
                 formula_oh_accum = True,
                 internal_d = 512, 
                 spect_bin_n = 512):

        
        super( MolAttentionNetElementOH, self).__init__()


        self.embed_g_feat = nn.Linear(g_feat_in, embedding_key_size)
        formula_encoding_n = np.sum(formula_oh_sizes)

        self.formula_to_oh = StructuredOneHot(formula_oh_sizes, formula_oh_accum)
        
        self.embed_formulae_feat = nn.Linear(formula_encoding_n, embedding_key_size)

        self.elt_type_lin = nn.Linear(g_feat_in, g_feat_in)
        
        self.norm2 = nn.LayerNorm(internal_d)

        self.combine_norm = nn.LayerNorm(formula_encoding_n +  g_feat_in + g_feat_in)
        self.f_combine_l1 = nn.Linear(formula_encoding_n +  g_feat_in + g_feat_in, internal_d)
        self.f_combine_l2 = nn.Linear(internal_d, internal_d)
        self.f_combine_score = nn.Linear(internal_d, 1)

        
        self.spect_bin_n = spect_bin_n
        
    def forward(self, vert_feat_in, vert_mask_in,
                possible_formulae_in, formulae_masses,
                vert_element_oh):
        """
        vert_feat_in : BATCH_N x ATOM_N x F
        vert_feat_mask : BATCH_N x ATOM_N  
        possible_formulae : BATCH_N x MAX_FORMULAE_N x FORMULA_ENCODING_N
        formula_masses : BATCH_N x MAX_FORMULAE_N x NUM_MASSES x 2  (mass, peak intensity) 
        vert_Element_oh: BATCH_N x ATOM_N x possible_elements
        """

        BATCH_N, ATOM_N, V_F = vert_feat_in.shape
        _, _, ELT_N = vert_element_oh.shape
        _, POSSIBLE_FORMULAE_N, ELT_N_2 = possible_formulae_in.shape

        assert ELT_N == ELT_N_2
        
        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1)

        # one-hot encode possible formulae
        possible_formulae_flat = possible_formulae_in.reshape(-1, possible_formulae_in.shape[-1])
        possible_formulae_flat_oh = self.formula_to_oh(possible_formulae_flat)
        possible_formulae = possible_formulae_flat_oh.reshape(possible_formulae_in.shape[0],
                                                              possible_formulae_in.shape[1], -1)
        
        
        ### reduce vertex features by atom type
        vertf_by_elt = torch.einsum('ijk,ijl->ilk', masked_vert_feat, vert_element_oh)
        #print(vertf_by_elt.shape,  (BATCH_N, ELT_N, V_F))
        assert vertf_by_elt.shape == (BATCH_N, ELT_N, V_F)
        vertf_by_elt_post = F.relu(self.elt_type_lin(vertf_by_elt))
        
        ## FIXME remember formula-in are not one-hot encoded: we should either norm the result or whatever
        elt_weighted_vert_f = torch.einsum("ijk,ilj->ilk", vertf_by_elt_post, possible_formulae_in.float())
        assert elt_weighted_vert_f.shape == (BATCH_N, POSSIBLE_FORMULAE_N, V_F)
        
        
        
        # encoded inputs
        vert_encoded = self.embed_g_feat(vert_feat_in)
        formulae_encoded = self.embed_formulae_feat(possible_formulae)

        dot_prod = (formulae_encoded.unsqueeze(1) * vert_encoded.unsqueeze(2)).sum(dim=-1)
        weighting = torch.softmax(dot_prod, dim=1)
        
        vert_att = masked_vert_feat.unsqueeze(3) * weighting.unsqueeze(2)
        vert_att_reduce = vert_att.sum(dim=1).permute(0, 2, 1) # (BATCH_N, FORMULAE_N, F)
        combined = torch.cat([possible_formulae, vert_att_reduce, elt_weighted_vert_f], -1)
        combined = self.combine_norm(combined)
        x = F.relu(self.f_combine_l1(combined))
        x = F.relu(self.f_combine_l2(x))
        x = self.norm2(x)
        formulae_scores = self.f_combine_score(x).squeeze(2)
        
        
        formulae_probs = torch.softmax(formulae_scores, dim=-1)

        mass_matrices = [create_mass_matrix_oh(formulae_masses[:, :, i, 0].round().long(),
                                               self.spect_bin_n,
                                               formulae_masses[:, :, i, 1]
        )\
                         .to(vert_feat_in.device) for i in range(formulae_masses.shape[2])]
        mass_matrix = torch.sum(torch.stack(mass_matrices, -1), -1)


        spect_out = torch.einsum("ij,ijk->ik", formulae_probs, mass_matrix)
        return spect_out











class MolAttentionNetMHA(nn.Module):
    """
    Per-vertex features to sparse points of peak, mass

    Input:
    BATCH_N x ATOM_N x F

    Output:
    BATCH_N x SPECT_BIN

    """
    def __init__(self, g_feat_in,
                 embedding_key_size = 16,
                 formula_oh_sizes = [20, 20, 20, 20, 20],
                 formula_oh_accum = True,
                 internal_d = 512,
                 prob_softmax = True,
                 num_heads = 4, 
                 
                 spect_bin_n = 512):

        
        super( MolAttentionNetMHA, self).__init__()

        self.norm = nn.LayerNorm(g_feat_in)

        self.embed_g_feat = nn.Linear(g_feat_in, embedding_key_size)
        formula_encoding_n = np.sum(formula_oh_sizes)


        self.mha1 = nn.MultiheadAttention(embed_dim = embedding_key_size,
                                          num_heads = num_heads)

        self.formula_to_oh = StructuredOneHot(formula_oh_sizes, formula_oh_accum)
        
        self.embed_formulae_feat = nn.Linear(formula_encoding_n, embedding_key_size)
        
        self.norm2 = nn.LayerNorm(internal_d)

        self.combine_norm = nn.LayerNorm(formula_encoding_n +  g_feat_in)
        self.f_combine_l1 = nn.Linear(formula_encoding_n +  g_feat_in, internal_d)
        self.f_combine_l2 = nn.Linear(internal_d, internal_d)
        self.f_combine_score = nn.Linear(internal_d, 1)

        
        
        self.spect_bin_n = spect_bin_n
        self.prob_softmax = prob_softmax
        
    def forward(self, vert_feat_in, vert_mask_in,
                possible_formulae, formulae_masses, vert_element_oh):
        """
        vert_feat_in : BATCH_N x ATOM_N x F
        vert_feat_mask : BATCH_N x ATOM_N  
        possible_formulae : BATCH_N x MAX_FORMULAE_N x FORMULA_ENCODING_N
        formula_masses : BATCH_N x MAX_FORMULAE_N x NUM_MASSES x 2  (mass, peak intensity) 
        """

        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1)

        # one-hot encode possible formulae
        possible_formulae_flat = possible_formulae.reshape(-1, possible_formulae.shape[-1])
        possible_formulae_flat_oh = self.formula_to_oh(possible_formulae_flat)
        possible_formulae = possible_formulae_flat_oh.reshape(possible_formulae.shape[0],
                                                              possible_formulae.shape[1], -1)
        

        
        # encoded inputs
        formulae_encoded = self.embed_formulae_feat(possible_formulae)
        
        vert_encoded = self.embed_g_feat(vert_feat_in)
        formulae_expanded = formulae_encoded.unsqueeze(1).permute(1, 0, 2) # .expand(-1, vert_encoded.shape[1],-1)
        vert_atom_first = vert_encoded.permute(1, 0, 2)
        attn_output, attn_weights = self.mha1(vert_atom_first,
                                              formulae_expanded, 
                                              vert_atom_first)
        attn_output = attn_output.permute(1, 0, 2)
        
        

        dot_prod = (formulae_encoded.unsqueeze(1) * attn_output.unsqueeze(2)).sum(dim=-1)
        weighting = torch.softmax(dot_prod, dim=1)
        
        vert_att = masked_vert_feat.unsqueeze(3) * weighting.unsqueeze(2)
        vert_att_reduce = vert_att.sum(dim=1).permute(0, 2, 1) # (BATCH_N, FORMULAE_N, F)
        combined = torch.cat([possible_formulae, vert_att_reduce], -1)
        combined = self.combine_norm(combined)
        x = F.relu(self.f_combine_l1(combined))
        x = F.relu(self.f_combine_l2(x))
        x = self.norm2(x)
        formulae_scores = self.f_combine_score(x).squeeze(2)
        
        if self.prob_softmax:
            formulae_probs = torch.softmax(formulae_scores, dim=-1)
        else:
            formulae_probs = torch.sigmoid(formulae_scores)

        mass_matrices = [create_mass_matrix_oh(formulae_masses[:, :, i, 0].round().long(),
                                               self.spect_bin_n,
                                               formulae_masses[:, :, i, 1]
        )\
                         .to(vert_feat_in.device) for i in range(formulae_masses.shape[2])]
        mass_matrix = torch.sum(torch.stack(mass_matrices, -1), -1)


        spect_out = torch.einsum("ij,ijk->ik", formulae_probs, mass_matrix)
        return spect_out


class MolAttentionNetOHSparse(nn.Module):
    """
    Per-vertex features to sparse points of peak, mass

    Input:
    BATCH_N x ATOM_N x F

    Output:
    BATCH_N x SPECT_BIN

    """
    def __init__(self, g_feat_in,
                 embedding_key_size = 16,
                 formula_oh_sizes = [20, 20, 20, 20, 20],
                 formula_oh_accum = True,
                 internal_d = 512,
                 prob_softmax = True,
                 g_embed_train = True,
                 g_embed_bias = True, 
                 formula_embed_train= True,
                 formula_embed_bias=True, 
                 spect_bin_n = 512):

        
        super( MolAttentionNetOHSparse, self).__init__()

        self.norm = nn.LayerNorm(g_feat_in)

        self.embed_g_feat = nn.Linear(g_feat_in, embedding_key_size, bias= g_embed_bias)
        formula_encoding_n = np.sum(formula_oh_sizes)

        self.formula_to_oh = StructuredOneHot(formula_oh_sizes, formula_oh_accum)
        
        self.embed_formulae_feat = nn.Linear(formula_encoding_n, embedding_key_size,
                                             bias=formula_embed_bias)

        if not formula_embed_train:
            self.embed_formulae_feat.weight.requires_grad = False
            self.embed_formulae_feat.bias.requires_grad = False
            
        
        self.norm2 = nn.LayerNorm(internal_d)

        self.combine_norm = nn.LayerNorm(formula_encoding_n +  g_feat_in)
        self.f_combine_l1 = nn.Linear(formula_encoding_n +  g_feat_in, internal_d)
        self.f_combine_l2 = nn.Linear(internal_d, internal_d)
        self.f_combine_score = nn.Linear(internal_d, 1)

        
        self.spect_bin_n = spect_bin_n
        self.prob_softmax = prob_softmax


    def forward(self, vert_feat_in, vert_mask_in,
                possible_formulae, formulae_masses,
                vert_element_oh, adj_oh, formula_frag_count):
        """
        vert_feat_in : BATCH_N x ATOM_N x F
        vert_feat_mask : BATCH_N x ATOM_N  
        possible_formulae : BATCH_N x MAX_FORMULAE_N x FORMULA_ENCODING_N
        formula_masses : BATCH_N x MAX_FORMULAE_N x NUM_MASSES x 2  (mass, peak intensity) 
        """

        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1)

        # one-hot encode possible formulae
        possible_formulae_flat = possible_formulae.reshape(-1, possible_formulae.shape[-1])
        possible_formulae_flat_oh = self.formula_to_oh(possible_formulae_flat)
        possible_formulae = possible_formulae_flat_oh.reshape(possible_formulae.shape[0],
                                                              possible_formulae.shape[1], -1)
        
        
        # encoded inputs
        vert_encoded = self.embed_g_feat(vert_feat_in)
        formulae_encoded = self.embed_formulae_feat(possible_formulae)

        dot_prod = (formulae_encoded.unsqueeze(1) * vert_encoded.unsqueeze(2)).sum(dim=-1)
        weighting = torch.softmax(dot_prod, dim=1)
        
        vert_att = masked_vert_feat.unsqueeze(3) * weighting.unsqueeze(2)
        vert_att_reduce = vert_att.sum(dim=1).permute(0, 2, 1) # (BATCH_N, FORMULAE_N, F)
        combined = torch.cat([possible_formulae, vert_att_reduce], -1)
        combined = self.combine_norm(combined)
        x = F.relu(self.f_combine_l1(combined))
        x = F.relu(self.f_combine_l2(x))
        x = self.norm2(x)
        formulae_scores = self.f_combine_score(x).squeeze(2)
        
        if self.prob_softmax:
            formulae_probs = torch.softmax(formulae_scores, dim=-1)
        else:
            formulae_probs = torch.sigmoid(formulae_scores)

        #t1 = time.time()
        sparse_mass_matrices = [create_mass_matrix_sparse(formulae_masses[:, :, i, 0].round().long(),
                                                   self.spect_bin_n,
                                                   formulae_masses[:, :, i, 1]
        )\
                         .to(vert_feat_in.device) for i in range(formulae_masses.shape[2])]

        
        sparse_mass_matrix = sparse_mass_matrices[0]
        for m in sparse_mass_matrices[1:]:
            sparse_mass_matrix += m

        spect_out = mat_matrix_sparse_mm(sparse_mass_matrix, formulae_probs)

        return {'spect_out': spect_out,
                'formulae_probs' : formulae_probs}


def create_mass_matrix_sparse(masses, SPECT_BIN_N, mass_intensities):
    BATCH_N, POSSIBLE_FORMULAE_N = masses.shape

    row_idx_offset = torch.arange(BATCH_N, device=masses.device)\
                     .unsqueeze(1).repeat(1, POSSIBLE_FORMULAE_N).flatten()*SPECT_BIN_N
    row_idx_offset = row_idx_offset.to(masses.device)
    row_idx = row_idx_offset + masses.flatten()

    col_idx = torch.arange(BATCH_N *POSSIBLE_FORMULAE_N, device=masses.device ).to(masses.device)
    idx = torch.stack([row_idx, col_idx], -1).T
    
    mat_val_oh = mass_intensities.flatten()
    
    #mat_val_oh_nz = (mat_val_oh > 1e-4)
    
    sparse_mat = torch.sparse_coo_tensor(idx, mat_val_oh,
                                         #idx[:, mat_val_oh_nz], mat_val_oh[mat_val_oh_nz], 
                                         (SPECT_BIN_N * BATCH_N , 
                                          BATCH_N* POSSIBLE_FORMULAE_N, ), device=masses.device)
    
    return sparse_mat


def mat_matrix_sparse_mm(sparse_matrix, formulae_probs):
    BATCH_N, _ = formulae_probs.shape
    f_flat =formulae_probs.flatten()
    #a = torch.mm(sparse_matrix, f_flat.unsqueeze(1))
    a = torch.sparse.mm(sparse_matrix, f_flat.unsqueeze(1))
    sparse_spect_out = a.reshape(BATCH_N, -1)
    return sparse_spect_out
    

def construct_sparse_mm_from_peak_info(peaks, spect_bin_n=512):
    assert peaks.shape[-1] == 2
    BATCH_N, SUBSET_N, PEAK_N, _ = peaks.shape
    sparse_mass_matrices = [
        create_mass_matrix_sparse(
            peaks[:, :, i, 0].round().long(),
            spect_bin_n,
            peaks[:, :, i, 1]
        ) for i in range(PEAK_N)
    ]

    sparse_mass_matrix = sparse_mass_matrices[0]
    for m in sparse_mass_matrices[1:]:
        sparse_mass_matrix += m
    
    return sparse_mass_matrix


class MolAttentionMultiTransform(nn.Module):
    """
    Per-vertex features to sparse points of peak, mass

    Input:
    BATCH_N x ATOM_N x F

    Output:
    BATCH_N x SPECT_BIN

    """
    def __init__(self, g_feat_in,
                 embedding_key_size = 16,
                 formula_oh_sizes = [20, 20, 20, 20, 20],
                 formula_oh_accum = True,
                 internal_d = 512,
                 attn_transform = 'softmax',
                 formulae_prob_transform = 'softmax',
                 spect_bin_n = 512):

        
        super( MolAttentionMultiTransform, self).__init__()

        self.norm = nn.LayerNorm(g_feat_in)

        self.embed_g_feat = nn.Linear(g_feat_in, embedding_key_size)
        formula_encoding_n = np.sum(formula_oh_sizes)

        self.formula_to_oh = StructuredOneHot(formula_oh_sizes, formula_oh_accum)
        
        self.embed_formulae_feat = nn.Linear(formula_encoding_n, embedding_key_size)
        
        self.norm2 = nn.LayerNorm(internal_d)

        self.combine_norm = nn.LayerNorm(formula_encoding_n +  g_feat_in)
        self.f_combine_l1 = nn.Linear(formula_encoding_n +  g_feat_in, internal_d)
        self.f_combine_l2 = nn.Linear(internal_d, internal_d)
        self.f_combine_score = nn.Linear(internal_d, 1)

        
        self.spect_bin_n = spect_bin_n

        if formulae_prob_transform == 'softmax':
            self.formulae_prob_transform = nn.Softmax(dim=-1)
        elif formulae_prob_transform == 'sigsoftmax':
            self.formulae_prob_transform = SigSoftmax(dim=-1)
        elif formulae_prob_transform == 'sigmoid':
            self.formulae_prob_transform = nn.Sigmoid()

        if attn_transform == 'softmax':
            self.attn_transform = nn.Softmax(dim=1)
        elif attn_transform == 'sigsoftmax':
            self.attn_transform = SigSoftmax(dim=1)
        elif attn_transform == 'sigmoid':
            self.attn_transform = nn.Sigmoid()




    def forward(self, vert_feat_in, vert_mask_in,
                possible_formulae, formulae_masses, vert_element_oh):
        """
        vert_feat_in : BATCH_N x ATOM_N x F
        vert_feat_mask : BATCH_N x ATOM_N  
        possible_formulae : BATCH_N x MAX_FORMULAE_N x FORMULA_ENCODING_N
        formula_masses : BATCH_N x MAX_FORMULAE_N x NUM_MASSES x 2  (mass, peak intensity) 
        """

        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1)

        # one-hot encode possible formulae
        possible_formulae_flat = possible_formulae.reshape(-1, possible_formulae.shape[-1])
        possible_formulae_flat_oh = self.formula_to_oh(possible_formulae_flat)
        possible_formulae = possible_formulae_flat_oh.reshape(possible_formulae.shape[0],
                                                              possible_formulae.shape[1], -1)
        
        
        # encoded inputs
        vert_encoded = self.embed_g_feat(vert_feat_in)
        formulae_encoded = self.embed_formulae_feat(possible_formulae)

        dot_prod = (formulae_encoded.unsqueeze(1) * vert_encoded.unsqueeze(2)).sum(dim=-1)
        weighting = self.attn_transform(dot_prod)
        
        vert_att = masked_vert_feat.unsqueeze(3) * weighting.unsqueeze(2)
        vert_att_reduce = vert_att.sum(dim=1).permute(0, 2, 1) # (BATCH_N, FORMULAE_N, F)
        combined = torch.cat([possible_formulae, vert_att_reduce], -1)
        combined = self.combine_norm(combined)
        x = F.relu(self.f_combine_l1(combined))
        x = F.relu(self.f_combine_l2(x))
        x = self.norm2(x)
        formulae_scores = self.f_combine_score(x).squeeze(2)
        
        formulae_probs = self.formulae_prob_transform(formulae_scores)

        #t1 = time.time()
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

        return spect_out





class SigSoftmax(nn.Module):
    """
    SigSoftmax from the paper - https://arxiv.org/pdf/1805.10829.pdf

    """
    def __init__(self, dim = 0):
        
        super( SigSoftmax, self).__init__()
        self.dim = dim


    def forward(self, logits):
        
        max_values = torch.max(logits, self.dim, keepdim = True)[0]
        exp_logits_sigmoided = torch.exp(logits - max_values) * torch.sigmoid(logits)
        sum_exp_logits_sigmoided = exp_logits_sigmoided.sum(self.dim, keepdim = True)
        return exp_logits_sigmoided / sum_exp_logits_sigmoided

    

class MolAttentionGRU(nn.Module):
    """
    Per-vertex features to sparse points of peak, mass

    Input:
    BATCH_N x ATOM_N x F

    Output:
    BATCH_N x SPECT_BIN

    """
    def __init__(self, g_feat_in,
                 spect_bin, 
                 embedding_key_size = 16,
                 formula_oh_sizes = [20, 20, 20, 20, 20],
                 formula_oh_accum = True,
                 formula_oh_normalize=False, 
                 
                 internal_d = 512,
                 prob_softmax = True,
                 g_embed_train = True,
                 g_embed_bias = True, 
                 formula_embed_train= True,
                 formula_embed_bias=True,
                 gru_layer_n = 1,
                 linear_layer_n = 2,
    ):

        
        super( MolAttentionGRU, self).__init__()

        self.spect_bin = spect_bin
        
        self.norm = nn.LayerNorm(g_feat_in)

        self.embed_g_feat = nn.Linear(g_feat_in, embedding_key_size, bias= g_embed_bias)
        formula_encoding_n = np.sum(formula_oh_sizes)

        self.formula_to_oh = StructuredOneHot(formula_oh_sizes, formula_oh_accum)
        
        self.embed_formulae_feat = nn.Linear(formula_encoding_n, embedding_key_size,
                                             bias=formula_embed_bias)
        self.formula_oh_normalize=formula_oh_normalize

        if not formula_embed_train:
            self.embed_formulae_feat.weight.requires_grad = False
            self.embed_formulae_feat.bias.requires_grad = False
            
        
        self.norm2 = nn.LayerNorm(internal_d)

        self.combine_layers = nn.ModuleList([nn.GRUCell(formula_encoding_n, g_feat_in) for _ in range(gru_layer_n)])
        
        self.f_combine_l1 = nn.Linear(g_feat_in, internal_d)
        self.f_combine_l2 = nn.Sequential(*[nn.Sequential(nn.Linear(internal_d, internal_d),
                                                          nn.ReLU()) for _ in range(linear_layer_n)])
        
        self.f_combine_score = nn.Linear(internal_d, 1)

        
        self.prob_softmax = prob_softmax


    def forward(self, vert_feat_in, vert_mask_in,
                possible_formulae,
                formulae_peaks_mass_idx,
                formulae_peaks_intensity, 
                vert_element_oh, adj_oh, formula_frag_count):
        """
        vert_feat_in : BATCH_N x ATOM_N x F
            computed global embedding for molecule
        vert_feat_mask : BATCH_N x ATOM_N  
            input mask for valid atoms

        possible_formulae : BATCH_N x MAX_FORMULAE_N x FORMULA_ENCODING_N
            'formulae_features'
        formulae_peaks_mass_idx : BATCH_N x MAX_FORMULAE_N x NUM_MASSES 
         formulae_peaks_intensity, 
        
        
        vert_element_oh: BATCH_N x MAX_ATOM_N=32 x MAX_ELEMENT_N=8
            one-hot encoding of which element corresponds to which vertex
        adj_oh: BATCH_N x N_CHANNELS x MAX_ATOM_N x MAX_ATOM_N
            one-hot adjacency matrix
        formula_frag_count:
            ?
        """

        BATCH_N = vert_feat_in.shape[0]
        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1)

        # one-hot encode possible formulae
        possible_formulae_flat = possible_formulae.reshape(-1, possible_formulae.shape[-1])
        possible_formulae_flat_oh = self.formula_to_oh(possible_formulae_flat)
        possible_formulae = possible_formulae_flat_oh.reshape(possible_formulae.shape[0],
                                                              possible_formulae.shape[1], -1)
        
        if self.formula_oh_normalize:
            possible_formulae = possible_formulae / possible_formulae.sum(axis=2).unsqueeze(2)
            
        # encoded inputs
        vert_encoded = self.embed_g_feat(vert_feat_in)
        formulae_encoded = self.embed_formulae_feat(possible_formulae)

        dot_prod = (formulae_encoded.unsqueeze(1) * vert_encoded.unsqueeze(2)).sum(dim=-1)
        weighting = torch.softmax(dot_prod, dim=1)
        
        vert_att = masked_vert_feat.unsqueeze(3) * weighting.unsqueeze(2)
        vert_att_reduce = vert_att.sum(dim=1).permute(0, 2, 1) # (BATCH_N, FORMULAE_N, F)

        x = vert_att_reduce

        for l in self.combine_layers:
            x = l(possible_formulae.reshape(-1, possible_formulae.shape[-1]),
                  x.reshape(-1, x.shape[-1]))\
                  .reshape(x.shape)
        x = F.relu(self.f_combine_l1(x))
        x = F.relu(self.f_combine_l2(x))
        x = self.norm2(x)
        formulae_scores = self.f_combine_score(x).squeeze(2)
        
        if self.prob_softmax:
            formulae_probs = torch.softmax(formulae_scores, dim=-1)
        else:
            formulae_probs = torch.sigmoid(formulae_scores)

        #t1 = time.time()

        # per batch
        out_y = []
        for batch_i in range(BATCH_N):
            sparse_mat_matrix = peak_indices_intensities_to_sparse_matrix(formulae_peaks_mass_idx[batch_i],
                                                                               formulae_peaks_intensity[batch_i], 
                                                                               self.spect_bin.get_num_bins())
            y = sparse_mat_matrix @ formulae_probs[batch_i]
            out_y.append(y)
        spect_out = torch.stack(out_y)
                
        # sparse_mass_matrices = [create_mass_matrix_sparse(formulae_masses[:, :, i, 0].round().long(),
        #                                            self.spect_bin_n,
        #                                            formulae_masses[:, :, i, 1]
        # )\
        #                  .to(vert_feat_in.device) for i in range(formulae_masses.shape[2])]

        
        # sparse_mass_matrix = sparse_mass_matrices[0]
        # for m in sparse_mass_matrices[1:]:
        #     sparse_mass_matrix += m

        # spect_out = mat_matrix_sparse_mm(sparse_mass_matrix, formulae_probs)
        # torch.cuda.synchronize()
        # t2 = time.time()
        # print(f"end of net took {(t2-t1)*1000:3.2f}ms ")

        return {'spect_out': spect_out,
                'formulae_probs' : formulae_probs}


class MolAttentionGRUNewSparse(nn.Module):
    """
    Per-vertex features to sparse points of peak, mass

    Input:
    BATCH_N x ATOM_N x F

    Output:
    BATCH_N x SPECT_BIN

    """
    def __init__(self, g_feat_in,
                 spect_bin, 
                 embedding_key_size = 16,
                 formula_oh_sizes = [20, 20, 20, 20, 20],
                 formula_oh_accum = True,
                 formula_oh_normalize=False, 
                 
                 internal_d = 512,
                 prob_softmax = True,
                 g_embed_train = True,
                 g_embed_bias = True, 
                 formula_embed_train= True,
                 formula_embed_bias=True,
                 gru_layer_n = 1,
                 linear_layer_n = 2,
    ):
        super(MolAttentionGRUNewSparse, self).__init__()

        self.spect_bin = spect_bin
        
        self.norm = nn.LayerNorm(g_feat_in)

        self.embed_g_feat = nn.Linear(g_feat_in, embedding_key_size, bias= g_embed_bias)
        formula_encoding_n = np.sum(formula_oh_sizes)

        self.formula_to_oh = StructuredOneHot(formula_oh_sizes, formula_oh_accum)
        
        self.embed_formulae_feat = nn.Linear(formula_encoding_n, embedding_key_size,
                                             bias=formula_embed_bias)
        self.formula_oh_normalize=formula_oh_normalize

        if not formula_embed_train:
            self.embed_formulae_feat.weight.requires_grad = False
            self.embed_formulae_feat.bias.requires_grad = False
            
        
        self.norm2 = nn.LayerNorm(internal_d)

        self.combine_layers = nn.ModuleList([nn.GRUCell(formula_encoding_n, g_feat_in) for _ in range(gru_layer_n)])
        
        self.f_combine_l1 = nn.Linear(g_feat_in, internal_d)
        self.f_combine_l2 = nn.Sequential(*[nn.Sequential(nn.Linear(internal_d, internal_d),
                                                          nn.ReLU()) for _ in range(linear_layer_n)])
        
        self.f_combine_score = nn.Linear(internal_d, 1)

        
        self.prob_softmax = prob_softmax


    def forward(self, vert_feat_in, vert_mask_in,
                possible_formulae,
                formulae_peaks_mass_idx,
                formulae_peaks_intensity, 
                vert_element_oh, adj_oh, formula_frag_count):
        """
        vert_feat_in : BATCH_N x ATOM_N x F
            computed global embedding for molecule
        vert_feat_mask : BATCH_N x ATOM_N  
            input mask for valid atoms

        possible_formulae : BATCH_N x MAX_FORMULAE_N x FORMULA_ENCODING_N
            'formulae_features'
        formulae_peaks_mass_idx : BATCH_N x MAX_FORMULAE_N x NUM_MASSES 
         formulae_peaks_intensity, 
        
        
        vert_element_oh: BATCH_N x MAX_ATOM_N=32 x MAX_ELEMENT_N=8
            one-hot encoding of which element corresponds to which vertex
        adj_oh: BATCH_N x N_CHANNELS x MAX_ATOM_N x MAX_ATOM_N
            one-hot adjacency matrix
        formula_frag_count:
            ?
        """

        BATCH_N = vert_feat_in.shape[0]
        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1)

        # one-hot encode possible formulae
        possible_formulae_flat = possible_formulae.reshape(-1, possible_formulae.shape[-1])
        possible_formulae_flat_oh = self.formula_to_oh(possible_formulae_flat)
        possible_formulae = possible_formulae_flat_oh.reshape(possible_formulae.shape[0],
                                                              possible_formulae.shape[1], -1)
        
        if self.formula_oh_normalize:
            possible_formulae = possible_formulae / possible_formulae.sum(axis=2).unsqueeze(2)
            
        # encoded inputs
        vert_encoded = self.embed_g_feat(vert_feat_in)
        formulae_encoded = self.embed_formulae_feat(possible_formulae)

        dot_prod = (formulae_encoded.unsqueeze(1) * vert_encoded.unsqueeze(2)).sum(dim=-1)
        weighting = torch.softmax(dot_prod, dim=1)
        
        vert_att = masked_vert_feat.unsqueeze(3) * weighting.unsqueeze(2)
        vert_att_reduce = vert_att.sum(dim=1).permute(0, 2, 1) # (BATCH_N, FORMULAE_N, F)

        x = vert_att_reduce

        for l in self.combine_layers:
            x = l(possible_formulae.reshape(-1, possible_formulae.shape[-1]),
                  x.reshape(-1, x.shape[-1]))\
                  .reshape(x.shape)
        x = F.relu(self.f_combine_l1(x))
        x = F.relu(self.f_combine_l2(x))
        x = self.norm2(x)
        formulae_scores = self.f_combine_score(x).squeeze(2)
        
        if self.prob_softmax:
            formulae_probs = torch.softmax(formulae_scores, dim=-1)
        else:
            formulae_probs = torch.sigmoid(formulae_scores)

        #t1 = time.time()

        # per batch
        out_y = []
        for batch_i in range(BATCH_N):
            sparse_mat_matrix = peak_indices_intensities_to_sparse_matrix(formulae_peaks_mass_idx[batch_i],
                                                                               formulae_peaks_intensity[batch_i], 
                                                                               self.spect_bin.get_num_bins())
            y = sparse_mat_matrix @ formulae_probs[batch_i]
            out_y.append(y)
        spect_out = torch.stack(out_y)

        # sparse_mass_matrices = [create_mass_matrix_sparse(formulae_masses[:, :, i, 0].round().long(),
        #                                            self.spect_bin_n,
        #                                            formulae_masses[:, :, i, 1]
        # )\
        #                  .to(vert_feat_in.device) for i in range(formulae_masses.shape[2])]

        
        # sparse_mass_matrix = sparse_mass_matrices[0]
        # for m in sparse_mass_matrices[1:]:
        #     sparse_mass_matrix += m

        # spect_out = mat_matrix_sparse_mm(sparse_mass_matrix, formulae_probs)
        # torch.cuda.synchronize()
        # t2 = time.time()
        # print(f"end of net took {(t2-t1)*1000:3.2f}ms ")

        return {'spect_out': spect_out,
                'formulae_probs' : formulae_probs}
    

class MolAttentionNetLowRank(nn.Module):
    """
    Per-vertex features to sparse points of peak, mass

    Input:
    BATCH_N x ATOM_N x F

    Output:
    BATCH_N x SPECT_BIN

    """
    def __init__(self, g_feat_in,
                 embedding_key_size = 16,
                 formula_oh_sizes = [20, 20, 20, 20, 20],
                 formula_oh_accum = True,
                 internal_d = 512,
                 prob_softmax = True,
                 g_embed_train = True,
                 g_embed_bias = True, 
                 formula_embed_train= True,
                 formula_embed_bias=True,
                 mixture_weight_method = 'method1', 
                 rank_n = 1, 
                 spect_bin_n = 512):

        
        super( MolAttentionNetLowRank, self).__init__()

        self.norm = nn.LayerNorm(g_feat_in)

        self.embed_g_feat = nn.Linear(g_feat_in, embedding_key_size, bias= g_embed_bias)
        formula_encoding_n = np.sum(formula_oh_sizes)

        self.formula_to_oh = StructuredOneHot(formula_oh_sizes, formula_oh_accum)
        
        self.embed_formulae_feat = nn.Linear(formula_encoding_n, embedding_key_size,
                                             bias=formula_embed_bias)

        if not formula_embed_train:
            self.embed_formulae_feat.weight.requires_grad = False
            self.embed_formulae_feat.bias.requires_grad = False
            

        
        self.out_encoding = nn.ModuleList([nn.Sequential(nn.LayerNorm(formula_encoding_n +  g_feat_in),
                                                         nn.Linear(formula_encoding_n +  g_feat_in, internal_d),
                                                         nn.ReLU(),
                                                         nn.Linear(internal_d, internal_d),
                                                         nn.ReLU(), 
                                                         nn.LayerNorm(internal_d)              ,
                                                         nn.Linear(internal_d, 1)) for _ in range(rank_n)])
        
        self.spect_bin_n = spect_bin_n
        self.prob_softmax = prob_softmax

        self.mixture_weight_method = mixture_weight_method
        if mixture_weight_method in ['method1', 'method2', 'method3', 'method4']:
            self.mixture_norm = nn.LayerNorm(g_feat_in)
            self.mixture_weights_l1 = nn.Linear(g_feat_in, g_feat_in)
            self.mixture_weights_l2 = nn.Linear(g_feat_in, rank_n)


    def forward(self, vert_feat_in, vert_mask_in,
                possible_formulae, formulae_masses,
                vert_element_oh, adj_oh):
        """
        vert_feat_in : BATCH_N x ATOM_N x F
        vert_feat_mask : BATCH_N x ATOM_N  
        possible_formulae : BATCH_N x MAX_FORMULAE_N x FORMULA_ENCODING_N
        formula_masses : BATCH_N x MAX_FORMULAE_N x NUM_MASSES x 2  (mass, peak intensity) 
        """

        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1)
        atom_counts = vert_mask_in.sum(dim=1)
        

        # one-hot encode possible formulae
        possible_formulae_flat = possible_formulae.reshape(-1, possible_formulae.shape[-1])
        possible_formulae_flat_oh = self.formula_to_oh(possible_formulae_flat)
        possible_formulae = possible_formulae_flat_oh.reshape(possible_formulae.shape[0],
                                                              possible_formulae.shape[1], -1)
        
        
        # encoded inputs
        vert_encoded = self.embed_g_feat(vert_feat_in)
        formulae_encoded = self.embed_formulae_feat(possible_formulae)

        dot_prod = (formulae_encoded.unsqueeze(1) * vert_encoded.unsqueeze(2)).sum(dim=-1)
        weighting = torch.softmax(dot_prod, dim=1)
        
        vert_att = masked_vert_feat.unsqueeze(3) * weighting.unsqueeze(2)
        vert_att_reduce = vert_att.sum(dim=1).permute(0, 2, 1) # (BATCH_N, FORMULAE_N, F)
        combined = torch.cat([possible_formulae, vert_att_reduce], -1)
        formulae_scores = torch.cat([l(combined) for l in self.out_encoding], -1)
        
        # combined = self.combine_norm(combined)
        # x = F.relu(self.f_combine_l1(combined))
        # x = F.relu(self.f_combine_l2(x))
        # x = self.norm2(x)
        # formulae_scores = self.f_combine_score(x).squeeze(2)

        if self.mixture_weight_method == 'method1':
            mixture_weights = torch.softmax(self.mixture_weights_l1(self.mixture_norm(masked_vert_feat).mean(dim=1)), dim=1)
        elif self.mixture_weight_method == 'method2':
            # vertex feats use max 
            mw = self.mixture_weights_l1(goodmax(self.mixture_norm(masked_vert_feat), dim=1))
            mixture_weights = torch.softmax(mw, dim=1)
        elif self.mixture_weight_method == 'method3':
            # proper weighting of vertex feats?
            mixture_weights = torch.softmax(self.mixture_weights_l1(self.mixture_norm(masked_vert_feat).sum(dim=1) / atom_counts.unsqueeze(-1)), dim=1)
        elif self.mixture_weight_method == 'method4':
            # proper weighting of vertex feats?
            mw = F.relu(self.mixture_weights_l1(masked_vert_feat)).sum(dim=1) / atom_counts.unsqueeze(-1)
            mw = self.mixture_norm(mw)
            mw = self.mixture_weights_l2(mw)
            mixture_weights = torch.softmax(mw, dim=1)

            
        #print("mixture_weights.shape=", mixture_weights.shape, "formulae_scores.shape=", formulae_scores.shape)
        #print("unsqueeze", mixture_weights.unsqueeze(1).shape)
        
        #mixture_weights_expand =  mixture_weights.unsqueeze(1).expand(-1, formulae_scores.shape[1], -1)
        
        formulae_probs = (torch.softmax(formulae_scores, dim=1) * mixture_weights.unsqueeze(1)).sum(dim=2)
        

        #t1 = time.time()
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

        return spect_out
    

class MolAttentionNetOuter(nn.Module):
    """
    Per-vertex features to sparse points of peak, mass

    Input:
    BATCH_N x ATOM_N x F

    Output:
    BATCH_N x SPECT_BIN

    """
    def __init__(self, g_feat_in,
                 embedding_key_size = 16,
                 formula_oh_sizes = [20, 20, 20, 20, 20],
                 formula_oh_accum = True,
                 internal_d = 512,
                 prob_softmax = True,
                 spect_bin_n = 512):

        
        super( MolAttentionNetOuter, self).__init__()

        self.norm = nn.LayerNorm(g_feat_in)

        self.embed_g_feat = nn.Linear(g_feat_in*2, embedding_key_size)
        formula_encoding_n = np.sum(formula_oh_sizes)

        self.formula_to_oh = StructuredOneHot(formula_oh_sizes, formula_oh_accum)
        self.possible_formulae_bn = nn.BatchNorm1d(formula_encoding_n)
        
        self.embed_formulae_feat = nn.Linear(formula_encoding_n, embedding_key_size)
        
        self.norm2 = nn.LayerNorm(internal_d)

        self.combine_norm = nn.LayerNorm(formula_encoding_n +  g_feat_in*2)
        self.f_combine_l1 = nn.Linear(formula_encoding_n +  g_feat_in*2, internal_d)
        self.f_combine_l2 = nn.Linear(internal_d, internal_d)
        self.f_combine_score = nn.Linear(internal_d, 1)

        
        self.spect_bin_n = spect_bin_n
        self.prob_softmax = prob_softmax
        
    def forward(self, vert_feat_in, vert_mask_in, 
                possible_formulae, formulae_masses,
                vert_element_oh, adj_oh):
        """
        vert_feat_in : BATCH_N x ATOM_N x F
        vert_feat_mask : BATCH_N x ATOM_N  
        possible_formulae : BATCH_N x MAX_FORMULAE_N x FORMULA_ENCODING_N
        formula_masses : BATCH_N x MAX_FORMULAE_N x NUM_MASSES x 2  (mass, peak intensity) 
        """

        BATCH_N, ATOM_N, VERT_F = vert_feat_in.shape
        _, MAX_FORMULAE_N, FORMULA_ENCODING_N = possible_formulae.shape
        
        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1)

        # one-hot encode possible formulae
        possible_formulae_flat = possible_formulae.reshape(-1, possible_formulae.shape[-1])
        possible_formulae_flat_oh = self.formula_to_oh(possible_formulae_flat)
        possible_formulae_flat_oh = self.possible_formulae_bn(possible_formulae_flat_oh)
        possible_formulae = possible_formulae_flat_oh.reshape(possible_formulae.shape[0],
                                                              possible_formulae.shape[1], -1)
        

        vert_feat_pairwise = torch.cat([masked_vert_feat.unsqueeze(1).expand(-1, ATOM_N, -1, -1), 
                                           masked_vert_feat.unsqueeze(2).expand(-1, -1, ATOM_N,  -1), ], -1)

        assert adj_oh.shape == (BATCH_N, 4, ATOM_N, ATOM_N)
        
        adj_oh_collapse = goodmax(adj_oh, dim=1)
        
        
        # encoded inputs
        vert_encoded_pairwise = self.embed_g_feat(vert_feat_pairwise) ## BATCH_N x ATOM_N x F
        
        formulae_encoded = self.embed_formulae_feat(possible_formulae) ## BATCH_N x MAX_FORMULA_N x F
        
        dot_prod = (formulae_encoded.unsqueeze(1).unsqueeze(1) * vert_encoded_pairwise.unsqueeze(3)).sum(dim=-1)
        dot_prod = dot_prod 
        

        dot_prod_flat = dot_prod.reshape(BATCH_N, ATOM_N * ATOM_N, MAX_FORMULAE_N)
        
        weighting_flat = torch.softmax(dot_prod_flat, dim=1)
        weighting = weighting_flat.reshape(BATCH_N, ATOM_N, ATOM_N, MAX_FORMULAE_N)
        

        vert_feat_pairwise = vert_feat_pairwise * adj_oh_collapse.unsqueeze(-1)
        #print('vert_feat_pairwise.shape=', vert_feat_pairwise.shape)

        vert_att_reduce = torch.einsum('nijf,nijd->ndf', vert_feat_pairwise, weighting)
        
        combined = torch.cat([possible_formulae, vert_att_reduce], -1)
        combined = self.combine_norm(combined)
        x = F.relu(self.f_combine_l1(combined))
        x = F.relu(self.f_combine_l2(x))
        x = self.norm2(x)
        formulae_scores = self.f_combine_score(x).squeeze(2)
        
        if self.prob_softmax:
            formulae_probs = torch.softmax(formulae_scores, dim=-1)
        else:
            formulae_probs = torch.sigmoid(formulae_scores)

        mass_matrices = [create_mass_matrix_oh(formulae_masses[:, :, i, 0].round().long(),
                                               self.spect_bin_n,
                                               formulae_masses[:, :, i, 1]
        )\
                         .to(vert_feat_in.device) for i in range(formulae_masses.shape[2])]
        mass_matrix = torch.sum(torch.stack(mass_matrices, -1), -1)


        spect_out = torch.einsum("ij,ijk->ik", formulae_probs, mass_matrix)
        return spect_out


    

class MolLesion(nn.Module):
    """
    Per-vertex features to sparse points of peak, mass

    Input:
    BATCH_N x ATOM_N x F

    Output:
    BATCH_N x SPECT_BIN

    """
    def __init__(self, g_feat_in,
                 embedding_key_size = 16,
                 formula_oh_sizes = [20, 20, 20, 20, 20],
                 formula_oh_accum = True,
                 internal_d = 512,
                 prob_softmax = True,
                 g_embed_train = True,
                 g_embed_bias = True, 
                 formula_embed_train= True,
                 formula_embed_bias=True,
                 norm = 'layer', 
                 vert_reduce_method = 'mean-naive', 
                 spect_bin_n = 512):

        
        super( MolLesion, self).__init__()

        if norm == 'layer':
            norm_class = nn.LayerNorm
        elif norm == 'batch':
            norm_class = BatchNorm1d
        self.norm = norm_class(g_feat_in)

        self.embed_g_feat = nn.Linear(g_feat_in, embedding_key_size, bias= g_embed_bias)
        formula_encoding_n = np.sum(formula_oh_sizes)

        self.formula_to_oh = StructuredOneHot(formula_oh_sizes, formula_oh_accum)
        
        self.embed_formulae_feat = nn.Linear(formula_encoding_n, embedding_key_size,
                                             bias=formula_embed_bias)

        # self.formulae_norm = nn.LayerNorm(formula_encoding_n)
        # self.formulae_collapse_l = nn.Linear(formula_encoding_n,formula_encoding_n)
        
        
        if not formula_embed_train:
            self.embed_formulae_feat.weight.requires_grad = False
            self.embed_formulae_feat.bias.requires_grad = False
            
        
        self.norm2 = norm_class(internal_d)

        self.combine_norm = norm_class(formula_encoding_n +  g_feat_in)
        self.f_combine_l1 = nn.Linear(formula_encoding_n +  g_feat_in, internal_d)
        self.f_combine_l2 = nn.Linear(internal_d, internal_d)
        self.f_combine_score = nn.Linear(internal_d, 1)

        
        self.spect_bin_n = spect_bin_n
        self.prob_softmax = prob_softmax

        self.vert_reduce_method = vert_reduce_method

    def forward(self, vert_feat_in, vert_mask_in,
                possible_formulae, formulae_masses,
                vert_element_oh, adj_oh,  formula_frag_count):
        """
        vert_feat_in : BATCH_N x ATOM_N x F
        vert_feat_mask : BATCH_N x ATOM_N  
        possible_formulae : BATCH_N x MAX_FORMULAE_N x FORMULA_ENCODING_N
        formula_masses : BATCH_N x MAX_FORMULAE_N x NUM_MASSES x 2  (mass, peak intensity) 
        """

        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1)

        _, MAX_FORMULAE_N, FORMULA_ENCODING_N = possible_formulae.shape
        # one-hot encode possible formulae
        possible_formulae_flat = possible_formulae.reshape(-1, possible_formulae.shape[-1])
        possible_formulae_flat_oh = self.formula_to_oh(possible_formulae_flat)
        possible_formulae = possible_formulae_flat_oh.reshape(possible_formulae.shape[0],
                                                              possible_formulae.shape[1], -1)
        
        possible_formulae_normed = possible_formulae
        formulae_num_atoms = possible_formulae.sum(dim=-1)

        # f_n = self.formulae_norm(F.relu(self.formulae_collapse_l(possible_formulae_normed )).sum(dim=1))
        # num_formulae =  (formulae_num_atoms > 0).float().sum(dim=1)

        # f_n = f_n / num_formulae.unsqueeze(-1) 
        
        # encoded inputs
        # vert_encoded = self.embed_g_feat(vert_feat_in)
        # formulae_encoded = self.embed_formulae_feat(possible_formulae)

        # dot_prod = (formulae_encoded.unsqueeze(1) * vert_encoded.unsqueeze(2)).sum(dim=-1)
        # weighting = torch.softmax(dot_prod, dim=1)
        
        #vert_att = masked_vert_feat.unsqueeze(3) * weighting.unsqueeze(2)
        # vert_att_reduce = vert_att.sum(dim=1).permute(0, 2, 1) # (BATCH_N, FORMULAE_N, F)
        if self.vert_reduce_method == 'mean-naive':
            
            vert_att_reduce = masked_vert_feat.mean(dim=1).unsqueeze(1).expand(-1, MAX_FORMULAE_N, -1)
        elif self.vert_reduce_method == 'sum-naive':
            
            vert_att_reduce = masked_vert_feat.sum(dim=1).unsqueeze(1).expand(-1, MAX_FORMULAE_N, -1)
        elif self.vert_reduce_method == 'max':
            
            vert_att_reduce = goodmax(masked_vert_feat, dim=1).unsqueeze(1).expand(-1, MAX_FORMULAE_N, -1)
        elif self.vert_reduce_method == 'mean-masked':
            
            vert_att_reduce = masked_vert_feat.sum(dim=1)
            num_vertices = vert_mask_in.sum(dim=1)
            vert_att_reduce = vert_att_reduce / num_vertices.unsqueeze(-1)

            vert_att_reduce = vert_att_reduce.unsqueeze(1).expand(-1, MAX_FORMULAE_N, -1)
        
        combined = torch.cat([possible_formulae, vert_att_reduce], -1)
        combined = self.combine_norm(combined)
        x1 = F.relu(self.f_combine_l1(combined))
        x2 = F.relu(self.f_combine_l2(x1))
        #x = self.norm2(x2) + x1
        x = self.norm2(x2)
        formulae_scores = self.f_combine_score(x).squeeze(2)
        
        if self.prob_softmax:
            formulae_probs = torch.softmax(formulae_scores, dim=-1)
        else:
            formulae_probs = torch.sigmoid(formulae_scores)

        #t1 = time.time()
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


    
class MolCombineFormulaPreReduce(nn.Module):
    """
    Per-vertex features to sparse points of peak, mass

    Input:
    BATCH_N x ATOM_N x F

    Output:
    BATCH_N x SPECT_BIN

    """
    def __init__(self, g_feat_in,
                 embedding_key_size = 16,
                 formula_oh_sizes = [20, 20, 20, 20, 20],
                 formula_oh_accum = True,
                 internal_d = 512,
                 prob_softmax = True,
                 g_embed_train = True,
                 g_embed_bias = True, 
                 formula_embed_train= True,
                 formula_embed_bias=True, 
                 spect_bin_n = 512):

        
        super( MolCombineFormulaPreReduce, self).__init__()

        self.norm = nn.LayerNorm(g_feat_in)

        self.embed_g_feat = nn.Linear(g_feat_in, embedding_key_size, bias= g_embed_bias)
        
        
        formula_encoding_n = np.sum(formula_oh_sizes)

        self.formula_to_oh = StructuredOneHot(formula_oh_sizes, formula_oh_accum)
        
        
        self.norm2 = nn.LayerNorm(internal_d)

        self.combine_norm = nn.LayerNorm(internal_d)
        self.f_combine_l1 = nn.Linear(internal_d, internal_d)
        self.f_combine_l2 = nn.Linear(internal_d, internal_d)
        self.f_combine_score = nn.Linear(internal_d, 1)

        self.embed_formulae_feat = nn.Linear(formula_encoding_n, embedding_key_size)

        self.vert_f_combine_l1 = nn.Linear(formula_encoding_n + g_feat_in, internal_d)
        self.vert_f_combine_l2 = nn.Linear(internal_d, internal_d)
        
                                           
        self.spect_bin_n = spect_bin_n
        self.prob_softmax = prob_softmax


    def forward(self, vert_feat_in, vert_mask_in,
                possible_formulae, formulae_masses,
                vert_element_oh, adj_oh):
        """
        vert_feat_in : BATCH_N x ATOM_N x F
        vert_feat_mask : BATCH_N x ATOM_N  
        possible_formulae : BATCH_N x MAX_FORMULAE_N x FORMULA_ENCODING_N
        formula_masses : BATCH_N x MAX_FORMULAE_N x NUM_MASSES x 2  (mass, peak intensity) 
        """

        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1)
        BATCH_N, ATOM_N, VERT_F = vert_feat_in.shape
        
        _, MAX_FORMULAE_N, FORMULA_ENCODING_N = possible_formulae.shape
        # one-hot encode possible formulae
        possible_formulae_flat = possible_formulae.reshape(-1, possible_formulae.shape[-1])
        possible_formulae_flat_oh = self.formula_to_oh(possible_formulae_flat)
        possible_formulae = possible_formulae_flat_oh.reshape(possible_formulae.shape[0],
                                                              possible_formulae.shape[1], -1) # BATCH_N, MAX_FORMULAE_N, FORMULAE_ENCODING
        


        vert_expand = masked_vert_feat.unsqueeze(2).expand(-1, -1, MAX_FORMULAE_N, -1)
        form_expand = possible_formulae.unsqueeze(1).expand(-1, ATOM_N, -1,-1)
        vert_form_cat = torch.cat([vert_expand, form_expand], -1)
        vf1 = F.relu(self.vert_f_combine_l1(vert_form_cat))
        vf2 = F.tanh(self.vert_f_combine_l2(vf1))
        
        

        combined = self.combine_norm(vf2.sum(dim=1))
        x = F.relu(self.f_combine_l1(combined))
        x = F.relu(self.f_combine_l2(x))
        x = self.norm2(x)
        formulae_scores = self.f_combine_score(x).squeeze(2)
        
        if self.prob_softmax:
            formulae_probs = torch.softmax(formulae_scores, dim=-1)
        else:
            formulae_probs = torch.sigmoid(formulae_scores)

        #t1 = time.time()
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

        return spect_out

    

class MolFormulaEltReduce(nn.Module):
    """
    Per-vertex features to sparse points of peak, mass

    Input:
    BATCH_N x ATOM_N x F

    Output:
    BATCH_N x SPECT_BIN

    """
    def __init__(self, g_feat_in,
                 embedding_key_size = 16,
                 formula_oh_sizes = [20, 20, 20, 20, 20],
                 formula_oh_accum = True,
                 internal_d = 512,
                 g_embed_train = True,
                 g_embed_bias = True, 
                 formula_embed_train= True,
                 formula_embed_bias=True, 
                 spect_bin_n = 512):

        
        super( MolFormulaEltReduce, self).__init__()

        self.norm = nn.LayerNorm(g_feat_in)

        #self.embed_g_feat = nn.Linear(g_feat_in, embedding_key_size, bias= g_embed_bias)
        formula_encoding_n = np.sum(formula_oh_sizes)

        self.formula_to_oh = StructuredOneHot(formula_oh_sizes, formula_oh_accum)
        
        # self.embed_formulae_feat = nn.Linear(formula_encoding_n, embedding_key_size,
        #                                      bias=formula_embed_bias)

        # if not formula_embed_train:
        #     self.embed_formulae_feat.weight.requires_grad = False
        #     self.embed_formulae_feat.bias.requires_grad = False
            

        self.vert_elt_lin = nn.Linear(g_feat_in, g_feat_in)

        self.norm2 = nn.LayerNorm(internal_d)

        self.embed_formulae = nn.Linear(formula_encoding_n, internal_d)
        
        self.combine_norm = nn.LayerNorm(g_feat_in + internal_d)
        self.f_l1 = nn.Linear(g_feat_in + internal_d, internal_d)
        self.f_l2 = nn.Linear(internal_d, internal_d)
        self.f_combine_score = nn.Linear(internal_d, 1)

        
        self.spect_bin_n = spect_bin_n


    def forward(self, vert_feat_in, vert_mask_in,
                possible_formulae, formulae_masses,
                vert_element_oh, adj_oh, formula_frag_count):
        """
        vert_feat_in : BATCH_N x ATOM_N x VERT_F
        vert_feat_mask : BATCH_N x ATOM_N  
        possible_formulae : BATCH_N x MAX_FORMULAE_N x FORMULA_ENCODING_N
        formula_masses : BATCH_N x MAX_FORMULAE_N x NUM_MASSES x 2  (mass, peak intensity) 
        vert_element_oh : BATCH_N x MAX_FORMULAE_N x ELEMENT_N 

        for this to work FORMULA_ENCODING_N == ELEMENT_N
        """

        BATCH_N, ATOM_N, VERT_F = vert_feat_in.shape

        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1)

        _, MAX_FORMULAE_N, FORMULA_ENCODING_N = possible_formulae.shape
        BATCH_N, _, ELEMENT_N = vert_element_oh.shape

        assert ELEMENT_N == FORMULA_ENCODING_N
        
        # one-hot encode possible formulae
        possible_formulae_flat = possible_formulae.reshape(-1, possible_formulae.shape[-1])
        possible_formulae_flat_oh = self.formula_to_oh(possible_formulae_flat)
        possible_formulae_accum = possible_formulae_flat_oh.reshape(possible_formulae.shape[0],
                                                              possible_formulae.shape[1], -1)


        vert_feat_contract_by_elt = torch.einsum('ijk,ije->iek', masked_vert_feat, vert_element_oh) # (BATCH_N, ELEMENT_N, F)
        vert_feat_contract_by_elt_mean = vert_feat_contract_by_elt / (vert_element_oh.sum(dim=1).unsqueeze(-1) + 0.001)
        #print(vert_feat_contract_by_elt_mean.shape)
        ## IDEA: Boolean of element counts
        ## IDEA: transforms inside of here
        vert_feat_contract_by_elt_mean = F.tanh(self.vert_elt_lin(vert_feat_contract_by_elt_mean))

        assert vert_feat_contract_by_elt_mean.shape == (BATCH_N, ELEMENT_N, VERT_F)

        ## IDEA: There are a lot of different ways of combining/weighting these things
        # now combine with formula
        formula_feat_contract = torch.einsum('ijk,ikl->ijl', possible_formulae.float(), vert_feat_contract_by_elt_mean)
        assert formula_feat_contract.shape == (BATCH_N, MAX_FORMULAE_N, VERT_F)
        formula_feat_contract = formula_feat_contract / (possible_formulae.float().sum(dim=2).unsqueeze(2) + 1)
        

        combined = self.combine_norm(torch.cat([formula_feat_contract, torch.tanh(self.embed_formulae(possible_formulae_accum))], -1))
        x = F.relu(self.f_l1(combined))
        x = F.relu(self.f_l2(x))
        x = self.norm2(x)
        formulae_scores = self.f_combine_score(x).squeeze(2)
        
        formulae_probs = torch.softmax(formulae_scores, dim=-1)


        #t1 = time.time()
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


    
    

class MolAttentionNetOHSparseHighway(nn.Module):
    """
    Per-vertex features to sparse points of peak, mass

    Input:
    BATCH_N x ATOM_N x F

    Output:
    BATCH_N x SPECT_BIN

    """
    def __init__(self, g_feat_in,
                 embedding_key_size = 16,
                 formula_oh_sizes = [20, 20, 20, 20, 20],
                 formula_oh_accum = True,
                 internal_d = 512,
                 prob_softmax = True,
                 g_embed_train = True,
                 g_embed_bias = True, 
                 formula_embed_train= True,
                 formula_embed_bias=True,
                 graph_layer_n= 4, 
                 spect_bin_n = 512):

        
        super( MolAttentionNetOHSparseHighway, self).__init__()

        g_feat_in = g_feat_in * graph_layer_n
        
        self.norm = nn.LayerNorm(g_feat_in)

        self.embed_g_feat = nn.Linear(g_feat_in, embedding_key_size, bias= g_embed_bias)
        formula_encoding_n = np.sum(formula_oh_sizes)

        self.formula_to_oh = StructuredOneHot(formula_oh_sizes, formula_oh_accum)
        
        self.embed_formulae_feat = nn.Linear(formula_encoding_n, embedding_key_size,
                                             bias=formula_embed_bias)

        if not formula_embed_train:
            self.embed_formulae_feat.weight.requires_grad = False
            self.embed_formulae_feat.bias.requires_grad = False
            
        
        self.norm2 = nn.LayerNorm(internal_d)

        self.combine_norm = nn.LayerNorm(formula_encoding_n +  g_feat_in)
        self.f_combine_l1 = nn.Linear(formula_encoding_n +  g_feat_in, internal_d)
        self.f_combine_l2 = nn.Linear(internal_d, internal_d)
        self.f_combine_score = nn.Linear(internal_d, 1)

        
        self.spect_bin_n = spect_bin_n
        self.prob_softmax = prob_softmax


    def forward(self, vert_feat_in, vert_mask_in,
                possible_formulae, formulae_masses,
                vert_element_oh, adj_oh):
        """
        vert_feat_in : BATCH_N x ATOM_N x F x VERT_LAYER_N
        vert_feat_mask : BATCH_N x ATOM_N  
        possible_formulae : BATCH_N x MAX_FORMULAE_N x FORMULA_ENCODING_N
        formula_masses : BATCH_N x MAX_FORMULAE_N x NUM_MASSES x 2  (mass, peak intensity) 
        """
        vert_feat_in = torch.sigmoid(vert_feat_in )
        
        BATCH_N, ATOM_N, VERT_F, VERT_LAYER_N = vert_feat_in.shape
        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1).unsqueeze(-1)        
        vert_feat_in_flat = vert_feat_in.reshape(BATCH_N, ATOM_N, -1)
        masked_vert_feat_flat = masked_vert_feat.reshape(BATCH_N, ATOM_N, -1)
        

        # one-hot encode possible formulae
        _, MAX_FORMULAE_N, FORMULAE_INPUT_INCODING_N = possible_formulae.shape
        possible_formulae_flat = possible_formulae.reshape(-1, possible_formulae.shape[-1])
        possible_formulae_flat_oh = self.formula_to_oh(possible_formulae_flat)
        possible_formulae = possible_formulae_flat_oh.reshape(possible_formulae.shape[0],
                                                              possible_formulae.shape[1], -1)
        
        
        # encoded inputs
        vert_encoded = self.embed_g_feat(vert_feat_in_flat)
        formulae_encoded = self.embed_formulae_feat(possible_formulae)

        dot_prod = (formulae_encoded.unsqueeze(1) * vert_encoded.unsqueeze(2)).sum(dim=-1)
        weighting = torch.softmax(dot_prod, dim=1)
        
        vert_att = masked_vert_feat_flat.unsqueeze(3) * weighting.unsqueeze(2)
        vert_att_reduce = vert_att.sum(dim=1).permute(0, 2, 1) # (BATCH_N, FORMULAE_N, F)
        combined = torch.cat([possible_formulae, vert_att_reduce], -1)
        combined = self.combine_norm(combined)
        x = F.relu(self.f_combine_l1(combined))
        x = F.relu(self.f_combine_l2(x))
        x = self.norm2(x)
        formulae_scores = self.f_combine_score(x).squeeze(2)
        formulae_num_atoms = possible_formulae.sum(dim=-1)
        assert formulae_num_atoms.shape == (BATCH_N, MAX_FORMULAE_N)
        formulae_present_mask = (formulae_num_atoms > 0).float()
        formulae_scores_masked = formulae_scores + (1-formulae_present_mask)*-100

        if self.prob_softmax:
            formulae_probs = torch.softmax(formulae_scores_masked, dim=-1)
        else:
            formulae_probs = torch.sigmoid(formulae_scores_masked)

        #t1 = time.time()
        sparse_mass_matrices = [create_mass_matrix_sparse(formulae_masses[:, :, i, 0].round().long(),
                                                   self.spect_bin_n,
                                                   formulae_masses[:, :, i, 1]
        )\
                         .to(vert_feat_in.device) for i in range(formulae_masses.shape[2])]

        
        sparse_mass_matrix = sparse_mass_matrices[0]
        for m in sparse_mass_matrices[1:]:
            sparse_mass_matrix += m

        spect_out = mat_matrix_sparse_mm(sparse_mass_matrix, formulae_probs)

        return {'spect_out': spect_out,
                'formulae_probs' : formulae_probs}


    

class MolAttentionNetOHSparseExtraFormulaFeat(nn.Module):
    """
    Per-vertex features to sparse points of peak, mass

    Input:
    BATCH_N x ATOM_N x F

    Output:
    BATCH_N x SPECT_BIN

    """
    def __init__(self, g_feat_in,
                 embedding_key_size = 16,
                 formula_oh_sizes = [20, 20, 20, 20, 20],
                 formula_oh_accum = True,
                 internal_d = 512,
                 prob_softmax = True,
                 g_embed_train = True,
                 g_embed_bias = True, 
                 formula_embed_train= True,
                 formula_embed_bias=True,
                 formula_frag_count = 1,
                 formula_frag_count_bool = False, 
                 spect_bin_n = 512):

        
        super( MolAttentionNetOHSparseExtraFormulaFeat, self).__init__()

        self.formula_frag_count = formula_frag_count
        self.formula_frag_count_bool = formula_frag_count_bool
        
        self.norm = nn.LayerNorm(g_feat_in)

        self.embed_g_feat = nn.Linear(g_feat_in, embedding_key_size, bias= g_embed_bias)
        formula_encoding_n = np.sum(formula_oh_sizes)

        if formula_frag_count > 0:
            formula_encoding_n += formula_frag_count
            
        self.formula_to_oh = StructuredOneHot(formula_oh_sizes, formula_oh_accum)
        
        self.embed_formulae_feat = nn.Linear(formula_encoding_n, embedding_key_size,
                                             bias=formula_embed_bias)

        if not formula_embed_train:
            self.embed_formulae_feat.weight.requires_grad = False
            self.embed_formulae_feat.bias.requires_grad = False
            
        
        self.norm2 = nn.LayerNorm(internal_d)

        self.combine_norm = nn.LayerNorm(formula_encoding_n +  g_feat_in)
        self.f_combine_l1 = nn.Linear(formula_encoding_n +  g_feat_in, internal_d)
        self.f_combine_l2 = nn.Linear(internal_d, internal_d)
        self.f_combine_score = nn.Linear(internal_d, 1)


        self.spect_bin_n = spect_bin_n
        self.prob_softmax = prob_softmax


    def forward(self, vert_feat_in, vert_mask_in,
                possible_formulae, formulae_masses,
                vert_element_oh, adj_oh, formula_frag_count):
        """
        vert_feat_in : BATCH_N x ATOM_N x F
        vert_feat_mask : BATCH_N x ATOM_N  
        possible_formulae : BATCH_N x MAX_FORMULAE_N x FORMULA_ENCODING_N
        formula_masses : BATCH_N x MAX_FORMULAE_N x NUM_MASSES x 2  (mass, peak intensity) 
        """

        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1)

        # one-hot encode possible formulae
        possible_formulae_flat = possible_formulae.reshape(-1, possible_formulae.shape[-1])
        possible_formulae_flat_oh = self.formula_to_oh(possible_formulae_flat)
        possible_formulae = possible_formulae_flat_oh.reshape(possible_formulae.shape[0],
                                                              possible_formulae.shape[1], -1)
        
        possible_formulae = torch.cat([possible_formulae, formula_frag_count], -1)
        
        # encoded inputs
        vert_encoded = self.embed_g_feat(vert_feat_in)
        formulae_encoded = self.embed_formulae_feat(possible_formulae)

        dot_prod = (formulae_encoded.unsqueeze(1) * vert_encoded.unsqueeze(2)).sum(dim=-1)
        weighting = torch.softmax(dot_prod, dim=1)
        
        vert_att = masked_vert_feat.unsqueeze(3) * weighting.unsqueeze(2)
        vert_att_reduce = vert_att.sum(dim=1).permute(0, 2, 1) # (BATCH_N, FORMULAE_N, F)
        combined = torch.cat([possible_formulae, vert_att_reduce], -1)
        combined = self.combine_norm(combined)
        x = F.relu(self.f_combine_l1(combined))
        x = F.relu(self.f_combine_l2(x))
        x = self.norm2(x)
        formulae_scores = self.f_combine_score(x).squeeze(2)
        
        if self.prob_softmax:
            formulae_probs = torch.softmax(formulae_scores, dim=-1)
        else:
            formulae_probs = torch.sigmoid(formulae_scores)

        #t1 = time.time()
        sparse_mass_matrices = [create_mass_matrix_sparse(formulae_masses[:, :, i, 0].round().long(),
                                                   self.spect_bin_n,
                                                   formulae_masses[:, :, i, 1]
        )\
                         .to(vert_feat_in.device) for i in range(formulae_masses.shape[2])]

        
        sparse_mass_matrix = sparse_mass_matrices[0]
        for m in sparse_mass_matrices[1:]:
            sparse_mass_matrix += m

        spect_out = mat_matrix_sparse_mm(sparse_mass_matrix, formulae_probs)

        return {'spect_out': spect_out,
                'formulae_probs' : formulae_probs}


    
    


class MolAttentionGRUDifferentSM(nn.Module):
    """
    Per-vertex features to sparse points of peak, mass

    Input:
    BATCH_N x ATOM_N x F

    Output:
    BATCH_N x SPECT_BIN

    """
    def __init__(self, g_feat_in,
                 embedding_key_size = 16,
                 formula_oh_sizes = [20, 20, 20, 20, 20],
                 formula_oh_accum = True,
                 internal_d = 512,
                 formulae_prob_transform = 'softmax',
                 g_embed_train = True,
                 g_embed_bias = True, 
                 formula_embed_train= True,
                 formula_embed_bias=True,
                 gru_layer_n = 1,
                 linear_layer_n = 2,
                 spect_bin_n = 512):

        
        super( MolAttentionGRUDifferentSM, self).__init__()

        self.norm = nn.LayerNorm(g_feat_in)

        self.embed_g_feat = nn.Linear(g_feat_in, embedding_key_size, bias= g_embed_bias)
        formula_encoding_n = np.sum(formula_oh_sizes)

        self.formula_to_oh = StructuredOneHot(formula_oh_sizes, formula_oh_accum)
        
        self.embed_formulae_feat = nn.Linear(formula_encoding_n, embedding_key_size,
                                             bias=formula_embed_bias)

        if not formula_embed_train:
            self.embed_formulae_feat.weight.requires_grad = False
            self.embed_formulae_feat.bias.requires_grad = False
            
        
        self.norm2 = nn.LayerNorm(internal_d)

        self.combine_layers = nn.ModuleList([nn.GRUCell(formula_encoding_n, g_feat_in) for _ in range(gru_layer_n)])
        
        self.f_combine_l1 = nn.Linear(g_feat_in, internal_d)
        self.f_combine_l2 = nn.Sequential(*[nn.Sequential(nn.Linear(internal_d, internal_d),
                                                          nn.ReLU()) for _ in range(linear_layer_n)])
        
        self.f_combine_score = nn.Linear(internal_d, 1)

        
        self.spect_bin_n = spect_bin_n

        if formulae_prob_transform == 'softmax':
            self.formulae_prob_transform = nn.Softmax(dim=-1)
        elif formulae_prob_transform == 'sigsoftmax':
            self.formulae_prob_transform = SigSoftmax(dim=-1)
        elif formulae_prob_transform == 'sigmoid':
            self.formulae_prob_transform = nn.Sigmoid()
        


    def forward(self, vert_feat_in, vert_mask_in,
                possible_formulae, formulae_masses,
                vert_element_oh, adj_oh, formula_frag_count):
        """
        vert_feat_in : BATCH_N x ATOM_N x F
        vert_feat_mask : BATCH_N x ATOM_N  
        possible_formulae : BATCH_N x MAX_FORMULAE_N x FORMULA_ENCODING_N
        formula_masses : BATCH_N x MAX_FORMULAE_N x NUM_MASSES x 2  (mass, peak intensity) 
        """

        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1)

        # one-hot encode possible formulae
        possible_formulae_flat = possible_formulae.reshape(-1, possible_formulae.shape[-1])
        possible_formulae_flat_oh = self.formula_to_oh(possible_formulae_flat)
        possible_formulae = possible_formulae_flat_oh.reshape(possible_formulae.shape[0],
                                                              possible_formulae.shape[1], -1)
        
        
        # encoded inputs
        vert_encoded = self.embed_g_feat(vert_feat_in)
        formulae_encoded = self.embed_formulae_feat(possible_formulae)

        dot_prod = (formulae_encoded.unsqueeze(1) * vert_encoded.unsqueeze(2)).sum(dim=-1)
        weighting = torch.softmax(dot_prod, dim=1)
        
        vert_att = masked_vert_feat.unsqueeze(3) * weighting.unsqueeze(2)
        vert_att_reduce = vert_att.sum(dim=1).permute(0, 2, 1) # (BATCH_N, FORMULAE_N, F)

        x = vert_att_reduce

        for l in self.combine_layers:
            x = l(possible_formulae.reshape(-1, possible_formulae.shape[-1]),
                  x.reshape(-1, x.shape[-1]))\
                  .reshape(x.shape)
        x = F.relu(self.f_combine_l1(x))
        x = F.relu(self.f_combine_l2(x))
        x = self.norm2(x)
        formulae_scores = self.f_combine_score(x).squeeze(2)
        
        formulae_probs = self.formulae_prob_transform(formulae_scores)

        #t1 = time.time()
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



    
class MolAttentionGRUNormalize(nn.Module):
    """
    Per-vertex features to sparse points of peak, mass

    Input:
    BATCH_N x ATOM_N x F

    Output:
    BATCH_N x SPECT_BIN

    """
    def __init__(self, g_feat_in,
                 embedding_key_size = 16,
                 formula_oh_sizes = [20, 20, 20, 20, 20],
                 formula_oh_accum = True,
                 formula_oh_normalize_1=False, 
                 formula_oh_normalize_2=False, 
                 
                 internal_d = 512,
                 prob_softmax = True,
                 g_embed_train = True,
                 g_embed_bias = True, 
                 formula_embed_train= True,
                 formula_embed_bias=True,
                 gru_layer_n = 1,
                 linear_layer_n = 2,
                 spect_bin_n = 512):

        
        super( MolAttentionGRUNormalize, self).__init__()

        self.norm = nn.LayerNorm(g_feat_in)

        self.embed_g_feat = nn.Linear(g_feat_in, embedding_key_size, bias= g_embed_bias)
        formula_encoding_n = np.sum(formula_oh_sizes)

        self.formula_to_oh = StructuredOneHot(formula_oh_sizes, formula_oh_accum)
        
        self.embed_formulae_feat = nn.Linear(formula_encoding_n, embedding_key_size,
                                             bias=formula_embed_bias)
        self.formula_oh_normalize_1 = formula_oh_normalize_1
        self.formula_oh_normalize_2 = formula_oh_normalize_2

        if not formula_embed_train:
            self.embed_formulae_feat.weight.requires_grad = False
            self.embed_formulae_feat.bias.requires_grad = False
            
        
        self.norm2 = nn.LayerNorm(internal_d)

        self.combine_layers = nn.ModuleList([nn.GRUCell(formula_encoding_n, g_feat_in) for _ in range(gru_layer_n)])
        
        self.f_combine_l1 = nn.Linear(g_feat_in, internal_d)
        self.f_combine_l2 = nn.Sequential(*[nn.Sequential(nn.Linear(internal_d, internal_d),
                                                          nn.ReLU()) for _ in range(linear_layer_n)])
        
        self.f_combine_score = nn.Linear(internal_d, 1)

        
        self.spect_bin_n = spect_bin_n
        self.prob_softmax = prob_softmax


    def forward(self, vert_feat_in, vert_mask_in,
                possible_formulae, formulae_masses,
                vert_element_oh, adj_oh, formula_frag_count):
        """
        vert_feat_in : BATCH_N x ATOM_N x F
        vert_feat_mask : BATCH_N x ATOM_N  
        possible_formulae : BATCH_N x MAX_FORMULAE_N x FORMULA_ENCODING_N
        formula_masses : BATCH_N x MAX_FORMULAE_N x NUM_MASSES x 2  (mass, peak intensity) 
        """

        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1)

        # one-hot encode possible formulae
        possible_formulae_flat = possible_formulae.reshape(-1, possible_formulae.shape[-1])
        possible_formulae_flat_oh = self.formula_to_oh(possible_formulae_flat)
        possible_formulae = possible_formulae_flat_oh.reshape(possible_formulae.shape[0],
                                                              possible_formulae.shape[1], -1)

        possible_formulae2 = possible_formulae
        if self.formula_oh_normalize_1:
            possible_formulae2 = possible_formulae / (possible_formulae.sum(axis=1).unsqueeze(1) + 0.1)
            
        if self.formula_oh_normalize_2:
            possible_formulae2 = possible_formulae / (possible_formulae.sum(axis=2).unsqueeze(2) + 0.1)
            
        # encoded inputs
        vert_encoded = self.embed_g_feat(vert_feat_in)
        formulae_encoded = self.embed_formulae_feat(possible_formulae)

        dot_prod = (formulae_encoded.unsqueeze(1) * vert_encoded.unsqueeze(2)).sum(dim=-1)
        weighting = torch.softmax(dot_prod, dim=1)
        
        vert_att = masked_vert_feat.unsqueeze(3) * weighting.unsqueeze(2)
        vert_att_reduce = vert_att.sum(dim=1).permute(0, 2, 1) # (BATCH_N, FORMULAE_N, F)

        x = vert_att_reduce

        for l in self.combine_layers:
            x = l(possible_formulae2.reshape(-1, possible_formulae2.shape[-1]),
                  x.reshape(-1, x.shape[-1]))\
                  .reshape(x.shape)
        x = F.relu(self.f_combine_l1(x))
        x = F.relu(self.f_combine_l2(x))
        x = self.norm2(x)
        formulae_scores = self.f_combine_score(x).squeeze(2)
        
        if self.prob_softmax:
            formulae_probs = torch.softmax(formulae_scores, dim=-1)
        else:
            formulae_probs = torch.sigmoid(formulae_scores)

        #t1 = time.time()
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

