import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import scipy.stats
import time
from torch.nn.parameter import Parameter

def goodmax(x, dim):
    return torch.max(x, dim=dim)[0]
 
def peak_indices_intensities_to_sparse_matrix(indices, intensities, max_bin):
    """
    Convert tensor of peaks (N x MAX_PEAK) and intensities (N x MAX_PEAK) into 
    a sparse max_bin x N matrix. 

    Note that a pytorch sparse matrix needs to know the full dimensions, hence
    specifying max_bin. 

    We do NOT check to make sure indices < max_bin. 

    """
    N, possible_peaks = indices.shape
    assert indices.dtype == torch.long
    assert intensities.shape == indices.shape

    device = indices.device
    col_idx = torch.arange(N, device=device).repeat(possible_peaks, 1).permute(1, 0).flatten()
    row_idx = indices.flatten()
    combined_idx = torch.stack([row_idx, col_idx])
    sparse_mat = torch.sparse_coo_tensor(combined_idx, values=intensities.flatten(), 
                                         size=(max_bin, N))

    return sparse_mat

class MaskedBatchNorm1d(nn.Module):
    def __init__(self, feature_n):
        """
        Batchnorm1d that skips some rows in the batch 
        """

        super(MaskedBatchNorm1d, self).__init__()
        self.feature_n = feature_n
        self.bn = nn.BatchNorm1d(feature_n)

    def forward(self, x, mask):
        assert x.shape[0] == mask.shape[0]
        assert mask.dim() == 1
        
        bin_mask = mask > 0
        y_i = self.bn(x[bin_mask])
        y = torch.zeros(x.shape, device=x.device)
        y[bin_mask] = y_i
        return y

class MaskedLayerNorm1d(nn.Module):
    def __init__(self, feature_n):
        """
        LayerNorm that skips some rows in the batch 
        """

        super(MaskedLayerNorm1d, self).__init__()
        self.feature_n = feature_n
        self.bn = nn.LayerNorm(feature_n)

    def forward(self, x, mask):
        assert x.shape[0] == mask.shape[0]
        assert mask.dim() == 1
        
        bin_mask = mask > 0
        y_i = self.bn(x[bin_mask])
        y = torch.zeros(x.shape, device=x.device)
        y[bin_mask] = y_i
        return y

class GraphInstanceNorm(nn.Module):
    def __init__(self, feature_n):
        """
        Batchnorm1d that can handle BATCH_N x VERT_N x F 
        """

        super(GraphInstanceNorm, self).__init__()
        self.feature_n = feature_n
        
    def forward(self, x, mask):

        div_eps = 1e-5
        BATCH_N, VERT_N, F = x.shape
        assert mask.shape == (BATCH_N, VERT_N)
        mask = mask + div_eps
        
        x_masked = x * mask.unsqueeze(-1)

        vert_count = mask.sum(dim=1)
        
        x_sum = x_masked.sum(dim=1)
        x_mean = x_sum / (vert_count.unsqueeze(-1) + div_eps)

        x_mean_minus = (x_masked - x_mean.unsqueeze(1)) * mask.unsqueeze(-1)

        x_mean_minus_sq = x_mean_minus **2
        x_mean_minus_sq_sum = x_mean_minus_sq.sum(dim=1)

        # normalize sum of squares by the num of present vertex subsets we summed over...
        x_std = x_mean_minus_sq_sum / (vert_count.unsqueeze(-1) + div_eps)
        assert x_std.shape == (BATCH_N, F)
        
        x_normed = x_mean_minus / torch.sqrt(x_std + div_eps).unsqueeze(1)

        return x_normed

class GraphMatLayer(nn.Module):
    def __init__(self, C, P , GS=1,  
                 noise=1e-6, agg_func=None, 
                 dropout=0.0, use_bias=True):
        """
        Pairwise layer -- takes a N x M x M x C matrix
        and turns it into a N x M x M x P matrix after
        multiplying with a graph matrix N x M x M
        
        if GS != 1 then there will be a per-graph-channel 
        linear layer

        Input: 
            G: adjacency matrix in shape (batch_size, n_channels, max_n_atoms, max_n_atoms)
                We usually have one channel per bond order.
            x: feature matrix in shape (batch_size, max_n_atoms, feature_dim=C)
        Output:
            xp: feature matrix in shape (batch_size, max_n_atoms, feature_dim=P)
        """
        super(GraphMatLayer, self).__init__()

        self.GS = GS
        self.noise=noise

        self.linlayers = nn.ModuleList()
        self.dropout = dropout
        self.dropout_layers = nn.ModuleList()
        for ll in range(GS):
            l = nn.Linear(C, P, bias=use_bias)
            if use_bias:
                l.bias.data.normal_(0.0, self.noise)
            l.weight.data.normal_(0.0, self.noise) #?!
            self.linlayers.append(l)
            if dropout > 0.0:
                self.dropout_layers.append(nn.Dropout(p=dropout))
            
        #self.r = nn.PReLU()
        self.r = nn.ReLU()
        self.agg_func = agg_func
 
    def forward(self, G, x):
        def apply_ll(i, x):
            y = self.linlayers[i](x)
            if self.dropout > 0:
                y = self.dropout_layers[i](y)
            return y

        multi_x = torch.stack([apply_ll(i,x) for i in range(self.GS)])
        # this is per-batch-element
        xout = torch.stack([torch.matmul(G[i], multi_x[:, i]) for i in range(x.shape[0])])

        x = self.r(xout)
        if self.agg_func is not None:
            x = self.agg_func(x, dim=1)
        return x
        

class GraphMatLayers(nn.Module):
    def __init__(self, input_feature_n, 
                 output_features_n, resnet=False, GS=1, 
                 norm=None,
                 force_use_bias = False, 
                 noise=1e-5, agg_func=None,
                 layer_class = 'GraphMatLayerFast', 
                 layer_config = {}):
        super(GraphMatLayers, self).__init__()
        
        self.gl = nn.ModuleList()
        self.resnet = resnet

        LayerClass = eval(layer_class)
        for li in range(len(output_features_n)):
            if li == 0:
                gl = LayerClass(input_feature_n, output_features_n[0],
                                noise=noise, agg_func=agg_func, GS=GS, 
                                use_bias=not norm or force_use_bias, 
                                **layer_config)
            else:
                gl = LayerClass(output_features_n[li-1], 
                                output_features_n[li], 
                                noise=noise, agg_func=agg_func, GS=GS, 
                                use_bias=not norm or force_use_bias, 
                                **layer_config)
            
            self.gl.append(gl)

        self.norm = norm
        if self.norm is not None:
            if self.norm == 'batch':
                Nlayer = MaskedBatchNorm1d
            elif self.norm == 'layer':
                Nlayer = MaskedLayerNorm1d
            self.bn = nn.ModuleList([Nlayer(f) for f in output_features_n])
            
        
    def forward(self, G, x, input_mask=None):
        for gi, gl in enumerate(self.gl):
            x2 = gl(G, x)
            if self.norm:
                x2 = self.bn[gi](x2.reshape(-1, x2.shape[-1]), 
                                 input_mask.reshape(-1)).reshape(x2.shape)

            if self.resnet:
                if x.shape == x2.shape:
                    x3 = x2 + x
                else:
                    x3 = x2
            else:
                x3 = x2
            x = x3
        

        return x


class GraphMatLayersNormAfterRes(nn.Module):
    """
    Same as GraphMatLayers but we do the norm
    after the resnet addition

    input_feature_n: the starting 
    """
    def __init__(self, input_feature_n, 
                 output_features_n, resnet=False, GS=1, 
                 norm=None,
                 force_use_bias = False, 
                 noise=1e-5, agg_func=None,
                 layer_class = 'GraphMatLayerFast', 
                 layer_config = {}):
        super(GraphMatLayersNormAfterRes, self).__init__()
        
        self.gl = nn.ModuleList()
        self.resnet = resnet
        LayerClass = eval(layer_class)

        fdims = [input_feature_n] + list(output_features_n)
        for li in range(len(fdims) - 1):
            gl = LayerClass(fdims[li], fdims[li + 1],
                            noise=noise, agg_func=agg_func, GS=GS, 
                            use_bias=not norm or force_use_bias, 
                            **layer_config)
            self.gl.append(gl)

        self.norm = norm
        if self.norm is not None:
            if self.norm == 'batch':
                Nlayer = MaskedBatchNorm1d
            elif self.norm == 'layer':
                Nlayer = MaskedLayerNorm1d
            self.bn = nn.ModuleList([Nlayer(f) for f in output_features_n])
        
    def forward(self, G, x, input_mask=None):
        for gi, gl in enumerate(self.gl):
            x2 = gl(G, x)
            # if self.norm:
            #     x2 = self.bn[gi](x2.reshape(-1, x2.shape[-1]), 
            #                      input_mask.reshape(-1)).reshape(x2.shape)

            if self.resnet:
                if x.shape == x2.shape:
                    x3 = x2 + x
                else:
                    x3 = x2
            else:
                x3 = x2

            if self.norm:
                # since we pad each graph up to max_n_atoms, if we norm
                # we need to norm only over valid atoms.
                # x3: (batch_size, max_n_atoms, feature_dim)
                # input_mask: (batch_size, max_n_atoms)
                x3 = self.bn[gi](x3.reshape(-1, x3.shape[-1]), 
                                 input_mask.reshape(-1)).reshape(x3.shape)
                
            x = x3

        return x

    
class GraphMatHighwayLayers(nn.Module):
    def __init__(self, input_feature_n, 
                 output_features_n, resnet=False, GS=1, 
                 noise=1e-5, agg_func=None):
        super(GraphMatHighwayLayers, self).__init__()
        
        self.gl = nn.ModuleList()
        self.resnet = resnet

        for li in range(len(output_features_n)):
            if li == 0:
                gl = GraphMatLayer(input_feature_n, output_features_n[0],
                                   noise=noise, agg_func=agg_func, GS=GS)
            else:
                gl = GraphMatLayer(output_features_n[li-1], 
                                   output_features_n[li], 
                                   noise=noise, agg_func=agg_func, GS=GS)
            
            self.gl.append(gl)

    def forward(self, G, x):
        highway_out = []
        for gl in self.gl:
            x2 = gl(G, x)
            if self.resnet:
                if x.shape == x2.shape:
                    x3 = x2 + x
                else:
                    x3 = x2
            else:
                x3 = x2
            x = x3
            highway_out.append(x2)

        return x, torch.stack(highway_out, -1)

def parse_agg_func(agg_func):
    if isinstance(agg_func, str):
        if agg_func == 'goodmax':
            return goodmax
        elif agg_func == 'sum':
            return torch.sum
        elif agg_func == 'mean':
            return torch.mean
        else:
            raise NotImplementedError()
    return agg_func


def log_normal_nolog(y, mu, std):
    element_wise =  -(y - mu)**2 / (2*std**2)  - std
    return element_wise

def log_student_t(y, mu, std, v=1.0):
    return -torch.log(1.0 + (y-mu)**2/(v * std)) - std

def log_normal(y, mu, std):
    element_wise =  -(y - mu)**2 / (2*std**2)  - torch.log(std)
    return element_wise


class GraphMatLayerFast(nn.Module):
    def __init__(self, C, P , GS=1,  
                 noise=1e-6, agg_func=None, 
                 dropout=0.0, use_bias=False,
                 nonlin = 'leakyrelu', 
                 ):
        """
        Pairwise layer -- takes a N x M x M x C matrix
        and turns it into a N x M x M x P matrix after
        multiplying with a graph matrix N x M x M
        
        if GS != 1 then there will be a per-graph-channel 
        linear layer
        """
        super(GraphMatLayerFast, self).__init__()

        self.GS = GS
        self.noise=noise

        self.linlayers = nn.ModuleList()
        for ll in range(GS):
            l = nn.Linear(C, P, bias=use_bias)
            if self.noise == 0:
                if use_bias:
                    l.bias.data.normal_(0.0, 1e-4)
                torch.nn.init.xavier_uniform_(l.weight)
            else:
                if use_bias:
                    l.bias.data.normal_(0.0, self.noise)
                l.weight.data.normal_(0.0, self.noise) #?!
            self.linlayers.append(l)
            
        #self.r = nn.PReLU()
        self.r = create_nonlin(nonlin)
        self.agg_func = agg_func
        self.dropout = dropout
 
    def forward(self, G, x):
        BATCH_N, CHAN_N,  MAX_N, _ = G.shape
        def apply_ll(i, x):
            y = self.linlayers[i](x)
            return y

        multi_x = torch.stack([apply_ll(i,x) for i in range(self.GS)], 0)
        # this is where the magic happens
        xout = torch.einsum("ijkl,jilm->jikm", [G, multi_x])
        if self.dropout > 0:
            xout = F.dropout(xout, p=self.dropout)
            
        xout = self.r(xout)
        
        # if GS != 1, this aggregation does something.
        if self.agg_func is not None:
            xout = self.agg_func(xout, dim=0)
        return xout
        




class GraphMatLayerFastPow(nn.Module):
    def __init__(self, C, P, GS=1,  
                 mat_pow = 1, 
                 mat_diag = False,
                 noise=1e-6, agg_func=None, 
                 use_bias=False, 
                 nonlin = None, 
                 dropout = 0.0, 
                 norm_by_neighbors=False, 
                 ):
        """

        """
        super(GraphMatLayerFastPow, self).__init__()

        self.GS = GS
        self.noise=noise

        self.linlayers = nn.ModuleList()
        for ll in range(GS):
            l = nn.Linear(C, P, bias=use_bias)
            self.linlayers.append(l)
        self.dropout_rate = dropout

        if self.dropout_rate > 0:
            self.dropout_layers = nn.ModuleList([nn.Dropout(self.dropout_rate) for _ in range(GS)])

        #self.r = nn.PReLU()
        self.nonlin = nonlin
        if self.nonlin == 'leakyrelu':
            self.r = nn.LeakyReLU()
        elif self.nonlin == 'sigmoid':
            self.r = nn.Sigmoid()
        elif self.nonlin == 'tanh':
            self.r = nn.Tanh()
        elif self.nonlin is None:
            pass
        else:
            raise ValueError(f'unknown nonlin {nonlin}')
            
        self.agg_func = agg_func
        self.mat_pow = mat_pow
        self.mat_diag = mat_diag

        self.norm_by_neighbors = norm_by_neighbors
 
    def forward(self, G, x):
        BATCH_N, CHAN_N,  MAX_N, _ = G.shape
        def apply_ll(i, x):
            y = self.linlayers[i](x)
            if self.dropout_rate > 0.0:
                y = self.dropout_layers[i](y)
            return y
        Gprod = G
        for mp in range(self.mat_pow -1):
            Gprod = torch.einsum("ijkl,ijlm->ijkm", G, Gprod)
        if self.mat_diag:
            Gprod = torch.eye(MAX_N).unsqueeze(0).unsqueeze(0).to(G.device) * Gprod
            
        multi_x = torch.stack([apply_ll(i,x) for i in range(self.GS)], 0)
        xout = torch.einsum("ijkl,jilm->jikm", [Gprod, multi_x])

        if self.norm_by_neighbors:
            G_neighbors = torch.clamp(G.sum(-1).permute(1, 0, 2), min=1)
            xout = xout / G_neighbors.unsqueeze(-1)

        if self.nonlin is not None:
            xout = self.r(xout)
        if self.agg_func is not None:
            xout = self.agg_func(xout, dim=0)
        return xout


class GraphMatLayerFastPowSwap(nn.Module):
    def __init__(self, C, P, GS=1,  
                 mat_pow = 1, 
                 mat_diag = False,
                 noise=1e-6, agg_func=None, 
                 use_bias=False, 
                 nonlin = None, 
                 dropout = 0.0, 
                 norm_by_neighbors=False, 
                 ):
        """

        """
        super(GraphMatLayerFastPowSwap, self).__init__()

        self.GS = GS
        self.noise=noise

        self.linlayers = nn.ModuleList()
        for ll in range(GS):
            l = nn.Linear(C, P, bias=use_bias)
            self.linlayers.append(l)
        self.dropout_rate = dropout

        if self.dropout_rate > 0:
            self.dropout_layers = nn.ModuleList([nn.Dropout(self.dropout_rate) for _ in range(GS)])

        #self.r = nn.PReLU()
        self.nonlin = nonlin
        if self.nonlin == 'leakyrelu':
            self.r = nn.LeakyReLU()
        elif self.nonlin is None:
            pass
        else:
            raise ValueError(f'unknown nonlin {nonlin}')
            
        self.agg_func = agg_func
        self.mat_pow = mat_pow
        self.mat_diag = mat_diag

        self.norm_by_neighbors = norm_by_neighbors
 
    def forward(self, G, x):
        BATCH_N, CHAN_N,  MAX_N, _ = G.shape
        def apply_ll(i, x):
            y = self.linlayers[i](x)
            if self.dropout_rate > 0.0:
                y = self.dropout_layers[i](y)
            return y
        Gprod = G
        for mp in range(self.mat_pow -1):
            Gprod = torch.einsum("ijkl,ijlm->ijkm", G, Gprod)
        if self.mat_diag:
            Gprod = torch.eye(MAX_N).unsqueeze(0).unsqueeze(0).to(G.device) * Gprod
            
        # multi_x = torch.stack([apply_ll(i,x) for i in range(self.GS)], 0)
        # xout = torch.einsum("ijkl,jilm->jikm", [Gprod, multi_x])
        # print("x.shape=", x.shape, "multi_x.shape=", multi_x.shape, 
        #       "Gprod.shape=", Gprod.shape, "xout.shape=", xout.shape)
        
        x_adj = torch.einsum("ijkl,ilm->jikm", [Gprod, x])
        xout = torch.stack([apply_ll(i, x_adj[i]) for i in range(self.GS)])
        # print("\nx.shape=", x.shape, 
        #       "x_adj.shape=", x_adj.shape, 
        #       "Gprod.shape=", Gprod.shape, 
        #       "xout.shape=", xout.shape)
        
                              
        if self.norm_by_neighbors:
            G_neighbors = torch.clamp(G.sum(-1).permute(1, 0, 2), min=1)
            xout = xout / G_neighbors.unsqueeze(-1)

        if self.nonlin is not None:
            xout = self.r(xout)
        if self.agg_func is not None:
            xout = self.agg_func(xout, dim=0)
        return xout

    
class GraphMatLayerFastPowSingleLayer(nn.Module):
    def __init__(self, C, P, GS=1,  
                 mat_pow = 1, 
                 mat_diag = False,
                 noise=1e-6, agg_func=None, 
                 use_bias=False, 
                 nonlin = None, 
                 dropout = 0.0, 
                 norm_by_neighbors=False, 
                 ):
        """

        """
        super(GraphMatLayerFastPowSingleLayer, self).__init__()

        self.GS = GS
        self.noise=noise

        self.l = nn.Linear(C, P, bias=use_bias)
        self.dropout_rate = dropout

        # if self.dropout_rate > 0:
        #     self.dropout_layers = nn.ModuleList([nn.Dropout(self.dropout_rate) for _ in range(GS)])

        #self.r = nn.PReLU()
        self.nonlin = nonlin
        if self.nonlin == 'leakyrelu':
            self.r = nn.LeakyReLU()
        elif self.nonlin is None:
            pass
        else:
            raise ValueError(f'unknown nonlin {nonlin}')
            
        self.agg_func = agg_func
        self.mat_pow = mat_pow
        self.mat_diag = mat_diag

        self.norm_by_neighbors = norm_by_neighbors
 
    def forward(self, G, x):
        BATCH_N, CHAN_N,  MAX_N, _ = G.shape
        def apply_ll(x):
            y = self.l(x)
            if self.dropout_rate > 0.0:
                y = self.dropout_layers(y)
            return y
        Gprod = G
        for mp in range(self.mat_pow -1):
            Gprod = torch.einsum("ijkl,ijlm->ijkm", G, Gprod)
        if self.mat_diag:
            Gprod = torch.eye(MAX_N).unsqueeze(0).unsqueeze(0).to(G.device) * Gprod
            
        # multi_x = torch.stack([apply_ll(i,x) for i in range(self.GS)], 0)
        # xout = torch.einsum("ijkl,jilm->jikm", [Gprod, multi_x])
        # print("x.shape=", x.shape, "multi_x.shape=", multi_x.shape, 
        #       "Gprod.shape=", Gprod.shape, "xout.shape=", xout.shape)
        
        x_adj = torch.einsum("ijkl,ilm->jikm", [Gprod, x])
        xout = torch.stack([apply_ll(x_adj[i]) for i in range(self.GS)])
        # print("\nx.shape=", x.shape, 
        #       "x_adj.shape=", x_adj.shape, 
        #       "Gprod.shape=", Gprod.shape, 
        #       "xout.shape=", xout.shape)
        
                              
        if self.norm_by_neighbors:
            G_neighbors = torch.clamp(G.sum(-1).permute(1, 0, 2), min=1)
            xout = xout / G_neighbors.unsqueeze(-1)

        if self.nonlin is not None:
            xout = self.r(xout)
        if self.agg_func is not None:
            xout = self.agg_func(xout, dim=0)
        return xout

    



def apply_masked_1d_norm(norm, x, mask):
    """
    Apply one of these norms and do the reshaping
    """
    F_N = x.shape[-1]
    x_flat = x.reshape(-1, F_N)
    mask_flat = mask.reshape(-1)
    out_flat = norm(x_flat, mask_flat)
    out = out_flat.reshape(*x.shape)
    return out


class GraphMatLayerFastPow2(nn.Module):
    def __init__(self, C, P, GS=1,  
                 mat_pow = 1, 
                 mat_diag = False,
                 noise=1e-6, agg_func=None, 
                 use_bias=False, 
                 nonlin = None, 
                 dropout = 0.0, 
                 norm_by_neighbors=False, 
                 ):
        """
        Two layer MLP 

        """
        super(GraphMatLayerFastPow2, self).__init__()

        self.GS = GS
        self.noise=noise

        self.linlayers1 = nn.ModuleList()
        self.linlayers2 = nn.ModuleList()
        
        for ll in range(GS):
            l = nn.Linear(C, P)
            self.linlayers1.append(l)
            l = nn.Linear(P, P)
            self.linlayers2.append(l)
        self.dropout_rate = dropout

        if self.dropout_rate > 0:
            self.dropout_layers = nn.ModuleList([nn.Dropout(self.dropout_rate) for _ in range(GS)])

        #self.r = nn.PReLU()
        self.nonlin = nonlin
        if self.nonlin == 'leakyrelu':
            self.r = nn.LeakyReLU()
        elif self.nonlin == 'sigmoid':
            self.r = nn.Sigmoid()
        elif self.nonlin == 'tanh':
            self.r = nn.Tanh()
        elif self.nonlin is None:
            pass
        else:
            raise ValueError(f'unknown nonlin {nonlin}')
            
        self.agg_func = agg_func
        self.mat_pow = mat_pow
        self.mat_diag = mat_diag

        self.norm_by_neighbors = norm_by_neighbors
 
    def forward(self, G, x):
        BATCH_N, CHAN_N,  MAX_N, _ = G.shape
        def apply_ll(i, x):
            y = F.relu(self.linlayers1[i](x))
            y = self.linlayers2[i](y)
            
            if self.dropout_rate > 0.0:
                y = self.dropout_layers[i](y)
            return y
        Gprod = G
        for mp in range(self.mat_pow -1):
            Gprod = torch.einsum("ijkl,ijlm->ijkm", G, Gprod)
        if self.mat_diag:
            Gprod = torch.eye(MAX_N).unsqueeze(0).unsqueeze(0).to(G.device) * Gprod
        multi_x = torch.stack([apply_ll(i,x) for i in range(self.GS)], 0)
        #print("Gprod.shape=", Gprod.shape, "multi_x.shape=", multi_x.shape)
        xout = torch.einsum("ijkl,jilm->jikm", [Gprod, multi_x])

        if self.norm_by_neighbors != False:
            G_neighbors = torch.clamp(G.sum(-1).permute(1, 0, 2), min=1)
            if self.norm_by_neighbors == 'sqrt':
                xout = xout / torch.sqrt(G_neighbors.unsqueeze(-1))
                
            else:
                xout = xout / G_neighbors.unsqueeze(-1)

        if self.nonlin is not None:
            xout = self.r(xout)
        if self.agg_func is not None:
            xout = self.agg_func(xout, dim=0)
        return xout

def create_nonlin(nonlin):
    if nonlin == 'leakyrelu':
        r = nn.LeakyReLU()
    elif nonlin == 'sigmoid':
        r = nn.Sigmoid()
    elif nonlin == 'tanh':
        r = nn.Tanh()
    elif nonlin == 'relu':
        r = nn.ReLU()
    elif nonlin == 'identity':
        r = nn.Identity()
    elif nonlin == 'prelu':
        r = nn.PReLU()
    elif nonlin.startswith('softmax'):
        dim = int(nonlin.split('-')[1])
        r = nn.Softmax(dim=dim)
    else:
        raise NotImplementedError(f'unknown nonlin {nonlin}')
    
    return r



class MLP(nn.Module):
    def __init__(self, layer_n=1, d=128,
                 input_d = None,
                 output_d = None,
                 nonlin='relu',
                 final_nonlin=True,
                 use_bias=True):
        super(MLP, self).__init__()

        ml = []
        for i in range(layer_n):
            in_d = d
            out_d = d
            if i == 0 and input_d is not None:
                in_d = input_d
            if (i == (layer_n -1)) and output_d is not None:
                out_d = output_d
            
            linlayer = nn.Linear(in_d, out_d, use_bias)

            ml.append(linlayer)
            nonlin_layer = create_nonlin(nonlin)
            if i == (layer_n -1) and not final_nonlin:
                pass
            else:
                ml.append(nonlin_layer)
        self.ml = nn.Sequential(*ml)
    def forward(self, x):
        return self.ml(x)
            
    





        
    
    

        

def render_gaussian_batch_points(x, w, eval_points, SIGMA=0.0001):
    y = x.unsqueeze(-1) - eval_points.unsqueeze(0).unsqueeze(0)
    y1 = torch.exp(-y**2 / SIGMA)
    
    y2 = y1 * w.unsqueeze(-1)
    ysum = y2.sum(dim=1)
    return ysum

def dense_to_sparse(prob_vects, sort=False):
    """
    Returns the sparse (loc, prob) representation
        of the dense prob vectors
    
    """
    
    BATCH_N, D = prob_vects.shape

    points = torch.arange(D).unsqueeze(0).repeat(BATCH_N, 1).to(prob_vects.device)
    probs = prob_vects
    if sort:
        idx = torch.argsort(prob_vects, dim=1, descending=True)
        probs = torch.gather(prob_vects, dim=1, index=idx)
        points = torch.gather(points, dim=1, index = idx)
        
    return points, probs

class GraphMatLayerFast2(nn.Module):
    def __init__(self, C, P , GS=1,  
                 noise=1e-6, agg_func=None, 
                 dropout=0.0, use_bias=False,
                 nonlin = 'leakyrelu',
                 swap_init = True
                 ):
        """
        Faster bias-free version of GMLFast 
        """
        super(GraphMatLayerFast2, self).__init__()

        self.GS = GS
        self.noise=noise
        
        if use_bias :
            print("use_bias=true not impelemnted")
            #raise NotImplementedError()

        self.weight = Parameter(torch.Tensor(GS, P, C))
            
        #self.r = nn.PReLU()
        self.r = create_nonlin(nonlin)
        self.agg_func = agg_func
        self.dropout = dropout

        # ?: what does this do
        if swap_init:
            torch.nn.init.kaiming_uniform_(self.weight.T) 
        else:
            torch.nn.init.kaiming_uniform_(self.weight) 
    def forward(self, G, x):
        BATCH_N, CHAN_N,  MAX_N, _ = G.shape

        multi_x = torch.einsum("ijk,lmk->lijm", x, self.weight)
        xout = torch.einsum("ijkl,jilm->jikm", [G, multi_x])
        if self.dropout > 0:
            xout = F.dropout(xout, p=self.dropout)
            
        xout = self.r(xout)
            
        if self.agg_func is not None:
            xout = self.agg_func(xout, dim=0)
        return xout

class GraphMatLayerFast3(nn.Module):
    def __init__(self, C, P , GS=1,  
                 noise=1e-6, agg_func=None, 
                 dropout=0.0, use_bias=False,
                 nonlin = 'leakyrelu',
                 swap_init = True
                 ):
        """
        Faster bias-free version of GMLFast 
        """
        super(GraphMatLayerFast3, self).__init__()

        self.GS = GS
        self.noise=noise
        
        self.weight = Parameter(torch.Tensor(GS, P, C))
        self.bias = Parameter(torch.Tensor(GS, 1, 1, P))
            
        #self.r = nn.PReLU()
        self.r = create_nonlin(nonlin)
        self.agg_func = agg_func
        self.dropout = dropout

        # ?: what does this do
        if swap_init:
            torch.nn.init.xavier_uniform_(self.weight.T) 
        else:
            torch.nn.init.xavier_uniform_(self.weight)

        torch.nn.init.constant_(self.bias, 0)
        
    def forward(self, G, x):
        BATCH_N, CHAN_N,  MAX_N, _ = G.shape

        multi_x = torch.einsum("ijk,lmk->lijm", x, self.weight)
        xout = torch.einsum("ijkl,jilm->jikm", [G, multi_x])

        xout = xout + self.bias 
        if self.dropout > 0:
            xout = F.dropout(xout, p=self.dropout)
            
        xout = self.r(xout)
            
        if self.agg_func is not None:
            xout = self.agg_func(xout, dim=0)
        return xout


class GraphMatLayersInstanceNorm(nn.Module):
    """
    Same as GraphMatLayers but we do an instance norm
    """
    def __init__(self, input_feature_n, 
                 output_features_n, resnet=False, GS=1, 
                 norm=None,
                 force_use_bias = False, 
                 noise=1e-5, agg_func=None,
                 layer_class = 'GraphMatLayerFast',
                 instance_norm_type = 'normal',
                 norm_after_res = True, 
                 return_every_n = 0, 
                 layer_config = {}):
        super(GraphMatLayersInstanceNorm, self).__init__()
        
        self.gl = nn.ModuleList()
        self.resnet = resnet
        self.norm_after_res = norm_after_res
        LayerClass = eval(layer_class)

        fdims = [input_feature_n] + list(output_features_n)
        for li in range(len(fdims) - 1):
            gl = LayerClass(fdims[li], 
                            fdims[li + 1], 
                            noise=noise, agg_func=agg_func, GS=GS, 
                            use_bias=not norm or force_use_bias, 
                            **layer_config)
            self.gl.append(gl)

        if instance_norm_type == 'normal':
            print("CREATING INSTANCE NORM NORMAL")
            self.g_norm = nn.ModuleList([GraphInstanceNorm(f) for f in output_features_n])
        elif instance_norm_type == 'param':
            raise NotImplementedError('Instance norm param was deleted.')
            
        self.return_every_n = return_every_n
        
    def forward(self, G, x, input_mask=None):

        intermediate = []
        
        for gi, gl in enumerate(self.gl):
            x2 = gl(G, x)
            # if self.norm:
            #     x2 = self.bn[gi](x2.reshape(-1, x2.shape[-1]), 
            #                      input_mask.reshape(-1)).reshape(x2.shape)
            if not self.norm_after_res:
                x2 = self.g_norm[gi](x2, input_mask)
                
            if self.resnet:
                if x.shape == x2.shape:
                    x3 = x2 + x
                else:
                    x3 = x2
            else:
                x3 = x2

            if self.norm_after_res:
                x3 = self.g_norm[gi](x3, input_mask)

            if self.return_every_n > 0:
                if (gi % self.return_every_n) == (self.return_every_n -1):
                    intermediate.append(x3)
                
            x = x3
        
        if self.return_every_n > 0:
            return torch.stack(intermediate, -1)
        else:
            return x
