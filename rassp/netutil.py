import torch
from rdkit import Chem
import rdkit.Chem.Descriptors
import torch.utils.data
import pickle
import zlib
import copy
import torch.autograd
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import time
import numpy as np
import pandas as pd
import os

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from rassp import util
from rassp import dataset
from rassp import featurize
from rassp.model import losses

default_atomicno = [1, 6, 7, 8, 9, 15, 16, 17]

### Create datasets and data loaders

default_feat_vert_args = dict(feat_atomicno_onehot=default_atomicno, 
                              feat_pos=False, feat_atomicno=True,
                              feat_valence=True, aromatic=True, 
                              hybridization=True, 
                              partial_charge=False, formal_charge=True,  
                              r_covalent=False,
                              total_valence_onehot=True, 
                              mmff_atom_types_onehot =False, 
                              r_vanderwals=False, 
                              default_valence=True, rings=True)


default_split_weights = [1, 1.5, 2, 3]

default_adj_args = dict(edge_weighted=False, 
                        norm_adj=True, add_identity=True, 
                        split_weights=default_split_weights)

default_subset_gen_config = {'name' : 'BandR',
                             'num_breaks' : 3}

DEFAULT_FEATURIZE_CONFIG = {'feat_vert_args' : default_feat_vert_args, 
                            'adj_args' : default_adj_args,
                            'subset_gen_config' : default_subset_gen_config}

DEFAULT_PRED_CONFIG = {}

def dict_combine(d1, d2):
    d1 = copy.deepcopy(d1)
    d1.update(d2)
    return d1

class CVSplit:
    def __init__(self, how, **args):
        self.how = how
        self.args = args

    def get_phase(self, mol, fp):
        if self.how == 'morgan_fingerprint_mod':
            mod = self.args['mod']
            mod_fp = fp % mod

            test = self.args.get('test', None)
            train = self.args.get('train', None)
            if (train is not None) and (test is not None):
                # both train and test are specified!
                # anything not in either becomes set to 'null'
                # and should get filtered out.
                if mod_fp in train:
                    return 'train'
                elif mod_fp in test:
                    return 'test'
                else:
                    return 'null'
            elif (train is None) and (test is not None):
                # default behavior: we specify test,
                # and everything not test is train
                if mod_fp in test:
                    return 'test'
                else:
                    return 'train'
            elif (train is not None) and (test is None):
                # we specify train,
                # and everything not train is test
                if mod_fp in train:
                    return 'train'
                else:
                    return 'test'

        else:
            raise ValueError(f"unknown method {self.how}")

def load_regular_dataset_recs(dataset_config, cv_splitter):
    
    filename = dataset_config['filename']
    phase = dataset_config.get('phase', 'train')
    dataset_spect_assign = dataset_config.get("spect_assign", True) 
    d = pd.read_parquet(filename)
    if dataset_config.get('subsample_to', 0) > 0:
        if len(d) > dataset_config['subsample_to']:
            print(f"subsampling data from {len(d)} to  {dataset_config['subsample_to']}")
            d = d.sample(dataset_config['subsample_to'],
                         random_state = dataset_config.get('subsample_seed', 0))

    spect_dict_field = dataset_config.get('spect_arrayfield', 'spect_binned')

    filter_max_n = dataset_config.get('filter_max_n', 0)
    filter_max_mass = dataset_config.get('filter_max_mass', 0)
    filter_bond_max_n = dataset_config.get('filter_bond_max_n', 0)

    filter_max_vert_subsets_n = dataset_config.get('filter_max_vert_subsets_n', 0)
    random_sample_vert_subsets = dataset_config.get('random_sample_vert_subsets', True)

    d['rdmol'] = d.rdmol.apply(Chem.Mol)

    if filter_max_n > 0:
        d['atom_n'] = d.rdmol.apply(lambda m: m.GetNumAtoms())

        print("filtering for atom max_n <=", filter_max_n, " from", len(d))
        d = d[d.atom_n <= filter_max_n]
        print("after filter length=", len(d))

    if filter_bond_max_n > 0:
        d['bond_n'] = d.rdmol.apply(lambda m: m.GetNumBonds())

        print("filtering for bond max_n <=", filter_bond_max_n, " from", len(d))
        d = d[d.bond_n <= filter_bond_max_n]
        print("after filter length=", len(d))

    if filter_max_mass > 0 :
        d['mass'] = d.rdmol.apply(Chem.Descriptors.ExactMolWt)
        d = d[d.mass <= filter_max_mass]

    cv_fp_field = dataset_config.get('cv_fp_field', 'morgan4_crc32')
    d_phase = d.apply(lambda row : cv_splitter.get_phase(row.rdmol, 
                                                         row[cv_fp_field]), 
                      axis=1)

    df = d[d_phase == phase]

    return df.to_dict('records')

def create_checkpoint_func(every_n, filename_str):
    def checkpoint(epoch_i, net, optimizer):
        if epoch_i > every_n:
            if epoch_i % every_n > 0:
                return {}
        else:
            if epoch_i % 5 != 0:
                return {}
                
        checkpoint_filename = filename_str.format(epoch_i = epoch_i)
        t1 = time.time()
        torch.save(net.state_dict(), checkpoint_filename + ".state")
        torch.save(net, checkpoint_filename + ".model")
        t2 = time.time()
        return {'savetime' : (t2-t1)}
    return checkpoint

def run_epoch(
        net, optimizer, criterion, dl, 
        pred_only = False, USE_CUDA=True,
        return_pred = False, desc="train", 
        print_shapes=False, progress_bar=True, 
        writer=None, epoch_i=None, res_skip_keys= [],
        clip_grad_value = None, scheduler=None,
        accumulate_steps=1,
        profile=False,
        automatic_mixed_precision=False,
        postprocess_batch_fn=None,
        return_subset_fields=False, 
):
    # if profile is set, we only run a single epoch
    if profile:
        import nvidia_dlprof_pytorch_nvtx as nvtx
        
        nvtx.init(enable_function_stack=True)
        profile_context = torch.autograd.profiler.emit_nvtx
    else:
        import contextlib
        profile_context = contextlib.suppress
    
    # if AMP is on, we need to scale gradients + do autocasting
    if automatic_mixed_precision:
        scaler = torch.cuda.amp.GradScaler()
        autocast_context = torch.cuda.amp.autocast
    else:
        import contextlib
        autocast_context = contextlib.suppress


    t1_total= time.time()

    ### DEBUGGING we should clean this up
    MAX_N = 64

    if not pred_only:
        net.train()
        optimizer.zero_grad()
        torch.set_grad_enabled(True)
    else:
        net.eval()
        if optimizer is not None:
            optimizer.zero_grad()
        torch.set_grad_enabled(False)

    accum_pred = []
    extra_loss_fields = {}

    running_loss = 0.0
    total_points = 0
    total_compute_time = 0.0
    if progress_bar:
        iterator =  tqdm(enumerate(dl), total=len(dl), desc=desc, leave=False)
    else:
        iterator = enumerate(dl)

    input_row_count = 0
    with profile_context():
        for i_batch, batch in iterator:
            
            t1 = time.time()
            if print_shapes:
                for k, v in batch.items():
                    print("{}.shape={}".format(k, v.shape))



            for k, v in batch.items():
                # hacky handling of sparse data
                if 'sparse' in k:
                    batch[k] = torch.sparse.FloatTensor(
                        v['inds'],
                        v['vals'],
                        v['shape'],
                    )
            if postprocess_batch_fn is not None:
                batch = postprocess_batch_fn(batch)
            batch_t = {k : util.move(v, USE_CUDA) for k, v in batch.items()}

            # with torch.autograd.detect_anomaly():
            # for k, v in batch_t.items():
            #     assert not torch.isnan(v).any()

            # TEST: automatic mixed-precision
            with autocast_context():
                res = net(**batch_t)

                input_mask_t = batch_t['input_mask']
                input_idx_t = batch_t['input_idx']
                if return_pred:
                    accum_pred_val = {}
                    if isinstance(res, dict):
                        for k, v in res.items():
                            if k not in res_skip_keys:
                                if isinstance(res[k], torch.Tensor):
                                    accum_pred_val[k] = res[k].cpu().detach().numpy()
                    else:
                        
                        accum_pred_val['res'] = res.cpu().detach().numpy()
                    # accum_pred_val['vert_pred_mask'] = vert_pred_mask_batch_t.cpu().detach().numpy()
                    # accum_pred_val['vert_pred'] = vert_pred_batch_t.cpu().detach().numpy()
                    # accum_pred_val['edge_pred_mask'] = edge_pred_mask_batch_t.cpu().detach().numpy()
                    # accum_pred_val['edge_pred'] = edge_pred_batch_t.cpu().detach().numpy()
                    accum_pred_val['input_idx'] = input_idx_t.cpu().detach().numpy().reshape(-1, 1)
                    accum_pred_val['input_mask'] = input_mask_t.cpu().detach().numpy()
                    accum_pred_val['true_spect'] =  batch_t['spect'].cpu().detach().numpy()
                    
                    # extra fields
                    for k, v in batch.items():
                        if k.startswith("passthrough_"):
                            accum_pred_val[k] = v.cpu().detach().numpy()
                    
                    # more_extra fields
                    for k, v in batch.items():
                        if k in ['formulae_features', 'formulae_masses', 'vert_element_oh']:
                            

                            accum_pred_val[k] = v.cpu().detach().numpy()
                        if return_subset_fields:
                            if k in ['atom_subsets', 'atom_subsets_peaks_mass_idx', 'atom_subsets_peaks_intensity']:
                                accum_pred_val[k] = v.cpu().detach().numpy()
                                
                    accum_pred.append(accum_pred_val)
                loss_dict = {}
                if criterion is None:
                    loss = 0.0
                else:
                    if 'spect_peak_prob' in batch:
                        loss = criterion(res, batch_t['spect_peak_prob'],
                                        batch_t['spect_peak_mass'],
                                        input_mask_t)
                    else:
                        loss = criterion(res, batch_t['spect'], 
                                        input_mask_t)
                    if isinstance(loss, dict):
                        loss_dict = loss
                        loss = loss_dict['loss']

            if not pred_only:
                if automatic_mixed_precision:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # for n, p in net.named_parameters():
                #     if 'weight' in n:
                #         writer.add_scalar(f"grads/{n}", torch.max(torch.abs(p.grad)), epoch_i)

                if clip_grad_value is not None:
                    nn.utils.clip_grad_value_(net.parameters(), clip_grad_value)

                # avg_grads = {}
                # for n, p in net.named_parameters():
                #     if(p.requires_grad) and ("bias" not in n):
                    
                #         avg_grads[n] = p.grad.detach().cpu().numpy()
                # pickle.dump(avg_grads, open("test.pickle", 'wb'))

                # apply gradients every `accumulate_steps` batches
                # TODO: it is good practice to do zero_grad => backward => step
                # but can we do this here? used to be after L426
                # allow the gradients to accumulate for `accumulate_steps` steps
                if accumulate_steps == 1 or (
                    i_batch > 0 and (i_batch - 1) % accumulate_steps == 0
                ):
                    if automatic_mixed_precision:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                        optimizer.zero_grad()

            train_points = batch['input_mask'].shape[0]
            if criterion is not None:
                running_loss += loss.item() * train_points
                for k, v in loss_dict.items():
                    extra_loss_fields[k] = extra_loss_fields.get(k, 0) + v.item() * train_points
                
            total_points += train_points

            t2 = time.time()
            total_compute_time += (t2-t1)

            input_row_count += batch['adj'].shape[0]

            # if scheduler is not None:
            if i_batch > 0 and (i_batch - 1) % accumulate_steps == 0 and scheduler is not None:
                scheduler.step()

    t2_total = time.time()
    
    #print('running_loss=', running_loss)
    total_points = max(total_points, 1)
    res =  {'timing' : 0.0, 
            'running_loss' : running_loss, 
            'total_points' : total_points, 
            'mean_loss' : running_loss / total_points,
            'runtime' : t2_total-t1_total, 
            'compute_time' : total_compute_time, 
            'run_efficiency' : total_compute_time / (t2_total-t1_total), 
            'pts_per_sec' : input_row_count / (t2_total-t1_total), 
            }


    for elf, v in extra_loss_fields.items():
        #print(f"extra loss fields {elf} = {v}")
        res[f'loss_total_{elf}'] = v
        res[f'loss_mean_{elf}'] = v/total_points

    if return_pred:
        keys = accum_pred[0].keys()
        for k in keys:
            accum_pred_v = np.vstack([a[k] for a in accum_pred])
            res[f'pred_{k}'] = accum_pred_v
            
    return res

def generic_runner(
    net, optimizer, scheduler, criterion, 
    dl_train, dl_test, 
    MAX_EPOCHS=1000, 
    epochs_trained=0,
    USE_CUDA=True, use_std=False, 
    writer=None, validate_funcs = None, 
    checkpoint_func = None, prog_bar=True,
    clip_grad_value = None, VALIDATE_EVERY = 1,
    accumulate_steps=1,
    profile=False,
    automatic_mixed_precision=False,
):
    # loss_scale = torch.Tensor(loss_scale)
    # std_scale = torch.Tensor(std_scale)

    res_skip_keys = ['g_in', 'g_decode']

    if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
        per_batch_scheduler = scheduler
    else:
        per_batch_scheduler = None

    for epoch_i in tqdm(range(epochs_trained, MAX_EPOCHS)):

        running_loss = 0.0
        total_compute_time = 0.0
        t1_total = time.time()

        net.train()
        t1_train = time.time()
        train_res = run_epoch(
            net, optimizer, criterion, dl_train, 
            pred_only = False, USE_CUDA=USE_CUDA, 
            return_pred=True, progress_bar=prog_bar,
            desc='train', writer=writer, epoch_i=epoch_i, 
            res_skip_keys = res_skip_keys,
            clip_grad_value = clip_grad_value,
            scheduler=per_batch_scheduler, 
            accumulate_steps=accumulate_steps,
            profile=profile,
            automatic_mixed_precision=automatic_mixed_precision,
        )
        t2_train = time.time()
        [v(train_res, "train_", epoch_i) for v in validate_funcs]
        t3_train_validate_func = time.time()
        #print(f"train epoch: {(t2_train - t1_train):3.1f}s, vals: {(t3_train_validate_func - t2_train):3.1f}s")
        
        if epoch_i % VALIDATE_EVERY == 0:
            net.eval()
            t1_validate = time.time()
            test_res = run_epoch(net, optimizer, criterion, dl_test, 
                                 pred_only = True, USE_CUDA=USE_CUDA, 
                                 progress_bar=prog_bar, 
                                 return_pred=True, desc='validate', 
                                 res_skip_keys=res_skip_keys,
                                 automatic_mixed_precision=automatic_mixed_precision,
                                 )
            t2_validate = time.time()
            [v(test_res, "validate_", epoch_i) for v in validate_funcs]
            t3_validate_func = time.time()
            #print(f"validate epoch: {(t2_validate - t1_validate):3.1f}s, vals: {(t3_validate_func - t2_validate):3.1f}s")
        
        if checkpoint_func is not None:
            t1_checkpoint = time.time()
            checkpoint_func(epoch_i = epoch_i, net =net, optimizer=optimizer)
            t2_checkpoint = time.time()
            #print(f"checckpoint {(t2_checkpoint - t1_checkpoint):3.1f} sec")
        if scheduler is not None and (per_batch_scheduler is None):
            scheduler.step()

class PredModel(object):
    def __init__(
        self,
        meta_filename,
        checkpoint_filename,
        USE_CUDA=False, 
        data_parallel=False,
        featurize_config_update={},
    ):
        meta = pickle.load(open(meta_filename, 'rb'))
        self.meta = meta 

        self.USE_CUDA = USE_CUDA

        if self.USE_CUDA:
            net = torch.load(checkpoint_filename)
        else:
            logger.warning(f'NOT USING GPU!')
            net = torch.load(checkpoint_filename, 
                             map_location=lambda storage, loc: storage)

        if hasattr(net, 'module'):
            # we originally saved model w nn.DataParallel wrapper
            # can we get rid of it?
            net = net.module
        
        if data_parallel:
            net = nn.DataParallel(net)

        self.net = net
        self.net.eval()
        
        self.spectrum_bins = featurize.msutil.binutils.create_spectrum_bins(**self.meta['spectrum_bin_config'])

        meta_config = self.meta['featurize_config']
        util.recursive_update(meta_config, featurize_config_update)
        self.featurizer_config = meta_config

    def pred(self, molecules,
             output_hist_bins = False,
             output_exact_peaks = False,
             output_formulae_probs = False,
             output_vert_subsets = False,
             batch_size = 32,
             progress_bar= False, MIN_PROB = 1e-6, 
             normalize_pred=True,
             dataloader_config = {}, 
             benchmark_dataloader=False,
    ):
        """
        

        output_hist_bins: Output the binned probs per the spectrum bin config. This is a 
        sparse output of mz, intensity

        output_exact_peaks: output the exact mass peaks computed from the subset exact peak
        calcs. Note that the same peak can appear multiple times in this dataset. 

        output_vert_subsets: output the implicated vertex subsets along with the probability
        of each subset. 
        """

        output = {}

        bin_centers = self.spectrum_bins.get_bin_centers()
        
        MOL_N = len(molecules)

        ds = dataset.WrapperDataset(molecules, self.spectrum_bins, self.featurizer_config)
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                         **dataloader_config)
        
        batch_iterator = dl
        if progress_bar:
            batch_iterator = tqdm(batch_iterator, unit_scale=batch_size)
        
        for batch_i, features_batch_tensor in enumerate(batch_iterator):

            if self.USE_CUDA:
                features_batch_tensor = {k: v.to('cuda') for k, v in features_batch_tensor.items()}

            if not benchmark_dataloader:
                # actually run the model
                res = self.net(**features_batch_tensor)

                if output_hist_bins:
                    if 'pred_binned' not in output:
                        output['pred_binned'] = []
                    
                    for p in res['spect']:
                        p = p.detach().cpu().numpy()
                        if normalize_pred:
                            p = p / max(np.sum(p), 1e-6)
                        nz_idx = p > MIN_PROB
                        nz_bin_centers = bin_centers[nz_idx]
                        nz_p = p[nz_idx]
                            
                        output['pred_binned'].append(np.stack([nz_bin_centers, nz_p], -1).astype(np.float32))
                
                if output_formulae_probs:
                    output['formulae_probs'] = res['formulae_probs']

                if output_exact_peaks:
                    if 'pred_exact' not in output:
                        output['pred_exact'] = []

                    formulae_peaks = features_batch_tensor['formulae_peaks'].cpu().numpy()
                    atom_subsets_formula_idx = features_batch_tensor['atom_subsets_formula_idx'].cpu().numpy()
                    subset_probs = res['subset_probs'].detach().cpu().numpy()

                    for batch_item in range(subset_probs.shape[0]):
                        peaks = formulae_peaks[batch_item]
                        idx = atom_subsets_formula_idx[batch_item]
                        probs = subset_probs[batch_item]
                        subset_peaks_intensities = peaks[idx]
                        subset_peaks = subset_peaks_intensities[:, :, 0]
                        subset_peaks_intensities = subset_peaks_intensities[:, :, 1]

                        total_intensities = subset_peaks_intensities * probs.reshape(-1, 1)

                        nz_idx = subset_peaks.flatten() > 0
                        nz_idx = nz_idx & (total_intensities.flatten() > MIN_PROB)
                        nz_intensities = total_intensities.flatten()[nz_idx]
                        if normalize_pred:
                            nz_intensities = nz_intensities / max(np.sum(nz_intensities), 1e-6)
                        output['pred_exact'].append(np.stack([subset_peaks.flatten()[nz_idx],
                                                            nz_intensities], -1))
                                                            
                if output_vert_subsets:
                    raise NotImplemented("Pred vertex subsets not implemented yet")

                # Memory cleanup for cuda. Yes, this is horrible.
                for t in features_batch_tensor.values():
                    del t
                del features_batch_tensor
                del res
            
            else:
                # do nothing, just load the batch
                pass

        return output
            
def rand_dict(d):
    p = {}
    for k, v in d.items():
        if isinstance(v, list):
            p[k] = v[np.random.randint(len(v))]
        else:
            p[k] = v
    return p



def create_optimizer(opt_params, net_params):
    opt_direct_params = {}
    optimizer_name = opt_params.get('optimizer', 'adam') 
    if optimizer_name == 'adam':
        for p in ['lr', 'amsgrad', 'eps', 'weight_decay', 'momentum']:
            if p in opt_params:
                opt_direct_params[p] = opt_params[p]

        optimizer = torch.optim.Adam(net_params, **opt_direct_params)
    elif optimizer_name == 'adamax':
        for p in ['lr', 'eps', 'weight_decay', 'momentum']:
            if p in opt_params:
                opt_direct_params[p] = opt_params[p]

        optimizer = torch.optim.Adamax(net_params, **opt_direct_params)
        
    elif optimizer_name == 'adagrad':
        for p in ['lr', 'eps', 'weight_decay', 'momentum']:
            if p in opt_params:
                opt_direct_params[p] = opt_params[p]

        optimizer = torch.optim.Adagrad(net_params, **opt_direct_params)
        
    elif optimizer_name == 'rmsprop':
        for p in ['lr', 'eps', 'weight_decay', 'momentum']:
            if p in opt_params:
                opt_direct_params[p] = opt_params[p]

        optimizer = torch.optim.RMSprop(net_params, **opt_direct_params)
        
    elif optimizer_name == 'sgd':
        for p in ['lr', 'momentum']:
            if p in opt_params:
                opt_direct_params[p] = opt_params[p]

        optimizer = torch.optim.SGD(net_params, **opt_direct_params)

    return optimizer


def create_loss(loss_params, USE_CUDA):
    loss_name = loss_params['loss_name']
    criterion = eval(loss_name)(**loss_params)
    return criterion


def create_kl_validate_func(config, writer):
    def val_func(input_res, prefix, epoch_i): # val, mask, truth):
        pred_spect = input_res['pred_spect']

        for field in ['mean_loss', 'timing']:
            writer.add_scalar(f"{prefix}/{field}", input_res[field], epoch_i)
        for k, v in input_res.items():
            if k.startswith("loss_"):

                writer.add_scalar(f"{prefix}/{k}", v, epoch_i)

    return val_func


def create_save_val_func(checkpoint_base_dir):
    def val_func(input_res, prefix, epoch_i): # val, mask, truth):
        #print("input_res.keys()=", list(input_res.keys()))

        if epoch_i > 10:
            if epoch_i % 10 != 0:
                return
            if epoch_i > 100:
                if epoch_i % 50 != 0:
                    return
            

        outfile = checkpoint_base_dir + f".{prefix}.{epoch_i:08d}.output"

        with open(outfile, 'wb') as fid:
            pickle.dump(input_res, fid)
            fid.close()
            
    return val_func



def create_dotprod_validate_func(config, writer):
    def val_func(input_res, prefix, epoch_i): # val, mask, truth):
        pred_spect = input_res['pred_spect']
        true_spect = input_res['pred_true_spect']


        exp_pred = config.get("exp_pred", True)
        exp_true = config.get("exp_true", True)

        if exp_pred:
            pred_spect = np.exp(pred_spect)
        if config.get("norm_pred", True):
            pred_spect = pred_spect / pred_spect.sum(axis=1).reshape(-1, 1)
            
        if exp_true:
            true_spect = np.exp(true_spect)
        if config.get("norm_true", True):
            true_spect = true_spect / true_spect.sum(axis=1).reshape(-1, 1)


        delta = pred_spect - true_spect
        delta_abs = np.abs(pred_spect - true_spect)

        l1_err = np.mean(delta_abs, axis=0).mean()
        
        writer.add_scalar(f"{prefix}/l1err", l1_err, epoch_i)


        pred_spect_l2norm = pred_spect / np.linalg.norm(pred_spect, axis=1).reshape(-1, 1)
        true_spect_l2norm = true_spect / np.linalg.norm(true_spect, axis=1).reshape(-1, 1)

        N= pred_spect.shape[1]

        for metric_name, mass_pow, intensity_pow in [('dp', 0.5, 0.5),
                                              ('sdp', 3.0, 0.6)]:
            
            mass_val_array = np.arange(N, dtype=np.float64)
            mass_val_array_pow = mass_val_array ** mass_pow


            pred_w = mass_val_array_pow * pred_spect_l2norm**intensity_pow
            true_w = mass_val_array_pow * true_spect_l2norm**intensity_pow

            metric = np.sum(pred_w * true_w, axis=1) / np.linalg.norm(pred_w, axis=1) / np.linalg.norm(true_w, axis=1)
            metric_mean = np.mean(metric)
            writer.add_scalar(f"{prefix}/{metric_name}", metric_mean, epoch_i)


        top_1_correct = np.mean(np.argmax(pred_spect, axis=1) == np.argmax(true_spect, axis=1))
        writer.add_scalar(f"{prefix}/top_1_correct", top_1_correct, epoch_i)


        mass_present_thold = 0.005

        invalid_mass = (pred_spect * (true_spect <= mass_present_thold)).sum(axis=1)

        writer.add_scalar(f"{prefix}/invalid_mass_mean", invalid_mass.mean(), epoch_i)
        try:
            writer.add_histogram(f"{prefix}/invalid_mass", invalid_mass, global_step=epoch_i, bins=np.linspace(0, 1, 21))
        except Exception as e:
            print("exception", str(e))
            pickle.dump({'invalid_mass' : invalid_mass,
                         'pred_spect' : pred_spect,
                         'true_spect' : true_spect}, open("invalid_mass.pickle", 'wb'))
            print("histogram crashed, invalid_mass saved to invalid_mass.pickle")

        writer.add_scalar(f"{prefix}/invalid_mass_max", invalid_mass.max(), epoch_i)
        
        
        
        #     writer.add_scalar(f"{prefix}/{field}", input_res[field], epoch_i)
    return val_func


def create_metadata_validate_func(config, writer):
    def val_func(input_res, prefix, epoch_i): 
        for field in ['runtime' ,
                      'compute_time' ,
                      'run_efficiency',
                      'pts_per_sec']:

            writer.add_scalar(f"{prefix}/{field}", input_res[field], epoch_i)

    return val_func


class SubsetSampler(torch.utils.data.Sampler):
    """
    A dataset sampler for sampling from a subset
    of data each epoch. 
    
    epoch_size: the size of the epoch 
    ds_size: the total size of the dataset
    
    this way you can always have epochs of the same size
    to compare training perf across different datasets. 
    """
    def __init__(self, epoch_size, ds_size, shuffle=False):
        self.epoch_size = epoch_size
        self.ds_size = ds_size
        self.pos = 0
        self.shuffle = shuffle
        self.compute_idx()

    def compute_idx(self):
        if self.shuffle:
            self.idx = np.random.permutation(self.ds_size)
        else:
            self.idx = np.arange(self.ds_size)
            
    def __len__(self):
        return self.epoch_size
    
    def __iter__(self):
        for i in range(self.epoch_size):
            yield self.idx[self.pos]
            self.pos = (self.pos + 1) % self.ds_size

            if self.pos == 0:
                self.compute_idx()
                
if __name__ == "__main__":
    meta_filename = 'checkpoints/jonas_db_debug.33555480.meta'
    checkpoint_filename = 'checkpoints/jonas_db_debug.33555480.00000000.model'

    pm = PredModel(meta_filename, checkpoint_filename, USE_CUDA=True,

    )

    mol1 = Chem.AddHs(Chem.MolFromSmiles("CCCCO"))
    mol2 = Chem.AddHs(Chem.MolFromSmiles("CCCCCCO"))

    res = pm.pred([mol1, mol2]*100, batch_size=8, 
                  output_hist_bins=True,
                  output_exact_peaks=False,
                  progress_bar=True,
                  dataloader_config={
                      'num_workers': 16,
                      'persistent_workers': True,
                  },

    )
    for k, v in res.items():
        if v is None:
            print("nothing for", k)
        else:
            print(k, len(v), v[0].shape, v[0].dtype)
