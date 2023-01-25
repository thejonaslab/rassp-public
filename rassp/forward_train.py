import click
import itertools
import json 
import numpy as np
import os
import pandas as pd
import pickle
import pickle
import re
import resource as res
import sys
import time
import torch
import yaml

from glob import glob
from natsort import natsorted
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors as rdMD
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from rassp import dataset
from rassp import netutil
from rassp import util
from rassp.msutil import binutils
from rassp.model import nets
from rassp.model import formulaenets
from rassp.model import subsetnets
from rassp.model import losses

USE_CUDA = os.environ.get('USE_CUDA', False)
DATALOADER_PIN_MEMORY = False

def train(
    exp_config_name,
    exp_config,
    exp_extra_name="",
    USE_CUDA=USE_CUDA,
    exp_config_filename=None,
    add_timestamp=True,
    profile=False,
    checkpoint_dir='checkpoints',
    checkpoint_every=50,
):
    """
    exp_config_name: str
        Base name of experiment yaml
    exp_config: dict
        Experiment config dictionary
    exp_extra_name: str
        Extra name to append to basename
        Full experiment name is {exp_config_name}.{exp_extra_name}.{time}
    exp_config_filename: str
        Path to experiment config we loaded
    profile: bool
        Whether or not to run profiler
    """
    EXP_NAME = exp_config_name
    if exp_extra_name is not None and len(exp_extra_name ) > 0:
        EXP_NAME += ".{}".format(exp_extra_name)

    if add_timestamp:
        MODEL_NAME = "{}.{:08d}".format(EXP_NAME,int(time.time() % 1e8))
    else:
        MODEL_NAME = EXP_NAME

    cluster_config = None
    if exp_config.get('cluster_config', None) is not None:
        cluster_config = exp_config['cluster_config']
    
    data_dir = None
    if cluster_config is not None:
        if cluster_config['using_cluster']:
            data_dir = cluster_config['cluster_data_dir']
        else:
            data_dir = cluster_config['data_dir']
    assert data_dir is not None

    if checkpoint_dir is not None:
        pass
    elif exp_config.get('checkpoint_dir', None) is not None:
        checkpoint_dir = exp_config['checkpoint_dir']
    elif cluster_config is not None:
        # try to get using_cluster and if so checkpoint dir there
        if cluster_config.get('using_cluster'):
            checkpoint_dir = cluster_config['checkpoint_dir']
            print(f'USING THE CLUSTER CHECKPOINT DIR: {checkpoint_dir}')
    if checkpoint_dir is None:
        checkpoint_dir = CHECKPOINT_DIR
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    print(f'using checkpoint_dir: {checkpoint_dir}')

    DATALOADER_NUM_WORKERS = exp_config.get('DATALOADER_NUM_WORKERS', 0)
    if cluster_config is not None and cluster_config.get('DATALOADER_NUM_WORKERS'):
        DATALOADER_NUM_WORKERS = cluster_config.get('DATALOADER_NUM_WORKERS')
        print(f'USING N_WORKERS={DATALOADER_NUM_WORKERS}')
    
    CHECKPOINT_BASENAME = os.path.join(checkpoint_dir, MODEL_NAME)
            
    with open(os.path.join(checkpoint_dir, 
                           "{}.yaml".format(MODEL_NAME,)), 'w') as fid:
        fid.write(open(exp_config_filename, 'r').read())
        
    np.random.seed(exp_config['seed'])

    MAX_N = exp_config['tgt_max_n']

    spect_bin_config = binutils.create_spectrum_bins(**exp_config['bin_config'])

    BATCH_SIZE = exp_config['batch_size']

    featurize_config_update = exp_config['featurize_config']
    featurize_config = netutil.DEFAULT_FEATURIZE_CONFIG
    util.recursive_update(featurize_config,
                          featurize_config_update)

    pred_config_update = exp_config['pred_config']
    pred_config = netutil.DEFAULT_PRED_CONFIG
    util.recursive_update(pred_config,
                          pred_config_update)

    exp_data = exp_config['exp_data']
    cv_func = netutil.CVSplit(**exp_data['cv_split'])

    t1_dataset = time.time()
    passthrough_config =  exp_config.get('passthroughs', {})
    datasets = {}
    for ds_config_i, dataset_config in enumerate(exp_data['data']):

        # override db_filename with data_dir cluster_config provided...
        dataset_config['db_filename'] = os.path.join(
            data_dir,
            dataset_config['db_filename']
        )
        assert os.path.exists(dataset_config['db_filename']), \
            f'provided db path {dataset_config["db_filename"]}did not exist...'

        basename = os.path.basename(dataset_config['db_filename'])
        ext = basename.split('.')[-1]
        if ext == 'db':
            ds = dataset.make_db_dataset(
                dataset_config,
                spect_bin_config, 
                featurize_config,
                pred_config,
                cv_func
            )
        elif ext == 'parquet' or ext == 'pq':
            ds = dataset.load_pq_dataset(
                dataset_config,
                spect_bin_config,
                featurize_config,
                pred_config,
            )
        else:
            assert False, f'Unsupported dataset extension {ext}, received {basename}'

        phase = dataset_config['phase']
        if phase not in datasets:
            datasets[phase] = []
        datasets[phase].append(ds)
        
    ds_train = datasets['train'][0] if len(datasets['train']) == 1 else torch.utils.data.ConcatDataset(datasets['train'])
    ds_test = datasets['test'][0] if len(datasets['test']) == 1 else torch.utils.data.ConcatDataset(datasets['test'])

    print("we are training with", len(ds_train))
    print("we are testing with", len(ds_test))

    dataloader_name = exp_config.get("dataloader_func",
                                 'torch.utils.data.DataLoader')

    dataloader_creator = eval(dataloader_name)

    epoch_size = exp_config.get('epoch_size', 8192)
    train_sampler = netutil.SubsetSampler(epoch_size, len(ds_train), shuffle=True)
    DATALOADER_PERSISTENT_WORKERS = False
    if DATALOADER_NUM_WORKERS == 0:
        DATALOADER_PERSISTENT_WORKERS = False
    # DATALOADER_PERSISTENT_WORKERS = False
    # if dataset_hparams.get('collate_fn', None) is not None:
    #     collate_fn = eval(dataset_hparams.get('collate_fn'))
    # else:
    collate_fn = None

    dl_train = dataloader_creator(
        ds_train,
        batch_size=BATCH_SIZE, 
        pin_memory=DATALOADER_PIN_MEMORY,
        sampler=train_sampler, 
        num_workers=DATALOADER_NUM_WORKERS,
        persistent_workers = DATALOADER_PERSISTENT_WORKERS,
        timeout=60 * 4,
        collate_fn=collate_fn,
    )
    
    test_sampler = netutil.SubsetSampler(epoch_size, len(ds_test), shuffle=True)
    dl_test = dataloader_creator(
        ds_test, batch_size=BATCH_SIZE, 
        pin_memory=DATALOADER_PIN_MEMORY,
        sampler=test_sampler, 
        num_workers=DATALOADER_NUM_WORKERS,
        persistent_workers = DATALOADER_PERSISTENT_WORKERS,
        timeout=60 * 4,
        collate_fn=collate_fn,
    )
    
    t2_dataset = time.time()
    print(f"it took {(t2_dataset-t1_dataset):3.1f}s to load the data")

    net_params = exp_config['net_params']
    net_name = exp_config['net_name']

    n_atom_feats = ds_test[0]['vect_feat'].shape[-1]
    net_params['g_feature_n'] = n_atom_feats

    net_params['GS'] = ds_test[0]['adj'].shape[0]
    net_params['spect_bin'] = spect_bin_config

    print(net_params)

    torch.manual_seed(exp_config['seed'])
    net = eval(net_name)(**net_params)
    torch.manual_seed(exp_config['seed'] + os.getpid())

    def get_module_nparams(module):
        n = 0
        for p in module.parameters():
            n += np.prod(p.shape)
        return n
    
    if hasattr(net, 'gml'):
        print(f'GCN module had {get_module_nparams(net.gml)} nparams')
    if hasattr(net, 'spect_out'):
        print(f'Spectral module had {get_module_nparams(net.spect_out)} nparams')
    print(f'Net overall had {get_module_nparams(net)} nparams')
    
    epochs_trained = 0
    ckpt_file = None
    if exp_config.get('attempt_load_recent_checkpoint') is not None:
        # attempt to load a recent checkpoint with same EXP_NAME first...
        print('looking to load a recent checkpoint...')
        state_files = glob(os.path.join(checkpoint_dir, f'{EXP_NAME}*.state'))
        if len(state_files) > 0:
            print(f'found {len(state_files)}, choosing the most recent one')
            ckpt_file = natsorted(state_files)[-1]
            epochs_trained = int(ckpt_file.split('.')[-2])
            print(f'using the one trained up to {epochs_trained} epochs')
    elif exp_config.get("load_checkpoint", None) is not None:
        # if we have specified a checkpoint to load, load it
        ckpt_file = exp_config['load_checkpoint']

    if ckpt_file is not None:
        print("LOADING CHECKPOINT", ckpt_file)

        # make sure to handle DataParallel case that prepends `module` to each tensor name
        state_dict = torch.load(ckpt_file)

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if len(k) >= 7 and k[:7] == 'module.':
                k = k[7:]
            new_state_dict[k] = v
        net.load_state_dict(new_state_dict)
    else:
        print("NOT LOADING FROM ANY CHECKPOINT")

    # net = net.to('cuda')
    net = util.move(net, USE_CUDA)
    if torch.cuda.device_count() > 1:
        print('USING DATAPARALLEL FOR MULTI GPU TRAINING')
        net = nn.DataParallel(net)

    print("NOT USING CUSTOM COLLATE_FN")

    loss_params = exp_config['loss_params']
    criterion = netutil.create_loss(loss_params, USE_CUDA)
    
    opt_params = exp_config['opt_params']
    optimizer = netutil.create_optimizer(opt_params, net.parameters())

    if 'freeze_control' in exp_config:
        # by default if fine tuning is on it will freeze everything
        for freeze_re in exp_config['freeze_control'].get('freeze_layer_regexs', [r'.+']):
            for n, p in net.named_parameters():
                if re.match(freeze_re, n):
                    p.requires_grad = False
                    
        for unfreeze_re in exp_config['freeze_control'].get('unfreeze_layer_regexs', []):
            for n, p in net.named_parameters():
                if re.match(unfreeze_re, n):
                    p.requires_grad = True
        
    for n, p in net.named_parameters():
        no_grad_str = "" if p.requires_grad else "[frozen]"
        print(f"{n:<50.50}  {str(tuple(p.shape)):<12.12} {no_grad_str}")
        
    scheduler_name = opt_params.get('scheduler_name', None)
    scheduler = None
    if scheduler_name == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    gamma= opt_params['scheduler_gamma'], 
                                                    step_size=opt_params['scheduler_step_size'])


    checkpoint_filename = CHECKPOINT_BASENAME + ".{epoch_i:08d}"
    print("checkpoint:", checkpoint_filename)

    checkpoint_every_n_epochs = exp_config.get('checkpoint_every_n_epochs', checkpoint_every)
    if cluster_config is not None and cluster_config.get('checkpoint_every_n_epochs') is not None:
        checkpoint_every_n_epochs = int(cluster_config.get('checkpoint_every_n_epochs'))
    checkpoint_func = netutil.create_checkpoint_func(checkpoint_every_n_epochs, checkpoint_filename)

    save_output_func = netutil.create_save_val_func(CHECKPOINT_BASENAME, )
    
    TENSORBOARD_DIR = exp_config.get('tblogdir', f"tensorboard.logs")
    writer = SummaryWriter("{}/{}".format(TENSORBOARD_DIR, MODEL_NAME))

    validate_config = exp_config.get('validate_config', {})
    validate_funcs = [save_output_func]
    for k, v in validate_config.items():
        if k == 'shift_uncertain_validate':
            validate_func = netutil.create_shift_uncertain_validate_func(v, writer)
        elif k == 'kl_validate':
            validate_func = netutil.create_kl_validate_func(v, writer)
        elif k == 'dotprod_validate':
            validate_func = netutil.create_dotprod_validate_func(v, writer)
        elif k == 'metadata':
            validate_func = netutil.create_metadata_validate_func(v, writer)

        validate_funcs.append(validate_func)
        

    metadata = {#'dataset_hparams' : dataset_hparams,
        'featurize_config': featurize_config,
        'pred_config' : pred_config, 
        'net_params' : net_params, 
        'opt_params' : opt_params, 
        'exp_data' : exp_data,
        'spectrum_bin_config' : spect_bin_config.config(), 
        #'meta_infile' : meta_infile, 
        'exp_config' : exp_config, 
        'validate_config': validate_config,
        'passthrough_config' : passthrough_config, 
        'max_n' : MAX_N,
        'net_name' : net_name, 
        'batch_size' : BATCH_SIZE, 
        'loss_params' : loss_params}
    

    print("MODEL_NAME=", MODEL_NAME)
    pickle.dump(metadata, 
                open(CHECKPOINT_BASENAME + ".meta", 'wb'))
    
    if profile:
        n_epochs = 1
    else:
        n_epochs = exp_config['max_epochs']

    netutil.generic_runner(
        net, optimizer, scheduler, criterion, 
        dl_train, dl_test, 
        MAX_EPOCHS=n_epochs,
        epochs_trained=epochs_trained,
        USE_CUDA=USE_CUDA, writer=writer, 
        validate_funcs= validate_funcs, 
        checkpoint_func= checkpoint_func,
        clip_grad_value=exp_config.get('clip_grad_value', None),
        VALIDATE_EVERY = exp_config.get('validate_every', 1),
        accumulate_steps=exp_config.get('accumulate_steps', 1),
        profile=profile,
        automatic_mixed_precision=exp_config.get('automatic_mixed_precision', False),
    )


@click.command()
@click.argument('exp_config_name')
@click.argument('exp_extra_name', required=False)
@click.option('--skip-timestamp', default=False, is_flag=True)
@click.option('--profile', default=False, is_flag=True)
def run(exp_config_name, exp_extra_name, skip_timestamp=False, profile=False):
    # raise file limits for pytorch processing
    file_limit_soft, file_limit_hard = res.getrlimit(res.RLIMIT_NOFILE)
    res.setrlimit(res.RLIMIT_NOFILE, (file_limit_hard, file_limit_hard))
    print("file limit is", file_limit_hard)

    exp_config = yaml.load(open(exp_config_name, 'r'), Loader=yaml.FullLoader)
    exp_name = os.path.basename(exp_config_name.replace(".yaml", ""))
    train(exp_name, exp_config, exp_extra_name,
          exp_config_filename=exp_config_name, add_timestamp=not skip_timestamp,
          profile=profile)

if __name__ == "__main__":
    run()
