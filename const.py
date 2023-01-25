import os

# python rassp/forward_evaluate_pipeline.py
# evaluate a trained model against a dataset and produce spectral predictions in `forward.preds`
FORWARD_EVAL_EXPERIMENTS = {
  # evaluating the 1-step demo model against smallmols sample
  'demo': {
    'dataset' : './sample_data/smallmols_cfm_pred_public_sample.parquet',
    'cv_method' : {
      'how': 'morgan_fingerprint_mod', 
      'mod' : 10,
      'test': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    },
    'normalize_pred': True,
    'streaming_save': True,
    'checkpoint': 'checkpoints/demo.first-test.48668593',  # edit this line to the checkpoint that you generate
    'batch_size': 6,
    'epoch': 0,
    'mol_id_type': str,  # either str or int, depending on your input dataset's `mol_id` column dtype
  },
  # evaluating pretrained FormulaNet against smallmols sample
  'demo-eval-best-formulanet': {
    'dataset' : './sample_data/smallmols_cfm_pred_public_sample.parquet',
    'cv_method' : {
      'how': 'morgan_fingerprint_mod', 
      'mod' : 10,
      'test': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    },
    'normalize_pred': True,
    'streaming_save': True,
    'checkpoint': 'models/formulanet',
    'batch_size': 6,
    'epoch': 740,
    'mol_id_type': str,  # either str or int, depending on your input dataset's `mol_id` column dtype
  },
}

# python analysis_pipeline.py
# given a set of spectral predictions + the true predictions, compute metrics over the entire set and produce output in `results.metrics`
DATA_DIR = "."
WORKING_DIR = "results.metrics"
td = lambda x: os.path.join(WORKING_DIR, x)
ANALYSIS_EXPERIMENTS = {
  'demo': {
    'true_spect' : './sample_data/smallmols_cfm_pred_public_sample.parquet',
    'pred_spect' : f'./forward.preds/demo.spect.sqlite',
    'phases': ['train', 'test'],
  },
  'demo-eval-best-formulanet': {
    'true_spect' : './sample_data/smallmols_cfm_pred_public_sample.parquet',
    'pred_spect' : f'./forward.preds/demo-eval-best-formulanet.spect.sqlite',
    'phases': ['train', 'test'],
  },
}

# python library_match_pipeline.py
# given a set of spectral predictions, compute library matching metrics
LIBRARY_MATCH_EXPERIMENTS = {
  'demo': {
    'main_library': './sample_data/smallmols_cfm_pred_public_sample.0.parquet',
    'query_library': './sample_data/smallmols_cfm_pred_public_sample.1.parquet',
    'exp_name': 'demo',
  },
  'demo-eval-best-formulanet': {
    'main_library': './sample_data/smallmols_cfm_pred_public_sample.0.parquet',
    'query_library': './sample_data/smallmols_cfm_pred_public_sample.1.parquet',
    'exp_name': 'demo-eval-best-formulanet',
  },
}