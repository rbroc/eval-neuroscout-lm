from transformers import (GPT2LMHeadModel, GPT2TokenizerFast)
from multiprocessing import Pool
import pandas as pd
import glob
from itertools import product
from lmeval.datasets import NarrativesDataset
from lmeval.engine import StridingForwardLM
import transformers
import argparse
import truecase

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0,
                    help='Which gpu to use')
parser.add_argument('--ctx-length', nargs='+', default=[25],
                    help='Context sizes', type=int)

transformers.logging.set_verbosity(50)

# Define parameters
transcripts = glob.glob('inputs/narratives/gentle/seinfeld/transcript*')
aligned = glob.glob('inputs/narratives/gentle/seinfeld/align.csv')
dataset_files = aligned

model_classes = [GPT2LMHeadModel]
model_ids = ['gpt2']
tokenizer_classes = [GPT2TokenizerFast]

model_parameters = list(zip(model_classes, 
                            model_ids, 
                            tokenizer_classes))
parameters = list(product(dataset_files, 
                          model_parameters))
parameters = [(i[0], *i[1]) for i in parameters]
    

# Define functions
def _make_dataset_id(datafile):
    ''' Extracts compact id'''
    ds_name_splits = datafile.split('/')
    narrative = ds_name_splits[3]
    ds_type = ds_name_splits[-1].split('.')[0]
    ds_id = '_'.join([narrative, ds_type])
    return ds_id


def _validate(datafile, 
              model_class, 
              model_id, 
              tokenizer_class, 
              ctx_length, 
              gpu=0,
              case_sensitive=True):
    ''' Main functon to run the validation'''
    files = glob.glob('outputs/narratives/*')
    n_files = len(files)
    dataset_name = _make_dataset_id(datafile)
    if '/' in model_id:
        model_id_log = model_id.split('/')[1]
    else:
        model_id_log = model_id
    if case_sensitive:
        log_id = f'{dataset_name}_{model_id_log}_{ctx_length}_cs.txt'    
    else:
        log_id = f'{dataset_name}_{model_id_log}_{ctx_length}_nocs.txt'          
    log_path = f'outputs/narratives/{log_id}' # sherlock
    if log_path not in files:
        tokenizer = tokenizer_class.from_pretrained(model_id)
        model = model_class.from_pretrained(model_id).to(device=f'cuda:{str(gpu)}')
        data = NarrativesDataset(datafile, dataset_name, case_sensitive=case_sensitive)
        engine = StridingForwardLM(context_length=ctx_length)
        result = engine.run(data, tokenizer, model, model_id, gpu)
        result.to_csv(log_path, sep='\t')
        print(f'{n_files+1} out of {len(parameters)}')
        return result
    

    # Run
if __name__=='__main__':
    args = parser.parse_args()
    pars = product(parameters, args.ctx_length)
    pars = [(*p[0], p[1]) for p in pars]
    for p in pars:
        _validate(*p, args.gpu)
