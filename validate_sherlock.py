from transformers import (GPT2LMHeadModel, GPT2TokenizerFast)
from multiprocessing import Pool
import pandas as pd
import glob
from itertools import product
from lmeval.datasets import NeuroscoutDataset
from lmeval.engine import StridingForwardLM
import transformers
import argparse


# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0,
                    help='Which gpu to use')
parser.add_argument('--ctx-length', nargs='+', default=[25],
                    help='Context sizes', type=int)

transformers.logging.set_verbosity(50)

# Define parameters
transcripts = glob.glob('inputs/sherlock/*srt')
aligned = glob.glob('inputs/sherlock/*txt')
dataset_files = transcripts + aligned

model_classes = [GPT2LMHeadModel]
model_ids = ['gpt2']
tokenizer_classes = [GPT2TokenizerFast]

model_parameters = list(zip(model_classes, 
                            model_ids, 
                            tokenizer_classes))
parameters = list(product(dataset_files, 
                          model_parameters))
parameters = [(i[0], *i[1]) for i in parameters]


def _validate(datafile, 
              model_class, 
              model_id, 
              tokenizer_class, 
              ctx_length, 
              gpu=0):
    ''' Main functon to run the validation'''
    files = glob.glob('outputs/sherlock/*')
    n_files = len(files)
    if '/' in model_id:
        model_id_log = model_id.split('/')[1]
    else:
        model_id_log = model_id
    if 'srt' in datafile:
        log_id = f'sherlock_transcript_{model_id_log}_{ctx_length}.txt'  
    else:
        log_id = f'sherlock_align_{model_id_log}_{ctx_length}.txt'  
    log_path = f'outputs/sherlock/{log_id}' # sherlock
    if log_path not in files:
        tokenizer = tokenizer_class.from_pretrained(model_id)
        model = model_class.from_pretrained(model_id).to(device=f'cuda:{str(gpu)}')
        if 'srt' in datafile:
            data = NeuroscoutDataset(datafile, 'sherlock_transcript')
        else:
            data = NeuroscoutDataset(datafile, 'sherlock_align')
        data.text = data.text.replace('That...', 'That... ')
        data.text = data.text.replace(' - ', '- ')
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
