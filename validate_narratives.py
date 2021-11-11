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
parser.add_argument('--fill-na', type=str, default='unk',
                    help='How to fill na')
parser.add_argument('--ds-type', type=str,
                    help='Which dataset (transcript or aligned)')
parser.add_argument('--ctx-length', nargs='+', default=[25],
                    help='Context sizes', type=int)
parser.add_argument('--case-sensitive', dest='case_sensitive', 
                    action='store_true')

transformers.logging.set_verbosity(50)

# Define functions
def _make_dataset_id(datafile):
    ''' Extracts compact id'''
    ds_name_splits = datafile.split('/')
    narrative = ds_name_splits[3]
    return narrative


def _validate(datafile, 
              model_class, 
              model_id, 
              tokenizer_class, 
              ctx_length, 
              gpu,
              dataset_type,
              case_sensitive,
              fill_na):
    ''' Main function to run the validation'''
    dataset_name = _make_dataset_id(datafile)
    if '/' in model_id:
        model_id_log = model_id.split('/')[1]
    else:
        model_id_log = model_id
    tokenizer = tokenizer_class.from_pretrained(model_id)
    model = model_class.from_pretrained(model_id).to(device=f'cuda:{str(gpu)}')
    data = NarrativesDataset(datafile, 
                             dataset_name, 
                             case_sensitive=case_sensitive,
                             fill_na=fill_na)
    log_id = f'{dataset_name}_{model_id_log}_{ctx_length}_{data.dataset_type}.txt'          
    log_path = f'outputs/narratives_top5/{log_id}'
    engine = StridingForwardLM(context_length=ctx_length)
    result = engine.run(data, 
                        tokenizer, 
                        model, model_id, 
                        gpu)
    result.to_csv(log_path, sep='\t')
    return result
    

    # Run
if __name__=='__main__':
    args = parser.parse_args()
    # Create datasets
    transcripts = glob.glob('inputs/narratives/gentle/*/transcript*')
    aligned = glob.glob('inputs/narratives/gentle/*/align.csv')
    if args.ds_type == 'aligned':
        dataset_files = aligned
    elif args.ds_type == 'transcripts':
        dataset_files = transcripts
    else:
        dataset_files = aligned + transcripts
    model_classes = [GPT2LMHeadModel] #[GPT2LMHeadModel, OpenAIGPTLMHeadModel]
    model_ids = ["gpt2"] #['gpt2', 'openai-gpt']
    tokenizer_classes = [GPT2TokenizerFast] # OpenAIGPTTokenizerFast
    # Set up parameters
    model_parameters = list(zip(model_classes, 
                                model_ids, 
                                tokenizer_classes))
    parameters = list(product(dataset_files, 
                              model_parameters))
    parameters = [(i[0], *i[1]) for i in parameters]
    pars = product(parameters, args.ctx_length)
    pars = [(*p[0], p[1]) for p in pars]
    # Run
    for i, p in enumerate(pars):
        _validate(*p, args.gpu, 
                  case_sensitive=args.case_sensitive,
                  dataset_type=args.ds_type,
                  fill_na=args.fill_na)
        print(f'{i+1} of {len(pars)}')
