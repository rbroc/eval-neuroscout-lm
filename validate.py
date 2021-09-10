from transformers import (GPT2LMHeadModel, GPT2TokenizerFast,
                          BertForMaskedLM, BertTokenizerFast,
                          DistilBertForMaskedLM, DistilBertTokenizerFast,
                          RobertaForMaskedLM, RobertaTokenizerFast,
                          BlenderbotForCausalLM, BlenderbotTokenizer,
                          BigBirdForMaskedLM, BigBirdTokenizer,
                          ElectraForMaskedLM, ElectraTokenizerFast,
                          CTRLLMHeadModel, CTRLTokenizer)
from multiprocessing import Pool
import pandas as pd
import glob
from itertools import product
from lmeval.datasets import NarrativesDataset
from lmeval.engine import StridingLM
import transformers

transformers.logging.set_verbosity(50)

# Define parameters
transcripts = glob.glob('inputs/narratives/gentle/*/transcript*')
aligned = glob.glob('inputs/narratives/gentle/*/align.csv')
dataset_files = transcripts + aligned
model_classes = [GPT2LMHeadModel,
                 BertForMaskedLM,
                 DistilBertForMaskedLM, 
                 RobertaForMaskedLM, 
                 BlenderbotForCausalLM, 
                 BigBirdForMaskedLM,
                 ElectraForMaskedLM]
model_ids = ['gpt2', 
             'bert-base-uncased', 
             'distilbert-base-uncased',
             'roberta-base', 
             'facebook/blenderbot-400M-distill', 
             'google/bigbird-roberta-base', 
             'google/electra-base-discriminator']
tokenizer_classes = [GPT2TokenizerFast,
                     BertTokenizerFast,
                     DistilBertTokenizerFast,
                     RobertaTokenizerFast,
                     BlenderbotTokenizer,
                     BigBirdTokenizer,
                     ElectraTokenizerFast]
model_parameters = list(zip(model_classes, 
                            model_ids, 
                            tokenizer_classes))
ctx_lengths = [5, 10, 15, 20]
parameters = list(product(dataset_files, 
                          model_parameters, 
                          ctx_lengths))
parameters = [(i[0], *i[1], i[2]) for i in parameters]
    
    
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
              ctx_length):
    ''' Main functon to run the validation'''
    files = glob.glob('outputs/narratives/*')
    n_files = len(files)
    dataset_name = _make_dataset_id(datafile)
    if '/' in model_id:
        model_id_log = model_id.split('/')[1]
    else:
        model_id_log = model_id
    log_id = f'{dataset_name}_{model_id_log}_{ctx_length}.txt'    
    log_path = f'outputs/narratives/{log_id}'
    if log_path not in files:
        tokenizer = tokenizer_class.from_pretrained(model_id)
        model = model_class.from_pretrained(model_id).to(device='cuda:0')
        data = NarrativesDataset(datafile, dataset_name)
        engine = StridingLM(context_length=ctx_length)
        result = engine.run(data, tokenizer, model, model_id)
        result.to_csv(log_path, sep='\t')
        print(f'{n_files+1} out of {len(parameters)}')
        return result

  
# Run
if __name__=='__main__':
    for p in parameters:
        print(len(parameters))
        _validate(*p)