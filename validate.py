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
import os

# Don't use gpus
os.environ['CUDA_VISIBLE_DEVICES']='-1'

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
    tokenizer = tokenizer_class.from_pretrained(model_id)
    model = model_class.from_pretrained(model_id)
    dataset_name = _make_dataset_id(datafile)
    data = NarrativesDataset(datafile, dataset_name)
    data.text = data.text[:100]
    engine = StridingLM(context_length=ctx_length)
    result = engine.run(data, tokenizer, model, model_id)
    # Log the data
    log_id = f'{dataset_name}_{model_id}_{ctx_length}.txt'
    result.to_csv(f'outputs/narratives/test_{log_id}',
                  sep='\t')
    # How many left?
    n_files = len(glob.glob('outputs/narratives/*'))
    print(f'{n_files} out of {len(parameters)}')
    return result

  
# Run
if __name__=='__main__':
    pool = Pool(20)
    results = pool.starmap(_validate, parameters)
    pool.close()