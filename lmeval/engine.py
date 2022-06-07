import torch 
from tqdm import tqdm
import pandas as pd
import time
import torch
from scipy.stats import entropy
from itertools import chain
import re
import random
import numpy as np


COLUMNS = ['dataset', 'model', 'context', 
           'top_id', 'top_token',
           'true_id', 'true_token',
           'target_id',
           'loss', 'entropy', 
           'prob_true', 'prob_predicted', 
           'top_5', 'top_10',
           'to_1', 'to_5', 'to_10', 'to_100', 
           'to_1000', 'bottom_1000', 'avg_all',
           'context_size', 'case_sensitive',
           'mask_idx']


class StridingForwardLM:
    ''' Engine to perform striding window forward LM over a dataset. '''
    def __init__(self, context_length):
        self.context_length = context_length
        self.softmax_fn = torch.nn.Softmax(dim=-1)
        
    def _split(self, whitespaced, tokenizer):
        ''' Make inputs from list of whitespace-separated tokens'''
        n_tokens = len(whitespaced)
        i_start = list(range(0, n_tokens-(self.context_length)))
        i_end = [i+(self.context_length) for i in i_start]
        split_tks = [tokenizer(' '.join(whitespaced[i_s:i_e]),
                               return_tensors='pt')['input_ids']
                     for i_s, i_e in zip(i_start,i_end)]
        targets = [whitespaced[i_e] for i_e in i_end]
        return split_tks, targets
    
    def _preprocess(self, text, tokenizer):
        ''' Preprocess text'''
        whitespaced = text.split()
        # Remove these characters
        for c in [r'—', r'–', r'-', r'—', 
                  r'…']: #r'.', 
            whitespaced = list(chain(*[w.split(c) 
                                       for w in whitespaced]))
        # Strip : and . from middle but keep if not in middle
        for c in [r':',r'.']:
            for i in range(len(whitespaced)):
                if whitespaced[i].endswith(c):
                    whitespaced[i] = whitespaced[i].replace(c, ' ').rstrip(' ') + c
                else:
                    whitespaced[i] = whitespaced[i].replace(c, ' ')
            whitespaced = list(chain(*[w.split() 
                                       for w in whitespaced]))
        # Remove these characters if alone 
        whitespaced = [w for w in whitespaced
                       if w not in [r'"', r'', r'”', r'’', r'’”', r',']]
        tokenized_lst, targets = self._split(whitespaced, tokenizer)
        return tokenized_lst, targets
    
    def _prepseq(self, list_item, tokenizer, true, gpu, model_name):
        ''' Prepare the input sequence. Note that true
            is the true word, not the token.
        '''
        input_ids = list_item.clone()
        ctx = tokenizer.decode(input_ids[0]) 
        true_ids = tokenizer.encode(' ' + true, return_tensors='pt')
        if 'opt-' in model_name:
            true_id = true_ids[:,1:2]
        else:
            true_id = true_ids[:,:1]
        target_ids = torch.tensor([-100]*(input_ids[0].shape[0]-1) + [true_id[0,0]]).to(device=f'cuda:{str(gpu)}')
        true_id = true_id.to(device=f'cuda:{str(gpu)}')
        true_token = tokenizer.decode(true_id[0,0])
        return input_ids, target_ids, ctx, true_token, true_id[0,0], -1
    
    def _compute_metrics(self, outputs, wd_id, tokenizer, mask_idx):
        ''' Compute metrics from model output and id of true token '''
        
        # Get loss and language modeling metrics
        loss = float(outputs.loss.cpu().detach().numpy())
        top_id = torch.argmax(outputs.logits[0,mask_idx,:], 
                              axis=-1)
        top_token = tokenizer.decode(top_id)
        softmaxed = self.softmax_fn(outputs.logits)
        prob_true = softmaxed[0,mask_idx,wd_id]
        prob_true = float(prob_true.cpu().detach().numpy())
        prob_predicted = float(softmaxed[0,mask_idx,top_id].cpu().detach().numpy())
        softmaxed = softmaxed[0,mask_idx,:].cpu().detach().numpy()
        entr = entropy(softmaxed)
        true_rank = softmaxed.argsort().argsort()[wd_id] 
        top_5 = int(true_rank >= tokenizer.vocab_size-5)
        top_10 = int(true_rank >= tokenizer.vocab_size-10)
        
        # Compute distribution mass metrics
        to_1 = float(softmaxed.max()) # edited
        s_sorted = np.sort(softmaxed) # edited
        to_5 = float(s_sorted[-5:-1].mean())
        to_10 = float(s_sorted[-10:-5].mean())
        to_100 = float(s_sorted[-100:-10].mean())
        to_1000 = float(s_sorted[-1000:-100].mean())
        bottom_1000 = float(s_sorted[:1000].mean())
        avg_all = float(softmaxed.mean())
        
        # Postprocess some metrics
        top_id = top_id.cpu().detach().numpy()
        
        # Gather metrics
        top_metrics = [top_id, top_token]
        lm_metrics = [loss, entr, prob_true, prob_predicted, top_5, top_10]
        dist_metrics = [to_1, to_5, to_10, to_100, to_1000, 
                        bottom_1000, avg_all]
        return top_metrics + lm_metrics + dist_metrics
             
    def run(self, dataset, tokenizer, model,
            model_name, 
            gpu=0):
        time.sleep(.5)
        results = []
        tokenized_lst, targets = self._preprocess(dataset.text, 
                                                  tokenizer)
        print(f'Running {model_name}, '
              f'{dataset.name}, {self.context_length}, '
              f'{len(tokenized_lst)}')
        for i in tqdm(range(len(tokenized_lst))):
            iids, tids, ctx, wd, wd_id, mask_idx = self._prepseq(tokenized_lst[i].to(device=f'cuda:{str(gpu)}'),
                                                                 tokenizer, 
                                                                 targets[i], gpu,
                                                                 model_name)
            outputs = model(iids, labels=tids)
            metrics = self._compute_metrics(outputs, wd_id, tokenizer, mask_idx)
            wd_id = wd_id.cpu().detach().numpy() # edited
            top_id, top_token = metrics[:2]
            metrics = metrics[2:]
            results.append((dataset.name, 
                            model_name, 
                            ctx, 
                            float(top_id), top_token,
                            float(wd_id), wd, 
                            targets[i], 
                            *metrics,
                            self.context_length,
                            dataset.dataset_type,
                            mask_idx))
        output = pd.DataFrame(results, columns=COLUMNS)
        return output
    
    
class StridingMLM(StridingForwardLM):
    ''' Engine for masked language models '''
    def __init__(self, mask_dict, **kwargs):
        super().__init__(**kwargs)
        self.mask_dict = mask_dict[str(self.context_length)]
    
    def _split(self, whitespaced, tokenizer):
        ''' Tokenization for MLM '''
        n_tokens = len(whitespaced)
        i_start = list(range(0, n_tokens-(self.context_length)))
        i_end = [i+(self.context_length) for i in i_start]
        split_tks = [] 
        targets = []
        for i_s, i_e in zip(i_start, i_end):
            # Encode
            tokenized = tokenizer(' '.join(whitespaced[i_s:i_e+1]))['input_ids']
            # Mask and get target
            mask_idx = self.mask_dict.pop(0)
            target = tokenizer.decode(tokenized[mask_idx])
            targets.append(target)
            # Replace in encoding
            tokenized = tokenized[:mask_idx] + [tokenizer.mask_token_id] + tokenized[mask_idx+1:]
            split_tks.append(torch.tensor([tokenized]))
        return split_tks, targets
        
    def _prepseq(self, list_item, tokenizer, true, gpu, model_name):
        ''' Prepare the input sequence for MLM '''
        input_ids = list_item.clone()
        ctx = tokenizer.decode(input_ids[0]) # also includes the mask
        true_id = tokenizer.encode(true, return_tensors='pt')[0,1]
        mask_idx = np.where(input_ids[0].cpu() == tokenizer.mask_token_id)[0][0]
        target_ids = [-100] * mask_idx + [true_id] 
        target_ids += [-100] * (input_ids[0].shape[0] - len(target_ids))
        target_ids = torch.tensor(target_ids).to(device=f'cuda:{str(gpu)}')
        true_id = true_id.to(device=f'cuda:{str(gpu)}')
        true_token = tokenizer.decode(true_id)
        return input_ids, target_ids, ctx, true_token, true_id, mask_idx
    
