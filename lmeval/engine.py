import torch 
from tqdm import tqdm
import pandas as pd
import time
import torch
from scipy.stats import entropy
from itertools import chain
import re
import numpy as np


COLUMNS = ['dataset', 'model', 'context', 
           'top_id', 'top_token',
           'true_id', 'true_token',
           'target_id',
           'loss', 'entropy', 
           'prob_true', 'prob_predicted', 
           'top_5', 'top_10',
           'to_1', 'to_5', 'to_10', 'to_100', 
            'to_1000', 'bottom_1000', 'avg_all'
           'context_size', 'case_sensitive']


class StridingForwardLM:
    ''' Engine to perform striding window forward LM over a dataset. '''
    def __init__(self, context_length=20):
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
    
    def _prepseq(self, list_item, tokenizer, true, gpu):
        ''' Prepare the input sequence. Note that true
            is the true word, not the token.
        '''
        input_ids = list_item.clone()
        ctx = tokenizer.decode(input_ids[0])
        true_id = tokenizer.encode(' ' + true, return_tensors='pt')[:,:1]
        target_ids = torch.tensor([-100]*(input_ids[0].shape[0]-1) + [true_id[0,0]]) # model-specific
        true_id = true_id.to(device=f'cuda:{str(gpu)}')
        true_token = tokenizer.decode(true_id[0,0]) # model-specific
        return input_ids, target_ids, ctx, true_token, true_id[0,0]
    
    def _compute_metrics(self, outputs, wd_id, tokenizer):
        ''' Compute metrics from model output and id of true token '''
        
        # Get loss and language modeling metrics
        loss = float(outputs.loss.cpu().detach().numpy()) # MS
        top_id = torch.argmax(outputs.logits[0,-1,:], 
                              axis=-1)
        top_token = tokenizer.decode(top_id) # MS
        softmaxed = self.softmax_fn(outputs.logits)
        prob_true = softmaxed[0,-1,wd_id]
        prob_true = float(prob_true.cpu().detach().numpy())
        prob_predicted = float(softmaxed[0,-1,top_id].cpu().detach().numpy())
        softmaxed = softmaxed[0,-1,:].cpu().detach().numpy()
        entr = entropy(softmaxed)
        true_rank = softmaxed.argsort().argsort()[wd_id] 
        top_5 = int(true_rank >= tokenizer.vocab_size-5)
        top_10 = int(true_rank >= tokenizer.vocab_size-10)
        
        # Compute distribution mass metrics
        to_1 = float(softmaxed.max().cpu().detach().numpy())
        s_sorted = np.sort(softmaxed.cpu().detach().numpy())
        to_5 = float(s_sorted[-5:-1].mean())
        to_10 = float(s_sorted[-10:-5].mean())
        to_100 = float(s_sorted[-100:-10].mean())
        to_1000 = float(s_sorted[-1000:-100].mean())
        bottom_1000 = float(s_sorted[:1000].mean())
        avg_all = float(softmaxed.mean())
        
        # Postprocess some metrics
        top_id = top_id.cpu().detach().numpy()
        wd_id = wd_id.cpu().detach().numpy()
        
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
            iids, tids, ctx, wd, wd_id = self._prepseq(tokenized_lst[i].to(device=f'cuda:{str(gpu)}'),
                                                       tokenizer, 
                                                       targets[i], gpu)
            outputs = model(iids, labels=tids)
            metrics = self._compute_metrics(outputs, wd_id, tokenizer)
            top_id, top_token = metrics[:2]
            metrics = metrics[2:]
            results.append((dataset.name, 
                            model_name, 
                            ctx, 
                            top_id, top_token,
                            wd_id, wd, 
                            targets[i], 
                            *metrics,
                            self.context_length,
                            dataset.dataset_type))
        output = pd.DataFrame(results, columns=COLUMNS)
        return output
    

class StridingMLM(StridingForwardLM):
    ''' Engine for masked LM models '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _split(self, whitespaced, tokenizer):
        n_tokens = len(whitespaced)
        i_start = list(range(0, n_tokens-(self.context_length)))
        i_end = [i+(self.context_length) for i in i_start]
        split_tks = []
        targets = []
        for i_s, i_e in zip(i_start, i_end):
            # Sample random integer in i_start, i_end
            # Replace that with '[MASK]'
            # Encode
            split.tks.append(tokenizer(' '.join(whitespaced[i_s:i_e+1]), # edit with correct whitespaced
                                       return_tensors='pt')['input_ids'])
            targets.append(whitespaced[i_e]) # edit correct index 
        return split_tks, targets
        
    def _prepseq(self, list_item, tokenizer, true, gpu):
        ''' Prepare the input sequence for MLM. '''
        pass
    
    def _compute_metrics(self):
        pass