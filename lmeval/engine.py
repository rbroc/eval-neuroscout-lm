import torch 
from tqdm import tqdm
import pandas as pd
import time
import torch
from scipy.stats import entropy


COLUMNS = ['dataset', 'type', 'model', 'context', 'target', 
           'top_predicted', 'loss', 'entropy', 
           'prob_true', 'prob_predicted', 'context_size']


class StridingForwardLM:
    ''' Class for striding forward LM over a dataset.
    '''
    def __init__(self, 
                 context_length=20):
        self.context_length = context_length
        self.softmax_fn = torch.nn.Softmax(dim=-1)
        
    def _split(self, whitespaced, tokenizer):
        n_tokens = len(whitespaced)
        i_start = list(range(0, n_tokens-(self.context_length)))
        i_end = [i+(self.context_length) for i in i_start]
        split_tks = [tokenizer(' '.join(whitespaced[i_s:i_e]), 
                               return_tensors='pt')['input_ids']
                     for i_s, i_e in zip(i_start,i_end)]
        targets = [whitespaced[i_e] for i in i_end]
        return split_tks, targets
    
    def _preprocess(self, text, tokenizer):
        whitespaced = text.split(' ')
        tokenized_lst, targets = self._split(whitespaced, tokenizer)
        return tokenized_lst, targets
    
    def _mask(self, list_item, tokenizer, true):
        input_ids = list_item.clone()
        ctx = tokenizer.decode(input_ids[0])
        target_ids = input_ids.clone()
        target_ids[:,:] = -100
        true_id = tokenizer.encode(true, return_tensors='pt')[:,:0] # get id for true token
        true_token = tokenizer.decode(true_id[0,0]).strip(' ') # get true token model can predict
        input_ids = torch.cat((input_ids, true_id), axis=-1) # append true token to the inputs
        target_ids = torch.cat((target_ids, true_id), axis=-1) # append to labels too
        return input_ids, target_ids, ctx, true_token, true_id[0,0]
    
    def _compute_metrics(self, outputs, wd_id, tokenizer):
        loss = float(outputs.loss.cpu().detach().numpy())
        top_id = torch.argmax(outputs.logits[0,-1,:], 
                                  axis=-1)
        top_token = tokenizer.decode(top_id).strip(' ')
        softmaxed = self.softmax_fn(outputs.logits)
        prob_true = softmaxed[0,-1,wd_id]
        prob_true = float(prob_true.cpu().detach().numpy())
        prob_predicted = float(softmaxed[0,-1,top_id].cpu().detach().numpy())
        entr = entropy(softmaxed[0,-1,:].cpu().detach().numpy())
        return top_token, loss, entr, prob_true, prob_predicted
        
    def run(self, dataset, tokenizer, model,
            model_name, gpu=0):
        time.sleep(.5)
        results = []
        tokenized_lst, targets = self._preprocess(dataset.text, 
                                                  tokenizer) # masking
        print(f'Running {model_name}, '
              f'{dataset.name}, {self.context_length}, '
              f'{len(tokenized_lst)}')
        for i in tqdm(range(len(tokenized_lst))):
            input_ids, target_ids, ctx, wd, wd_id = self._mask(tokenized_lst[i].to(device=f'cuda:{str(gpu)}'), # edited
                                                               tokenizer, 
                                                               targets[i]) # make masking
            outputs = model(input_ids, labels=target_ids)
            metrics = self._compute_metrics(outputs, wd_id, tokenizer)
            results.append((dataset.name, 
                            dataset.dataset_type,
                            model_name, 
                            ctx, wd, *metrics,
                            self.context_length))
        output = pd.DataFrame(results, columns=COLUMNS)
        return output
