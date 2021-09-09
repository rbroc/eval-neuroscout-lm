import torch 
from tqdm import tqdm
import pandas as pd
import time
from scipy.stats import entropy

class StridingLM:
    ''' Runs striding forward LM over a dataset'''
    def __init__(self, 
                 context_length=20):
        self.context_length = context_length
        
    def _split(self, lst):
        i_start = range(0, len(lst)-self.context_length)
        i_end = range(self.context_length, len(lst))
        split_lst = [lst[i_s:i_e+1] 
                     for i_s, i_e in zip(i_start,i_end)]
        return split_lst
    
    def _preprocess(self, text, tokenizer):
        tokenized = tokenizer(text, return_tensors='pt')
        tokenized_lst = self._split(tokenized)
        return tokenized_lst
    
    def run(self, dataset, tokenizer, model,
            model_name):
        print(f'Running {model_name},'
              f'{dataset.name}, {self.context_length}')
        time.sleep(.5)
        results = []
        softmax_fn = torch.nn.Softmax(dim=-1)
        tokenized_lst = self._preprocess(dataset.text, 
                                         tokenizer)
        
        for i in range(len(tokenized_lst)): # tqdm?
            # Inference
            t = tokenized_lst[i]
            input_ids = t['input_ids']
            target_ids = input_ids.clone()
            if any([mid in model_name
                    for mid in ['bert', 'bigbird', 'electra']]):
                input_ids[0][-1] = tokenizer.mask_token_id
            target_ids[:,:-1] = -100
            outputs = model(input_ids, labels=target_ids)
            # Compute metrics
            ctx = tokenizer.decode(input_ids[0][:-1])
            wd = tokenizer.decode(target_ids[0][-1])
            loss = float(outputs.loss.detach().numpy())
            top_id = torch.argmax(outputs.logits[0,-1,:], 
                                  axis=-1)
            top_token = tokenizer.decode(top_id).strip(' ')
            softmaxed = softmax_fn(outputs.logits)
            prob_true = softmaxed[0,-1,target_ids[0][-1]]
            prob_true = float(prob_true.detach().numpy())
            prob_predicted = float(softmaxed[0,-1,top_id].detach().numpy())
            entr = entropy(softmaxed[0,-1,:].detach().numpy())
            results.append((dataset.name, 
                            dataset.dataset_type,
                            model_name, 
                            ctx, wd, top_token, loss,
                            entr, prob_true, prob_predicted,
                            self.context_length))
        output = pd.DataFrame(results, 
                              columns=['dataset', 
                                       'type',
                                       'model',
                                       'context', 'target', 
                                       'top_predicted', 
                                       'loss', 'entropy',
                                       'prob_true',
                                       'prob_predicted',
                                       'context_size'])
        
        return output