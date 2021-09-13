import torch 
from tqdm import tqdm
import pandas as pd
import time
from scipy.stats import entropy

# TO DOS:
# If LM:
    # Make lists with n+1 words
    # Trim input ids and targets
    # Save last word to look at true word
# - Add "is_top_5" measure
# - Set up same comparison for reading brain project (scrambled vs. not scrambled)

class StridingLM:
    ''' Runs striding forward LM over a dataset'''
    def __init__(self, 
                 context_length=20):
        self.context_length = context_length
        
    def _split(self, tokenized, model_name):
        n_tokens = tokenized['input_ids'].shape[-1]
        i_start = range(0, n_tokens-self.context_length)
        i_end = range(self.context_length, 
                      n_tokens)
        split_tks = [tokenized['input_ids'][:,i_s:i_e+1] 
                     for i_s, i_e in zip(i_start,i_end)]
        return split_tks
    
    def _preprocess(self, text, tokenizer, model_name):
        tokenized = tokenizer(text, return_tensors='pt')#.to(device='cuda:0')
        tokenized_lst = self._split(tokenized, model_name)
        return tokenized_lst
    
    def run(self, dataset, tokenizer, model,
            model_name):
        time.sleep(.5)
        results = []
        softmax_fn = torch.nn.Softmax(dim=-1)
        tokenized_lst = self._preprocess(dataset.text, 
                                         tokenizer, 
                                         model_name)
        print(f'Running {model_name}, '
              f'{dataset.name}, {self.context_length}, '
              f'{len(tokenized_lst)}')
        for i in tqdm(range(len(tokenized_lst))): # tqdm
            # Inference
            input_ids = tokenized_lst[i].clone()
            target_ids = input_ids.clone()
            if any([mid in model_name
                    for mid in ['bert', 'bigbird', 'electra']]):
                input_ids[0][-1] = tokenizer.mask_token_id
            target_ids[:,:-1] = -100
            outputs = model(input_ids, labels=target_ids)
            # Compute metrics
            ctx = tokenizer.decode(input_ids[0][:-1])
            wd = tokenizer.decode(target_ids[0][-1])
            loss = float(outputs.loss.cpu().detach().numpy())
            top_id = torch.argmax(outputs.logits[0,-1,:], 
                                  axis=-1)
            top_token = tokenizer.decode(top_id).strip(' ')
            softmaxed = softmax_fn(outputs.logits)
            prob_true = softmaxed[0,-1,target_ids[0][-1]]
            prob_true = float(prob_true.cpu().detach().numpy())
            prob_predicted = float(softmaxed[0,-1,top_id].cpu().detach().numpy())
            entr = entropy(softmaxed[0,-1,:].cpu().detach().numpy())
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