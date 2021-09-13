import torch 
from tqdm import tqdm
import pandas as pd
import time
from scipy.stats import entropy


COLUMNS = ['dataset', 'type', 'model', 'context', 'target', 
           'top_predicted', 'loss', 'entropy', 
           'prob_true', 'prob_predicted', 'context_size']


class StridingMLM:
    ''' Class for striding forward LM over a dataset 
        Applies to BERT-like models, which need explicit masking 
        of the last token.
    '''
    def __init__(self, 
                 context_length=20):
        self.context_length = context_length
        self.softmax_fn = torch.nn.Softmax(dim=-1)
        
    def _split(self, tokenized, model_name):
        n_tokens = tokenized['input_ids'].shape[-1]
        i_start = list(range(0, 
                             n_tokens-(self.context_length+1)))
        i_end = [i+(self.context_length+1) for i in i_start]
        split_tks = [tokenized['input_ids'][:,i_s:i_e]
                     for i_s, i_e in zip(i_start,i_end)]
        return split_tks
    
    def _preprocess(self, text, tokenizer, model_name):
        tokenized = tokenizer(text, return_tensors='pt').to(device='cuda:0')
        tokenized_lst = self._split(tokenized, model_name)
        return tokenized_lst
    
    def _mask(self, list_item, tokenizer):
        input_ids = list_item.clone()
        target_ids = input_ids.clone()
        input_ids[0][-1] = tokenizer.mask_token_id
        target_ids[:,:-1] = -100
        ctx = tokenizer.decode(input_ids[0][:-1])
        wd_id = target_ids[0][-1]
        wd = tokenizer.decode(wd_id)
        return input_ids, target_ids, ctx, wd, wd_id
    
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
            model_name):
        time.sleep(.5)
        results = []
        tokenized_lst = self._preprocess(dataset.text, 
                                         tokenizer, 
                                         model_name)
        print(f'Running {model_name}, '
              f'{dataset.name}, {self.context_length}, '
              f'{len(tokenized_lst)}')
        for i in tqdm(range(len(tokenized_lst))):
            input_ids, target_ids, ctx, wd, wd_id = self._mask(tokenized_lst[i],
                                                               tokenizer)
            outputs = model(input_ids, labels=target_ids)
            metrics = self._compute_metrics(outputs, wd_id, tokenizer)
            results.append((dataset.name, 
                            dataset.dataset_type,
                            model_name, 
                            ctx, wd, *metrics,
                            self.context_length))
        output = pd.DataFrame(results, columns=COLUMNS)
        return output
 

class StridingForwardLM(StridingMLM):
    ''' Class for striding forward LM over a dataset 
        Applies to GPT-like models.
    '''
    def __init__(self, 
                 context_length=20):
        super().__init__(context_length=context_length)
    
    def _mask(self, list_item, tokenizer):
        input_ids = list_item[:,:-1].clone()
        target_ids = input_ids.clone()
        target_ids[:,:-1] = -100
        ctx = tokenizer.decode(input_ids[0][:-1])
        wd_id = list_item[0,-1]
        wd = tokenizer.decode(wd_id)
        return input_ids, target_ids, ctx, wd, wd_id
   

    
