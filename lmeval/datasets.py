import pandas as pd
import pysrt
import numpy as np

class NeuroscoutDataset:
    def __init__(self, filename, name=None):
        if filename.endswith('.srt'):
            self.dataset_type = 'transcript'
            data = pysrt.open(filename)
            text_list = [d.text for d in data]
            self.text = ' '.join(text_list)
            self.text = self.text.replace('\n',' ')
            self.text = self.text.replace(r'\n',' ')
            self.text = self.text.replace('  ',' ')
        else:
            self.dataset_type = 'aligned'
            data = pd.read_csv(filename, sep='\t', skip_blank_lines=False)
            data = data.sort_values(by='onset')
            self.text = ' '.join(data.text.tolist())
        if name is not None:
            self.name = name
        else:
            self.name = filename
            
            
class NarrativesDataset:
    def __init__(self, 
                 filename,
                 name=None, 
                 case_sensitive=False,
                 fill_na='unk'):
        if 'align' in filename:
            data = pd.read_csv(filename, header=None, 
                               skip_blank_lines=False)
            if case_sensitive:
                text = data.iloc[:,0].tolist() 
                self.dataset_type = 'align_upper'
            else:
                if fill_na == 'unk':
                    text = data.iloc[:,1].fillna('<unk>').tolist()
                    self.dataset_type = 'align_lower_unk'
                elif fill_na == 'replace':
                    data.iloc[:,1].fillna(data.iloc[:,0].str.lower(), 
                                          inplace=True) #replace na w/ other column
                    data.iloc[:,1] = np.where(data.iloc[:,1] == '<unk>',
                                              data.iloc[:,0].str.lower(), 
                                              data.iloc[:,1]) # replace unk with other column
                    self.dataset_type = 'align_lower_nounk'
                    text = data.iloc[:,1].tolist()
            self.text = ' '.join(text) 
        elif 'transcript' in filename:
            self.text = open(filename).read()
            if case_sensitive:
                self.text = self.text.lower()
                self.dataset_type = 'transcript_cased'
            else:
                self.dataset_type = 'transcript'      
        else:
            raise ValueError('unable to determine type')
        if name is not None:
            self.name = name
        else:
            self.name = filename
