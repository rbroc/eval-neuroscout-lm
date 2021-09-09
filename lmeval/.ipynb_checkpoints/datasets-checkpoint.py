import pandas as pd
import pysrt

class NeuroscoutDataset:
    def __init__(self, filename, name=None):
        if filename.endswith('.srt'):
            self.dataset_type = 'transcript'
            data = pysrt.open(filename)
            self.text = ' '.join([d.text.replace('\n', ' ') 
                                  for d in data])
        else:
            self.dataset_type = 'aligned'
            data = pd.read_csv(filename, sep='\t')
            data = data.sort_values(by='onset')
            self.text = ' '.join(data.text.tolist())
        if name is not None:
            self.name = name
        else:
            self.name = filename
            
            
class NarrativesDataset:
    def __init__(self, filename,
                 name=None, 
                 case_sensitive=False):
        if 'align' in filename:
            self.dataset_type = 'aligned'
            data = pd.read_csv(filename, header=None)
            if case_sensitive:
                self.text = data.iloc[:,0].tolist().dropna()
            else:
                self.text = data.iloc[:,1].tolist().dropna()
        elif 'transcript' in filename:
            self.dataset_type = 'transcript'
            self.text = open(filename).read()
        else:
            raise ValueError('unable to determine type')
        if name is not None:
            self.name = name
        else:
            self.name = filename