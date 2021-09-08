import pandas as pd
import pysrt

class Dataset:
    def __init__(self, filename, name=None):
        if filename.endswith('.srt'):
            data = pysrt.open(filename)
            self.text = ' '.join([d.text.replace('\n', ' ') 
                                  for d in data])
        else:
            data = pd.read_csv(filename, sep='\t')
            data = data.sort_values(by='onset')
            self.text = ' '.join(data.text.tolist())
        if name is not None:
            self.name = name
        else:
            self.name = filename