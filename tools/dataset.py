from torch.utils.data import Dataset
import logging
import pandas as pd
from PIL import Image
import numpy as np

class CsvDataset(Dataset):
    def __init__(self, csv_path, transforms, img_key='link', caption_key='caption', tokenizer=None):
        logging.debug(f'Loading csv data from {csv_path}.')
        df = pd.read_csv(csv_path)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        while True:
            try: #################################### 异常机制 #############
                images = self.transforms(Image.open(str(self.images[idx])))
                texts = self.tokenize([str(self.captions[idx])])[0]
                break
            except:
                idx = np.random.randint(0, len(self.captions))
        return {'img_tensor': images, 'text_tensor': texts, 'text': self.captions[idx]}