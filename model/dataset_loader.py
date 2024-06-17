import pandas as pd
import os

def load_dataset(data_folder='./data'):
    dfs = dict()
    for split in ['train', 'val']:
        dfs[split] = pd.read_csv(
            os.path.join(
                data_folder,
                f'captions_{split}.tsv'),
            sep='\t')

    return dfs
