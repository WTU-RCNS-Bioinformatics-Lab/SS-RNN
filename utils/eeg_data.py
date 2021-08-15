import pandas as pd
import numpy as np
import os
from collections import Counter



if __name__ == '__main__':
    path = os.path.join(os.path.dirname(__file__), '..', 'dataset/eeg_data.csv')
    eeg_data = pd.read_csv(path, index_col=0)
    print(eeg_data.shape)
    print(Counter(eeg_data['y']))