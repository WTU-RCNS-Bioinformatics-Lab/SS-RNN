import pandas as pd
import os
from collections import Counter





if __name__ == '__main__':
    path = os.path.join(os.path.dirname(__file__), '..', 'dataset/ecg_data.csv')
    ecg_data = pd.read_csv(path, index_col=0)
    print(len(ecg_data))
    print(Counter(ecg_data['label']))
