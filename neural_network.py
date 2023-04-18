import torch
import pandas as pd

from tabular_data import Data_Preparation
from torch.utils.data import Dataset


class AirbnbDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = pd.read_csv('/Users/apple/Documents/GitHub/Data_Science_Airbnb/airbnb_datasets/clean_tabular_data.csv')
        self.X, self.y = Data_Preparation.load_airbnb('Price_Night', self.data)
        self.X = self.X.select_dtypes(include =['float64', 'int64'])
        # print(self.X)
        # print(self.y)
    
    def __getitem__(self, index):
        return (torch.tensor(self.X.iloc[index]), torch.tensor(self.y.iloc[index]))
    
    def __len__(self):
        return len(self.X)
    

dataset = AirbnbDataset()
print(dataset[10])
print(len(dataset))
    