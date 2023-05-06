import glob
import itertools
import heapq
import json
import os
from pathlib import Path
import time
import torch
import torch.nn.functional as F
import pandas as pd
import yaml

from collections import OrderedDict as OrderedDict
from datetime import datetime
from tabular_data import Data_Preparation
from torchmetrics import R2Score
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter


class AirbnbNightlyPriceRegressionDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = pd.read_csv('/Users/apple/Documents/GitHub/Data_Science_Airbnb/airbnb_datasets/clean_tabular_data.csv')
        self.data.drop('4c917b3c-d693-4ee4-a321-f5babc728dc9', inplace=True)
        self.data['bedrooms'] = self.data
        
        self.X, self.y = Data_Preparation.load_airbnb('Price_Night', self.data)
        self.X = self.X.select_dtypes(include =['float64', 'int64'])
    



if __name__ == '__main__':
    dataset = AirbnbNightlyPriceRegressionDataset()
    print(dataset)

# Use load_dataset function to get a new airbnb dataset where 'y' is the integer number of bedrooms.
# Include 'Category' as part of features (X).

#  Run entire pipeline to train entire pipeline and find the best model.