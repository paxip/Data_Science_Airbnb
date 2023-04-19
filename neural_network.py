
import torch
import torch.nn.functional as F
import pandas as pd

from tabular_data import Data_Preparation

from torch.utils.data import DataLoader, Dataset, random_split



class AirbnbNightlyPriceRegressionDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = pd.read_csv('/Users/apple/Documents/GitHub/Data_Science_Airbnb/airbnb_datasets/clean_tabular_data.csv')
        self.X, self.y = Data_Preparation.load_airbnb('Price_Night', self.data)
        self.X = self.X.select_dtypes(include =['float64', 'int64'])
    
    def __getitem__(self, index):
        return (torch.tensor(self.X.iloc[index]), torch.tensor(self.y.iloc[index]))
    
    def __len__(self):
        return len(self.X)



class AirbnbNightlyPriceRegressionModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_layer = torch.nn.Linear(9, 1)
        pass

    def forward(self, X):
        #  use the layers to process the features.
        return self.linear_layer(X)
    


if __name__ == '__main__':
    dataset = AirbnbNightlyPriceRegressionDataset()

    train_set, test_set, validation_set = random_split(dataset, [0.7, 0.15, 0.15])
    batch_size = 4
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    batch = next(iter(train_loader))

    # X, y = example
    # X = X.to(torch.float32)
    # y = y.to(torch.float32)

    model = AirbnbNightlyPriceRegressionModel()
    # print(model(X))

    def train(model, epochs=10):
        for epoch in range(epochs):
            for batch in train_loader:
                X, y = batch
                X = X.to(torch.float32)
                y = y.to(torch.float32)
                y = torch.unsqueeze(y, 1)
                # # `print(X)` and `print(y)` are used to print the input features and target variable
                # of a batch respectively. This is done to check if the data is being loaded
                # correctly into the model.
                # print(X)
                # print(y)
                prediction = model(X)
                loss = F.mse_loss(prediction, y)
                loss.backward()
                print(loss)
            
                
                break 

    train(model)

   


    
