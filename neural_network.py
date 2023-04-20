
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
        

    def forward(self, X):
        #  use the layers to process the features.
        return self.linear_layer(X)
    


if __name__ == '__main__':
    dataset = AirbnbNightlyPriceRegressionDataset()

    train_set, test_set, validation_set = random_split(dataset, [0.7, 0.15, 0.15])
    batch_size = 4
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    # batch = next(iter(train_loader))

    model = AirbnbNightlyPriceRegressionModel()
    # print(model(X))

    def train(model, epochs=10):

        optimiser = torch.optim.SGD(model.parameters(), lr=0.0001, weight_decay=0.001, momentum=0.07)

        for epoch in range(epochs):
            for batch in train_loader:
                X, y = batch
                X = X.to(torch.float32)
                y = y.to(torch.float32)
                y = torch.unsqueeze(y, 1)
                y_prediction = model(X)
                loss = F.mse_loss(y_prediction, y)
                loss.backward()
                print(f'Train loss is {loss.item()}')
                optimiser.step()
                optimiser.zero_grad()
                
            for batch in validation_loader:
                X, y = batch
                X = X.to(torch.float32)
                y = y.to(torch.float32)
                y = torch.unsqueeze(y, 1)
                y_validation = model(X)
                loss = F.mse_loss(y_validation, y)
                loss.backward()
                print(f'Validation loss is {loss.item()}')
                optimiser.step()
                optimiser.zero_grad()

            

    train(model)

    # Complete the training loop so that it iterates through ever batch.
    # Optimise model parameters.
    # Evaluate model performance on the validation dataset after each epoch.


   


    
