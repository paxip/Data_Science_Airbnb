
import torch
import torch.nn.functional as F
import pandas as pd

from tabular_data import Data_Preparation
from torchmetrics import R2Score
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter


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

        optimiser = torch.optim.SGD(model.parameters(), lr=0.0001, weight_decay=0.01, momentum=0.1)

        writer = SummaryWriter()

        batch_index = 0

        for epoch in range(epochs):
            for batch in train_loader:
                X, y = batch
                X = X.to(torch.float32)
                y = y.to(torch.float32)
                y = torch.unsqueeze(y, 1)
                y_prediction = model(X)
                train_loss = F.mse_loss(y_prediction, y)
                train_loss.backward()
                print(f'Train loss is {train_loss.item()}')
                R2_train = R2Score()
                R2_train = R2_train(y_prediction, y)
                optimiser.step()
                optimiser.zero_grad()
                writer.add_scalar('train loss', train_loss.item(), batch_index)
                writer.add_scalar('train accuracy', R2_train, batch_index)       
                    
                
            for batch in validation_loader:
                X, y = batch
                X = X.to(torch.float32)
                y = y.to(torch.float32)
                y = torch.unsqueeze(y, 1)
                y_validation = model(X)
                validation_loss = F.mse_loss(y_validation, y)
                validation_loss.backward()
                print(f'Validation loss is {validation_loss.item()}')
                R2_validation = R2Score()
                R2_validation = R2_validation(y_validation, y)
                optimiser.step()
                optimiser.zero_grad()
                writer.add_scalar('validation loss', validation_loss.item(), batch_index)
                writer.add_scalar('validation accuracy', R2_validation, batch_index)

                batch_index += 1
                
    train(model)

    

   


    
