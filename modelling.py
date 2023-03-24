from tabular_data import load_airbnb
from tabular_data import Data_Preparation























if __name__ == '__main__':
       airbnb_df = Data_Preparation.clean_tabular_data()
       X,y = load_airbnb('Price_Night', airbnb_df)
       print(y.shape)