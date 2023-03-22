from ast import literal_eval
import pandas as pd


class Data_Preparation:
    # def __init__(self):
        
    def remove_rows_with_missing_ratings(self):
        airbnb_df.dropna(subset=['Cleanliness_rating', 'Accuracy_rating', 'Communication_rating', 'Location_rating', 'Check-in_rating', 'Value_rating'], inplace=True)
        return airbnb_df

    def clean_description_strings(self, text):
        try:
            description_list = literal_eval(text)
            description_list.remove('About this space')
            description_list = ''.join(description_list)
            return description_list
        except Exception as e:
                return text 

    def combine_description_strings(self):
        airbnb_df.dropna(subset=['Description'], inplace=True)
        airbnb_df['Description'] = airbnb_df['Description'].apply(self.clean_description_strings) 
        airbnb_df['Description'].replace([r"\\n", "\n", r"\'"], [" "," ",""], regex=True, inplace=True)
        return airbnb_df
      
    def set_default_feature_values(self):
        values = {"guests": 1, "beds": 1, "bathrooms": 1, "bedrooms": 1}
        airbnb_df.fillna(value=values, inplace=True)
        return airbnb_df

    def clean_tabular_data(self, airbnb_df):
        self.remove_rows_with_missing_ratings()
        self.combine_description_strings()
        self.set_default_feature_values()

if __name__ == '__main__':
    airbnb_df = pd.read_csv('listing.csv', index_col='ID')
    airbnb_df.drop('Unnamed: 19', axis=1, inplace=True)
    Data_processor = Data_Preparation()
    Data_processor.clean_tabular_data(airbnb_df)

    airbnb_df.to_csv(r'/Users/apple/Documents/GitHub/Data_Science_Airbnb/airbnb_datasets/clean_tabular_data.csv') 
    

