from ast import literal_eval
import pandas as pd


class Data_Preparation:
    
        
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

    def clean_tabular_data(self):
        self.remove_rows_with_missing_ratings()
        self.combine_description_strings()
        self.set_default_feature_values()
      
    def load_airbnb(label:str, df):
        features=df.drop(label, axis=1)
        labels=df[label]
        return(features, labels)
    
if __name__ == '__main__':
    airbnb_df = pd.read_csv('/Users/apple/Documents/GitHub/Data_Science_Airbnb/airbnb_datasets/listing.csv', index_col='ID')
    airbnb_df.drop('Unnamed: 19', axis=1, inplace=True)
    Data_processor = Data_Preparation()
    Data_processor.clean_tabular_data()
    airbnb_df.drop('4c917b3c-d693-4ee4-a321-f5babc728dc9', inplace=True)
    airbnb_df['guests'] = airbnb_df['guests'].astype('int64')
    airbnb_df['bedrooms'] = airbnb_df['bedrooms'].astype('int64')
    airbnb_df.to_csv(r'/Users/apple/Documents/GitHub/Data_Science_Airbnb/airbnb_datasets/clean_tabular_data.csv') 
    

    
    
 
    

