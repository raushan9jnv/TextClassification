import pandas as pd
import numpy as np

class TextClassificationModel():
    def __init__(self,dataset,features):
        self.dataset = dataset
        self.features = features
        # self.features_columns = features_columns
        # self.target_column = target_column
    
    def print_df(self):
        print(self.dataset)

    # def preprocessing():

        # Handling null values
#     import pandas as pd

# class DataCleaner:
#     def __init__(self, dataset, features, fillwith):
#         self.dataset = dataset
#         self.features = features
#         self.fillwith = fillwith
        
    def handle_nulls(self, fillwith):
        for feature in self.features:
            if self.dataset[feature].dtype in ['int64', 'float64']:
                if fillwith == 'mean':
                    self.dataset[feature].fillna(self.dataset[feature].mean(), inplace=True)
                elif fillwith == 'median':
                    self.dataset[feature].fillna(self.dataset[feature].median(), inplace=True)
                elif fillwith == 'zero':
                    self.dataset[feature].fillna(0, inplace=True)
            elif self.dataset[feature].dtype == 'object':
                if fillwith == 'mode':
                    self.dataset[feature].fillna(self.dataset[feature].mode()[0], inplace=True)
                elif fillwith == 'missing':
                    self.dataset[feature].fillna('Missing', inplace=True)
            elif self.dataset[feature].dtype == 'bool':
                self.dataset[feature].fillna(False, inplace=True)
            elif self.dataset[feature].dtype == 'datetime64[ns]':
                self.dataset[feature].fillna(pd.to_datetime('1970-01-01'), inplace=True)

# # usage example:
# dc = DataCleaner(pd.read_csv("data.csv"), ['age', 'income', 'gender', 'is_student', 'birthdate'], 'mode')
# dc.handle_nulls()
# print(dc.dataset)

        
        
# df= pd.read_csv('master_cop_data.csv')
# df= pd.read_csv('train.csv')
# tc = TextClassificationModel(df,)
# tc.print_df()

# import pandas as pd

# create a DataFrame with different data types
data = {
    'age': [25, 30, np.nan, 40, 50,np.nan],
    'income': [50000, 60000, np.nan, 80000, np.nan,np.nan],
    'gender': ['male', 'female', 'male', 'female', 'male',np.nan],
    'is_student': [True, False, np.nan, False, True,np.nan],
    'birthdate': ['1990-01-01', '1985-05-01', '2000-03-01', '1995-01-01', '1980-12-01'],
    'birthdate': [pd.to_datetime('1990-01-01'), pd.to_datetime('1985-05-01'), pd.to_datetime('2000-03-01'), pd.to_datetime('1995-01-01'), pd.to_datetime('1980-12-01'),np.nan]
}
df = pd.DataFrame(data)
print('Original Dataframe:')
print(df)

# use the DataCleaner class to handle null values in specified features
dc = TextClassificationModel(df, ['age', 'income', 'gender', 'is_student', 'birthdate'])
dc.handle_nulls('mean')
print('Dataframe after handling null values:')
print(df)
