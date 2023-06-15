import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class PreprocessingTools:
    def __init__(self, dataset_filename, start_column_index=None):
        self.dataset_name = dataset_filename
        if (start_column_index is not None):
            self.X, self.y = self.import_selected_columns_dataset(start_column_index)
        else:
            self.X, self.y = self.import_entire_dataset()
    
    def import_entire_dataset(self):
        dataset = pd.read_csv(self.dataset_name) 
        X = dataset.iloc[: , :-1].values 
        y = dataset.iloc[: , -1].values
        return X, y
    
    def import_selected_columns_dataset(self, start_column_index):
        dataset = pd.read_csv(self.dataset_name) 
        X = dataset.iloc[: , start_column_index: -1].values 
        y = dataset.iloc[: , -1].values
        return X, y

    def split_dataset(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=0)
        return X_train, X_test, y_train, y_test
    
    def encode_dependent_variable(self, column_index):
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [column_index])], remainder='passthrough')
        self.X = np.array(ct.fit_transform(self.X))
        return self.X
