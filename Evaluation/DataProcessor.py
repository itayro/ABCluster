
import os
import pandas as pd
import json
import numpy as np
import glob

CONFIG_PATH = 'config.json'


class DataProcessor:
    """
        Manager of reading, manipulating and preparing data-sets for evaluation
    """

    DATA_SETS_DIR = "InputCSVs/"
    MANIPULATED_DATA_SETS_DIR = "ManInputCSVs/"

    def __init__(self, config_path):
        if not os.path.exists(CONFIG_PATH):
            raise Exception("{} is not a valid path to a file".format(config_path))
        with open(config_path) as f:
            self.data_info = json.load(f)
            for ds in self.data_info:
                for key in ['name', 'address', 'categorical_related_cols', 'categorical_distinct_cols',
                            'missing_values', 'to_remove_cols', 'target_col']:
                    if key not in ds:
                        raise Exception("{} not in one of the objects, revisit configure file".format(key))

    # Turn categorical column in DF to numeric values
    @staticmethod
    def turn_to_numeric_single_col(df, col_name, values_in_col=""):
        # if already numeric- return
        if df[col_name].dtype in ['float64', 'int64']:
            return
        if values_in_col == "":
            curr_values = df[col_name].unique()
        else:
            curr_values = values_in_col
        curr_values = list(filter(lambda v: v == v, curr_values))
        df[col_name] = df[col_name].apply(lambda x: curr_values.index(x) if x in curr_values else x)

    # Read a single data set from uci to a local csv
    def __single_uci_to_csv(self, address, name, target_col, delim=None):
        if os.path.exists("{}{}.csv".format(self.DATA_SETS_DIR, name)):
            print("a csv for {} already exists - skipped".format(name))
            return
        if delim is not None:
            df = pd.read_csv(address, header=None, delimiter='  ')
        else:
            df = pd.read_csv(address, header=None)
        # Making sure the target column is last
        if target_col != -1:
            cols = list(df.columns)
            cols = cols[:target_col] + cols[target_col+1:] + [cols[target_col]]
            df = df[cols]
        df.to_csv("{}{}.csv".format(self.DATA_SETS_DIR, name), header=None, index=None)

    # Read all data sets from uci (using config file) to csv
    def multiple_uci_to_csv(self):
        if not os.path.exists(self.DATA_SETS_DIR):
            os.makedirs(self.DATA_SETS_DIR)
        for ds in self.data_info:
            if 'delimeter' in ds:
                self.__single_uci_to_csv(ds['address'], ds['name'], ds['target_col'], delim=ds['delimeter'])
            else:
                self.__single_uci_to_csv(ds['address'], ds['name'], ds['target_col'])

    # Turn categorical features to numeric using one hot encoding for distinct and normally for related
    def __manipulate_single_csv(self, name, categorical_related_cols, categorical_distinct_cols, missing, to_remove):
        df = pd.read_csv('{}{}.csv'.format(self.DATA_SETS_DIR, name), header=None)
        class_col = df.columns[-1]

        # if needed, drop rows with missing values
        if bool(missing):
            df.replace('?', np.nan, inplace=True)
            df.dropna(inplace=True)

        for key in categorical_related_cols:
            self.turn_to_numeric_single_col(df, df.columns[int(key)], categorical_related_cols[key])

        df = pd.get_dummies(df, columns=categorical_distinct_cols)
        df = df[[c for c in df if c != class_col] + [class_col]]

        # TODO: find out how to work with string values in class columns
        self.turn_to_numeric_single_col(df, df.columns[-1])

        # if needed, drop irrelevant cols (like id)
        if len(to_remove) > 0:
            df.drop(df.columns[to_remove], axis=1, inplace=True)

        df.to_csv('{}{}.csv'.format(self.MANIPULATED_DATA_SETS_DIR, name), header=None, index=None)

    # Turn categorical features to numeric using one hot encoding for distinct and normally for related - all data sets
    def manipulate_multiple_csv(self):
        if not os.path.exists(self.MANIPULATED_DATA_SETS_DIR):
            os.makedirs(self.MANIPULATED_DATA_SETS_DIR)
        for ds in self.data_info:
            self.__manipulate_single_csv(ds['name'],
                                         ds['categorical_related_cols'],
                                         ds['categorical_distinct_cols'],
                                         ds['missing_values'],
                                         ds['to_remove_cols'])

    # Read csv and return X,y
    def __get_X_y(self, name):
        df = pd.read_csv('{}{}.csv'.format(self.MANIPULATED_DATA_SETS_DIR, name), header=None)
        return df.drop(df.columns[[-1]], axis=1).values, df[[df.columns[-1]]].values.ravel()

    # Returns data as dictionary of names and X,y : {name1 : (X1,y1), ..., nameN : (XN,yN)}
    def get_multiple_X_y_with_names(self):
        ret = {}
        for ds in self.data_info:
            ret[ds['name']] = self.__get_X_y(ds['name'])
        return ret