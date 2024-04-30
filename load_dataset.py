import pandas as pd
import os
def load_dataset(dataset_dir):
    """
    用pandas read json直接读起来。
    """
    file_names = ['cosponsors.json','members.json','votes.json']
    pathes = [os.path.join(dataset_dir, file_name) for file_name in file_names]
    data = [pd.read_json(path) for path in pathes]
    return data

def split_data(df):
    pass