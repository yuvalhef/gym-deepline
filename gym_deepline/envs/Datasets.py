import pandas as pd
from sklearn.utils import shuffle
import os
abs_path = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')

class Metadata:
    def __init__(self):
        self.attributes = []
        self.target = -1


class dataset:
    def __init__(self, path, name, metadata):
        self.id = 0
        self.dataframe = pd.read_csv(path).infer_objects()
        self.name = name
        self.metadata = metadata
        self.dataframe = shuffle(self.dataframe, random_state=0).reset_index(drop=True)
        if self.dataframe.shape[0] > 1500:
            self.dataframe = self.dataframe.iloc[: 1500, :]
        if not self.metadata.attributes:
            self.X = self.dataframe.iloc[:, :-1]
        self.Y = self.dataframe.iloc[:, -1].values


def load_all_datasets(mode='train', task='classification'):

    ds = {}
    dir_path = abs_path + '/datasets/' + task + '/' + mode + '/'
    for file in os.listdir(dir_path):
        meta = Metadata()
        meta.attributes = None
        meta.target = -1
        data = dataset(dir_path+file, file.split('.')[0], meta)
        data.id = 1
        ds[file.split('.')[0]] = data

    return ds
