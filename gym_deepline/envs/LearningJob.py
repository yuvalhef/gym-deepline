from . metrics import Accuracy, MSE
from . import Datasets as ds
import numpy as np

metrics = [Accuracy().name, Accuracy(balanced=False).name, MSE().name]
# datasets = ds.load_all_datasets()
# # ds_list = [d.name for d in datasets.values()]
tasks = ['Classification', 'Regression']


class Learning_Job:
    def __init__(self, task, metric, dataset, name):
        self.task = task
        self.metric = metric
        self.dataset = dataset
        self.name = name
        self.lj_vector = None
        self.base_scores = None

    def to_vector(self):
        metric_vec = np.zeros(len(metrics))
        metric_vec[metrics.index(self.metric.name)] = 1
        task_vec = np.zeros(len(tasks))
        task_vec[tasks.index(self.task)] = 1
        self.lj_vector = np.concatenate((task_vec, metric_vec))
        return self.lj_vector


def load_all_learning_jobs(mode='train', task='Classification', metric=Accuracy()):

    leaning_jobs = {}
    datasets = ds.load_all_datasets(mode=mode, task=task)

    for dataset in datasets.values():
        name = dataset.name
        lj = Learning_Job(task, metric, datasets[name], name)
        leaning_jobs[name] = lj

    return leaning_jobs