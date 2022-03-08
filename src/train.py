import re

import numpy as np
import pandas as pd
import seaborn as sns
import mlflow

from ruamel.yaml import YAML
from joblib import Parallel, delayed
from mlflow.tracking import MlflowClient
from scipy.sparse import coo_matrix
from ast import literal_eval as make_tuple
from utils import mlflow_server

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

import sklearn.tree as tree
from sklearn.model_selection import ParameterGrid, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier

ESTIMATORS = {
    'linear': LogisticRegression,
    'naive_bayes': GaussianNB,
    'random_forest': RandomForestClassifier,
    'xgbclassifier': XGBClassifier,
    'knn': KNeighborsClassifier,
    'mlpclassifier': MLPClassifier,
    'dummy': DummyClassifier
}

SCORES = {
    'accuracy': accuracy_score,
    'f1_macro': lambda *args, **kwargs: f1_score(*args, **kwargs, average='macro'),
    'precision_macro': lambda *args, **kwargs: precision_score(*args, **kwargs, average='macro'),
    'recall_macro': lambda *args, **kwargs: recall_score(*args, **kwargs, average='macro'),
}
# SCORES = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
VALIDATION_JOBS = 12
GRID_JOBS = 0


def scorer(clf, X, y):
    y_pred = clf.predict(X)
    scores = {k: fun(y, y_pred) for k, fun in SCORES.items()}
    cm = confusion_matrix(y, y_pred)
    cm = pd.DataFrame(cm).stack().to_dict()
    cm = {str(k): v for k, v in cm.items()}
    return {**scores, **cm}


def parse_parameters(parameters):
    return {
        param: [
            value if not value == 'None' else None
            for value in value_list]
        for param, value_list in parameters.items()
    }


def parse_scores(scores):
    clean_scores, cm_dict = {}, {}
    estimators = scores.pop('estimator')
    for k, v in scores.items():
        try:
            _, index = re.split('test_', k)
            index = make_tuple(index)
            cm_dict[index] = v
        except ValueError:
            clean_scores[k] = v
    indices = np.array(list(cm_dict.keys()))
    rows, cols = indices[:, 0], indices[:, 1]
    data = np.array(list(cm_dict.values()))
    print(rows.shape, cols.shape, data.shape)
    shape = int(np.sqrt(cols.shape[0]))
    cms = [
        coo_matrix((data[:, k], (rows, cols)), shape=(shape, shape)).toarray()
        for k in range(data.shape[1])
    ]
    cm = np.mean(cms, axis=0)
    cm /= np.sum(cm)
    return clean_scores, cm, estimators


class Trainer:
    def __init__(self, param_path):
        self.exp_name = None
        self.grids = None
        self.datapath = None
        self.target_feature = None
        self.parse_grids(param_path)
        self.build_dataset()
        self.setup_mlflow()

    def parse_grids(self, param_path):
        with open(param_path, 'r') as param_stream:
            params = YAML().load(param_stream)
        self.datapath = params['data']['path']
        self.target_feature = params['data']['target']
        self.grids = {model: parse_parameters(parameters) for model, parameters in params['models'].items()}
        self.exp_name = params['experiment']

    def train(self):
        for model in self.grids:
            print('---------------------------------------- {} -----------------------------------------'.format(model))
            instance_model = ESTIMATORS[model]()
            self.grid_search(model, instance_model, self.grids[model])

    def grid_search(self, model_name, model, params):
        return self.evaluate_grid(model_name, model, params)

    def evaluate_grid(self, model_name, model, params):
        if params.get('parallelization'):
            grid_jobs, val_jobs, alg_jobs = params.pop('parallelization')
        else:
            grid_jobs, val_jobs, alg_jobs = GRID_JOBS, VALIDATION_JOBS, None
        grid = ParameterGrid(params)
        runs = list(iter(grid))

        def fit_score(model, params):
            print(model, '\t', params)
            run = client.create_run(experiment_id=self.exp_id)
            client.log_param(run.info.run_id, 'name', model_name)
            client.log_param(run.info.run_id, 'datapath', self.datapath)
            client.log_param(run.info.run_id, 'target', self.target_feature)
            for k, v in params.items():
                client.log_param(run.info.run_id, k, v)
            if alg_jobs:
                model.set_params(**params, n_jobs=alg_jobs)
            else:
                model.set_params(**params)
            dec_jobs = 0
            condition = True
            while condition:
                jobs = {'n_jobs': val_jobs - dec_jobs} if val_jobs - dec_jobs > 0 else {}
                try:
                    print('Validation with {} jobs'.format(val_jobs - dec_jobs))
                    scores = cross_validate(model, self.X, self.y, scoring=scorer,
                                            cv=10, return_estimator=True, **jobs)
                    condition = False
                except (OSError, MemoryError) as e:
                    print(e)
                    print('Insufficent resources')
                    dec_jobs += 1
                    condition = val_jobs >= dec_jobs

            scores, cm, estimators = parse_scores(scores)
            for key, value in scores.items():
                client.log_metric(run.info.run_id, key, value=np.mean(value))
            ax = sns.heatmap(cm, annot=True, fmt=".1f")
            ax.figure.set_size_inches(18, 18)
            client.log_figure(run.info.run_id, figure=ax.figure, artifact_file="conf_matrix.png")

        client = MlflowClient()
        if grid_jobs == 0:
            for param in runs:
                fit_score(model, param)
        else:
            parallel = Parallel(n_jobs=grid_jobs)
            parallel(delayed(fit_score)(model, param) for param in runs)

    def build_dataset(self):
        dataset = pd.read_csv(self.datapath)
        y = dataset.pop(self.target_feature)
        self.X = dataset
        self.y = y

    def setup_mlflow(self):
        mlflow_server()
        print('Server started!')
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        print('URI set!')
        exp_info = MlflowClient().get_experiment_by_name(self.exp_name)
        self.exp_id = exp_info.experiment_id if exp_info else MlflowClient().create_experiment(self.exp_name)
        print('Experiment set')
        # mlflow.sklearn.autolog(max_tuning_runs=None, log_models=True, log_post_training_metrics=True)
        print('Autolog on')


if __name__ == '__main__':
    parampath = 'parameters.yaml'
    trainer = Trainer(parampath)
    trainer.train()
