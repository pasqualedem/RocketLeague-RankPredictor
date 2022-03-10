import pandas as pd
import mlflow

from ruamel.yaml import YAML
from mlflow.tracking import MlflowClient
from utils import mlflow_server
from grid import GridSearch, Evaluator

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


# SCORES = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']


def parse_parameters(parameters):
    return {
        param: [
            value if not value == 'None' else None
            for value in value_list]
        for param, value_list in parameters.items()
    }


class Trainer:
    def __init__(self, param_path):
        self.exp_name = None
        self.grids = None
        self.datapath = None
        self.target_feature = None
        self.parse_grids(param_path)
        self.build_dataset()
        self.setup_mlflow()
        self.evluator = Evaluator(self.exp_id, self.datapath, self.target_feature, self.X, self.y)

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
            self.evluator.fit(model, instance_model, self.grids[model])

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
