import re

import numpy as np
import seaborn as sns

from joblib import Parallel, delayed
from mlflow.tracking import MlflowClient
from scipy.sparse import coo_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import ParameterGrid, cross_validate, train_test_split
from ast import literal_eval

from utils import log_matrix_artifact

VALIDATION_JOBS = 12
GRID_JOBS = 0
RANDOM_STATE = 42

SCORES = {
    'mae': mean_absolute_error,
    'rmse': lambda *args, **kwargs: mean_squared_error(*args, **kwargs, squared=False),
}
# METRICS = {
#     'roc_auc': lambda *args, **kwargs: roc_auc_score(args, **kwargs, multi_class='ovr'),
# }

LEAD_METRIC = 'rmse'


def scorer(clf, X, y):
    y_pred = clf.predict(X)
    scores = {k: fun(y, y_pred) for k, fun in SCORES.items()}
    # # cm = confusion_matrix(y, y_pred)
    # # cm = pd.DataFrame(cm).stack().to_dict()
    # # cm = {str(k): v for k, v in cm.items()}
    # return {**scores, **cm}
    return scores


def parse_scores(scores):
    clean_scores, cm_dict = {}, {}
    for k, v in scores.items():
        try:
            _, index = re.split('test_', k)
            index = literal_eval(index)
            cm_dict[index] = v
        except ValueError:
            clean_scores[k] = v
    # indices = np.array(list(cm_dict.keys()))
    # rows, cols = indices[:, 0], indices[:, 1]
    # data = np.array(list(cm_dict.values()))
    # print(rows.shape, cols.shape, data.shape)
    # shape = int(np.sqrt(cols.shape[0]))
    # cms = [
    #     coo_matrix((data[:, k], (rows, cols)), shape=(shape, shape)).toarray()
    #     for k in range(data.shape[1])
    # ]
    # cm = np.mean(cms, axis=0)
    # cm /= np.sum(cm)
    return clean_scores  # , cm


class Evaluator:
    def __init__(self, exp_id, datapath, target_feature, validate, X, y):
        self.X = X
        self.y = y
        self.exp_id = exp_id
        self.datapath = datapath
        self.target_feature = target_feature
        self.client = MlflowClient()
        self.validate = validate
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.33, random_state=RANDOM_STATE, stratify=y)

    def fit(self, model_name, model, params):
        grid_searcher = GridSearch(self, model_name, model, params)
        if self.validate:
            scores_dict = grid_searcher.fit()
            params, scores = min(scores_dict, key=lambda elem: elem[1]['test_' + LEAD_METRIC])
        else:
            print('SKIPPING VALIDATION!!')
            if len(grid_searcher.runs) > 1:
                raise NotImplementedError('Not implemented Grid search in test phase')
            params = grid_searcher.runs[0]

        run = self.client.create_run(experiment_id=self.exp_id)
        self.client.log_param(run.info.run_id, 'name', model_name)
        self.client.log_param(run.info.run_id, 'datapath', self.datapath)
        self.client.log_param(run.info.run_id, 'target', self.target_feature)
        self.client.log_param(run.info.run_id, 'type', "test")
        for k, v in params.items():
            self.client.log_param(run.info.run_id, k, v)

        model.set_params(**params)
        model.fit(self.X_train, self.y_train)
        predicts = model.predict(self.X_test)
        log_matrix_artifact(self.client, run, predicts, 'prediction.csv')
        self.calculate_metrics(predicts, run)

    def calculate_metrics(self, predicts, run):
        scores = {'test_' + score: fun(self.y_test, predicts) for score, fun in SCORES.items()}
        for key, value in scores.items():
            self.client.log_metric(run.info.run_id, key, value)
        # cm = confusion_matrix(self.y_test, predicts)
        # metrics = {metric: fun(self.y_test, predicts) for metric, fun in METRICS.items()}
        # self.client.log_metric(run.info.run_id, 'auc', metrics['auc'])
        # log_matrix_artifact(self.client, run, metrics['roc_auc'], 'roc')

        # ax = sns.heatmap(cm, annot=True, fmt=".1f")
        # ax.figure.set_size_inches(18, 18)
        # self.client.log_figure(run.info.run_id, figure=ax.figure, artifact_file="conf_matrix.png")
        # log_matrix_artifact(self.client, run, cm, 'confusion_matrix.csv')


class GridSearch:
    def __init__(self, evaluator, model_name, model, param_grid):
        self.evaluator = evaluator
        if param_grid.get('parallelization'):
            self.grid_jobs, self.val_jobs, self.alg_jobs = param_grid.pop('parallelization')
        else:
            self.grid_jobs, self.val_jobs, self.alg_jobs = GRID_JOBS, VALIDATION_JOBS, None
        grid = ParameterGrid(param_grid)
        self.runs = list(iter(grid))
        self.scores = []
        self.model_name = model_name
        self.model = model

    def fit(self):
        if self.grid_jobs == 0:
            for id, param in enumerate(self.runs):
                self.fit_score(self.model, param)
        else:
            parallel = Parallel(n_jobs=self.grid_jobs)
            parallel(delayed(self.fit_score)(self.model, self.model_name, id, param)
                     for id, param in enumerate(self.runs))
        return self.scores

    def fit_score(self, model, params):
        print(model, '\t', params)
        run = self.evaluator.client.create_run(experiment_id=self.evaluator.exp_id)
        self.evaluator.client.log_param(run.info.run_id, 'name', self.model_name)
        self.evaluator.client.log_param(run.info.run_id, 'datapath', self.evaluator.datapath)
        self.evaluator.client.log_param(run.info.run_id, 'target', self.evaluator.target_feature)
        self.evaluator.client.log_param(run.info.run_id, 'type', "validation")
        for k, v in params.items():
            self.evaluator.client.log_param(run.info.run_id, k, v)
        if self.alg_jobs:
            model.set_params(**params, n_jobs=self.alg_jobs)
        else:
            model.set_params(**params)
        dec_jobs = 0
        condition = True
        scores = None
        while condition:
            jobs = {'n_jobs': self.val_jobs - dec_jobs} if self.val_jobs - dec_jobs > 0 else {}
            try:
                print('Validation with {} jobs'.format(self.val_jobs - dec_jobs))
                scores = cross_validate(model, self.evaluator.X_train, self.evaluator.y_train, scoring=scorer,
                                        cv=10, **jobs)
                condition = False
            except (OSError, MemoryError) as e:
                print(e)
                print('Insufficent resources')
                dec_jobs += 1
                condition = self.val_jobs >= dec_jobs

        if scores is not None:
            scores = {score: np.mean(values) for score, values in scores.items()}
            self.scores.append((params, scores))
            for key, value in scores.items():
                self.evaluator.client.log_metric(run.info.run_id, key, value=value)
        else:
            print('WARNING VALIDATION FAILED')
