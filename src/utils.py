import os
import random
import subprocess
import time
import datetime
import json

from tqdm import tqdm

import numpy as np
import seaborn as sns
import pandas as pd

TFORMAT = "%Y-%m-%dT%H:%M:%SZ"


def str_time_prop(start, end, time_format, prop):
    """Get a time at a proportion of a range of two formatted times.

    start and end should be strings specifying times formatted in the
    given format (strftime-style), giving an interval [start, end].
    prop specifies how a proportion of the interval to be taken after
    start.  The returned time will be in the specified format.
    """

    stime = time.mktime(time.strptime(start, time_format))
    etime = time.mktime(time.strptime(end, time_format))

    ptime = stime + prop * (etime - stime)

    return time.strftime(time_format, time.localtime(ptime))


def random_date(start, end=None):
    prop = random.random()
    if not end:
        end = time.time()
        end = time.strftime(TFORMAT, time.localtime(end))
    return str_time_prop(start, end, TFORMAT, prop)


def next_hours(date, hours):
    dtime = time.mktime(time.strptime(date, TFORMAT))
    ntime = datetime.datetime.fromtimestamp(dtime) + datetime.timedelta(hours=hours)
    ntime = ntime.timestamp()
    return time.strftime(TFORMAT, time.localtime(ntime))


def progress_sleep(seconds):
    for _ in tqdm(range(600)):
        time.sleep(1)


def remaining_time(elapsed):
    remaining = datetime.timedelta(hours=1) - datetime.timedelta(elapsed)
    return remaining.seconds


def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def folder_iterator(path):
    """
    Iterates over files in a folder or list of folders
    :param path: folder path
    :return: File iterator
    """
    if type(path) == list:
        folder = path.pop()
        if len(path) > 0:
            for file in folder_iterator(path):
                yield file
    else:
        folder = path
    files = os.listdir(folder)
    for file in files:
        yield os.path.join(folder, file)


class AverageMeter:
    def __init__(self):
        self.i = 0
        self.sum = 0

    def avg(self):
        try:
            return self.sum / self.i
        except ZeroDivisionError:
            return 0

    def update(self, value):
        self.sum += value
        self.i += 1


def mlflow_server():
    cmd = "mlflow server"
    cmd_env = cmd_env = os.environ.copy()
    child = subprocess.Popen(
        cmd, env=cmd_env, universal_newlines=True, stdin=subprocess.PIPE,
    )
    
    
def calculate_plot_bar(data, name, value_name=None, rotation=0):
    labels, counts = np.unique(data[name], return_counts=True)
    labels = list(map(lambda x: str(x), labels))
    source = pd.DataFrame({name: labels, 'counts': counts})
    plot_bar(source, name, value_name=value_name, rotation=rotation)

def plot_bar(data, name, value_name=None, rotation=0):
    if value_name is None:
        value_name = 'counts'
    ax = sns.barplot(x=name, y=value_name, data=data)
    sns.set(rc={'figure.figsize':(15,15)})
    ax.bar_label(ax.containers[0])
    plt.xticks(rotation=rotation)
    plt.show()