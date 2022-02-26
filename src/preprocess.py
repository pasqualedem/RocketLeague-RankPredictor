import os
from tqdm import tqdm
from utils import load_json, folder_iterator

import pandas as pd


def prepare_data(inpath):
    """

    :param inpath: folder or list of folders containing json data
    :param outpath: csv output file
    :return:
    """
    if not type(inpath) == list:
        inpath = [inpath]

    dataframe = pd.concat([
        prepare_replay(load_json(file))
        for file in tqdm(folder_iterator(inpath))
    ])
    return dataframe


def prepare_replay(replay):
    return pd.DataFrame([{
        **{k: v for stat in player['stats'].values() for k, v in stat.items()},
        **{'tier': player['rank']['tier']}
    }
        for player in
        filter(
            lambda player: player.get('rank') is not None,
            replay['blue']['players'] + replay['orange']['players']

        )])


def fix_data(data):
    # Column is Nan if no goals are taken
    data['goals_against_while_last_defender'].fillna(0, inplace=True)

    # Clean data
    na_possible_cols = ['percent_closest_to_ball', 'percent_farthest_from_ball', 'percent_most_forward',
                        'avg_distance_to_mates', 'time_most_back']
    for na_col in na_possible_cols:
        data.drop(data[data[na_col].isna()].index, inplace=True)
        data.reset_index(inplace=True, drop=True)

    return data


if __name__ == '__main__':
    outfile = "data/preprocessed/dataset.csv"
    base_folder = "data/raw"
    paths = os.listdir(base_folder)
    paths = [os.path.join(base_folder, folder) for folder in paths]
    dataframe = prepare_data(paths)
    dataframe.to_csv(outfile, index=False)

    # dataframe = pd.read_csv(outfile)
    outfile = "data/preprocessed/dataset_fixed.csv"
    dataframe = fix_data(dataframe)
    dataframe.to_csv(outfile, index=False)
