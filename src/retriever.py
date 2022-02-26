import os.path
import time

import requests

from utils import random_date, next_hours, save_json, AverageMeter, remaining_time, progress_sleep
from ranks import RANKS

token = '7rAVl0019gzHIQFoBTRtIlgr4YOkP7xZiMK7gAgI'

url = "https://ballchasing.com/api/replays"

FIRST_DATE = "2021-08-15T00:00:00Z"
SECON_DATE = "2022-02-20T00:00:00Z"

MIN_RANK = "champion-1"
MAX_RANK = "champion-3"
PRO = "false"

REP_PER_RANK = 200
REP_COUNT = 20
SLEEP = 0.6

BASE_FOLDER = 'data/raw'

PARAMETERS = {
    "count": REP_COUNT,
    "playlist": "ranked-standard",
    "min-rank": MIN_RANK,
    "max-rank": MAX_RANK,
    "pro": PRO,
    "replay-date-after": "2022-02-07T00:00:00Z",
}


class RateLimitError(Exception):
    pass


def retrieve_replays(parameters):
    try:
        r = requests.get(url, params=parameters, headers={
            'Authorization': token})
    except ValueError:
        return None
    except requests.exceptions.ChunkedEncodingError:
        return None
    if not r.status_code == 200:
        print('WARNING: Status {}'.format(r.status_code))
        return None
    return r.json()


def request_replay(id):
    try:
        r = requests.get(url + "/" + id, headers={
            'Authorization': token})
    except ValueError:
        return None
    except requests.exceptions.ChunkedEncodingError:
        return None
    if not r.status_code == 200:
        print('WARNING: Status {}'.format(r.status_code))
        if r.status_code == 429:
            raise RateLimitError
        return None
    return r.json()


def get_single_replays(replay_list, save_path):
    for replay in replay_list:
        replay_json = request_replay(replay['id'])
        if not replay_json:
            return
        print('Got: {}'.format(replay['id']))
        time.sleep(SLEEP)
        os.makedirs(save_path, exist_ok=True)
        pathfile = os.path.join(save_path, replay['id'] + '.json')
        save_json(replay_json, pathfile)


def replay_exist(save_folder, id):
    return os.path.exists(
        os.path.join(save_folder, id + '.json')
    )


class ReplayRetriever:
    def __init__(self, ranks):
        self.ranks = ranks
        os.makedirs(BASE_FOLDER, exist_ok=True)
        self.init_timer = time.time()

    def get_parameters(self, min_rank, max_rank, pros, delta_hours):
        date_after = random_date(FIRST_DATE, SECON_DATE)
        date_before = next_hours(date_after, hours=delta_hours)
        paramaters = PARAMETERS.copy()
        print('Random date: {}'.format(date_after))
        paramaters["replay-date-after"] = date_after
        paramaters["replay-date-before"] = date_before
        paramaters['min-rank'] = min_rank
        paramaters['max-rank'] = max_rank
        paramaters['pros'] = pros
        return paramaters

    def get_replays_over_rank(self, min_rank, max_rank, pros, delta_hours):
        remaining = REP_PER_RANK
        expected_iterations = remaining // REP_COUNT
        avg_retrieved = AverageMeter()
        save_folder = os.path.join(BASE_FOLDER, "{}-{}".format(min_rank, max_rank))
        print('------------------------ CURRENT {} - {} ------------------------'.format(min_rank, max_rank))
        while remaining > 0:
            parameters = self.get_parameters(min_rank, max_rank, pros, delta_hours)
            try:
                replays = retrieve_replays(parameters)
            except RateLimitError:
                self.cooldown()
                continue
            if replays:
                retrieved = len(replays['list'])
                counted = replays['count'] if replays.get('count') else 0
                replays_to_get = list(filter(lambda x: not replay_exist(save_folder, x['id']), replays['list']))
                n_considered = len(replays_to_get)
                print('Match filters   : {} \n'
                      'Retrieved       : {} \n'
                      'Considered      :  {} \t avg. considered: {} \n'
                      'Expected its    : {} \t Actual its     : {} \n'
                      'Remaining       : {} \n'
                      .format(counted,
                              retrieved,
                              n_considered, avg_retrieved.avg(),
                              expected_iterations, avg_retrieved.i,
                              remaining))
                avg_retrieved.update(n_considered)
                if n_considered > 0:
                    remaining -= n_considered
                    try:
                        get_single_replays(replays_to_get, save_folder)
                    except RateLimitError:
                        self.cooldown()
                        continue
                else:
                    print('Considered 0 replays, skip')
                    time.sleep(SLEEP)
            else:
                time.sleep(SLEEP)
            if avg_retrieved.i > 10 and avg_retrieved.avg() < 1:
                break

    def cycle_ranks(self):
        while True:
            for rank in self.ranks:
                self.get_replays_over_rank(rank.min_rank, rank.max_rank, rank.pros, rank.time_range)

    def cooldown(self):
        elapsed = time.time() - self.init_timer
        cooldown = remaining_time(elapsed)
        progress_sleep(cooldown)
        self.init_timer = time.time()


if __name__ == '__main__':
    retr = ReplayRetriever(RANKS)
    retr.cycle_ranks()