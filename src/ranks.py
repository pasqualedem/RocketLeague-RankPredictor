class Rank:
    def __init__(self, min_rank, max_rank, time_range, pros='false'):
        self.min_rank = min_rank
        self.max_rank = max_rank
        self.pros = pros
        self.time_range = time_range


RANKS = [
    Rank('bronze-1', 'bronze-3', 400),
    Rank('silver-1', 'silver-3', 10),
    Rank('gold-1', 'gold-3', 5),
    Rank('platinum-1', 'platinum-3', 5),
    Rank('diamond-1', 'diamond-3', 5),
    Rank('champion-1', 'champion-3', 5),
    Rank('grand-champion', '', 5),
    Rank('grand-champion', '', 5, 'true'),
]
