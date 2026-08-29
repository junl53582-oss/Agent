from research_v22 import replay as parent
from research_v22r1.schedule import schedule_from_parent
from research_v28.replay import confidence_optimize, portfolio_input


MODES = ("v16_replay", "v29_sector_tail")
SCORE_COLUMNS = {"v16_replay": "v16_score", "v29_sector_tail": "v29_score"}


def run_replay(scores, book, membership, schedule, settings, progress=None, checkpoint=None):
    old = (parent.MODES, parent.SCORE_COLUMNS, parent.portfolio_input, parent.optimize_v16)
    try:
        parent.MODES, parent.SCORE_COLUMNS = MODES, SCORE_COLUMNS
        parent.portfolio_input, parent.optimize_v16 = portfolio_input, confidence_optimize
        return parent.run_replay(scores, book, membership, schedule, settings, progress, checkpoint)
    finally:
        parent.MODES, parent.SCORE_COLUMNS, parent.portfolio_input, parent.optimize_v16 = old


load_scores = parent.load_scores
attach_volatility = parent.attach_volatility
compare_control = parent.compare_control
