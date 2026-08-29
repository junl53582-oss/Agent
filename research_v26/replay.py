from research_v22 import replay as parent
from research_v22r1.schedule import schedule_from_parent


MODES = ("v16_replay", "directional_probability")
SCORE_COLUMNS = {"v16_replay": "v16_score", "directional_probability": "directional_probability_score"}


def run_replay(scores, book, membership, schedule, settings, progress=None, checkpoint=None):
    old_modes, old_columns = parent.MODES, parent.SCORE_COLUMNS
    try:
        parent.MODES, parent.SCORE_COLUMNS = MODES, SCORE_COLUMNS
        return parent.run_replay(scores, book, membership, schedule, settings, progress, checkpoint)
    finally:
        parent.MODES, parent.SCORE_COLUMNS = old_modes, old_columns


load_scores = parent.load_scores
attach_volatility = parent.attach_volatility
compare_control = parent.compare_control

