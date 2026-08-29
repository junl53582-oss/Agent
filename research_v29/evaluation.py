from research_v28.evaluation import conversion_gate, ranking_gate, selection_gate


def evaluate_three_gates(scores, equity, settings):
    ranking = ranking_gate(scores, score="v29_score")
    selection = selection_gate(scores, score="v29_score", quantile=settings.tail_quantile)
    candidate = equity[equity["mode"].eq("v29_sector_tail")].copy()
    control = equity[equity["mode"].eq("v16_replay")].copy()
    conversion = conversion_gate(candidate, control, settings.active_drawdown_floor, settings.maximum_tracking_error)
    return {"ranking": ranking, "selection": selection, "portfolio_conversion": conversion,
            "all_three_passed": ranking["passed"] and selection["passed"] and conversion["passed"]}
