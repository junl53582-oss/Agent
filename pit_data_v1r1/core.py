from dataclasses import dataclass
from pathlib import Path

from pit_data_v1.core import ObservationSettings as ParentSettings
from pit_data_v1.core import observe as parent_observe

from .source import fetch_flow_pages


@dataclass(frozen=True)
class ObservationSettings(ParentSettings):
    version: str = "pit-data-v1r1"
    data_root: Path = Path("data/pit_observations_v1r1")
    artifact_root: Path = Path("artifacts/pit_data_v1r1")


def observe(target_date=None, *, now=None, settings=None, expectation_fetcher=None, flow_fetcher=None):
    settings = settings or ObservationSettings()
    kwargs = {"now": now, "settings": settings, "flow_fetcher": flow_fetcher or fetch_flow_pages}
    if expectation_fetcher is not None:
        kwargs["expectation_fetcher"] = expectation_fetcher
    return parent_observe(target_date, **kwargs)
