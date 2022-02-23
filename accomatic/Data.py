from typing import Dict, List

import pandas as pd

from accomatic.Model import Model
from accomatic.Observation import Observation


class Data:
    _obs: Observation
    _models: List["Model"]
    _model_count: int

    def __init__(self, obs, models):
        self._obs = obs
        self._model_count = len(models)
        self._models = models

    @property
    def obs(self) -> Observation:
        return self._obs

    @property
    def models(self) -> List["Model"]:
        return self._models

    @property
    def count(self) -> int:
        return self._model_count

