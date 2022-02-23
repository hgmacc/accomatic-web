from accomatic.Model import Model
from accomatic.Observation import Observation

from typing import List, Dict
import pandas as pd


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

"""

Clip all model output to obs dataset.

:return: overlap: dict = {'beg': datetime, 'end' : datetime}
overlap = self._obs.time_extent
        
    @property
    def date_overlap(self) -> List[str]:
        for mod in self._models:
            if mod.time_extent['beg'] >= self._obs.time_extent['beg']:
                overlap['beg'] = mod.time_extent['beg']

            if mod.time_extent['end'] <= self._obs.time_extent['end']:
                overlap['end'] = mod.time_extent['end']

        return overlap
"""


