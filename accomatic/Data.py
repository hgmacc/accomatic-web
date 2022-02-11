from accomatic.Model import Model
from accomatic.Observation import Observation

from typing import List

class Data(Model, Observation):
    _obs: Observation  # Data contents
    _models: List["Model"]  # Must be unique
    _model_count: int

    def __init__(self, obs=Observation, models=List["Model"]):
        self._obs = obs
        self._models = models
        self._model_count = len(models)


    @property
    def obs(self) -> Observation:
        return self._obs

    @property
    def models(self) -> List["Model"]:
        return self._models

    @property
    def count(self) -> int:
        return self._model_count

    @property
    def date_overlap(self) -> List[str]:
        """
        Clip all model output to obs dataset.

        :return: overlap: dict = {'beg': datetime, 'end' : datetime}
        """
        overlap = self._obs.time_extent
        for mod in self._models:
            if mod.time_extent['beg'] > overlap['beg']:
                overlap['beg'] = mod.time_extent['beg']

            if mod.time_extent['end'] < overlap['end']:
                overlap['end'] = mod.time_extent['end']

        return overlap
