from accomatic.Model import *
from accomatic.Observation import *

from typing import List


class Data(Model, Observation):
    # [observation_df, [models]]

    obs: Observation  # Data contents
    models: List["Model"]  # Must be unique
    model_count: int

    def __init__(self, obs=Observation, models=List["Model"]):
        self.obs = obs
        self.models = models
        self.model_count = len(models)

    def get_obs(self) -> Observation:
        return self.obs

    def get_models(self) -> List["Model"]:
        return self.models

    def get_model_count(self) -> int:
        return self.model_count

    def date_overlap(self) -> List[str]:
        # Returns overlap between all df
        overlap: List[str]

        return overlap
