from accomatic.Data import *
from accomatic.Settings import *
from accomatic.Sites import *


class Experiment:
    _data: Data
    _sett: Settings
    _sites: Sites

    def __init__(self, data, sett, sites):
        self._data = data
        self._sett = sett
        self._sites = sites

    @property
    def data(self) -> Data:
        return self._data

    @property
    def sett(self) -> Settings:
        return self._sett

    @property
    def sites(self) -> List["Sites"]:
        return self._sites

    def run(self) -> None:
        if self.sett.exp_acco:
            # Runs accordance test
            pass
        if self.sett.exp_season:
            # Runs season test
            pass
        if self.sett.exp_terrain:
            # Runs terrain test
            pass

