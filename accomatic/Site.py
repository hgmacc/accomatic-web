from typing import List
import pandas as pd


class Site:
    _name: str
    _elevation: float
    _terrain: str
    _lat: float
    _lon: float

    def __init__(self, name, terr, lon, lat, ele):
        self._name = name
        self._terrain = terr
        self._lon = lon
        self._lat = lat
        self._elevation = ele

    def __eq__(self, other):
        if isinstance(other, Site):
            return self.name == other.name

    @property
    def name(self):
        return self._name
