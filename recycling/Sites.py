from typing import List

import pandas as pd


class Site:
    _name: str
    _elevation: float
    _terrain: str
    _lat: float
    _lon: float

    def __init__(self, name="", terr="", lon=0, lat=0, ele=0):
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


class Sites:
    _file_path: str
    _list: List["Site"]
    _count: int

    def __init__(self, file_path=""):
        df = pd.read_csv(file_path, header=0)
        self._list = []
        for row in df.index:
            site = Site(
                df["station_name"][row],
                df["ground_type"][row],
                df["longitude_dd"][row],
                df["latitude_dd"][row],
                df["elevation_m"][row],
            )
            self._list.append(site)

    @property
    def list(self) -> List["Site"]:
        return self._list

    @property
    def count(self) -> int:
        return len(self._list)
