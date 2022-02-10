from typing import List
import pandas as pd
from accomatic.Site import *


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
