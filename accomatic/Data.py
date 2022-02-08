"""
inherits from Model, Observation
list of models + observation
clsmethods: getDateOverlap

"""

import os
import sqlite3
import pandas as pd

from typing import List

class NotFound(Exception):
    pass


class Data():
    name: str   # Must be unique
    df: pd.DataFrame    # Data contents
    stats: dict     # Individual results ultimatly stored here

    def __init__(self, age=0):
        self._age = age


    def get_name(self):
        return self._name


    def set_name(self, name):
        self._name = name





    def list(cls) -> List['Article']:
        con = sqlite3.connect(os.getenv('DATABASE_NAME', 'database.db'))
        con.row_factory = sqlite3.Row

        cur = con.cursor()
        cur.execute("SELECT * FROM articles")

        records = cur.fetchall()
        articles = [cls(**record) for record in records]
        con.close()

        return articles

    def save(self) -> 'Article':
        with sqlite3.connect(os.getenv('DATABASE_NAME', 'database.db')) as con:
            cur = con.cursor()
            cur.execute(
                "INSERT INTO articles (id,author,title,content) VALUES(?, ?, ?, ?)",
                (self.id, self.author, self.title, self.content)
            )
            con.commit()

        return self



raj = Model()

# setting the age using setter
raj.set_age(21)

# retrieving age using getter
print(raj.get_age())

print(raj._age)