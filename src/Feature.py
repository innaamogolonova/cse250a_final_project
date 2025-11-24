import pandas as pd
import numpy as np


class Feature:

    name: str
    columns: list[str]
    map_important_pref = {"[0-15]": 0, "[16-20]": 1, "[21-100]": 2}
    map_regular = {"[0-5]": 0, "[6-8]": 1, "[9-10]": 2}
    raw_data: pd.DataFrame
    encoded_data: np.ndarray

    def __init__(self, name, columns, data, n_x_state=3) -> None:
        self.name = name
        self.columns = columns
        self.raw_data = data
        self.N_X_STATE = n_x_state

    def discretization(self, column_name, value) -> int:

        if "important" in column_name or "pref" in column_name:
            return self.map_important_pref[value]
        else:
            return self.map_regular[value]

    def encode(self) -> None:
        encoded_cols = []

        for column in self.columns:
            encoded_col = self.raw_data[column].apply(
                lambda v: self.discretization(column, v)
            )
            encoded_cols.append(encoded_col.values)

        data = np.vstack(encoded_cols).T
        powers = self.N_X_STATE ** np.arange(data.shape[1])[::-1]
        self.encoded_data = (data * powers).sum(axis=1)
