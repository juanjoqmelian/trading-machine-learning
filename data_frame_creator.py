import pandas as pd


class DataFrameCreator:

    def read(self, file_location) -> pd.DataFrame:
        return pd.read_csv(file_location)

    def from_date_range(self, start_date, end_date) -> pd.DataFrame:
        dates = pd.date_range(start_date, end_date)
        return pd.DataFrame(index=dates)
