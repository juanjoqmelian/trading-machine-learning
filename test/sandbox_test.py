import unittest

from pandas import DataFrame

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from data_frame_creator import DataFrameCreator


class SandboxTest(unittest.TestCase):

    def test_read_csv(self):
        creator = DataFrameCreator()
        csv: DataFrame = creator.read('../resources/GOOG.csv')
        print(csv[2:4])

    def test_max_close(self):
        creator = DataFrameCreator()
        csv: DataFrame = creator.read('../resources/GOOG.csv')
        print(csv['Close'].max())


    def test_show_plot(self):
        creator = DataFrameCreator()
        csv: DataFrame = creator.read('../resources/GOOG.csv')
        csv['AdjClose'].plot()
        plt.show()

    def test_show_two_columns_plot(self):
        creator = DataFrameCreator()
        csv: DataFrame = creator.read('../resources/GOOG.csv')
        csv[['Close', 'AdjClose']].plot()
        plt.show()

    def test_create_dataframe_from_date_range(self):
        creator = DataFrameCreator()
        data_frame = creator.from_date_range('2018-01-01', '2019-01-01')
        print(data_frame)

    def test_join_two_data_frames(self):
        dates = pd.date_range('2012-09-07', '2012-09-11')
        data_frame_from_dates = pd.DataFrame(index=dates)
        goog = pd.read_csv('../resources/GOOG.csv', index_col='Date', parse_dates=True, na_values=0.0)
        print(goog)
        specific_dates_from_goog = data_frame_from_dates.join(goog, how='inner')
        print(specific_dates_from_goog)
        specific_dates_from_goog['Volume'].plot()
        plt.show()

    def test_numpy_arrays_init(self):
        one_d_array = np.array([2, 3, 4, 5])
        print(type(one_d_array))
        print(one_d_array)
        two_d_array = np.array([(2, 3, 4), (5, 6, 7)])
        print(type(two_d_array))
        print(two_d_array)
        empty_array = np.empty(5)
        print(empty_array)
        ones_array = np.ones((5, 2))
        print(ones_array)
        ones_ints_array = np.ones((5, 2), dtype=np.int_)
        print(ones_ints_array)
