import unittest

from pandas import DataFrame

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

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

    def test_numpy_random_arrays(self):
        random_array = np.random.random([6, 6])
        print(random_array)

    def test_normal_distribution(self):
        normal = np.random.normal(50, 10, size=(5, 5))
        print(normal)
        plt.plot(normal)
        plt.show()

    def test_fixed_random_array(self):
        np.random.seed(222)
        random_array = np.random.randint(0, 100, size=(6, 6))
        print(random_array)
        print("Sum of columns : {}".format(random_array.sum(axis=0)))
        print("Sum of rows : {}".format(random_array.sum(axis=1)))
        print("Sum of all elements : {}".format(random_array.sum()))

    def test_max_min_and_mean(self):
        np.random.seed(222)
        random_array = np.random.randint(0, 100, size=(6, 6))
        print(random_array)
        print("Max number for each column : {}".format(random_array.max(axis=0)))
        print("Max number for each row : {}".format(random_array.max(axis=1)))
        print("Max number for all elements : {}".format(random_array.max()))
        print("Min number for each column : {}".format(random_array.min(axis=0)))
        print("Min number for each row : {}".format(random_array.min(axis=1)))
        print("Min number for all elements : {}".format(random_array.min()))
        print("Mean for all elements : {}".format(random_array.mean()))

    def test_timing(self):
        random_big_array = np.random.randint(0, 100, size=(1000, 10000))
        t1 = time.time()
        print("Mean of array is : {}".format(str(random_big_array.mean())))
        t2 = time.time()
        print("Time taken to calculate mean was : {} seconds".format(t2 - t1))

    def test_array_masking(self):
        array = np.array([2, 8, 20, 50, 30, 1, 0, 7, 4, 40, 42])
        mean = array.mean()
        print("Mean is : {}".format(mean))
        print(array[array<mean])
        array[array<mean] = mean
        print(array)

