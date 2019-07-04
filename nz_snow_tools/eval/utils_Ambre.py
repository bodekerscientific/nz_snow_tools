import numpy as np

def maxmin(time, swe):
    """
    return the maximum, the minimum and the date of each one of a series of values

    :param time: time serie
    :param swe: snow water equivalent series
    :return:
    """
    assert time.ndim == 1 and swe.ndim == 1 and len(time) == len(swe)

    maximum = np.max(swe)
    minimum = np.min(swe)

    index_of_maximum = np.where(swe == maximum)
    index_of_minimum = np.where(swe == minimum)

    date_max = time[index_of_maximum][0]
    date_min = time[index_of_minimum][0]

    return maximum, minimum, date_max, date_min