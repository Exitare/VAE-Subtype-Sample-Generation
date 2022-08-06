import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple


class Scaling:

    @staticmethod
    def standardize(data: pd.DataFrame, features: list, scaler=None) -> (pd.DataFrame, any):
        """
        @param data The data to be normalized
        @param features The features of the dataset
        @param scaler The scaler to use
        """

        # Input data contains some zeros which results in NaN (or Inf)
        # values when their log10 is computed. NaN (or Inf) are problematic
        # values for downstream analysis. Therefore, zeros are replaced by
        # a small value; see the following thread for related discussion.
        # https://www.researchgate.net/post/Log_transformation_of_values_that_include_0_zero_for_statistical_analyses2

        data = data.where(data != 0, other=1e-32)
        # print(np.any(np.isnan(data)))

        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(data)

        data = scaler.transform(data)

        return pd.DataFrame(columns=features, data=data), scaler

    @staticmethod
    def normalize(data: pd.DataFrame, features: list, feature_range: Tuple = None, scaler=None) -> (pd.DataFrame, any):
        if feature_range is not None and scaler is None:
            scaler = MinMaxScaler(feature_range=feature_range)
            scaler.fit(data)
        elif feature_range is None and scaler is None:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler.fit(data)

        data = scaler.transform(data)

        return pd.DataFrame(columns=features, data=data), scaler
