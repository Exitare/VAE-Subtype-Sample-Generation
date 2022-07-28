import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple


class Preprocessing:
    available_options = ["s", "min"]

    @staticmethod
    def normalize(data: pd.DataFrame, features: list, method: str = "s", feature_range: Tuple = None,
                  scaler=None) -> (pd.DataFrame, any):
        """
        @param data The data to be normalized
        @param features The features of the dataset
        @param method The method which should be used for normalization
        @param feature_range Feature range. Is required for min max scaler for example.
        If no range is provided a default -1 to 1 is being used
        """

        # Input data contains some zeros which results in NaN (or Inf)
        # values when their log10 is computed. NaN (or Inf) are problematic
        # values for downstream analysis. Therefore, zeros are replaced by
        # a small value; see the following thread for related discussion.
        # https://www.researchgate.net/post/Log_transformation_of_values_that_include_0_zero_for_statistical_analyses2

        data = data.where(data != 0, other=1e-32)
        # data = data.apply(lambda x: np.log10(x) if np.issubdtype(x.dtype, np.number) else x)
        data = np.log10(data)

        # filter numeric columns
        # num_cols = data.columns[data.dtypes.apply(lambda c: np.issubdtype(c, np.number))]

        if method == "s":
            if scaler is None:
                scaler = StandardScaler()
                scaler.fit(data)

            data = scaler.transform(data)

        elif method == "min":
            if feature_range is not None and scaler is None:
                scaler = MinMaxScaler(feature_range=feature_range)
                scaler.fit(data)
            elif feature_range is None and scaler is None:
                scaler = MinMaxScaler(feature_range=(-1, 1))
                scaler.fit(data)

            data = scaler.transform(data)

        else:
            raise f"Please provide a valid normalization method. Select one of these options: " \
                  f"{[option for option in Preprocessing.available_options]}"

        return pd.DataFrame(columns=features, data=data), scaler
