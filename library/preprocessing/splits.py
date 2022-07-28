import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple


class SplitHandler:

    @staticmethod
    def create_splits(input_data: pd.DataFrame, without_val: bool = False) -> Tuple:
        """
        Creates train val test split of the data provided
        @param input_data: The input data which should be split
        @param without_val: If true only a test and a train set will be created
        @return: A tuple containing all the sets.
        """

        # No validation set will be created
        if without_val:
            return train_test_split(input_data, test_size=0.2, random_state=1, shuffle=True)

        # Create validation set
        X_dev, X_val = train_test_split(input_data, test_size=0.05, random_state=1, shuffle=True)
        X_train, X_test = train_test_split(X_dev, test_size=0.25, random_state=1, shuffle=True)
        return X_train, X_val, X_test
