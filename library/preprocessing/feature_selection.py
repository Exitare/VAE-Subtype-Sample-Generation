import pandas as pd


class FeatureSelector:
    @staticmethod
    def feature_selection(input_data: pd.DataFrame):
        input_data.drop(columns='Cell_line', inplace=True)

        return input_data
