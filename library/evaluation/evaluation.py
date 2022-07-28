import pandas as pd
from sklearn.metrics import r2_score
from pathlib import Path
import mlflow


class Evaluation:

    @staticmethod
    def calculate_r2_score(test_data: pd.DataFrame, reconstructed_data: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Calculates the r2 scores for the given parameters
        @param test_data: The input data to evaluate
        @param reconstructed_data: The reconstructed data to evaluate
        @param columns: The columns of the dataset
        @return: Returns a dataframe containing all r2 scores
        """
        r2_scores = pd.DataFrame(columns=["Column", "Score"])
        recon_test = pd.DataFrame(data=reconstructed_data, columns=columns)
        test_data = pd.DataFrame(data=test_data, columns=columns)

        for column in columns:
            ground_truth_marker = test_data[f"{column}"]
            reconstructed_marker = recon_test[f"{column}"]

            score = r2_score(ground_truth_marker, reconstructed_marker)
            r2_scores = r2_scores.append(
                {
                    "Column": column,
                    "Score": score
                }, ignore_index=True
            )

        return r2_scores
