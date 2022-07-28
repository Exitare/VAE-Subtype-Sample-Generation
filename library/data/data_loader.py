from typing import Optional
import pandas as pd
from pathlib import Path


class DataLoader:

    @staticmethod
    def load_data(file_name: str) -> Optional[pd.DataFrame]:
        """
        Depending on the suffix (file extensions) loads a tsv or a csv dataframe
        """
        path: Path = Path(file_name)
        if path.suffix == '.tsv':
            return pd.read_csv(file_name, sep='\t', index_col=0)
        else:
            return pd.read_csv(file_name)
