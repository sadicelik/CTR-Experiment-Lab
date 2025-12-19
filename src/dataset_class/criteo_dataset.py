import os

import pandas as pd
from torch.utils.data import Dataset


class CriteoDataset(Dataset):
    def __init__(self, data_path: str):
        """
        Criteo Click-Through Rate Prediction Dataset

        Parameters
        ----------
        data_path : str
            Path to the Criteo dataset.
        """
        self.data_path = data_path

        if not os.path.exists(data_path):
            raise FileNotFoundError(data_path)

        self.data = None

        self.load_data()

    def load_data(self) -> None:
        print(f"CriteoDataset: Loading data from: {self.data_path}")
        self.data = pd.read_csv(self.data_path)
        print(f"CriteoDataset: Loaded data from: {self.data_path}")
