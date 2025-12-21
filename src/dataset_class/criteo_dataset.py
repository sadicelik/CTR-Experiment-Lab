import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import Dataset


class CriteoDataset(Dataset):
    """
    Criteo Click-Through Rate Prediction Dataset

    Parameters
    ----------
    data_path : str
        Path to the Criteo dataset.
    sample_size : int, optional
        Number of samples to use for training. (default: None)
    seed : int, optional
        Random seed for reproducibility. (default: 1773)

    References
    ---------
        https://www.kaggle.com/datasets/mrkmakr/criteo-dataset?resource=download
    """

    DENSE_FEATURES = ["I" + str(i) for i in range(1, 14)]
    SPARSE_FEATURES = ["C" + str(i) for i in range(1, 27)]
    TARGET = ["label"]
    CRITEO_FEATURES = TARGET + DENSE_FEATURES + SPARSE_FEATURES

    def __init__(
        self, data_path: str = None, sample_size: int = None, seed: int = 1773
    ):

        if not data_path or not os.path.exists(data_path):
            data_path = os.path.join("data", "criteo_kaggle", "train.csv")
            print(f"CriteoDataset: Using default data path: {data_path}")

        self.data_path = data_path
        self.data = None
        self.sample_size = sample_size
        self.seed = seed

        self.load_data()
        self.preprocess()

        if self.sample_size:
            print(f"CriteoDataset: Sampling {self.sample_size} rows from the dataset")
            self.data = self.data.sample(n=self.sample_size, random_state=self.seed)

    def load_data(self):
        print(f"CriteoDataset: Loading data from: {self.data_path}")
        self.data = pd.read_csv(
            self.data_path, header=None, sep="\t", names=self.CRITEO_FEATURES
        )
        print(f"CriteoDataset: Loaded data from: {self.data_path}")

    def preprocess(self):
        print("CriteoDataset: Filling missing data...")
        # Dense features missing imputation --- maybe need more inspection
        self.data[self.DENSE_FEATURES] = self.data[self.DENSE_FEATURES].fillna(
            0,
        )
        # Sparse features missing imputation --- need string here
        self.data[self.SPARSE_FEATURES] = self.data[self.SPARSE_FEATURES].fillna(
            "-1",
        )

        print(
            f"CriteoDataset: Encoding sparse features and transforming dense features..."
        )
        mms = MinMaxScaler(feature_range=(0, 1))
        self.data[self.DENSE_FEATURES] = mms.fit_transform(
            self.data[self.DENSE_FEATURES]
        )

        for feat in self.SPARSE_FEATURES:
            lbe = LabelEncoder()
            self.data[feat] = lbe.fit_transform(self.data[feat])


if __name__ == "__main__":
    dataset = CriteoDataset(sample_size=1_000_000, seed=1773)
