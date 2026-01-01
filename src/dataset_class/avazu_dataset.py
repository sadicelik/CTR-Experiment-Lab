import os

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from tqdm import tqdm

from ..utils.torch_utils import get_device


class AvazuCTRDataset(Dataset):
    """
    Avazu Click-Through Rate Prediction Dataset

    Parameters
    ----------
        data_path : str
            Path to Avazu click-through rate prediction dataset.
        drop_hour : bool, optional
            Whether to drop the original hour column.
        parse_hour : bool, optional
            Whether to parse the hour column.
        drop_id : bool, optional
            Whether to drop the original id column.
        sample_size : int, optional
            Number of samples selected out from the dataset.
        seed : int, optional
            Random seed for reproducibility.

    References
    ---------
        https://www.kaggle.com/c/avazu-ctr-prediction
    """

    TARGET = "click"

    ID_FEATURE = ["id"]

    CAT_FEATURES = [
        "C1",
        "banner_pos",
        "site_id",
        "site_domain",
        "site_category",
        "app_id",
        "app_domain",
        "app_category",
        "device_id",
        "device_ip",
        "device_model",
        "device_type",
        "device_conn_type",
        "C14",
        "C15",
        "C16",
        "C17",
        "C18",
        "C19",
        "C20",
        "C21",
    ]

    EXTRA_FEATURE_NAMES = ["hour_of_day"]

    def __init__(
        self,
        data_path: str = None,
        is_preprocess: bool = True,
        drop_hour: bool = True,
        parse_hour: bool = False,
        drop_id: bool = True,
        sample_size: int = None,
        seed: int = 1773,
        test_size: float = None,
    ) -> None:
        # Initialization
        self.data_path = data_path
        self.device = self._get_device()
        self.is_preprocess = is_preprocess
        self.drop_hour = drop_hour
        self.parse_hour = False if drop_hour else parse_hour
        self.drop_id = drop_id
        self.sample_size = sample_size
        self.seed = seed
        self.test_size = test_size

        self.load_data()

        if self.is_preprocess:
            self.preprocess()

        if self.sample_size:
            print(f"AvazuCTRDataset: Sampling {self.sample_size} rows from the dataset")
            sampled_data = self.data.sample(n=self.sample_size, random_state=self.seed)
            self.data = sampled_data.reset_index(drop=True)

            data_fpath = os.path.join(
                "data", "avazu", f"avazu_sample_{self.sample_size}.csv"
            )
            self.data.to_csv(data_fpath, index=False)
            print(f"AvazuCTRDataset: Saved sampled data to: {data_fpath}")

        if self.test_size:
            self.record_train_test_split()

        self.prepare()

    def load_data(self) -> None:
        print(f"AvazuCTRDataset: Loading data from: {self.data_path}")
        self.data = pd.read_csv(self.data_path)
        print(f"AvazuCTRDataset: Loaded data from: {self.data_path}")

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int):
        x = torch.tensor(
            self.x[index], dtype=torch.int64, device=self.device
        )  # Cast int for other models, float for MLP
        y = torch.tensor(
            self.y[index], dtype=torch.float64, device=self.device
        )  # Cast float for loss calculation
        return x, y

    ############################ DATA PREPROCESSING ############################

    def preprocess(self) -> None:
        """Preprocessing operations."""
        if self.drop_hour:
            self.data.drop(columns=["hour"], inplace=True)
            print("AvazuCTRDataset: Dropped original hour column.")

        if self.parse_hour:
            self.parse_avazu_hour()

        if self.drop_id:
            self.data.drop(columns=["id"], inplace=True)
            print("AvazuCTRDataset: Dropped original id column.")

        self.label_encoding()

    def parse_avazu_hour(self, drop_hour: bool = True) -> None:
        """
        Avazu "hour" is YYMMDDHH as int. Example: 14102100 -> 2014-10-21 00:00
        """
        # Datetime conversion and parsing for hour feature
        self.data["hour"] = pd.to_datetime(
            self.data["hour"], format="%y%m%d%H", errors="coerce", utc=True
        )

        # No need for year and month here, all records are in October, 2014
        # And also for day because we are training with days from past to predict a future day
        # self.data["day"] = self.data["hour"].dt.day
        # self.data["weekday"] = self.data["hour"].dt.weekday
        self.data["hour_of_day"] = self.data["hour"].dt.hour

        print(f"AvazuCTRDataset: Parsed hour into hour_of_day")

        if drop_hour:
            self.data.drop(columns=["hour"], inplace=True)
            print("AvazuCTRDataset: Dropped original hour column.")

    def label_encoding(self) -> None:
        """Label encoding categorical features."""
        print(f"AvazuCTRDataset: Label encoding categorical features.")

        for feature in tqdm(self.CAT_FEATURES):
            lbe = LabelEncoder()
            self.data[feature] = lbe.fit_transform(self.data[feature])

        print(
            f"AvazuCTRDataset: Label encoded categorical features {list(self.CAT_FEATURES)}."
        )

    def _get_field_dims(self):
        """Gets dimensions of features for embedding."""
        for feature in self.x.columns:
            self.feature_unique_size[feature] = self.x[feature].nunique()

        return self.feature_unique_size.values()

    def prepare(self):
        """Sets x,y and field_dims values for the model."""
        self.x = self.data.drop(columns=[self.TARGET])
        self.y = self.data[self.TARGET]

        self.feature_unique_size = {}
        self.field_dims = self._get_field_dims()

        self.x = self.x.values
        self.y = self.y.values

    ############################ HELPER FUNCTIONS ############################

    @staticmethod
    def _get_device():
        """Get device for tensor operations."""
        device = get_device()
        print(f"AvazuCTRDataset: Tensors will be created on device {device}.")
        return device

    def record_train_test_split(self):
        print(f"AvazuCTRDataset: Splitting data into train and test sets.")
        avazu_train, avazu_test = train_test_split(
            self.data, test_size=self.test_size, random_state=self.seed
        )
        train_fpath = os.path.join(
            "data", "avazu", f"avazu_train_{len(avazu_train)}_{self.seed}.csv"
        )
        test_fpath = os.path.join(
            "data", "avazu", f"avazu_test_{len(avazu_test)}_{self.seed}.csv"
        )
        avazu_train.to_csv(train_fpath, index=False)
        avazu_test.to_csv(test_fpath, index=False)

        print(f"AvazuCTRDataset: Saved train data to: {train_fpath}")
        print(f"AvazuCTRDataset: Saved test data to: {test_fpath}")
