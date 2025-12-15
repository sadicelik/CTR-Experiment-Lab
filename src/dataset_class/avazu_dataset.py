import os

import pandas as pd
import torch
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
        label_encode : bool, optional
            Whether to label encode categorical features.

    Reference
    ---------
        https://www.kaggle.com/c/avazu-ctr-prediction
    """

    TARGET_COLUMN = "click"

    ID_FEATURE_NAME = ["id"]

    CAT_FEATURE_NAMES = [
        "id",
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
        data_path: str,
        drop_hour: bool = False,
        parse_hour: bool = True,
        drop_id: bool = False,
        label_encode: bool = True,
    ) -> None:

        if not os.path.exists(data_path):
            raise FileNotFoundError(data_path)

        # Initialization
        self.data_path = data_path
        self.data = None
        self.device = self._get_device()
        self.load_data()
        self.init()

        self.x = self.data.drop(columns=[self.TARGET_COLUMN])
        self.y = self.data[self.TARGET_COLUMN]

        # Preprocessing
        self.drop_hour = drop_hour
        self.parse_hour = False if drop_hour else parse_hour
        self.drop_id = drop_id
        self.label_encode = label_encode

        if self.drop_hour:
            self.x.drop(columns=["hour"], inplace=True)
            print("AvazuCTRDataset: Dropped original hour column.")

        if self.parse_hour:
            self.parse_avazu_hour()

        if self.drop_id:
            self.x.drop(columns=["id"], inplace=True)
            print("AvazuCTRDataset: Dropped original id column.")

        if label_encode:
            self.label_encoding()

        self.field_dims = self._get_field_dims()

        self.x = self.x.values
        self.y = self.y.values

    def load_data(self) -> None:
        print(f"AvazuCTRDataset: Loading data from: {self.data_path}")
        self.data = pd.read_csv(self.data_path)
        print(f"AvazuCTRDataset: Loaded data from: {self.data_path}")

    def init(self) -> None:
        self.feature_names = list(self.data.columns)
        self.n_samples = self.data.shape[0]
        self.n_features = self.data.shape[1]
        self.feature_unique_size = {}

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int):
        x = torch.tensor(
            self.x[index], dtype=torch.int64, device=self.device
        )  # Cast int for other models, float for MLP
        y = torch.tensor(
            self.y[index], dtype=torch.float64, device=self.device
        )  # Cast float for loss calculation
        return x, y

    ############################ DATA PREPROCESSING ############################

    def parse_avazu_hour(self, drop_hour: bool = True) -> None:
        """
        Avazu "hour" is YYMMDDHH as int. Example: 14102100 -> 2014-10-21 00:00
        """
        # Datetime conversion and parsing for hour feature
        self.x["hour"] = pd.to_datetime(
            self.x["hour"], format="%y%m%d%H", errors="coerce", utc=True
        )

        # No need for year and month here, all records are in October, 2014
        # And also for day because we are training with days from past to predict a future day
        # self.x["day"] = self.x["hour"].dt.day
        # self.x["weekday"] = self.x["hour"].dt.weekday
        self.x["hour_of_day"] = self.x["hour"].dt.hour

        print(f"AvazuCTRDataset: Parsed hour into hour_of_day")

        if drop_hour:
            self.x.drop(columns=["hour"], inplace=True)
            print("AvazuCTRDataset: Dropped original hour column.")

    def label_encoding(self) -> None:
        """Label encoding categorical features."""
        for feature in tqdm(self.x.columns):
            lbe = LabelEncoder()
            self.x[feature] = lbe.fit_transform(self.x[feature])

        print(
            f"AvazuCTRDataset: Label encoded categorical features {list(self.x.columns)}."
        )

    def _get_field_dims(self):
        """Gets dimensions of features for embedding."""
        for feature in self.x.columns:
            self.feature_unique_size[feature] = self.x[feature].nunique()

        return self.feature_unique_size.values()

    ############################ HELPER FUNCTIONS ############################

    @staticmethod
    def _get_device():
        """Get device for tensor operations."""
        device = get_device()
        print(f"AvazuCTRDataset: Tensors will be created on device {device}.")
        return device
