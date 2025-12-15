import datetime
import os

import pandas as pd
from sklearn.model_selection import train_test_split


def criteo_sample_data(
    data_path: str,
    use_sklearn_split: bool = False,
    train_size: float | None = 0.9,
    test_size: float | None = 0.1,
    train_sample_size: float = None,
    test_sample_size: float = None,
    save: bool = False,
    seed: int = 1773,
    date_format: str = "%Y%m%d_%H%M",
):
    """
    Sample Criteo data from a given data path.

    Train data is sampled from the beginning of the dataset
    Test data is sampled from the end of the dataset based on test_sample_size * 2

    Parameters
    ----------
    data_path : str
        The path to the data to be sampled.
    use_sklearn_split : bool, optional
        Whether to use scikit-learn's train_test_split method for splitting the data. Defaults to False.
    train_size : float, optional
        The size of the training sample. Example: 0.9.
    test_size : float, optional
        The size of the test sample. Example: 0.1.
    train_sample_size : float
        The size of the training sample. Example: 10_000_000.
    test_sample_size : float
        The size of the test sample. Example: 2_000_000.
    save : bool, optional
        Whether to save the sampled data. Defaults to False.
    seed : int, optional
        The random seed to be used for sampling. Defaults to 1773.
    date_format : str, optional
        The format of the date and time to be included in the filename. Defaults to "%Y%m%d_%H%M".
    """
    criteo_features = (
        ["label"] + [f"I{i}" for i in range(1, 14)] + [f"C{i}" for i in range(1, 27)]
    )
    data = pd.read_csv(data_path, header=None, sep="\t", names=criteo_features)

    if use_sklearn_split:
        train_data, test_data = train_test_split(
            data, train_size=train_size, test_size=test_size, random_state=seed
        )

    else:
        # Split the data into train and test pools
        train_pool = data.iloc[: (len(data) - (test_sample_size * 2))]
        test_pool = data.tail(test_sample_size * 2)

        print(f"CRITEO: Train pool shape: {train_pool.shape}")
        print(f"CRITEO: Test pool shape: {test_pool.shape}")

        # Sample from the pools
        train_data = train_pool.sample(n=train_sample_size, random_state=seed)
        test_data = test_pool.sample(n=test_sample_size, random_state=seed)

    print(f"CRITEO: Train data shape: {train_data.shape}")
    print(f"CRITEO: Test data shape: {test_data.shape}")

    if save:
        now = datetime.datetime.now()

        train_fname = (
            f"criteo_train_{now.strftime(date_format)}_{train_sample_size}.csv"
        )
        test_fname = f"criteo_test_{now.strftime(date_format)}_{test_sample_size}.csv"

        train_fpath = os.path.join("data", "criteo_kaggle", train_fname)
        test_fpath = os.path.join("data", "criteo_kaggle", test_fname)

        train_data.to_csv(train_fpath, index=False)
        test_data.to_csv(test_fpath, index=False)

        print(f"CRITEO: Train data saved to {train_fpath}")
        print(f"CRITEO: Test data saved to {test_fpath}")

    return train_data, test_data
