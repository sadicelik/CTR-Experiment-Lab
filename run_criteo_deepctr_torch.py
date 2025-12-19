import argparse
import datetime
import os

import pandas as pd
from deepctr_torch.inputs import DenseFeat, SparseFeat, get_feature_names
from deepctr_torch.models import DeepFM
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from src.utils.logging_utils import *


def main(config: dict, logger) -> None:

    dense_features = ["I" + str(i) for i in range(1, 14)]
    sparse_features = ["C" + str(i) for i in range(1, 27)]
    target = ["label"]
    criteo_features = target + dense_features + sparse_features

    # ----- Data path -----
    if config["data_path"]:
        criteo_path = config["data_path"]
    else:
        criteo_path = os.path.join("data", "criteo_kaggle", "train.csv")

    logger.info(f"Loading the Criteo data from {criteo_path}")

    criteo_data = pd.read_csv(criteo_path, header=None, sep="\t", names=criteo_features)

    # ----- Sampling -----
    if config["sample_size"]:
        criteo_data = criteo_data.sample(
            n=config["sample_size"], random_state=config["seed"]
        )
    else:
        pass

    # Categorical feature missing imputation --- need string here
    criteo_data[sparse_features] = criteo_data[sparse_features].fillna(
        "-1",
    )
    # Dense feature missing imputation --- maybe need more inspection
    criteo_data[dense_features] = criteo_data[dense_features].fillna(
        0,
    )

    # ----- Label Encoding for sparse features,and do simple Transformation for dense features ----- #

    logger.info(f"Encoding sparse features and transforming dense features...")

    for feat in sparse_features:
        lbe = LabelEncoder()
        criteo_data[feat] = lbe.fit_transform(criteo_data[feat])

    mms = MinMaxScaler(feature_range=(0, 1))
    criteo_data[dense_features] = mms.fit_transform(criteo_data[dense_features])

    # ----- Count #unique features for each sparse field, and record dense feature field name ----- #

    fixlen_feature_columns = [
        SparseFeat(
            feat,
            vocabulary_size=criteo_data[feat].max() + 1,
            embedding_dim=config["embedding_dim"],
        )
        for feat in sparse_features
    ] + [DenseFeat(feat, 1) for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # ----- Generate input data for model ----- #

    criteo_train, criteo_test = train_test_split(
        criteo_data, test_size=0.2, random_state=config["seed"]
    )
    train_model_input = {name: criteo_train[name] for name in feature_names}
    test_model_input = {name: criteo_test[name] for name in feature_names}

    # ----- Define Model, train, predict and evaluate ----- #

    model = DeepFM(
        linear_feature_columns=linear_feature_columns,
        dnn_feature_columns=dnn_feature_columns,
        dnn_hidden_units=config["dnn_hidden_units"],
        seed=config["seed"],
        dnn_dropout=config["dnn_dropout"],
        dnn_activation=config["dnn_activation"],
        dnn_use_bn=config["dnn_use_bn"],
        device=config["device"],
        task="binary",
        l2_reg_embedding=1e-5,
    )

    model.compile(
        "adagrad", "binary_crossentropy", metrics=["binary_crossentropy", "auc"]
    )

    history = model.fit(
        train_model_input,
        criteo_train[target].values,
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        verbose=2,  # 0 for non, 1 for progress bar, 2 for every epoch
        validation_split=0.2,
    )
    pred_ans = model.predict(test_model_input, batch_size=256)

    logger.info(
        f"TEST: BCE Loss: {round(log_loss(criteo_test[target].values, pred_ans), 4)} | ROC AUC: {round(roc_auc_score(criteo_test[target].values, pred_ans), 4)} "
    )


if __name__ == "__main__":

    now = datetime.datetime.now()
    logfname = f"criteo_deepfm_{now.strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logger(os.path.join("src", "logs", logfname))

    parser = argparse.ArgumentParser(description="Criteo - DeepCTR DeepFM Training")

    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to the data. If not provided, the default path will be used.",
    )

    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Size of the sample to use. If not provided, the full data will be used.",
    )

    parser.add_argument(
        "--embedding_dim", type=int, default=4, help="Dimension of the embedding layer"
    )
    parser.add_argument(
        "--dnn_hidden_units",
        type=tuple,
        default=(400, 400, 400),
        help="Hidden dimensions of the MLP layer",
    )
    parser.add_argument("--dnn_dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument(
        "--dnn_activation", type=str, default="relu", help="Activation function"
    )
    parser.add_argument("--dnn_use_bn", type=bool, default=False, help="Whether use BN")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use. If not provided, cpu will be used.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16000, help="Batch size")
    parser.add_argument("--seed", type=int, default=1773, help="Random seed")

    args = parser.parse_args()

    config = {
        "data_path": args.data_path,
        "sample_size": args.sample_size,
        "embedding_dim": args.embedding_dim,
        "dnn_hidden_units": args.dnn_hidden_units,
        "dnn_dropout": args.dnn_dropout,
        "dnn_activation": args.dnn_activation,
        "dnn_use_bn": args.dnn_use_bn,
        "device": args.device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }

    logger.info("Config:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    main(config=config, logger=logger)

    close_logger(logger)
