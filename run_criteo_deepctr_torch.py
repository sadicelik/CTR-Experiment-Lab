import argparse
import datetime
import os

from deepctr_torch.inputs import DenseFeat, SparseFeat, get_feature_names
from deepctr_torch.models import DeepFM
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from src.dataset_class.criteo_dataset import CriteoDataset
from src.utils.logging_utils import *


def main(config: dict, logger) -> None:
    if config["data_path"]:
        criteo_path = config["data_path"]
    else:
        criteo_path = os.path.join("data", "criteo_kaggle", "train.csv")

    logger.info(f"Loading the Criteo data from {criteo_path}")

    criteo_dataset = CriteoDataset(
        data_path=criteo_path, sample_size=config["sample_size"], seed=config["seed"]
    )

    logger.info(f"Preprocessed the Criteo data.")

    # ----- Generate input features ----- #

    fixlen_feature_columns = [
        SparseFeat(
            feat,
            vocabulary_size=criteo_dataset.data[feat].max() + 1,
            embedding_dim=config["embedding_dim"],
        )
        for feat in criteo_dataset.SPARSE_FEATURES
    ] + [DenseFeat(feat, 1) for feat in criteo_dataset.DENSE_FEATURES]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # ----- Generate input data for model ----- #

    criteo_train, criteo_test = train_test_split(
        criteo_dataset.data, test_size=config["test_size"], random_state=config["seed"]
    )
    train_model_input = {name: criteo_train[name] for name in feature_names}
    test_model_input = {name: criteo_test[name] for name in feature_names}

    # ----- Define Model, train, predict and evaluate ----- #

    model = DeepFM(
        linear_feature_columns=linear_feature_columns,
        dnn_feature_columns=dnn_feature_columns,
        dnn_hidden_units=config["dnn_hidden_units"],
        l2_reg_linear=config["l2_reg_linear"],
        l2_reg_embedding=config["l2_reg_embedding"],
        l2_reg_dnn=config["l2_reg_dnn"],
        init_std=config["init_std"],
        seed=config["seed"],
        dnn_dropout=config["dnn_dropout"],
        dnn_activation=config["dnn_activation"],
        dnn_use_bn=config["dnn_use_bn"],
        device=config["device"],
        task="binary",
    )

    model.compile(
        config["optimizer"],
        "binary_crossentropy",
        metrics=["binary_crossentropy", "auc"],
    )

    history = model.fit(
        train_model_input,
        criteo_train[criteo_dataset.TARGET].values,
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        verbose=2,  # 0 for non, 1 for progress bar, 2 for every epoch
        validation_split=config["val_size"],
    )
    pred_ans = model.predict(test_model_input, batch_size=256)

    logger.info(
        f"TEST: BCE Loss: {round(log_loss(criteo_test[criteo_dataset.TARGET].values, pred_ans), 4)} | ROC AUC: {round(roc_auc_score(criteo_test[criteo_dataset.TARGET].values, pred_ans), 4)} "
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

    parser.add_argument("--test_size", type=float, default=0.2, help="Test size ratio")

    parser.add_argument(
        "--val_size", type=float, default=0.2, help="Validation size ratio"
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
    parser.add_argument(
        "--l2_reg_linear",
        type=float,
        default=1e-5,
        help="L2 regularizer of linear part",
    )
    parser.add_argument(
        "--l2_reg_embedding",
        type=float,
        default=1e-5,
        help="L2 regularizer of embedding part",
    )
    parser.add_argument(
        "--l2_reg_dnn", type=float, default=0, help="L2 regularizer of DNN part"
    )
    parser.add_argument("--init_std", type=float, default=1e-5, help="Initializer std")
    parser.add_argument("--dnn_dropout", type=float, default=0.9, help="Dropout rate")
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
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16000, help="Batch size")
    parser.add_argument("--seed", type=int, default=1773, help="Random seed")

    args = parser.parse_args()

    config = {
        "data_path": args.data_path,
        "sample_size": args.sample_size,
        "test_size": args.test_size,
        "val_size": args.val_size,
        "embedding_dim": args.embedding_dim,
        "dnn_hidden_units": args.dnn_hidden_units,
        "l2_reg_linear": args.l2_reg_linear,
        "l2_reg_embedding": args.l2_reg_embedding,
        "l2_reg_dnn": args.l2_reg_dnn,
        "init_std": args.init_std,
        "dnn_dropout": args.dnn_dropout,
        "dnn_activation": args.dnn_activation,
        "dnn_use_bn": args.dnn_use_bn,
        "device": args.device,
        "optimizer": args.optimizer,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }

    logger.info("Config:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    main(config=config, logger=logger)

    close_logger(logger)
