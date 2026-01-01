import os

from src.dataset_class.avazu_dataset import AvazuCTRDataset
from src.trainer.model_generator import ModelGenerator
from src.utils.constants import LOGS_PATH
from src.utils.logging_utils import *

MODEL_NAME = "lr"
SAMPLE_SIZE = 1_000_000
TEST_SIZE = 0.2
EPOCHS = 10
BATCH_SIZE = 512
SEED = 1773


if __name__ == "__main__":

    # avazu_train_dataset = AvazuCTRDataset(
    #     data_path=os.path.join("data", "avazu", "train.csv"),
    #     is_preprocess=False,
    #     sample_size=1_000_000,
    #     seed=SEED,
    # )

    # avazu_test_dataset = AvazuCTRDataset(
    #     data_path=os.path.join("data", "avazu", "train.csv"),
    #     is_preprocess=True,
    #     sample_size=200_000,
    #     seed=SEED,
    # )

    # ---------------------- MAIN ---------------------- #

    avazu_train_dataset = AvazuCTRDataset(
        data_path=os.path.join("data", "avazu", "avazu_sample_1000000.csv"),
        is_preprocess=True,
    )

    # avazu_train_dataset = AvazuCTRDataset(
    #     data_path=os.path.join("data", "avazu", "avazu_sample_800000.csv"),
    #     is_preprocess=False,
    # )

    # avazu_test_dataset = AvazuCTRDataset(
    #     data_path=os.path.join("data", "avazu", "avazu_sample_200000.csv"),
    #     is_preprocess=False,
    # )

    FIELD_DIMS = avazu_train_dataset.field_dims

    print(
        f"Total field dimensions of {len(avazu_train_dataset.field_dims)} features: {sum(avazu_train_dataset.field_dims)}"
    )

    log_path = os.path.join(
        LOGS_PATH,
        f"avazu_{MODEL_NAME}_epochs_{EPOCHS}_bs_{BATCH_SIZE}_seed_{SEED}.log",
    )
    logger = setup_logger(log_path)

    trainer = ModelGenerator(
        model_name=MODEL_NAME,
        field_dims=FIELD_DIMS,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        seed=SEED,
        logger=logger,
    )
    trainer.train_test(avazu_train_dataset, avazu_train_dataset)

    close_logger(logger)
