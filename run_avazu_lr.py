import os

from src.dataset_class.avazu_dataset import AvazuCTRDataset
from src.trainer.model_generator import ModelGenerator
from src.utils.constants import LOGS_PATH
from src.utils.logging_utils import *

if __name__ == "__main__":
    avazu_train_path = os.path.join("data", "avazu", "avazu_train_800k.csv")
    avazu_train_dataset = AvazuCTRDataset(
        avazu_train_path, drop_hour=True, drop_id=True
    )

    avazu_test_path = os.path.join("data", "avazu", "avazu_test_200k.csv")
    avazu_test_dataset = AvazuCTRDataset(
        data_path=avazu_test_path, drop_hour=True, drop_id=True
    )

    MODEL_NAME = "lr"
    FIELD_DIMS = avazu_train_dataset.field_dims
    EPOCHS = 10
    BATCH_SIZE = 512
    SEED = 1773

    print(
        f"Total field dimensions of {len(avazu_train_dataset.field_dims)} features: {sum(avazu_train_dataset.field_dims)}"
    )

    log_path = os.path.join(
        LOGS_PATH,
        f"avazu_drop_id_hour_{MODEL_NAME}_epochs_{EPOCHS}_bs_{BATCH_SIZE}_seed_{SEED}.log",
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
    trainer.train_test(avazu_train_dataset, avazu_test_dataset)

    close_logger(logger)
