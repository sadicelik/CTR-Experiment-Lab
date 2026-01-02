import json
import os

from src.dataset_class.avazu_dataset import AvazuCTRDataset
from src.trainer.model_generator import ModelGenerator
from src.utils.constants import CONFIGS_PATH, LOGS_PATH
from src.utils.logging_utils import *

if __name__ == "__main__":
    # ---------------------- MAIN ---------------------- #

    with open(os.path.join(CONFIGS_PATH, "avazu_fm.json"), "r") as f:
        config = json.load(f)

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
        f"avazu_{config["MODEL_NAME"]}_epochs_{config["EPOCHS"]}_bs_{config["BATCH_SIZE"]}_seed_{config["SEED"]}.log",
    )
    logger = setup_logger(log_path)

    logger.info(f"Config: {json.dumps(config, indent=4)}")

    # ---------------------- FM ---------------------- #

    trainer = ModelGenerator(
        model_name=config["MODEL_NAME"],
        field_dims=FIELD_DIMS,
        embed_dim=config["EMBED_DIM"],
        epochs=config["EPOCHS"],
        batch_size=config["BATCH_SIZE"],
        seed=config["SEED"],
        logger=logger,
    )

    # ---------------------- DeepFM ---------------------- #

    # trainer = ModelGenerator(
    #     model_name=config["MODEL_NAME"],
    #     field_dims=FIELD_DIMS,
    #     embed_dim=config["EMBED_DIM"],
    #     mlp_hidden_dims=config["HIDDEN_DIMS"],
    #     epochs=config["EPOCHS"],
    #     batch_size=config["BATCH_SIZE"],
    #     seed=config["SEED"],
    #     logger=logger,
    # )

    trainer.train_test(avazu_train_dataset, avazu_train_dataset)

    close_logger(logger)
