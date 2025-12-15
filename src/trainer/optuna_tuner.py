import datetime
import logging
import os
import random

import numpy as np
import optuna
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader

from ..models.deepembed import DeepEmbed
from ..models.fint import FINT
from ..utils.constants import LOGS_PATH
from ..utils.torch_utils import get_device, reset_weights


class OptunaTuner:
    """
    Optuna tuning class for model hyperparameter tunings.

    Parameters
    ----------
        train_dataset :
            The dataset the optimization takes place.
        model_name : str
            Choose a model to optimize with simple mini-batch iterations.
        field_dims : list of int, optional
            Dimensions of each categorical feature's unique values.
        seed : int default = 1773
            Seed for reproducibility.
    """

    def __init__(
        self,
        train_dataset,
        model_name: str,
        field_dims=None,
        seed: int = 1773,
    ) -> None:
        self.train_dataset = train_dataset

        self.model_name = model_name
        self.field_dims = field_dims

        # Reproducibility
        self.seed = seed
        self.set_seeds()

        self.model = None
        self.optimizer = None
        self.loss_function = torch.nn.BCEWithLogitsLoss()

        # Set up the device for training cuda if available
        self.device = self._get_device()

        self._init_tune_logs()

    def objective(self, trial):
        if self.model_name == "lr":
            params = {}
            raise NotImplementedError
        elif self.model_name == "deep-embed":
            # Hidden layers
            n_layers = trial.suggest_int("n_layers", 1, 5)

            hidden_dims = []
            for i in range(n_layers):
                hidden_dims.append(
                    trial.suggest_int(f"hidden_dim_{i}", 32, 512, log=True)
                )
            hidden_dims = tuple(hidden_dims)

            params = {
                "embed_dim": trial.suggest_int("embed_dim", 4, 128, log=True),
                "hidden_dims": hidden_dims,
                "dropout": trial.suggest_float("dropout", 0.0, 0.5),
                "optimizer_name": trial.suggest_categorical("optimizer", ["Adam"]),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-5, 1e-2, log=True
                ),
                "epochs": trial.suggest_int("epochs", 5, 20),
                "batch_size": trial.suggest_categorical(
                    "batch_size", [256, 512, 1024, 2048, 4096]
                ),
            }
            self.model = DeepEmbed(
                field_dims=self.field_dims,
                embed_dim=params["embed_dim"],
                hidden_dims=params["hidden_dims"],
                dropout=params["dropout"],
                output_layer=True,
            )
            self.optimizer = getattr(torch.optim, params["optimizer_name"])(
                self.model.parameters(), lr=params["learning_rate"]
            )
        elif self.model_name == "fint":
            # Hidden layers
            n_layers = trial.suggest_int("n_layers", 1, 5)

            hidden_dims = []
            for i in range(n_layers):
                hidden_dims.append(
                    trial.suggest_int(f"hidden_dim_{i}", 32, 512, log=True)
                )
            hidden_dims = tuple(hidden_dims)

            params = {
                "embed_dim": trial.suggest_int("embed_dim", 4, 128, log=True),
                "fint_layers": trial.suggest_int("n_fint_layer", 1, 5),
                "hidden_dims": hidden_dims,
                "dropout": trial.suggest_float("dropout", 0.0, 0.5),
                "optimizer_name": trial.suggest_categorical("optimizer", ["Adam"]),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-5, 1e-2, log=True
                ),
                "epochs": trial.suggest_int("epochs", 5, 20),
                "batch_size": trial.suggest_categorical(
                    "batch_size", [256, 512, 1024, 2048, 4096]
                ),
            }

            self.model = FINT(
                field_dims=self.field_dims,
                embed_dim=params["embed_dim"],
                fint_layers=params["fint_layers"],
                hidden_dims=params["hidden_dims"],
                dropout=params["dropout"],
            )
            self.optimizer = getattr(torch.optim, params["optimizer_name"])(
                self.model.parameters(), lr=params["learning_rate"]
            )
        else:
            raise NotImplementedError

        # Reset parameters
        self.model.apply(reset_weights)
        self.model.to(self.device, dtype=torch.float64)

        self.logger.info(f"OPTUNA TUNER: PyTorch: {self.model}")

        self.logger.info(f"###################-STARTED TRAINING-###################")

        train_loader = DataLoader(
            self.train_dataset, batch_size=params["batch_size"], shuffle=True
        )

        for epoch in range(1, params["epochs"] + 1):
            epoch_loss = self._train_one_epoch(train_loader)

            self.logger.info(
                f"OPTUNA TUNER: TRAIN:\t Epoch: {epoch:3d} | BCE Loss: {epoch_loss:.5f}"
            )

        self.logger.info(f"###################-FINISHED TRAINING-###################")

        model_loss = epoch_loss

        return model_loss

    def _train_one_epoch(self, train_loader: DataLoader = None):
        # Record epoch loss as the average of the batch losses
        epoch_loss = 0.0

        # Set model to train mode
        self.model.train()

        # Iterate over mini batches
        for _, (x_train_batch, y_train_batch) in enumerate(train_loader):
            # Forward Pass
            out = self.model(x_train_batch)  # logits
            # Loss Calculation
            batch_loss = self.loss_function(out, y_train_batch)
            # Backward Pass
            self.optimizer.zero_grad()
            batch_loss.backward()
            # Parameter Update
            self.optimizer.step()
            # Update epoch loss
            epoch_loss += batch_loss.item()

        # One epoch of training complete, calculate average training epoch loss
        epoch_loss /= len(train_loader)

        return epoch_loss

    def tune(self, n_trials: int = 10, storage_name: str = None) -> None:
        """
        Main tuning function which takes objective and optimizes.

        Parameters
        ----------
        n_trials : int
            Number of trials that different models are created.
        storage_name : str
            Name of the storage file to record db data.
        """
        storage = f"sqlite:///{storage_name}" if storage_name else None

        # Create a study object and optimize the objective function.
        study = optuna.create_study(
            direction="minimize",
            study_name=self.get_optuna_study_name(),
            sampler=TPESampler(),
            pruner=MedianPruner(),
            storage=storage,
        )
        study.optimize(self.objective, n_jobs=1, n_trials=n_trials)

        best_trial = study.best_trial

        self.logger.info(f"Number of finished trials: {len(study.trials)}")
        self.logger.info(f"Best trial:")
        self.logger.info(f"  Value: {best_trial.value}")
        self.logger.info(f"  Params: ")
        for key, value in best_trial.params.items():
            self.logger.info(f"    {key}: {value}")

        self._finish_tune_logs()

    ################################# HELPER FUNCTIONS #################################

    def _init_tune_logs(self) -> None:
        """Save Optuna tuning logs into a log file."""
        if not os.path.exists(LOGS_PATH):
            os.mkdir(LOGS_PATH)

        logfname = self.create_logfname()
        logfpath = os.path.join(LOGS_PATH, logfname)

        # Create a logger and configure it to log to a file
        self.logger = optuna.logging.get_logger("optuna")
        self.file_handler = logging.FileHandler(logfpath)
        self.logger.addHandler(self.file_handler)

    def _finish_tune_logs(self) -> None:
        """Close the logger to ensure all logs are flushed to the file"""
        self.logger.removeHandler(self.file_handler)
        self.file_handler.close()

    def get_optuna_study_name(self) -> str:
        now = datetime.datetime.now()

        if self.model_name == "deep-embed":
            return (
                f"Avazu-DeepEmbed-Hyperparameter-Tuning-{now.strftime('%Y%m%d%H%M%S')}"
            )
        elif self.model_name == "fint":
            return f"Avazu-FINT-Hyperparameter-Tuning-{now.strftime('%Y%m%d%H%M%S')}"

    def create_logfname(self) -> str:
        now = datetime.datetime.now()

        if self.model_name == "deep-embed":
            return f"Avazu_DeepEmbed_{now.strftime('%Y%m%d%H%M%S')}.log"
        elif self.model_name == "fint":
            return f"Avazu_FINT_{now.strftime('%Y%m%d%H%M%S')}.log"

    def set_seeds(self):
        """Set random seeds for libraries."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

    @staticmethod
    def _get_device():
        """Get device for tensor operations."""
        device = get_device()
        print(f"MODEL GENERATOR: PyTorch: Training model on device {device}.")
        return device
