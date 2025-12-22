import os
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from ..dataset_class.avazu_dataset import AvazuCTRDataset
from ..models.deepembed import DeepEmbed
from ..models.fint import FINT
from ..models.lr import LogisticRegression
from ..models.mlp import MLP
from ..utils.constants import MODELS_PATH
from ..utils.torch_utils import get_device, reset_weights


class ModelGenerator:
    """
    Model Generator Trainer with mini-batch optimizations.

    Parameters
    ----------
        model_name : str
            Choose a model to train with simple mini-batch iterations.
        field_dim : ()
            Dimensions of each categorical feature's unique values.
        embed_dim : int
            Dimension of features to be embed.
        mlp_input_dim : int
            Placeholder.
        mlp_hidden_dims :
            Hidden dimensions of MLP component.
        dropout : float
            Dropout used in MLP parts of the models.
        lr : float, default=1e-3
            Learning rate for the optimizer.
        epochs : int
            Number of times the data is optimized.
        batch_size : int
            Mini batch size for the training.
        seed : int, default = 1773
            Seed for reproducibility.
        logger : None
            A logger object to record console outputs to log files.
        verbose : bool
            If True, self.logger.info verbose information during training.
        save_model : bool
            If True, save the trained model.
    """

    def __init__(
        self,
        model_name: str,
        field_dims=None,
        embed_dim: int = None,
        fint_layers: int = None,
        mlp_input_dim: int = None,
        mlp_hidden_dims=None,
        dropout: float = 0.2,
        lr: float = 1e-3,
        epochs: int = 100,
        batch_size: int = None,
        seed: int = 1773,
        logger=None,
        verbose: bool = True,
        save_model: bool = False,
    ):
        # Model attributes
        self.model_name = model_name

        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.fint_layers = fint_layers
        self.mlp_input_dim = mlp_input_dim
        self.mlp_hidden_dims = mlp_hidden_dims
        self.droput = dropout

        # Trainer attributes
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        # Reproducibility
        self.seed = seed
        self.set_seeds()

        self.logger = logger
        self.verbose = verbose

        # Model saving
        self.save_model = save_model

        self.model = None
        self.optimizer = None
        self.loss_function = torch.nn.BCEWithLogitsLoss()

        self.init()

        # Set up the device for training cuda if available
        self.device = self._get_device()
        self.model.to(self.device, dtype=torch.float64)

        if self.verbose:
            self.logger.info(f"MODEL GENERATOR: PyTorch: {self.model}")

    def init(self):
        self.model, self.optimizer = self._init_model()

    def _init_model(self):
        if self.model_name == "mlp":  # 800.000 samples, 26 features
            model = MLP(input_dim=self.mlp_input_dim, hidden_dims=self.mlp_hidden_dims)
        elif self.model_name == "lr":  # 800.000 samples, 23 features, 1 target
            model = LogisticRegression(field_dims=self.field_dims)
        elif self.model_name == "deep-embed":
            model = DeepEmbed(
                field_dims=self.field_dims,
                embed_dim=self.embed_dim,
                hidden_dims=self.mlp_hidden_dims,
                dropout=self.droput,
                output_layer=True,
            )
        elif self.model_name == "fint":
            model = FINT(
                field_dims=self.field_dims,
                embed_dim=self.embed_dim,
                fint_layers=self.fint_layers,
                hidden_dims=self.mlp_hidden_dims,
                dropout=self.droput,
            )
        else:
            raise NotImplementedError

        # Reset parameters
        model.apply(reset_weights)

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        return model, optimizer

    ################################# TRAIN #################################

    def _train_one_epoch(self, train_loader: DataLoader = None):
        # Record epoch loss as the average of the batch losses
        epoch_loss = 0.0
        epoch_roc_auc = 0.0

        # Set model to train mode
        self.model.train()

        # Iterate over mini batches
        for _, (x_train_batch, y_train_batch) in enumerate(train_loader):
            # Forward Pass
            out = self.model(x_train_batch)  # logits
            # Loss Calculation
            batch_loss = self.loss_function(out, y_train_batch)
            # Score Calculation
            batch_metrics = self._calculate_performance_metrics(
                y_true=y_train_batch, y_pred=out, y_logits=out
            )
            # Backward Pass
            self.optimizer.zero_grad()
            batch_loss.backward()
            # Parameter Update
            self.optimizer.step()
            # Update epoch loss
            epoch_loss += batch_loss.item()
            epoch_roc_auc += batch_metrics["ROC-AUC"]

        # One epoch of training complete, calculate average training epoch loss
        epoch_loss /= len(train_loader)
        epoch_roc_auc /= len(train_loader)

        return epoch_loss, epoch_roc_auc

    def train(self, train_dataset: AvazuCTRDataset):
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        # Metrics
        train_losses = []
        train_roc_aucs = []

        self.logger.info(f"###################-STARTED TRAINING-###################")

        for epoch in range(1, self.epochs + 1):
            train_epoch_loss, train_epoch_roc_auc = self._train_one_epoch(
                train_loader=train_loader
            )
            train_losses.append(train_epoch_loss)
            train_roc_aucs.append(train_epoch_roc_auc)

            if self.verbose:
                self.logger.info(
                    f"TRAIN:\t Epoch: {epoch:3d} | BCE Loss: {train_epoch_loss:.5f} | ROC-AUC: {train_epoch_roc_auc:.5f}"
                )

        self.logger.info(f"###################-FINISHED TRAINING-###################")

        if self.save_model:
            self.save_pt_model()

    ################################# TEST #################################

    def test(self, test_dataset: AvazuCTRDataset, batch_size=None):
        test_loader = DataLoader(
            test_dataset, batch_size=len(test_dataset), shuffle=False
        )  # Use lenght of test dataset as the batch size

        self.model.eval()

        # Metrics
        batch_losses = []
        performance_metrics = {"ROC-AUC": []}

        self.logger.info(f"###################-STARTED TESTING-###################")

        with torch.no_grad():
            for _, (x_test_batch, y_test_batch) in enumerate(test_loader):
                # Forwad Pass
                out = self.model(x_test_batch)  # logits
                # Loss Calculation
                batch_loss = self.loss_function(out, y_test_batch)
                # Convert logits to predicted labels (threshold=0.5)
                preds = torch.sigmoid(out)
                predicted = (preds >= 0.5).float()

                # Performance metrics
                batch_metrics = self._calculate_performance_metrics(
                    y_true=y_test_batch, y_pred=predicted, y_logits=out
                )
                # Record
                batch_losses.append(batch_loss.item())
                for metric, value in batch_metrics.items():
                    performance_metrics[metric].append(value)

        test_loss = np.mean(batch_losses)
        test_roc_auc = np.mean(performance_metrics["ROC-AUC"])

        if self.verbose:
            self.logger.info(
                f"TEST:\t BCE Loss: {test_loss:.5f} | ROC-AUC: {test_roc_auc:.5f}"
            )

        self.logger.info(f"###################-FINISHED TESTING-###################")

    def _calculate_performance_metrics(
        self, y_true: torch.Tensor, y_pred: torch.Tensor, y_logits: torch.Tensor
    ):
        """
        Calculate test performance metrics for binary classification.
        """
        # Conver to cpu and numpy tensors
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
        y_logits = y_logits.detach().cpu().numpy()

        # Scikit-learn performance metrics
        metrics = {}
        metrics["ROC-AUC"] = roc_auc_score(y_true=y_true, y_score=y_logits)

        return metrics

    def recommend_ads(self, dataset, top_n: int = 5) -> pd.DataFrame:
        """Recommend top  most relevant items for any given user."""
        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

        self.model.eval()

        print(f"###################-STARTED RECOMMENDING ITEMS-###################")

        with torch.no_grad():
            for x_test_batch, _ in dataloader:
                # Forward Pass
                out = self.model(x_test_batch)  # logits
                preds = torch.sigmoid(out)
                preds = preds.detach().cpu().numpy()

            # Get top n relevant items
            items_df = pd.DataFrame({"ad_id": dataset.data["id"], "prediction": preds})
            top_items = items_df.sort_values("prediction", ascending=False).head(top_n)

        print(f"###################-FINISHED RECOMMENDING ITEMS-###################")

        return top_items

    ################################# TRAIN - TEST #################################

    def train_test(self, train_dataset: AvazuCTRDataset, test_dataset: AvazuCTRDataset):
        self.train(train_dataset=train_dataset)
        self.test(test_dataset=test_dataset)

    ################################# HELPER FUNCTIONS #################################

    def set_seeds(self):
        """Set random seeds for libraries."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

    @staticmethod
    def _get_device() -> torch.device:
        """Get device for tensor operations."""
        device = get_device()
        print(f"MODEL GENERATOR: PyTorch: Training model on device {device}.")
        return device

    def save_pt_model(self):
        """Save the trained PyTorch model into .pt file."""
        os.makedirs(MODELS_PATH, exist_ok=True)

        timestamp = int(time.time())
        fname = f"{self.model_name}_{timestamp}"
        fname += ".pt"
        fpath = os.path.join(MODELS_PATH, fname)

        # Save model state dict and metadata
        torch.save({"model_state_dict": self.model.state_dict()}, fpath)

        if self.verbose:
            print(f"MODEL GENERATOR: Model saved to {fname}")
