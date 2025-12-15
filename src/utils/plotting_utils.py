import math
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def multi_distplot(
    df: pd.DataFrame,
    numerical_features: List[str],
    n: int = 2,
    bins: int = 50,
    dpi: int = 300,
) -> None:
    """
    Plots distributions of numerical features in a grid layout.
    """
    rows = math.ceil(len(numerical_features) / n)

    plt.figure(figsize=(15, 3 * rows), dpi=dpi)

    for i, feature in enumerate(numerical_features, 1):
        plt.subplot(rows, n, i)
        sns.histplot(df[feature], bins=bins, kde=True, stat="density")
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Density")

    plt.tight_layout()
    plt.show()


def multi_boxplot(
    df: pd.DataFrame,
    numerical_features: List[str],
    n: int = 2,
    dpi: int = 300,
) -> None:
    """
    Plots boxplots of numerical features in a grid layout.
    """
    rows = math.ceil(len(numerical_features) / n)

    plt.figure(figsize=(15, 3 * rows), dpi=dpi)

    for i, feature in enumerate(numerical_features, 1):
        plt.subplot(rows, n, i)
        sns.boxplot(x=df[feature], color="skyblue")
        plt.title(f"Boxplot of {feature}")
        plt.xlabel(feature)

    plt.tight_layout()
    plt.show()
