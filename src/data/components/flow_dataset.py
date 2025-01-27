import os

import pandas as pd
from torch import float32, tensor
from torch.utils.data import TensorDataset

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class FlowDataset:
    """Abstract class containing preprocessing logic for loading 3D flow data in the form of velocity gradients from CSV"""

    def _preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocesses the dataframe by transforming / cleaning data.

        :param df: The dataframe to preprocess.
        :return: The preprocessed dataframe.
        """
        # exclude gradients with invalid values
        df = df.apply(pd.to_numeric, errors="coerce")
        log.info("Cleaned dataframe by converting invalid values to NaN.")

        df = df.dropna()
        log.info("Dropped rows with NaN values.")

        # shuffle the dataframe (seed is set by hydra, see :file:`src/train.py`)
        df = df.sample(frac=1).reset_index(drop=True)
        log.info("Shuffled the dataframe.")
        return df


class FlowDatasetResVort(TensorDataset, FlowDataset):
    """FlowDataset class for loading 3D flow data in the form of velocity gradients from CSV files.
    Gradients are stored as tensors of shape (10, 1) (incl. ResVort)."""

    def __init__(self, data_dir: str, dataset_name: str):
        """Initializes the FlowDataset class.

        :param data_dir: The directory where the dataset is stored.
        :param dataset_name: The name of the dataset file.
        """
        columns = [
            "A11",
            "A21",
            "A31",
            "A12",
            "A22",
            "A32",
            "A13",
            "A23",
            "A33",
            "ResVort",
        ]

        log.info(f"Loading dataset from {os.path.join(data_dir, dataset_name)}.")
        flow_data = pd.read_csv(os.path.join(data_dir, dataset_name), usecols=columns)
        log.info(f"Loaded dataset from {os.path.join(data_dir, dataset_name)}.")

        flow_data = self._preprocess_dataframe(flow_data)
        log.info("Preprocessed the dataset.")

        flow_data_tensor = tensor(flow_data.loc[:, "A11":"A33"].values, dtype=float32)
        res_vort_tensor = tensor(flow_data["ResVort"].values, dtype=float32).unsqueeze(1)

        super().__init__(flow_data_tensor, res_vort_tensor)


class FlowDataSetComplete(TensorDataset, FlowDataset):
    """FlowDataset class for loading 3D flow data in the form of velocity gradients from CSV files.
    Gradients are stored as tensors of shape (12, 1) (incl. ResVort, ResStrain, Shear)."""

    def __init__(self, data_dir, dataset_name):
        """Initializes the FlowDatasetComplete class.

        :param data_dir: The directory where the dataset is stored.
        :param dataset_name: The name of the dataset file.
        """
        columns = [
            "A11",
            "A21",
            "A31",
            "A12",
            "A22",
            "A32",
            "A13",
            "A23",
            "A33",
            "ResVort",
            "ResStrain",
            "Shear",
        ]

        log.info(f"Loading dataset from {os.path.join(data_dir, dataset_name)}.")
        flow_data = pd.read_csv(os.path.join(data_dir, dataset_name), usecols=columns)
        log.info(f"Loaded dataset from {os.path.join(data_dir, dataset_name)}.")

        flow_data = self._preprocess_dataframe(flow_data)
        log.info("Preprocessed the dataset.")

        flow_data_tensor = tensor(flow_data.loc[:, "A11":"A33"].values, dtype=float32)
        res_vort_tensor = tensor(flow_data.loc[:, "ResVort":"Shear"].values, dtype=float32)

        super().__init__(flow_data_tensor, res_vort_tensor)
