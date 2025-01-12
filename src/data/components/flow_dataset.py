import os
import pandas as pd
from torch.utils.data import TensorDataset
from torch import tensor, float32

from src.utils import RankedLogger


log = RankedLogger(__name__, rank_zero_only=True)


class FlowDataset(TensorDataset):
    """FlowDataset class for loading 3D flow data in the form of velocity gradients from CSV files.
    Gradients are stored as tensors of shape (9, 1)."""

    def __init__(self, data_dir: str, dataset_name: str):
        """Initializes the FlowDataset class.

        :param data_dir: The directory where the dataset is stored.
        :param dataset_name: The name of the dataset file.
        """
        columns = (
            [
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
            ],
        )

        flow_data = pd.read_csv(os.path.join(data_dir, dataset_name))
        log.info(f"Loaded dataset from {os.path.join(data_dir, dataset_name)}.")

        flow_data = self._preprocess_dataframe(flow_data)
        log.info("Preprocessed the dataset.")

        flow_data = tensor(flow_data.values, dtype=float32)

        super().__init__(flow_data)

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