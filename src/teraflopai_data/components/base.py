from abc import ABC, abstractmethod
from typing import Any, Optional

import daft
from daft import DataType, col


class ScoreFilter(ABC):
    """Base class for applying Score and Filter to DataFrame."""

    def __init__(
        self,
        input_column: str,
        output_column: Optional[str] = None,
        daft_dtype: DataType = None,
        threshold: Optional[float] = None,
    ):
        self.input_column = input_column
        self.output_column = output_column
        self.daft_dtype = daft_dtype
        self.threshold = threshold

    @abstractmethod
    def _score(self) -> Any:
        """Create and return the score."""
        pass

    @abstractmethod
    def _filter(self) -> Any:
        """Fitler the Dataframe depending on the score."""
        pass

    def __call__(self, df: daft.DataFrame) -> daft.DataFrame:
        """Apply the score and filter to the dataframe."""
        df = df.with_column(
            self.output_column,
            col(self.input_column).apply(
                lambda x: self._score(x), return_dtype=self.daft_dtype
            ),
        )
        if self.threshold:
            df = self._filter(df, self.threshold)
        return df
