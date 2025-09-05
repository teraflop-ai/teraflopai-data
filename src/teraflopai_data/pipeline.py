from typing import Callable

import daft


class Pipeline:
    def __init__(self, ops: Callable):
        self.ops = ops

    def __call__(self, df: daft.DataFrame) -> daft.DataFrame:
        for op in self.ops:
            df = op(df)
        return df
