from typing import Optional

import daft
import numpy as np
import torch
from daft import DataType
from loguru import logger

from src.teraflopai_data.components.distributed_base import Distributed


def create_sentence_transformer_udf(
    model_name: str,
    max_seq_length: Optional[int] = None,
    batch_size: Optional[int] = None,
    concurrency: Optional[int] = None,
    num_cpus: Optional[int] = None,
    num_gpus: Optional[int] = None,
):
    @daft.udf(
        return_dtype=DataType.list(DataType.float32()),
        concurrency=concurrency,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        batch_size=batch_size,
    )
    class SentenceTransformersUDF:
        def __init__(
            self,
            model_name: str = model_name,
            batch_size: int = batch_size,
            device: str = "cuda",
            convert_to_tensor: bool = False,
            dtype: torch.dtype = torch.bfloat16,
            attn_implementation: str = "sdpa",
            max_seq_length: Optional[int] = max_seq_length,
            token: str = None,
            show_progress_bar: bool = False,
        ):
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                model_name,
                device=device,
                token=token,
                model_kwargs={
                    "dtype": dtype,
                    "attn_implementation": attn_implementation,
                },
            )
            self.model = torch.compile(self.model)

            if max_seq_length is not None:
                logger.info(f"Max sequence length is set to: {max_seq_length}")
                self.model.max_seq_length = max_seq_length

            self.convert_to_tensor = convert_to_tensor
            self.device = device
            self.show_progress_bar = show_progress_bar
            self.batch_size = batch_size

        def __call__(self, text_col: daft.DataFrame) -> daft.DataFrame:
            embeddings = self.model.encode(
                text_col.to_pylist(),
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_tensor=self.convert_to_tensor,
                device=self.device,
            )
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()
            embeddings = embeddings.astype(np.float32)
            return embeddings

    return SentenceTransformersUDF.with_init_args(
        model_name=model_name,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
    )


class SentenceTransformersEmbed(Distributed):
    def __init__(
        self,
        model_name: str,
        max_seq_length: Optional[int] = None,
        batch_size: int = 1,
        input_column: str = None,
        output_column: Optional[str] = "text_embedding",
        concurrency: Optional[int] = None,
        num_cpus: Optional[int] = None,
        num_gpus: Optional[int] = None,
    ):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        super().__init__(
            input_column=input_column,
            output_column=output_column,
            batch_size=batch_size,
            concurrency=concurrency,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
        )

    def _udf(self):
        return create_sentence_transformer_udf(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            batch_size=self.batch_size,
            concurrency=self.concurrency,
            num_cpus=self.num_cpus,
            num_gpus=self.num_gpus,
        )
