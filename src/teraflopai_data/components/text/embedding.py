from typing import Optional

import daft
import numpy as np
import torch
from daft import DataType, col

from src.teraflopai_data.components.distributed_base import Distributed


def create_sentence_transformer_udf(
    model_name: str,
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
            torch_dtype: torch.dtype = torch.bfloat16,
            attn_implementation: str = "sdpa",
            set_seq_len: bool = True,
            max_seq_length: Optional[int] = None,
            token: str = None,
            show_progress_bar: bool = False,
        ):
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                model_name,
                device=device,
                token=token,
                model_kwargs={
                    "torch_dtype": torch_dtype,
                    "attn_implementation": attn_implementation,
                },
            )

            if set_seq_len:
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
        model_name=model_name, batch_size=batch_size
    )


class SentenceTransformersEmbed(Distributed):
    def __init__(
        self,
        model_name: str,
        batch_size: int,
        input_column: str = None,
        output_column: Optional[str] = "text_embedding",
        concurrency: Optional[int] = None,
        num_cpus: Optional[int] = None,
        num_gpus: Optional[int] = None,
    ):
        self.model_name = model_name
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
            batch_size=self.batch_size,
            concurrency=self.concurrency,
            num_cpus=self.num_cpus,
            num_gpus=self.num_gpus,
        )
