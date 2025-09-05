from typing import Optional

import daft
import torch
from daft import DataType

from src.teraflopai_data.components.distributed_base import Distributed


def create_finewebedu_udf(
    model_name: str,
    batch_size: Optional[int] = None,
    concurrency: Optional[int] = None,
    num_cpus: Optional[int] = None,
    num_gpus: Optional[int] = None,
):
    @daft.udf(
        return_dtype=DataType.int8(),
        concurrency=concurrency,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        batch_size=batch_size,
    )
    class FinewebEduUDF:
        def __init__(
            self,
            model_name: str = model_name,
            device: str = "cuda",
            torch_dtype: torch.dtype = torch.bfloat16,
            attn_implementation: str = "sdpa",
        ):
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.device = device

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                attn_implementation=attn_implementation,
            ).to(self.device)
            self.model.compile()
            self.model.eval()

        def __call__(self, text_col: daft.DataFrame) -> daft.DataFrame:
            inputs = self.tokenizer(
                text_col.to_pylist(),
                return_tensors="pt",
                padding="longest",
                truncation=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits.squeeze(-1).float().cpu().numpy()

            scores = [int(round(max(0, min(score, 5)))) for score in logits]
            return scores

    return FinewebEduUDF.with_init_args(
        model_name=model_name,
    )


class FinewebEduClassifier(Distributed):
    def __init__(
        self,
        model_name: str = "HuggingFaceTB/fineweb-edu-classifier",
        batch_size: int = 1,
        input_column: str = None,
        output_column: Optional[str] = "finewebedu_score",
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
        return create_finewebedu_udf(
            model_name=self.model_name,
            batch_size=self.batch_size,
            concurrency=self.concurrency,
            num_cpus=self.num_cpus,
            num_gpus=self.num_gpus,
        )
