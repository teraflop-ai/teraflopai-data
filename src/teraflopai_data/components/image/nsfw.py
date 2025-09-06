from typing import Optional

import daft
import torch
from daft import DataType
from PIL import Image

from teraflopai_data.components.distributed_base import Distributed


def create_falcon_nsfw_udf(
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
    class FalconsNSFWUDF:
        def __init__(
            self,
            model_name: str = model_name,
            device: str = "cuda",
        ):
            from transformers import AutoImageProcessor, AutoModelForImageClassification

            self.device = device

            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForImageClassification.from_pretrained(model_name).to(
                self.device
            )
            self.model.compile()
            self.model.eval()

        def __call__(self, images: daft.DataFrame) -> daft.DataFrame:
            inputs = self.processor(
                images=[Image.fromarray(img) for img in images], return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs).logits
            predicted_labels = outputs.argmax(-1)
            scores = [p.cpu().item() for p in predicted_labels]
            return scores

    return FalconsNSFWUDF.with_init_args(
        model_name=model_name,
    )


class FalconsNSFWClassifier(Distributed):
    def __init__(
        self,
        model_name: str = "Falconsai/nsfw_image_detection",
        batch_size: int = 1,
        input_column: str = None,
        output_column: Optional[str] = "falconsnsfw_score",
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
        return create_falcon_nsfw_udf(
            model_name=self.model_name,
            batch_size=self.batch_size,
            concurrency=self.concurrency,
            num_cpus=self.num_cpus,
            num_gpus=self.num_gpus,
        )
