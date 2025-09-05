from typing import Optional

import daft
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
from daft import DataType
from PIL import Image

from src.teraflopai_data.components.distributed_base import Distributed
from src.teraflopai_data.models.owl_watermark import DetectorModelOwl


def create_owl_watermark_udf(
    model_name: str,
    batch_size: Optional[int] = None,
    concurrency: Optional[int] = None,
    num_cpus: Optional[int] = None,
    num_gpus: Optional[int] = None,
):
    @daft.udf(
        return_dtype=DataType.string(),
        concurrency=concurrency,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        batch_size=batch_size,
    )
    class OwlWatermarkUDF:
        def __init__(
            self,
            model_name: str = model_name,
            device: str = "cuda",
        ):
            self.device = device

            self.model = DetectorModelOwl(model_name)
            self.model.load_state_dict(
                torch.hub.load_state_dict_from_url(
                    "https://huggingface.co/TeraflopAI/owl-watermark-joycaption/resolve/main/far5y1y5-8000.pt",
                    map_location="cpu",
                    progress=True,
                )
            )
            self.model.to(self.device)
            self.model.compile()
            self.model.eval()

        def __call__(self, images: daft.DataFrame) -> daft.DataFrame:
            input_images = torch.stack(
                [self.processor(Image.fromarray(img)) for img in images]
            )
            with torch.no_grad():
                (logits,) = self.model(input_images)
            probs = F.softmax(logits, dim=1)
            predictions = torch.argmax(probs.cpu(), dim=1)
            scores = [pred.item() == 1 for pred in predictions]
            return daft.Series.from_pylist(scores)

        def processor(self, image: Image.Image):
            big_side = max(image.size)
            new_image = Image.new("RGB", (big_side, big_side), (128, 128, 128))
            new_image.paste(image, (0, 0))

            preped = new_image.resize((960, 960), Image.BICUBIC)

            preped = TVF.pil_to_tensor(preped)
            preped = preped / 255.0
            input_image = TVF.normalize(
                preped,
                [0.48145466, 0.4578275, 0.40821073],
                [0.26862954, 0.26130258, 0.27577711],
            )
            return input_image.to(self.device)

    return OwlWatermarkUDF.with_init_args(
        model_name=model_name,
    )


class OwlWatermarkClassifier(Distributed):
    def __init__(
        self,
        model_name: str = "google/owlv2-base-patch16-ensemble",
        batch_size: int = 1,
        input_column: str = None,
        output_column: Optional[str] = "owlwatermark_score",
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
        return create_owl_watermark_udf(
            model_name=self.model_name,
            batch_size=self.batch_size,
            concurrency=self.concurrency,
            num_cpus=self.num_cpus,
            num_gpus=self.num_gpus,
        )
