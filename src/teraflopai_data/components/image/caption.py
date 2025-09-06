from typing import Optional

import daft
from daft import DataType
from PIL import Image

from teraflopai_data.components.distributed_base import Distributed

CAPTION = """\
<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\
f"Write a detailed description of what is in the image?<|im_end|>\n\
<|im_start|>assistant\n"""


def create_vllm_image_caption_udf(
    model_name: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
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
    class VllmImageCaptionUDF:
        def __init__(
            self,
            model_name: str = model_name,
            max_tokens: int = max_tokens,
            temperature: float = temperature,
        ):
            from vllm import LLM, SamplingParams

            self.vllm_engine = LLM(model=model_name)
            self.sampling_params = SamplingParams(
                max_tokens=max_tokens, temperature=temperature
            )

        def __call__(self, images: daft.DataFrame) -> daft.DataFrame:
            images = [Image.fromarray(img) for img in images]
            vllm_inputs = [
                {
                    "prompt": CAPTION,
                    "multi_modal_data": {"image": image},
                }
                for image in images
            ]
            outputs = self.vllm_engine.generate(vllm_inputs, self.sampling_params)
            generated_text = []
            for output in outputs:
                generated_text.append(output.outputs[0].text)
            return generated_text

    return VllmImageCaptionUDF.with_init_args(
        model_name=model_name, max_tokens=max_tokens, temperature=temperature
    )


class VllmImageCaption(Distributed):
    def __init__(
        self,
        model_name,
        max_tokens=256,
        temperature=0.2,
        batch_size: int = 1,
        input_column: str = None,
        output_column: Optional[str] = "vllm_caption",
        concurrency: Optional[int] = None,
        num_cpus: Optional[int] = None,
        num_gpus: Optional[int] = None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        super().__init__(
            input_column=input_column,
            output_column=output_column,
            batch_size=batch_size,
            concurrency=concurrency,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
        )

    def _udf(self):
        return create_vllm_image_caption_udf(
            model_name=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            batch_size=self.batch_size,
            concurrency=self.concurrency,
            num_cpus=self.num_cpus,
            num_gpus=self.num_gpus,
        )
