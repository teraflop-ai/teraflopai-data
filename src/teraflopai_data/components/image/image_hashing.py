from enum import Enum
from typing import Optional

import daft
import imagehash
from PIL import Image

from src.teraflopai_data.components.distributed_base import Distributed


class HashingAlgorithm(str, Enum):
    """Available hashing algorithms."""

    AVERAGE = "average"
    PERCEPTUAL = "perceptual"
    DIFFERENCE = "difference"
    WAVELET = "wavelet"
    CROP_RESISTANT = "crop_resistant"


def create_image_hasher_udf(
    hashing_algorithm: HashingAlgorithm,
    hash_size: Optional[int] = None,
    batch_size: Optional[int] = None,
    concurrency: Optional[int] = None,
    num_cpus: Optional[int] = None,
    num_gpus: Optional[int] = None,
):
    @daft.udf(
        return_dtype=daft.DataType.string(),
        concurrency=concurrency,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        batch_size=batch_size,
    )
    class ImageHasherUDF:
        def __init__(
            self,
            hashing_algorithm: HashingAlgorithm,
            hash_size: Optional[int] = None,
        ):
            self.hashing_algorithm = hashing_algorithm
            self.hash_size = hash_size

        def __call__(self, images):
            return [self.hash_image(Image.fromarray(img)) for img in images]

        def hash_image(self, image):
            """
            Args:
                hashing_algorithm: The selected image hashing algorithm to use
                hash_size: Size of the image hash

            Returns:
                Selected Image Hasher
            """
            kwargs = {}
            if self.hash_size is not None:
                kwargs["hash_size"] = self.hash_size

            match self.hashing_algorithm:
                case HashingAlgorithm.AVERAGE:
                    return imagehash.average_hash(image, **kwargs)
                case HashingAlgorithm.PERCEPTUAL:
                    return imagehash.phash(image, **kwargs)
                case HashingAlgorithm.DIFFERENCE:
                    return imagehash.dhash(image, **kwargs)
                case HashingAlgorithm.WAVELET:
                    return imagehash.whash(image, **kwargs)
                case HashingAlgorithm.CROP_RESISTANT:
                    return imagehash.crop_resistant_hash(image)
                case _:
                    raise ValueError(
                        f"Please select a valid hashing algorithm: {self.hashing_algorithm}"
                    )

    return ImageHasherUDF.with_init_args(
        hashing_algorithm=hashing_algorithm, hash_size=hash_size
    )


class ImageHasher(Distributed):
    def __init__(
        self,
        hashing_algorithm: HashingAlgorithm,
        hash_size: Optional[int] = None,
        input_column: str = None,
        output_column: Optional[str] = "image_hash",
        batch_size: Optional[int] = None,
        concurrency: Optional[int] = None,
        num_cpus: Optional[int] = None,
        num_gpus: Optional[int] = None,
    ):
        self.hashing_algorithm = hashing_algorithm
        self.hash_size = hash_size
        super().__init__(
            input_column=input_column,
            output_column=output_column,
            batch_size=batch_size,
            concurrency=concurrency,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
        )

    def _udf(self):
        """Apply the image hasher UDF to the dataframe."""
        return create_image_hasher_udf(
            hashing_algorithm=self.hashing_algorithm,
            hash_size=self.hash_size,
            batch_size=self.batch_size,
            concurrency=self.concurrency,
            num_cpus=self.num_cpus,
            num_gpus=self.num_gpus,
        )
