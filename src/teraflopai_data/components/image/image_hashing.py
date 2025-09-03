from enum import Enum
from typing import Optional

import daft
import imagehash


class HashingAlgorithm(str, Enum):
    """Available hashing algorithms."""

    AVERAGE = "average"
    PERCEPTUAL = "perceptual"
    DIFFERENCE = "difference"
    WAVELET = "wavelet"
    CROP_RESISTANT = "crop_resistant"


@daft.udf(return_dtype=daft.DataType.string())
class ImageHasher:
    def __init__(
        self, hashing_algorithm: HashingAlgorithm, hash_size: Optional[int] = None
    ):
        self.hashing_algorithm = hashing_algorithm
        self.hash_size = hash_size

    def __call__(self, images):
        return [self.hash_image(img) for img in images]

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
                    f"Please select a valide hashing algorithm: {self.hashing_algorithm}"
                )
