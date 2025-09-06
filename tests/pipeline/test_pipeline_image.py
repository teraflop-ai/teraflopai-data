import daft
from daft import col

from teraflopai_data.components.image.image_hashing import ImageHasher
from teraflopai_data.components.image.nsfw import FalconsNSFWClassifier
from teraflopai_data.pipeline import Pipeline

df = daft.from_pydict(
    {
        "urls": [
            "https://live.staticflickr.com/65535/53671838774_03ba68d203_o.jpg",
            "https://live.staticflickr.com/65535/53671700073_2c9441422e_o.jpg",
            "https://live.staticflickr.com/65535/53670606332_1ea5f2ce68_o.jpg",
            "https://live.staticflickr.com/65535/53671838039_b97411a441_o.jpg",
            "https://live.staticflickr.com/65535/53671698613_0230f8af3c_o.jpg",
            "https://live.staticflickr.com/65535/53671838774_03ba68d203_o.jpg",
            "https://live.staticflickr.com/65535/53671700073_2c9441422e_o.jpg",
            "https://live.staticflickr.com/65535/53670606332_1ea5f2ce68_o.jpg",
            "https://live.staticflickr.com/65535/53671838039_b97411a441_o.jpg",
            "https://live.staticflickr.com/65535/53671698613_0230f8af3c_o.jpg",
        ],
    }
)

hasher = ImageHasher(
    input_column="image",
    hashing_algorithm="wavelet",
    concurrency=1,
    num_cpus=6,
)

classifier = FalconsNSFWClassifier(
    input_column="image", batch_size=8, concurrency=1, num_cpus=6, num_gpus=1
)

df = df.with_column("image_bytes", col("urls").url.download(on_error="null"))
df = df.with_column("image", col("image_bytes").image.decode())

pipeline = Pipeline(
    ops=[classifier, hasher],
)

df = pipeline(df)
df.show()
