import daft
from daft import col

from teraflopai_data.components.image.blur import Blur

df = daft.from_pydict(
    {
        "urls": [
            "https://live.staticflickr.com/65535/53671838774_03ba68d203_o.jpg",
            "https://live.staticflickr.com/65535/53671700073_2c9441422e_o.jpg",
            "https://live.staticflickr.com/65535/53670606332_1ea5f2ce68_o.jpg",
            "https://live.staticflickr.com/65535/53671838039_b97411a441_o.jpg",
            "https://live.staticflickr.com/65535/53671698613_0230f8af3c_o.jpg",
        ],
    }
)

blur_filter = Blur(input_column="image", threshold=150.0)

df = df.with_column("image_bytes", col("urls").url.download(on_error="null"))
df = df.with_column("image", col("image_bytes").image.decode())
df = blur_filter(df)
df.show()
