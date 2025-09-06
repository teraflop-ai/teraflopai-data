import daft
from daft import col

from teraflopai_data.components.image.watermark import OwlWatermarkClassifier

df = daft.from_pydict(
    {
        "urls": [
            "https://live.staticflickr.com/65535/53671838774_03ba68d203_o.jpg",
            "https://live.staticflickr.com/65535/53671700073_2c9441422e_o.jpg",
            "https://1.img-dpreview.com/files/p/E~TS590x0~articles/6322652598/watermarkprotection",
            "https://www.shutterstock.com/shutterstock/photos/563966644/display_1500/stock-photo-watermark-reflected-on-water-and-blue-sky-563966644.jpg",
        ],
    }
)

classifier = OwlWatermarkClassifier(
    input_column="image", batch_size=8, concurrency=1, num_cpus=6, num_gpus=1
)

df = df.with_column("image_bytes", col("urls").url.download(on_error="null"))
df = df.with_column("image", col("image_bytes").image.decode())
df = classifier(df)
df.show()
