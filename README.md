# teraflopai-data

A petabyte scale data processing framework for AI models using Daft + Ray.

## Installation
```python
pip install teraflopai-data
```

### Pipeline
```python
import daft

from src.teraflopai_data.components.text.embedding import SentenceTransformersEmbed
from src.teraflopai_data.components.text.fineweb_edu import FinewebEduClassifier
from src.teraflopai_data.pipeline import Pipeline

df = daft.from_pydict(
    {
        "text": [
            "My mother told me",
            "Someday I will buy",
            "Galleys with good oars",
            "Sail to distant shores",
        ],
    }
)

classifier = FinewebEduClassifier(
    input_column="text",
    batch_size=4,
    concurrency=1,
    num_cpus=6,
    num_gpus=1,
)

embedder = SentenceTransformersEmbed(
    input_column="text",
    model_name="all-MiniLM-L6-v2",
    batch_size=4,
    concurrency=1,
    num_cpus=6,
    num_gpus=1,
)

pipeline = Pipeline(
    ops=[classifier, embedder],
)

df = pipeline(df)
df.show()
```

### Text

### Image

Image Hashing
```python
import daft
from daft import col

from src.teraflopai_data.components.image.image_hashing import ImageHasher

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

hasher = ImageHasher(
    input_column="image",
    hashing_algorithm="wavelet",
    concurrency=1,
    num_cpus=6,
)

df = df.with_column("image_bytes", col("urls").url.download(on_error="null"))
df = df.with_column("image", col("image_bytes").image.decode())
df = hasher(df)
df = df.drop_duplicates("image_hash")
df.show()
```
### Video

## Slurm

