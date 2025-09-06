import daft

from teraflopai_data.components.text.embedding import SentenceTransformersEmbed
from teraflopai_data.components.text.fineweb_edu import FinewebEduClassifier
from teraflopai_data.pipeline import Pipeline

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
