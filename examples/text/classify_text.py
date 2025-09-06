import daft

from teraflopai_data.components.text.fineweb_edu import FinewebEduClassifier

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
df = classifier(df)
df.show()
