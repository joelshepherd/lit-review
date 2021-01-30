from os import path
from simpletransformers.classification import ClassificationModel
import pandas as pd

model = ClassificationModel(
    "roberta",
    "model/outputs/best",
    args={
        "fp16": False,
        "use_multiprocessing": False,
    },
)

df = pd.read_json("data/ready/predict.json", orient="records")

sentences = df["text"].tolist()
labels, _ = model.predict(sentences)

df.insert(2, "predicted_labels", labels, True)

with open("data/done/predict.json", "w") as file:
    df.to_json(file, orient="records")
