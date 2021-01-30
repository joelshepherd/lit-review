import pandas as pd
import json

with open("data/raw/articles.csv") as file:
    df = pd.read_csv(file, dtype="string")

df = df[(df["notes"] == " ")]

df["text"] = df["title"] + "\n" + df["abstract"]
df["label"] = None

df = df[["text", "label"]]

with open("data/ready/predict.json", "w") as file:
    df.to_json(file, orient="records")
