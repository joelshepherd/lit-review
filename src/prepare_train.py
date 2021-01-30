import pandas as pd
import json

with open("data/raw/articles.csv") as file:
    df = pd.read_csv(file, dtype="string")

df = df[(df["notes"] != " ")]

df["text"] = df["title"] + "\n" + df["abstract"]
df["label"] = df.apply(
    lambda row: 1
    if row["notes"].startswith(' RAYYAN-INCLUSION: {"Nicole"=>"Excluded"}')
    else 0,
    axis=1,
)

df = df[["text", "label"]]

with open("data/ready/articles.json", "w") as file:
    df.to_json(file, orient="records")
