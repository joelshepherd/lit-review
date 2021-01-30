import pandas as pd

df = pd.read_json("data/ready/articles.json", orient="records")

train_set = df.sample(frac=0.8, random_state=0)
test_set = df.drop(train_set.index)

train_set.to_csv("data/train/train.csv")
test_set.to_csv("data/train/test.csv")
