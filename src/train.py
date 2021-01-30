import torch
from os import path
import pandas as pd
from sklearn import metrics
from simpletransformers.classification import ClassificationModel

print("CUDA available?")
print(torch.cuda.is_available())

train_df = pd.read_csv(
    "data/train/train.csv",
    header=0,
    names=["id", "text", "labels"],
    dtype={"text": "string"},
)
test_df = pd.read_csv(
    "data/train/test.csv",
    header=0,
    names=["id", "text", "labels"],
    dtype={"text": "string"},
)

model = ClassificationModel(
    "roberta",
    "roberta-base",
    args={
        "fp16": False,
        "num_train_epochs": 5,
        "evaluate_during_training": True,
        # dirs
        "cache_dir": "model/cache",
        "output_dir": "model/outputs",
        "best_model_dir": "model/outputs/best",
        "tensorboard_dir": "model/runs",
    },
)

model.train_model(
    train_df,
    eval_df=test_df,
    acc=metrics.accuracy_score,
    cr=metrics.classification_report,
)
