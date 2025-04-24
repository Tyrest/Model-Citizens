import functools
import math
import os
import sys
from typing import List

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from rich.prompt import Confirm
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)

sys.path.append(".")

from progress_bar import progress_bar

from load_nlvr import load_nlvr

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

VAL_OUTPUT_PATH = "tyler/data/unimodal_gpt_val_scores.csv"
TEST_OUTPUT_PATH = "tyler/data/unimodal_gpt_test_scores.csv"


@functools.cache
def get_score(sentence: str) -> float:
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an expert at determine whether a given sentence is true or false. Evaluate the following sentence and provide your answer. Ouput only the label (True/False).",
            },
            {
                "role": "user",
                "content": f"Is the following sentence true or false? {sentence}",
            },
        ],
        logprobs=True,
        max_tokens=1,
        top_logprobs=5,
    )

    true_score = 0
    false_score = 0

    for token in completion.choices[0].logprobs.content[0].top_logprobs[::-1]:
        if token.token.lower() == "true":
            true_score = math.exp(token.logprob)
        elif token.token.lower() == "false":
            false_score = math.exp(token.logprob)

    if true_score == 0 and false_score == 0:
        return None

    normalized_score = true_score / (true_score + false_score)

    return normalized_score


def score_df(df: pd.DataFrame) -> List[float]:
    scores = []
    for _, row in progress_bar(
        df.iterrows(), description="Scoring sentences...", total=len(df)
    ):
        score = get_score(row["sentence"])
        if score is None:
            print("Error scoring: ", row["identifier"])
        scores.append(score)
    return scores


def main():
    _, val_df, test_df = load_nlvr()

    if Confirm.ask(
        f"Do you want to re-score the validation and test sets? (len {len(val_df)} and {len(test_df)})"
    ):
        val_df["score"] = score_df(val_df)
        test_df["score"] = score_df(test_df)
    else:
        val_df = pd.read_csv(VAL_OUTPUT_PATH)
        test_df = pd.read_csv(TEST_OUTPUT_PATH)

    # Select threshold that maximizes F1 score
    precision, recall, thresholds = precision_recall_curve(
        val_df["label"], val_df["score"]
    )
    f1 = 2 * (precision * recall) / (precision + recall)
    threshold = thresholds[f1.argmax()]
    print("Best Threshold:", threshold)

    val_df["prediction"] = val_df["score"] > threshold
    test_df["prediction"] = test_df["score"] > threshold

    columns = [
        "label",
        "prediction",
        "score",
        "sentence",
        "identifier",
        "left",
        "right",
    ]

    val_df[columns].to_csv(VAL_OUTPUT_PATH, index=False)
    test_df[columns].to_csv(TEST_OUTPUT_PATH, index=False)

    print("=" * 80)

    print("Validation Accuracy:", accuracy_score(val_df["label"], val_df["prediction"]))
    print("Validation AUC:", roc_auc_score(val_df["label"], val_df["score"]))
    # print("Validation Report:")
    # print(classification_report(val_df["label"], val_df["prediction"]))

    print("-" * 80)

    print("Test Accuracy:", accuracy_score(test_df["label"], test_df["prediction"]))
    print("Test AUC:", roc_auc_score(test_df["label"], test_df["score"]))
    # print("Test Report:")
    # print(classification_report(test_df["label"], test_df["prediction"]))

    print("=" * 80)


if __name__ == "__main__":
    main()
