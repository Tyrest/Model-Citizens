import base64
import functools
import math
import os
import sys
from io import BytesIO
from typing import List

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
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

COST_PER_TOKEN = 0.15 / 1_000_000

VAL_OUTPUT_PATH = "tyler/data/multimodal_gpt_val_scores.csv"
TEST_OUTPUT_PATH = "tyler/data/multimodal_gpt_test_scores.csv"


def encode_image(image_path):
    with Image.open(image_path) as img:
        img = img.resize((512, 512), Image.Resampling.LANCZOS)

        buffered = BytesIO()
        img.save(buffered, format="PNG")
        buffered.seek(0)

        encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return encoded_string


@functools.cache
def get_score(sentence: str, left: str, right: str) -> float:
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Ouput only the label (True/False).",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Is the following sentence true or false? {sentence}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encode_image(left)}",
                            "detail": "low",
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encode_image(right)}",
                            "detail": "low",
                        },
                    },
                ],
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

    return normalized_score, completion.usage.prompt_tokens


def score_df(df: pd.DataFrame) -> List[float]:
    scores = []
    total_tokens = 0
    for _, row in progress_bar(
        df.iterrows(), description="Scoring sentences...", total=len(df)
    ):
        score, tokens = get_score(row["sentence"], row["left"], row["right"])
        if score is None:
            print("Error scoring: ", row["identifier"])
        scores.append(score)
        total_tokens += tokens
        if len(scores) % 10 == 0:
            print(
                f"Labels Scored: {len(scores)}, Current Tokens: {total_tokens}, Current Cost: ${COST_PER_TOKEN * total_tokens:.5f}"
            )
    return scores


def main():
    _, val_df, test_df = load_nlvr()

    val_df = val_df.head(100)
    test_df = test_df.head(100)

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
