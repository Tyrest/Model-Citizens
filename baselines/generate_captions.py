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

# client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
client = OpenAI(
    api_key="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"
)

COST_PER_TOKEN = 2.5 / 1_000_000

VAL_OUTPUT_PATH = "tyler/data/multimodal_gpt_val_scores.csv"
TEST_OUTPUT_PATH = "tyler/data/multimodal_gpt_test_scores.csv"
TEST_OUTPUT_INTERMEDIATE_PATH = "tyler/data/multimodal_gpt_test_scores_intermediate.csv"


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
        model="gpt-4o",
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
    with open(TEST_OUTPUT_INTERMEDIATE_PATH, "w") as f:
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
            f.write(f"{row['identifier']},{score}\n")
    return scores


def main():
    _, val_df, test_df = load_nlvr()

    # val_df = val_df.head(1)
    # test_df = test_df.head(5)

    print(val_df)

    if Confirm.ask(f"Do you want to re-score the test set? (len {len(test_df)})"):
        test_df["score"] = score_df(test_df)
    else:
        scores_df = pd.read_csv(TEST_OUTPUT_INTERMEDIATE_PATH)

        print(test_df)
        print(scores_df)

        test_df = test_df.merge(scores_df, on="identifier")
        print(test_df)
        # rename score_y to score

        test_df = test_df.rename(columns={"score_y": "score"})
        # remove rows with missing scores
        # test_df = test_df[~test_df["score"].isnull()]

    # # Select threshold that maximizes F1 score
    # precision, recall, thresholds = precision_recall_curve(
    #     val_df["label"], val_df["score"]
    # )
    # f1 = 2 * (precision * recall) / (precision + recall)
    # threshold = thresholds[f1.argmax()]
    # print("Best Threshold:", threshold)
    threshold = 0.5

    # val_df["prediction"] = val_df["score"] > threshold
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

    # val_df[columns].to_csv(VAL_OUTPUT_PATH, index=False)
    test_df[columns].to_csv(TEST_OUTPUT_PATH, index=False)

    test_df = pd.read_csv(TEST_OUTPUT_PATH)

    print("=" * 80)

    # print("Validation Accuracy:", accuracy_score(val_df["label"], val_df["prediction"]))
    # print("Validation AUC:", roc_auc_score(val_df["label"], val_df["score"]))
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
