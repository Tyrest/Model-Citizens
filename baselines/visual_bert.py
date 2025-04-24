import sys

sys.path.append(".")

import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import models, transforms
from torchvision.models import resnet50, ResNet50_Weights
from transformers import AutoTokenizer, VisualBertForVisualReasoning
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from transformers import CLIPProcessor, CLIPModel
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch

from load_nlvr import load_nlvr


DATA_DIR = "data/nlvr/nlvr2/data"
TEST_IMAGE_DIR = "data/test1"
TEST_FILE = "test1.json"


def _process_test_df(df, image_dir):
    df = df[["label", "sentence", "identifier"]].copy()
    df["left"] = df["identifier"].apply(
        lambda x: os.path.join(image_dir, "-".join(x.split("-")[:-1] + ["img0.png"]))
    )
    df["right"] = df["identifier"].apply(
        lambda x: os.path.join(image_dir, "-".join(x.split("-")[:-1] + ["img1.png"]))
    )
    return df


# Load dataset
test_df = pd.read_json(os.path.join(DATA_DIR, TEST_FILE), lines=True)
test_df = _process_test_df(test_df, TEST_IMAGE_DIR)

# Load model
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = VisualBertForVisualReasoning.from_pretrained("uclanlp/visualbert-nlvr2").eval()

# Load Faster R-CNN for object detection
faster_rcnn = fasterrcnn_resnet50_fpn(pretrained=True)
faster_rcnn = torch.nn.Sequential(*list(faster_rcnn.children())[:-1])
faster_rcnn.eval()
image_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# Function to extract visual features
def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image_transform(image).unsqueeze(0)
    print(image.shape)
    with torch.no_grad():
        features = faster_rcnn(image)
    return features


predictions = []

batch_bar = tqdm(
    total=len(test_df),
    dynamic_ncols=True,
    leave=False,
    position=0,
    desc="Processing batches",
)

total_accuracy = 0

with torch.no_grad():
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        # Process images
        visual_embed_left = extract_features(row["left"])
        visual_embed_right = extract_features(row["right"])

        print(visual_embed_left.shape, visual_embed_right.shape)
        break

        # Combine visual embeddings
        visual_embeds = torch.cat(
            [visual_embed_left, visual_embed_right], dim=0
        ).unsqueeze(0)
        visual_embeds = projection_layer(visual_embeds)

        visual_token_type_ids = torch.ones(
            (1, visual_embeds.shape[1]), dtype=torch.long
        )
        visual_attention_mask = torch.ones(
            (1, visual_embeds.shape[1]), dtype=torch.float
        )

        # Process text
        text = row["sentence"]
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128,
        )

        inputs.update(
            {
                "visual_embeds": visual_embeds,
                "visual_token_type_ids": visual_token_type_ids,
                "visual_attention_mask": visual_attention_mask,
            }
        )

        outputs = model(**inputs)
        logits = outputs.logits
        pred_label = logits.argmax(-1).item()
        label_str = "true" if pred_label == 1 else "false"
        predictions.append(label_str)

        total_accuracy += 1 if pred_label == row["label"].lower() else 0
        batch_bar.set_postfix(
            {
                "label": row["label"],
                "prediction": label_str,
                "accuracy": total_accuracy / (len(predictions)),
            }
        )

        batch_bar.update()


test_df["prediction"] = predictions
test_df["correct"] = test_df.apply(
    lambda row: row["prediction"] == str(row["label"]).lower(), axis=1
)
accuracy = test_df["correct"].mean()

test_df.to_csv("nlvr2_test_predictions_visualbert.csv", index=False)

print(f"Inference complete. Accuracy: {accuracy:.4f}.")

# Accuracy: 0.5011
