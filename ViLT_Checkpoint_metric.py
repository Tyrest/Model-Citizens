import os
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, ViltModel, ViltConfig
import torch.nn.functional as F
from sklearn.metrics import silhouette_score
import torch.nn as nn
import numpy as np
from safetensors.torch import load_file

from transformers import (
    ViltModel,
    ViltConfig,
    PreTrainedModel,
    AutoProcessor,
    Trainer,
    TrainingArguments
)

# ------------------ Config ------------------
DATA_DIR = "data/nlvr/nlvr2/data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
TRAIN_FILE = "train.json"
VAL_FILE = "dev.json"

TEST_FILE = "test1.json"
TEST_IMAGE_DIR = "data/test1"

LABEL2ID = {"False": 0, "True": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# ------------------ Custom Model Definition ------------------
class ViltForNLVR2(nn.Module):
    def __init__(self, pretrained_model="dandelin/vilt-b32-mlm", num_labels=2):
        super().__init__()
        self.vilt = ViltModel.from_pretrained(pretrained_model)
        hidden_size = self.vilt.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels)
        )

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, pixel_values=None, labels=None):
        outputs = self.vilt(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
        )
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        return {"loss": loss, "logits": logits}


class ViltFineTuner(PreTrainedModel):
    def __init__(self, config: ViltConfig):
        super().__init__(config)
        self.model = ViltForNLVR2(pretrained_model="dandelin/vilt-b32-mlm", num_labels=config.num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, pixel_values=None, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            labels=labels,
        )

processor = AutoProcessor.from_pretrained("dandelin/vilt-b32-mlm")
config = ViltConfig.from_pretrained("dandelin/vilt-b32-mlm")
config.num_labels = 2
model = ViltFineTuner(config)

# # Your data setup
# DATA_DIR = os.path.join("data", "nlvr", "nlvr2", "data")
# TEST_IMAGE_DIR = "data/test1"
# TEST_FILE = "test1.json"

def _process_test_df(df, image_dir):
    df = df[["label", "sentence", "identifier"]].copy()
    df["left"] = df["identifier"].apply(
        lambda x: os.path.join(image_dir, "-".join(x.split("-")[:-1] + ["img0.png"]))
    )
    df["right"] = df["identifier"].apply(
        lambda x: os.path.join(image_dir, "-".join(x.split("-")[:-1] + ["img1.png"]))
    )
    return df

# Load data
test_df = pd.read_json(os.path.join(DATA_DIR, TEST_FILE), lines=True)
test_df = _process_test_df(test_df, TEST_IMAGE_DIR)
# test_df = test_df.head(10)  # for testing purposes

# Checkpoint directory
CHECKPOINT_DIR = "/Users/angelaqu/Desktop/checkpoints"
checkpoint_paths = sorted([
    os.path.join(CHECKPOINT_DIR, d)
    for d in os.listdir(CHECKPOINT_DIR)
    if os.path.isdir(os.path.join(CHECKPOINT_DIR, d))
])

# Save results
results = []

for ckpt_path in checkpoint_paths:
    # print(f"\n Loading checkpoint: {ckpt_path}")
    # checkpoint = torch.load(os.path.join(ckpt_path, "training_args.bin") , map_location=torch.device("cpu"))
    # model.load_state_dict(checkpoint)
    print(ckpt_path)
    model_path = ckpt_path
    model.load_state_dict(load_file(model_path + '/model.safetensors', device="cpu"))

    # print(f"\n Running checkpoint: {ckpt_path}")
    # processor = AutoProcessor.from_pretrained(ckpt_path)
    # model = ViltModel.from_pretrained(ckpt_path)
    # model.eval()

    cosine_scores = []
    all_embeddings = []
    entropy_scores = []

    with torch.no_grad():
        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        # for _, row in tqdm(test_df.head(10).iterrows(), total=10):
            image1 = Image.open(row["left"]).convert("RGB")
            image2 = Image.open(row["right"]).convert("RGB")
            text = row["sentence"]

            encoding = processor(
                [image1, image2],
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=40
            )

            # Cosine Similarity
            text_inputs = {
                "input_ids": encoding["input_ids"],
                "token_type_ids": encoding["token_type_ids"],
            }
            text_embed_tokens = model.model.vilt.embeddings.text_embeddings(**text_inputs)
            text_embed = text_embed_tokens.mean(dim=1)

            image_embed_tokens = model.model.vilt.embeddings.patch_embeddings(encoding["pixel_values"])
            image_embed_tokens = image_embed_tokens.flatten(2).transpose(1, 2)
            image_embed = image_embed_tokens.mean(dim=1)
            combined_image_embed = image_embed.mean(dim=0, keepdim=True)

            cosine_sim = F.cosine_similarity(text_embed, combined_image_embed).item()
            cosine_scores.append(cosine_sim)

            # Attention Entropy
            text_input_ids = encoding["input_ids"].expand(2, -1)
            text_token_type_ids = encoding["token_type_ids"].expand(2, -1)
            text_attention_mask = encoding["attention_mask"].expand(2, -1)
            pixel_values = encoding["pixel_values"]

            outputs = model.model.vilt(
                input_ids=text_input_ids,
                token_type_ids=text_token_type_ids,
                attention_mask=text_attention_mask,
                pixel_values=pixel_values,
                output_attentions=True,
                return_dict=True
            )

            attentions = outputs.attentions
            layer_entropies = []
            for layer_attn in attentions:
                if isinstance(layer_attn, tuple):
                    layer_attn = layer_attn[0]
                entropy = -torch.sum(layer_attn * torch.log(layer_attn + 1e-8), dim=-1)
                entropy = entropy.mean(dim=-1).mean(dim=-1)
                layer_entropies.append(entropy)

            sample_entropies = torch.stack(layer_entropies).mean(dim=0)
            mean_entropy = sample_entropies.mean().item()
            entropy_scores.append(mean_entropy)

            # Silhouette embeddings
            vilt_out = model.model.vilt(
                input_ids=text_input_ids,
                token_type_ids=text_token_type_ids,
                attention_mask=text_attention_mask,
                pixel_values=pixel_values
            )
            pooled_output = vilt_out.pooler_output
            pooled_cat = torch.cat((pooled_output[0], pooled_output[1]), dim=0)
            all_embeddings.append(pooled_cat)

    # Compute metrics
    mean_cosine = sum(cosine_scores) / len(cosine_scores)
    final_entropy = sum(entropy_scores) / len(entropy_scores)
    X = torch.stack(all_embeddings).cpu().numpy()
    label_map = {"True": 1, "False": 0}
    labels = test_df["label"].map(label_map).astype(int).tolist()
    sil_score = silhouette_score(X, labels, metric="cosine")

    # Save results
    results.append({
        "checkpoint": os.path.basename(ckpt_path),
        "cosine_similarity": round(mean_cosine, 4),
        "attention_entropy": round(final_entropy, 4),
        "silhouette_score": round(sil_score, 4)
    })

    print(f"Cosine: {mean_cosine:.4f}, Entropy: {final_entropy:.4f}, Silhouette: {sil_score:.4f}")

# # Export results to CSV
# results_df = pd.DataFrame(results)
# results_df.to_csv("vilt_checkpoint_metrics.csv", index=False)
# print("Saved all metrics to vilt_checkpoint_metrics.csv")
