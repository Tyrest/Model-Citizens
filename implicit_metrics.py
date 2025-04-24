import os
from typing import Any, Union

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, ViltForImagesAndTextClassification

import torch.nn.functional as F
from sklearn.metrics import silhouette_score
import numpy as np
from safetensors.torch import load_file
from torchvision import transforms

from load_nlvr import load_nlvr
# from ViLT_finetune import AdversarialTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------ Config ------------------
# DATA_DIR = "data/nlvr/nlvr2/data"
# IMAGE_DIR = os.path.join(DATA_DIR, "images")
# TEST_FILE = "test1.json"
# TEST_IMAGE_DIR = "data/test1" # Assuming test images are in this separate dir as per original script

# LABEL2ID = {"False": 0, "True": 1}
# ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# Checkpoint directory from the finetuning script
CHECKPOINT_BASE_DIR = "checkpoints"

# Find checkpoint paths
checkpoint_paths = []
if os.path.isdir(CHECKPOINT_BASE_DIR):
    checkpoint_paths = sorted(
        [
            os.path.join(CHECKPOINT_BASE_DIR, d, "checkpoint-2700")
            for d in os.listdir(CHECKPOINT_BASE_DIR)
        ]
    )
else:
    print(f"Warning: Checkpoint base directory '{CHECKPOINT_BASE_DIR}' not found.")

print(f"Found checkpoints: {checkpoint_paths}")

model_name = "dandelin/vilt-b32-finetuned-nlvr2"
processor = AutoProcessor.from_pretrained(model_name)

image_transform = transforms.Compose(
    [
        transforms.Resize((384, 384)),  # ViLT's default size
    ]
)

# Load test data
train_df, val_df, test_df = load_nlvr()
test_df = test_df[:100]

# Save results
results = []

model = ViltForImagesAndTextClassification.from_pretrained("dandelin/vilt-b32-finetuned-nlvr2")
model.to(device)

for ckpt_path in checkpoint_paths:
    print(f"\n--- Processing checkpoint: {ckpt_path} ---")
    try:
        model = ViltForImagesAndTextClassification.from_pretrained(ckpt_path)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model from {ckpt_path}: {e}")
        continue

    cosine_scores = []
    all_embeddings = []
    entropy_scores = []

    with torch.no_grad():
        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            # for _, row in tqdm(test_df.head(10).iterrows(), total=10):
            image1 = image_transform(Image.open(row["left"]).convert("RGB"))
            image2 = image_transform(Image.open(row["right"]).convert("RGB"))
            text = row["sentence"]

            encoding = processor(
                [image1, image2],
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=40,
            )

            encoding.to(device)

            # Cosine Similarity
            text_inputs = {
                "input_ids": encoding["input_ids"],
                "token_type_ids": encoding["token_type_ids"],
            }
            text_embed_tokens = model.vilt.embeddings.text_embeddings(
                **text_inputs
            )
            text_embed = text_embed_tokens.mean(dim=1)

            image_embed_tokens = model.vilt.embeddings.patch_embeddings(
                encoding["pixel_values"]
            )
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

            outputs = model.vilt(
                input_ids=text_input_ids,
                token_type_ids=text_token_type_ids,
                attention_mask=text_attention_mask,
                pixel_values=pixel_values,
                output_attentions=True,
                return_dict=True,
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
            vilt_out = model.vilt(
                input_ids=text_input_ids,
                token_type_ids=text_token_type_ids,
                attention_mask=text_attention_mask,
                pixel_values=pixel_values,
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
    results.append(
        {
            "checkpoint": os.path.basename(ckpt_path),
            "cosine_similarity": round(mean_cosine, 4),
            "attention_entropy": round(final_entropy, 4),
            "silhouette_score": round(sil_score, 4),
        }
    )

    print(
        f"Cosine: {mean_cosine:.4f}, Entropy: {final_entropy:.4f}, Silhouette: {sil_score:.4f}"
    )

results_df = pd.DataFrame(results)
results_df.to_csv("implicit_checkpoint_metrics.csv", index=False)