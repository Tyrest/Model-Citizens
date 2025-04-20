# adversarial.py
# ===================================================
# Implements embedding-space adversarial training utilities for ViLT + VILLA
# ===================================================

import os
import torch
import torch.nn.functional as F
import pandas as pd

# =============================
# Dataset Loader for NLVR2
# =============================

DATA_DIR = os.path.join("data", "nlvr", "nlvr2", "data")
TRAIN_FILE = "train.json"
DEV_FILE = "dev.json"
TEST_FILE = "test1.json"
TEST_FILE_2 = "test2.json"

TRAIN_IMAGE_DIR = os.path.join("data", "images", "train")
VAL_IMAGE_DIR = os.path.join("data", "dev")
TEST_IMAGE_DIR = os.path.join("data", "test1")
TEST_IMAGE_2_DIR = os.path.join("data", "test2")

def _process_val_df(df, image_dir):
    df = df[["label", "sentence", "identifier"]].copy()
    df["left"] = df["identifier"].apply(
        lambda x: os.path.join(image_dir, "-".join(x.split("-")[:-1] + ["img0.png"]))
    )
    df["right"] = df["identifier"].apply(
        lambda x: os.path.join(image_dir, "-".join(x.split("-")[:-1] + ["img1.png"]))
    )
    return df

def _process_train_df(df):
    df = df[["label", "sentence", "identifier", "directory"]].copy()

    def get_image_path(row, suffix):
        return os.path.join(
            TRAIN_IMAGE_DIR,
            str(row["directory"]),
            "-".join(row["identifier"].split("-")[:-1] + [suffix]),
        )

    df["left"] = df.apply(lambda row: get_image_path(row, "img0.png"), axis=1)
    df["right"] = df.apply(lambda row: get_image_path(row, "img1.png"), axis=1)
    return df

def load_nlvr():
    val_df = pd.read_json(os.path.join(DATA_DIR, DEV_FILE), lines=True)
    val_df = _process_val_df(val_df, VAL_IMAGE_DIR)

    test_df = pd.read_json(os.path.join(DATA_DIR, TEST_FILE), lines=True)
    test_df = _process_val_df(test_df, TEST_IMAGE_DIR)

    test_2_df = pd.read_json(os.path.join(DATA_DIR, TEST_FILE_2), lines=True)
    test_2_df = _process_val_df(test_2_df, TEST_IMAGE_2_DIR)

    train_df = pd.read_json(os.path.join(DATA_DIR, TRAIN_FILE), lines=True)
    train_df = _process_train_df(train_df)

    return train_df, val_df, test_df, test_2_df

# =============================
# Adversarial Training Utilities
# =============================

def kl_div(p_logits, q_logits):
    """
    Bidirectional KL-divergence loss.
    Encourages output distribution consistency.
    """
    p = F.log_softmax(p_logits, dim=-1)
    q = F.softmax(q_logits, dim=-1)
    return F.kl_div(p, q, reduction="batchmean") + F.kl_div(F.log_softmax(q_logits, dim=-1), F.softmax(p_logits, dim=-1), reduction="batchmean")

def generate_adv_embedding(model, inputs, epsilon=1e-5, alpha=1e-1, k=3):
    """
    FreeLB-style adversarial perturbation in embedding space.
    Args:
        model: model with .vilt and .classifier
        inputs: input dict (token ids, attention mask, pixel values, labels)
        epsilon: perturbation bound
        alpha: step size
        k: PGD steps
    Returns:
        perturbed input embeddings
    """
    # Get word embeddings (detach from graph)
    inputs_embeds = model.vilt.embeddings(input_ids=inputs["input_ids"]).detach()

    # Init small random noise (requires_grad for PGD)
    delta = torch.zeros_like(inputs_embeds).uniform_(-epsilon, epsilon).requires_grad_(True)

    for _ in range(k):
        perturbed_embeds = inputs_embeds + delta

        # Forward pass with perturbed embeddings
        outputs = model(
            input_ids=None,
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs.get("token_type_ids", None),
            pixel_values=inputs["pixel_values"],
            labels=inputs["labels"],
            input_embeds=perturbed_embeds
        )
        loss = outputs["loss"]

        # Backprop into delta
        loss.backward()
        delta.data = (delta + alpha * delta.grad.sign()).clamp(-epsilon, epsilon)
        delta.grad.zero_()

    # Return perturbed embedding for use in final forward
    return (inputs_embeds + delta).detach()
