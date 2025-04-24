import os
from typing import Any, Union
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from load_nlvr import load_nlvr

from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Trainer,
    TrainingArguments,
    ViltForImagesAndTextClassification,
)
from torchvision import transforms
import wandb

# ------------------ Config ------------------
DATA_DIR = "/local/data/nlvr/nlvr2/data"

LABEL2ID = {"False": 0, "True": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Image transformations (can be adjusted)
image_transform = transforms.Compose(
    [
        transforms.Resize((384, 384)),  # ViLT's default size
    ]
)


class NLVR2Dataset(Dataset):
    def __init__(self, df, processor, image_transform, is_test=False):
        self.df = df
        self.processor = processor
        self.image_transform = image_transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            # Load and transform images
            image1 = self.image_transform(Image.open(row["left"]).convert("RGB"))
            image2 = self.image_transform(Image.open(row["right"]).convert("RGB"))

            sentence = row["sentence"]

            # Process inputs for ViLT
            encoding = self.processor(
                images=[image1, image2],  # Pass as a list
                text=sentence,
                padding="max_length",
                truncation=True,
                max_length=40,  # Max sequence length for text
                return_tensors="pt",
            )
            encoding = {k: v.squeeze(0) for k, v in encoding.items()}

            if not self.is_test:
                label_str = row.get("label", None)
                if label_str is not None:
                    encoding["labels"] = torch.tensor(LABEL2ID[label_str])
                else:
                    print(
                        f"Warning: Missing or invalid label for index {idx}. Assigning default label 0."
                    )
                    encoding["labels"] = torch.tensor(0)

        except FileNotFoundError as e:
            print(f"Error loading image for index {idx}: {e}")
            return None
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            return None

        return encoding


# --- Using ViltForImagesAndTextClassification directly ---
model_name = "dandelin/vilt-b32-finetuned-nlvr2"
processor = AutoProcessor.from_pretrained(model_name)
model = ViltForImagesAndTextClassification.from_pretrained(model_name)
model.to(device)

# ------------------ Load Data ------------------
print("Loading NLVR data...")
try:
    train_df, val_df, test_df = load_nlvr()
    print(
        f"Loaded {len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples."
    )
except Exception as e:
    print(f"Error loading data: {e}")
    print(
        "Please ensure the data is correctly downloaded and paths are correct in load_nlvr.py"
    )
    exit()

# ------------------ Create Datasets ------------------
print("Creating datasets...")
train_dataset = NLVR2Dataset(train_df, processor, image_transform)
val_dataset = NLVR2Dataset(val_df, processor, image_transform)

# ------------------ Training Arguments ------------------
wandb.init(project="vilt-nlvr2")

training_args = TrainingArguments(
    output_dir="./vilt-nlvr2-finetuned-run",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    num_train_epochs=4,
    learning_rate=3e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=25,
    save_total_limit=4,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=torch.cuda.is_available(),
    report_to=["wandb"],  # enable logging to Weights & Biases
    dataloader_num_workers=12,
)


# ------------------ Evaluation Metric ------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}


# ------------------ Adversarial Training ------------------
class AdversarialTrainer(Trainer):
    def __init__(
        self,
        adversarial_epsilon_text=3e-2,
        adversarial_epsilon_image=1e-2,
        adversarial_target="both",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.adversarial_epsilon_text = adversarial_epsilon_text
        self.adversarial_epsilon_image = adversarial_epsilon_image
        if adversarial_target not in ["text", "image", "both"]:
            raise ValueError(
                "adversarial_target must be one of 'text', 'image', or 'both'"
            )
        self.adversarial_target = adversarial_target
        print(
            f"Adversarial training enabled with epsilon_text={self.adversarial_epsilon_text}, "
            f"epsilon_image={self.adversarial_epsilon_image} targeting '{self.adversarial_target}' embeddings."
        )

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch=None,
    ) -> torch.Tensor:
        """
        Perform a training step with optional Fast Gradient Method (FGM) adversarial training.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        # --- Standard Forward and Backward Pass ---
        with self.compute_loss_context_manager():
            loss = self.compute_loss(
                model, inputs, num_items_in_batch=num_items_in_batch
            )

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        self.accelerator.backward(loss)

        # --- Adversarial Training Step (FGSM) ---
        # Check if adversarial training is enabled for either modality
        if self.adversarial_epsilon_text > 0 or self.adversarial_epsilon_image > 0:
            # References to embedding layers
            text_emb = model.vilt.embeddings.text_embeddings.word_embeddings
            patch_emb = (
                model.vilt.embeddings.patch_embeddings.projection
            )  # Image embeddings

            # Store original embeddings and calculate perturbations
            original_text_emb = (
                text_emb.weight.data.clone()
                if self.adversarial_target in ["text", "both"]
                and self.adversarial_epsilon_text > 0
                else None
            )
            original_patch_emb = (
                patch_emb.weight.data.clone()
                if self.adversarial_target in ["image", "both"]
                and self.adversarial_epsilon_image > 0
                else None
            )
            delta_text = None
            delta_patch = None

            if (
                self.adversarial_target in ["text", "both"]
                and self.adversarial_epsilon_text > 0
                and text_emb.weight.grad is not None
            ):
                grad_text = text_emb.weight.grad.detach()
                delta_text = self.adversarial_epsilon_text * grad_text.sign()
                text_emb.weight.data.add_(delta_text)

            if (
                self.adversarial_target in ["image", "both"]
                and self.adversarial_epsilon_image > 0
                and patch_emb.weight.grad is not None
            ):
                grad_patch = patch_emb.weight.grad.detach()
                delta_patch = self.adversarial_epsilon_image * grad_patch.sign()
                patch_emb.weight.data.add_(delta_patch)

            # --- Adversarial Forward and Backward Pass ---
            if (
                delta_text is not None or delta_patch is not None
            ):  # Only proceed if perturbation happened
                with self.compute_loss_context_manager():
                    loss_adv = self.compute_loss(
                        model, inputs, num_items_in_batch=num_items_in_batch
                    )

                if self.args.gradient_accumulation_steps > 1:
                    loss_adv = loss_adv / self.args.gradient_accumulation_steps

                self.accelerator.backward(loss_adv)

                # --- Restore Original Embeddings ---
                if delta_text is not None and original_text_emb is not None:
                    text_emb.weight.data = original_text_emb
                if delta_patch is not None and original_patch_emb is not None:
                    patch_emb.weight.data = original_patch_emb

                # Optional: Average the losses for reporting, though the gradients are accumulated
                loss = (
                    loss + loss_adv
                ) / 2  # This might misrepresent the actual gradient magnitude being used

        return loss.detach()  # Return the original loss for logging purposes


# ------------------ Initialize Trainer ------------------
trainer = AdversarialTrainer(
    adversarial_epsilon_text=0.03,
    adversarial_epsilon_image=0.01,
    adversarial_target="both",  # Choose "text", "image", or "both"
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# # -------------------- Evaluate on Validation Set --------------------
# print("Evaluating on validation set...")
# val_result = trainer.evaluate(val_dataset)
# print(f"Validation Accuracy: {val_result['eval_accuracy']:.4f}")
# print("Validation results:", val_result)

# ------------------ Train and Save ------------------
print("Starting training...")
trainer.train()
print("Training finished.")

print("Saving the best model...")
trainer.save_model("./vilt-nlvr2-best-finetuned")
processor.save_pretrained("./vilt-nlvr2-best-finetuned")
print("Model and processor saved.")

# ------------------ Evaluate on Validation Set ------------------
print("Evaluating on validation set...")
val_result = trainer.evaluate(val_dataset)
print(f"Validation Accuracy: {val_result['eval_accuracy']:.4f}")

print("Script finished.")
