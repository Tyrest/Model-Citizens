import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoProcessor, ViltConfig
from tqdm import tqdm
from load_nlvr import load_nlvr
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    ViltModel,
    ViltConfig,
    PreTrainedModel,
    AutoProcessor,
    Trainer,
    TrainingArguments
)
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms

# -------- Settings --------
EPSILON = 1e-1  # perturbation budget
ADV_STEPS = 3   # K steps of PGD
ALPHA = 1e-1    # step size
BATCH_SIZE = 8
EPOCHS = 4
LR = 5e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL2ID = {"False": 0, "True": 1}

resize_transform = transforms.Compose([
    transforms.Resize((384, 384)),
])

class NLVR2Dataset(Dataset):
    def __init__(self, df, processor, is_test=False):
        self.df = df
        self.processor = processor
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image1 = resize_transform(Image.open(row["left"]).convert("RGB"))
        image2 = resize_transform(Image.open(row["right"]).convert("RGB"))
        concat_image = Image.new("RGB", (image1.width + image2.width, image1.height))
        concat_image.paste(image1, (0, 0))
        concat_image.paste(image2, (image1.width, 0))

        sentence = row["sentence"]

        encoding = self.processor(
            concat_image,
            sentence,
            padding="max_length",
            truncation=True,
            max_length=40,
            return_tensors="pt",
        )
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        if not self.is_test:
            encoding["labels"] = torch.tensor(LABEL2ID[row["label"]])
        return encoding
    
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

# -------- Data & Model --------
train_df, val_df, _, _ = load_nlvr()
processor = AutoProcessor.from_pretrained("dandelin/vilt-b32-mlm")
config = ViltConfig.from_pretrained("dandelin/vilt-b32-mlm")
config.num_labels = 2

model = ViltFineTuner(config).to(DEVICE)
train_dataset = NLVR2Dataset(train_df, processor)
val_dataset = NLVR2Dataset(val_df, processor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
global_step = 0

# -------- Training with VILLA --------
def compute_kl(p_logits, q_logits):
    p = F.log_softmax(p_logits, dim=-1)
    q = F.softmax(q_logits, dim=-1)
    return F.kl_div(p, q, reduction='batchmean') + F.kl_div(F.log_softmax(q_logits, dim=-1), F.softmax(p_logits, dim=-1), reduction='batchmean')

writer = SummaryWriter(log_dir="./runs/vilt_nlvr2")

model.train()
for epoch in range(EPOCHS):
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch in pbar:
        batch = {k: v.to(DEVICE) for k, v in batch.items() if k in ["input_ids", "attention_mask", "token_type_ids", "pixel_values", "labels"]}


        # -------- Clean Forward --------
        clean_output = model(**batch)
        clean_loss = clean_output["loss"]
        clean_logits = clean_output["logits"]

        # -------- Create Perturbation --------
        delta = {}
        if "pixel_values" in batch:
            delta["pixel_values"] = torch.zeros_like(batch["pixel_values"], dtype=torch.float).uniform_(-EPSILON, EPSILON).to(DEVICE)
            delta["pixel_values"].requires_grad = True


        adv_loss = 0
        for _ in range(ADV_STEPS):
            adv_batch = batch.copy()
            if "input_ids" in delta:
                adv_batch["input_ids"] = batch["input_ids"] + delta["input_ids"]
            if "pixel_values" in delta:
                adv_batch["pixel_values"] = batch["pixel_values"] + delta["pixel_values"]

            adv_output = model(**adv_batch)
            adv_logits = adv_output["logits"]
            loss_adv = F.cross_entropy(adv_logits, batch["labels"])
            kl_loss = compute_kl(clean_logits.detach(), adv_logits)
            total_loss = loss_adv + ALPHA * kl_loss
            total_loss.backward()

            # update delta
            for k in delta:
                grad = delta[k].grad
                delta[k] = (delta[k] + ALPHA * grad.sign()).detach()
                delta[k].clamp_(-EPSILON, EPSILON)
                delta[k].requires_grad = True

            adv_loss += loss_adv.item()

        optimizer.step()
        optimizer.zero_grad()
        pbar.set_postfix(loss=clean_loss.item(), adv_loss=adv_loss / ADV_STEPS)
        writer.add_scalar("Loss/train_clean", clean_loss.item(), global_step)
        writer.add_scalar("Loss/train_adv", adv_loss / ADV_STEPS, global_step)
        global_step += 1
writer.close()

# -------- Evaluation --------
def evaluate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            preds = outputs["logits"].argmax(dim=1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
    acc = correct / total
    print(f"Validation Accuracy: {acc:.4f}")

evaluate()
