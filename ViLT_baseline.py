import os
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from load_nlvr import load_nlvr

from torch.utils.data import Dataset
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

from torchvision import transforms

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


train_df, val_df, test_df, _ = load_nlvr()

# ------------------ Load Processor & Model ------------------
processor = AutoProcessor.from_pretrained("dandelin/vilt-b32-mlm")
config = ViltConfig.from_pretrained("dandelin/vilt-b32-mlm")
config.num_labels = 2
model = ViltFineTuner(config)

train_dataset = NLVR2Dataset(train_df, processor)
val_dataset = NLVR2Dataset(val_df, processor)
test_dataset = NLVR2Dataset(test_df, processor, is_test=True)

# ------------------ Training Arguments ------------------
training_args = TrainingArguments(
    output_dir="./vilt-nlvr2-custom",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=False,
)

# ------------------ Evaluation Metric ------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# ------------------ Initialize Trainer ------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=processor,
    compute_metrics=compute_metrics,
)

# ------------------ Train and Save ------------------
# trainer.train()
# trainer.save_model("./vilt-nlvr2-finetuned")
from safetensors.torch import load_file
state_dict = load_file("./checkpoints/checkpoint-5400/model.safetensors")
model.load_state_dict(state_dict)
print("Model loaded from checkpoint.")  
# ------------------ Evaluate on Validation Set ------------------
import ipdb
ipdb.set_trace()
val_result = trainer.evaluate(val_dataset)
print(f"Validation Accuracy: {val_result['eval_accuracy']:.4f}")

# ------------------ Run Inference on Test Set ------------------
print("Running inference on test1...")

model.eval()
predictions = []

with torch.no_grad():
    for idx in tqdm(range(len(test_df))):
        row = test_df.iloc[idx]
        image1 = Image.open(row["left"]).convert("RGB")
        image2 = Image.open(row["right"]).convert("RGB")
        sentence = row["sentence"]

        encoding = processor(
            [image1, image2],
            sentence,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=40
        )
        encoding = {k: v for k, v in encoding.items()}

        outputs = model(**encoding)
        pred_label = ID2LABEL[outputs["logits"].argmax(-1).item()]
        predictions.append(pred_label)

# ------------------ Save Test Predictions ------------------
test_df["prediction"] = predictions
test_df.to_csv("nlvr2_test1_predictions.csv", index=False)
print("Test predictions saved to nlvr2_test1_predictions.csv")
