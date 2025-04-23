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
from safetensors.torch import load_file

# ------------------ Config ------------------
DATA_DIR = "/local/data/nlvr/nlvr2/data"
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

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, pixel_values=None, labels=None, epsilon=0.0):
        # enable gradient for pixel_values if adversarial epsilon is set
        if labels is not None and epsilon > 0.0 and pixel_values is not None:
            pixel_values.requires_grad_(True)
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
            # compute FGSM adversarial loss if epsilon>0
            if epsilon > 0.0 and pixel_values is not None:
                # compute gradient wrt pixel_values
                grads = torch.autograd.grad(loss, pixel_values, retain_graph=True)[0]
                # generate adversarial pixel values
                adv_pixels = pixel_values + epsilon * grads.sign()
                adv_pixels = adv_pixels.detach()
                # clamp pixel values to valid range
                adv_pixels = torch.clamp(adv_pixels, 0, 1)
                # forward pass with adversarial pixels
                adv_outputs = self.vilt(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    pixel_values=adv_pixels,
                )
                adv_pooled = adv_outputs.pooler_output
                adv_logits = self.classifier(adv_pooled)
                loss_adv = loss_fn(adv_logits, labels)
                # combine losses and logits
                loss = loss_adv
                logits = adv_logits
        return {"loss": loss, "logits": logits}


class ViltFineTuner(PreTrainedModel):
    def __init__(self, config: ViltConfig):
        super().__init__(config)
        self.model = ViltForNLVR2(pretrained_model="dandelin/vilt-b32-mlm", num_labels=config.num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, pixel_values=None, labels=None, epsilon=0.0):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            labels=labels,
            epsilon=epsilon,
        )


train_df, val_df, test_df = load_nlvr()

# ------------------ Load Processor & Model ------------------
processor = AutoProcessor.from_pretrained("dandelin/vilt-b32-mlm")
config = ViltConfig.from_pretrained("dandelin/vilt-b32-mlm")
config.num_labels = 2
model = ViltFineTuner(config)
model.load_state_dict(load_file("./checkpoint-5400/model.safetensors"))

train_dataset = NLVR2Dataset(train_df, processor)
val_dataset = NLVR2Dataset(val_df, processor)
test_dataset = NLVR2Dataset(test_df, processor, is_test=True)

# ------------------ Training Arguments ------------------
training_args = TrainingArguments(
    output_dir="./vilt-nlvr2-adversarial-image",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    num_train_epochs=4,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    save_total_limit=4,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=False,
    dataloader_num_workers=12,
)

# ------------------ Evaluation Metric ------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# ------------------ Custom FGSM Trainer ------------------
class FGSMTrainer(Trainer):
    def __init__(self, epsilon=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        # use adversarial epsilon only in training mode
        epsilon_val = self.epsilon if model.training else 0.0
        outputs = model(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            token_type_ids=inputs.get("token_type_ids"),
            pixel_values=inputs.get("pixel_values"),
            labels=labels,
            epsilon=epsilon_val,
        )
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

# ------------------ Initialize Trainer ------------------
# set FGSM epsilon for adversarial perturbation
epsilon_val = 0.03 # adjust as needed
trainer = FGSMTrainer(
    epsilon=epsilon_val,
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=processor,
    compute_metrics=compute_metrics,
)

torch.cuda.empty_cache()

# ------------------ Train and Save ------------------
trainer.train()
trainer.save_model("./vilt-nlvr2-adversarial-image-final")

# ------------------ Evaluate on Validation Set ------------------
val_result = trainer.evaluate(val_dataset)
print(f"Validation Accuracy: {val_result['eval_accuracy']:.4f}")
