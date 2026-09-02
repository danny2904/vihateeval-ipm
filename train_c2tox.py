"""
Train C2Tox: a small router MLP that combines the frozen OC and IC branches
(see train_oc.py / train_ic.py) via a learned per-sample weighted average.

Minimal, self-contained reproduction script for the ViHateEval dataset.
Requires OC and IC checkpoints already trained (same seed, same data split).

Usage:
    python train_c2tox.py --oc-checkpoint oc_model.pth --ic-checkpoint ic_model.pth --data-dir .
"""
import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer, ViTModel, ViTImageProcessor, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

TEXT_MODEL_NAME = "vinai/phobert-base"
IMAGE_MODEL_NAME = "google/vit-base-patch16-224"
LABEL_MAP = {"Normal": 0, "Offensive": 1, "Hate Speech": 2}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(data_dir):
    df = pd.read_csv(os.path.join(data_dir, "data.csv"))
    df["label"] = df["label"].map(LABEL_MAP)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    return train_df, val_df, test_df


class DualTextDataset(Dataset):
    """Provides both OC text (comment only) and IC text (title+thumb+comment)
    plus the shared thumbnail image, needed for the combined router model."""

    def __init__(self, df, image_dir, tokenizer, max_length=256):
        self.df = df
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        oc_enc = self.tokenizer(
            str(row["comment_text"]),
            truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt",
        )
        ic_text = f"[TITLE] {row['video_title']} [THUMB] {row['thumbnail_text']} [COMMENT] {row['comment_text']}"
        ic_enc = self.tokenizer(
            ic_text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt",
        )
        try:
            image = Image.open(os.path.join(self.image_dir, row["img"])).convert("RGB")
            image_tensor = self.transform(image)
        except Exception:
            image_tensor = torch.zeros(3, 224, 224)
        return {
            "oc_input_ids": oc_enc["input_ids"].flatten(),
            "oc_attention_mask": oc_enc["attention_mask"].flatten(),
            "ic_input_ids": ic_enc["input_ids"].flatten(),
            "ic_attention_mask": ic_enc["attention_mask"].flatten(),
            "image": image_tensor,
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
        }


class PhoBERTViTClassifier(nn.Module):
    """Same architecture as in train_oc.py / train_ic.py — must match to load checkpoints."""

    def __init__(self, num_classes=3, dropout=0.3):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(TEXT_MODEL_NAME)
        self.image_model = ViTModel.from_pretrained(IMAGE_MODEL_NAME)
        self.hidden_size = self.text_model.config.hidden_size + self.image_model.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, image):
        text_features = self.text_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        image_features = self.image_model(pixel_values=image).pooler_output
        fused = torch.cat([text_features, image_features], dim=1)
        return self.classifier(self.dropout(fused))


class Router(nn.Module):
    """Small MLP that outputs a 2-way softmax weight (alpha_OC, alpha_IC)
    from the OC branch's fused text+image features."""

    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 2),
        )

    def forward(self, h):
        return F.softmax(self.mlp(h), dim=-1)


class C2Tox(nn.Module):
    """Combines frozen OC + IC branches via a learned router:
    logits = alpha_OC * oc_logits + alpha_IC * ic_logits."""

    def __init__(self, oc_checkpoint, ic_checkpoint, dropout=0.3, device="cpu"):
        super().__init__()
        self.oc_model = PhoBERTViTClassifier(dropout=dropout)
        self.ic_model = PhoBERTViTClassifier(dropout=dropout)
        self.oc_model.load_state_dict(torch.load(oc_checkpoint, map_location=device))
        self.ic_model.load_state_dict(torch.load(ic_checkpoint, map_location=device))
        self.router = Router(self.oc_model.hidden_size)
        for p in self.oc_model.parameters():
            p.requires_grad = False
        for p in self.ic_model.parameters():
            p.requires_grad = False

    def forward(self, oc_input_ids, oc_attention_mask, ic_input_ids, ic_attention_mask, image):
        oc_logits = self.oc_model(oc_input_ids, oc_attention_mask, image)
        ic_logits = self.ic_model(ic_input_ids, ic_attention_mask, image)
        with torch.no_grad():
            oc_text_features = self.oc_model.text_model(input_ids=oc_input_ids, attention_mask=oc_attention_mask).pooler_output
            oc_image_features = self.oc_model.image_model(pixel_values=image).pooler_output
            router_features = torch.cat([oc_text_features, oc_image_features], dim=1)
        alpha = self.router(router_features)  # (batch, 2)
        combined_logits = alpha[:, 0:1] * oc_logits + alpha[:, 1:2] * ic_logits
        return combined_logits


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            oc_ids = batch["oc_input_ids"].to(device)
            oc_mask = batch["oc_attention_mask"].to(device)
            ic_ids = batch["ic_input_ids"].to(device)
            ic_mask = batch["ic_attention_mask"].to(device)
            image = batch["image"].to(device)
            labels = batch["label"].to(device)
            with autocast():
                logits = model(oc_ids, oc_mask, ic_ids, ic_mask, image)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=".", help="Directory containing data.csv")
    parser.add_argument("--image-dir", default=None, help="Directory containing images (default: <data-dir>/images_blurred)")
    parser.add_argument("--oc-checkpoint", required=True)
    parser.add_argument("--ic-checkpoint", required=True)
    parser.add_argument("--output", default="c2tox_router.pth")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    args = parser.parse_args()

    image_dir = args.image_dir or os.path.join(args.data_dir, "images_blurred")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df, val_df, test_df = load_split(args.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)

    def make_loader(df, shuffle):
        ds = DualTextDataset(df, image_dir, tokenizer, args.max_length)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=4)

    train_loader = make_loader(train_df, True)
    val_loader = make_loader(val_df, False)
    test_loader = make_loader(test_df, False)

    model = C2Tox(args.oc_checkpoint, args.ic_checkpoint, dropout=args.dropout, device=device).to(device)
    optimizer = torch.optim.AdamW(model.router.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    best_f1, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.oc_model.eval()
        model.ic_model.eval()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"[C2Tox router] epoch {epoch}"):
            oc_ids = batch["oc_input_ids"].to(device)
            oc_mask = batch["oc_attention_mask"].to(device)
            ic_ids = batch["ic_input_ids"].to(device)
            ic_mask = batch["ic_attention_mask"].to(device)
            image = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            with autocast():
                logits = model(oc_ids, oc_mask, ic_ids, ic_mask, image)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.router.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item()

        val_metrics = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: train_loss={running_loss / len(train_loader):.4f}, val={val_metrics}")
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.router.state_dict().items()}

    model.router.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)
    print(f"C2Tox TEST metrics: {test_metrics}")

    torch.save(best_state, args.output)
    print(f"Saved best router checkpoint to {args.output}")


if __name__ == "__main__":
    main()
