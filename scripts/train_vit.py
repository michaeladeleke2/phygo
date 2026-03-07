#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
)
import evaluate


# ----------------------------
# Dataset that matches your structure:
# data/train/spectrogram/<subject>/<gesture>/*.png
# ----------------------------
class SpectrogramBySubjectDataset(Dataset):
    def __init__(self, root_dir: str, subjects: list[str], transform=None, is_train=True, train_split=0.8, seed=42):
        self.root_dir = Path(root_dir)
        self.subjects = subjects
        self.transform = transform
        self.is_train = is_train
        self.train_split = train_split
        self.seed = seed

        # gestures/classes discovered globally from folder names
        self.label_names = self._discover_gestures()
        self.label2id = {name: i for i, name in enumerate(self.label_names)}
        self.id2label = {i: name for name, i in self.label2id.items()}

        self.items = []  # list of (img_path, label_id)

        for subj in subjects:
            subj_dir = self.root_dir / subj
            if not subj_dir.exists():
                continue
            for gesture_dir in subj_dir.iterdir():
                if not gesture_dir.is_dir():
                    continue
                gesture = gesture_dir.name
                if gesture not in self.label2id:
                    continue
                for img_path in gesture_dir.glob("*.png"):
                    self.items.append((img_path, self.label2id[gesture]))

        # If single subject mode (train and val subjects are the same), do random split
        if len(self.subjects) == 1 and len(set(self.subjects)) == 1:
            self._apply_random_split()
        
        print(f"[Dataset] subjects={len(subjects)} | images={len(self.items)} | classes={len(self.label_names)} | split={'train' if is_train else 'val'}")
        print(f"[Dataset] classes: {self.label_names}")
    
    def _apply_random_split(self):
        """Apply random 80/20 split when using single subject"""
        rng = np.random.default_rng(self.seed)
        indices = np.arange(len(self.items))
        rng.shuffle(indices)
        
        split_point = int(len(self.items) * self.train_split)
        
        if self.is_train:
            keep_indices = indices[:split_point]
        else:
            keep_indices = indices[split_point:]
        
        self.items = [self.items[i] for i in keep_indices]

    def _discover_gestures(self):
        gestures = set()
        for subj_dir in self.root_dir.iterdir():
            if not subj_dir.is_dir():
                continue
            for gesture_dir in subj_dir.iterdir():
                if gesture_dir.is_dir():
                    gestures.add(gesture_dir.name)
        return sorted(list(gestures))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, label = self.items[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {"pixel_values": image, "labels": label}


def collate_fn(batch):
    pixel_values = torch.stack([x["pixel_values"] for x in batch])
    labels = torch.tensor([x["labels"] for x in batch], dtype=torch.long)
    return {"pixel_values": pixel_values, "labels": labels}


# ----------------------------
# Metrics: accuracy + macro F1
# ----------------------------
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
    return {"accuracy": acc, "f1_macro": f1}


# ----------------------------
# Subject split helper
# ----------------------------
def split_subjects(root_dir: str, val_subjects: int = 1, seed: int = 42):
    """Modified to handle single subject case and exclude temporary folders"""
    root = Path(root_dir)
    
    # Exclude temporary prediction folders
    EXCLUDE_SUBJECTS = ['predict_temp', 'temp', 'test']
    
    subjects = sorted([
        d.name for d in root.iterdir() 
        if d.is_dir() and d.name not in EXCLUDE_SUBJECTS
    ])
    
    if len(subjects) == 0:
        raise ValueError(f"No valid subjects found in {root_dir}. Check your data folder structure.")
    
    if len(subjects) < 2:
        # Single subject - use the same subject for train and val
        # The dataset class will handle the random split
        print(f"⚠️ Only 1 subject found ({subjects[0]}). Using random 80/20 split instead.")
        return subjects, subjects
    
    # Multiple subjects - do proper subject-wise split
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    val_subjects = max(1, min(val_subjects, len(subjects) - 1))
    val = subjects[:val_subjects]
    train = subjects[val_subjects:]
    return train, val


def build_transforms(processor, train: bool):
    # Spectrogram-safe augments:
    # - mild crop/resize
    # - mild brightness/contrast
    # - mild affine
    # Avoid left-right flip unless you KNOW time reversal is valid.
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop((224, 224), scale=(0.85, 1.0)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05),
            transforms.RandomAffine(degrees=3, translate=(0.02, 0.02), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
        ])


def train_model(
    data_dir="data/train/spectrogram",
    output_dir="data/model_hf",
    model_name="google/vit-base-patch16-224",
    epochs=15,
    batch_size=8,
    lr=2e-5,
    val_subjects=1,
    seed=42,
    fp16=False,
):
    data_dir = str(Path(data_dir))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)

    # Check for local bundled model FIRST
    repo_root = Path(__file__).parent.parent
    local_model_path = repo_root / "models" / "vit-base-patch16-224"
    
    if local_model_path.exists():
        print("\n[1/5] Using bundled model (offline mode)...")
        print(f"📦 Loading from: {local_model_path}")
        print("   ✅ No internet required!")
        model_to_load = str(local_model_path)
    else:
        print("\n[1/5] Bundled model not found...")
        print(f"📥 Will download from HuggingFace: {model_name}")
        print("   ⚠️  Requires internet connection")
        model_to_load = model_name
    
    processor = AutoImageProcessor.from_pretrained(model_to_load)

    print("\n[2/5] Splitting subjects (no leakage)...")
    train_subj, val_subj = split_subjects(data_dir, val_subjects=val_subjects, seed=seed)
    print(f"Train subjects: {train_subj}")
    print(f"Val subjects:   {val_subj}")

    print("\n[3/5] Loading datasets...")
    train_ds = SpectrogramBySubjectDataset(
        data_dir, train_subj, transform=build_transforms(processor, train=True),
        is_train=True, seed=seed
    )
    val_ds = SpectrogramBySubjectDataset(
        data_dir, val_subj, transform=build_transforms(processor, train=False),
        is_train=False, seed=seed
    )
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("Train/val dataset is empty. Check your folder structure and images.")

    num_labels = len(train_ds.label_names)

    print("\n[4/5] Loading pretrained model...")
    model = AutoModelForImageClassification.from_pretrained(
        model_to_load,
        num_labels=num_labels,
        id2label=train_ds.id2label,
        label2id=train_ds.label2id,
        ignore_mismatched_sizes=True,
    )

    # NOTE: ViT often benefits from slightly higher LR on the head.
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir=str(output_dir / "logs"),
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        remove_unused_columns=False,
        report_to="none",
        fp16=fp16,
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    print("\n[5/5] Training...")
    trainer.train()
    metrics = trainer.evaluate()
    print("\nFinal metrics:", metrics)

    # Save final model + processor
    final_dir = output_dir / "model"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))

    # Save labels in your existing format too
    labels_txt = output_dir / "labels.txt"
    with labels_txt.open("w") as f:
        for i, name in enumerate(train_ds.label_names):
            f.write(f"{i} {name}\n")

    labels_json = output_dir / "labels.json"
    with labels_json.open("w") as f:
        json.dump({"id2label": train_ds.id2label, "label2id": train_ds.label2id}, f, indent=2)

    print("\n✅ Saved:")
    print(f"  - HF model:  {final_dir}")
    print(f"  - labels:    {labels_txt}")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/train/spectrogram")
    ap.add_argument("--output-dir", default="data/model_hf")
    ap.add_argument("--model", default="google/vit-base-patch16-224")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--val-subjects", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    train_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_subjects=args.val_subjects,
        seed=args.seed,
        fp16=args.fp16,
    )


if __name__ == "__main__":
    main()