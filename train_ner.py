import os
import random
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)
from seqeval.metrics import precision_score, recall_score, f1_score

MODEL_NAME = "DeepPavlov/rubert-base-cased"
MODEL_DIR = "models/rubert-ner"
SEED = 42
EPOCHS = 6  # reduced for quicker training; adjust if needed
LR = 5e-5
BATCH = 8
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01

random.seed(SEED)
torch.manual_seed(SEED)


def read_bio(path: str = "gold_bio.tsv"):
    sentences = []
    current_tokens = []
    current_labels = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if current_tokens:
                    sentences.append((current_tokens, current_labels))
                    current_tokens, current_labels = [], []
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            _, tok, lab = parts
            current_tokens.append(tok)
            current_labels.append(lab)
    if current_tokens:
        sentences.append((current_tokens, current_labels))
    return sentences


def build_label_list(sentences):
    labels = {"O"}
    for _, labs in sentences:
        labels.update(labs)
    return sorted(labels)


@dataclass
class NERDataset(torch.utils.data.Dataset):
    encodings: Dict[str, Any]
    labels: List[List[int]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def align_labels(tokenizer, tokens_list, labels_list, label2id):
    encodings = tokenizer(
        tokens_list,
        is_split_into_words=True,
        return_offsets_mapping=False,
        padding=True,
        truncation=True,
        max_length=256,
    )
    aligned_labels = []
    for i, labels in enumerate(labels_list):
        word_ids = encodings.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[labels[word_idx]])
            else:
                # subsequent wordpiece inside the same token -> ignore for loss
                label_ids.append(-100)
            previous_word_idx = word_idx
        aligned_labels.append(label_ids)
    return encodings, aligned_labels


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = logits.argmax(-1)
    true_labels = []
    true_preds = []
    for pred, lab in zip(preds, labels):
        sent_preds = []
        sent_labels = []
        for p_id, l_id in zip(pred.tolist(), lab.tolist()):
            if l_id == -100:
                continue
            sent_preds.append(id2label[p_id])
            sent_labels.append(id2label[l_id])
        true_preds.append(sent_preds)
        true_labels.append(sent_labels)
    prec = precision_score(true_labels, true_preds)
    rec = recall_score(true_labels, true_preds)
    f1 = f1_score(true_labels, true_preds)
    return {"precision": prec, "recall": rec, "f1": f1}


if __name__ == "__main__":
    train_sentences = read_bio("train_bio.tsv")
    val_sentences = read_bio("val_bio.tsv")
    if not train_sentences or not val_sentences:
        raise RuntimeError("Не найдены train_bio.tsv или val_bio.tsv. Сначала запустите generate_dataset_with_labels.py")

    label_list = build_label_list(train_sentences + val_sentences)
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    train_tokens = [t for t, _ in train_sentences]
    train_labels = [l for _, l in train_sentences]
    eval_tokens = [t for t, _ in val_sentences]
    eval_labels = [l for _, l in val_sentences]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    enc_train, lab_train = align_labels(tokenizer, train_tokens, train_labels, label2id)
    enc_eval, lab_eval = align_labels(tokenizer, eval_tokens, eval_labels, label2id)

    train_ds = NERDataset(enc_train, lab_train)
    eval_ds = NERDataset(enc_eval, lab_eval)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(label_list), id2label=id2label, label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LR,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH,
        num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p),
    )

    print(f"Using CUDA: {torch.cuda.is_available()}")
    trainer.train()
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print("Model trained and saved to", MODEL_DIR)
