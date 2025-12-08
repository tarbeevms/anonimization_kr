import csv
import re
from pathlib import Path

import numpy as np

import kr  # reuse regexes and ner_predictor
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import os


LABEL_MAP = {
    "PERSON": "NAME",  # nlp_ner returns PERSON, мы считаем это именем/фамилией
    "PASSPORT": "PASSPORT",
    "INN": "INN",
    "SNILS": "SNILS",
    "CARD": "CARD",
    "PHONE": "PHONE",
    "URL": "URL",
    "MARRIAGE": "MARRIAGE",
    "CADASTRAL": "CADASTRAL",
}


def tokenize_with_spans(text: str):
    return list(re.finditer(r"\S+", text))


MODEL_DIR = "models/rubert-ner"
_fine_tuned_model = None
_fine_tuned_tokenizer = None


def load_fine_tuned():
    global _fine_tuned_model, _fine_tuned_tokenizer
    if _fine_tuned_model is not None:
        return _fine_tuned_model, _fine_tuned_tokenizer
    if not os.path.isdir(MODEL_DIR):
        return None, None
    _fine_tuned_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    _fine_tuned_model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    return _fine_tuned_model, _fine_tuned_tokenizer


def predict_spans(text):
    spans = []

    model, tok = load_fine_tuned()
    if model and tok:
        token_spans = tokenize_with_spans(text)
        words = [m.group(0) for m in token_spans]
        encoding = tok(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        with torch.no_grad():
            logits = model(**encoding).logits
        preds = logits.argmax(-1).squeeze(0).tolist()
        word_ids = encoding.word_ids(batch_index=0)
        word_labels = []
        for wid, pid in zip(word_ids, preds):
            if wid is None:
                continue
            if len(word_labels) <= wid:
                word_labels.append(model.config.id2label[pid])
        for (start, end), lab in zip([(m.start(), m.end()) for m in token_spans], word_labels):
            if lab != "O":
                spans.append((start, end, lab.split("-", 1)[-1] if "-" in lab else lab))

    # regex-based structured override (higher priority)
    struct = kr.extract_structured_pd(text)
    label_map_struct = {
        "urls": "URL",
        "phones": "PHONE",
        "inns": "INN",
        "passports": "PASSPORT",
        "cards": "CARD",
        "snils": "SNILS",
        "marriage": "MARRIAGE",
        "cadastral": "CADASTRAL",
    }
    for key, label in label_map_struct.items():
        for s, e, _ in struct.get(key, []):
            spans.append((s, e, label))

    # fallback NER for persons (if no fine-tuned model)
    if model is None:
        ents = kr.ner_predictor(text)
        for ent in ents:
            spans.append((ent["start"], ent["end"], "NAME"))
    return spans


def build_pred_bio(texts):
    sentences = []
    for text in texts:
        char_labels = ["O"] * len(text)
        for s, e, lbl in predict_spans(text):
            mapped = LABEL_MAP.get(lbl, lbl)
            for pos in range(s, min(e, len(char_labels))):
                char_labels[pos] = mapped

        token_spans = tokenize_with_spans(text)
        sent_labels = []
        for t in token_spans:
            s, e = t.start(), t.end()
            slice_labels = char_labels[s:e] if e <= len(char_labels) else char_labels[s:]
            label = "O"
            for lab in slice_labels:
                if lab != "O":
                    label = lab
                    break
            if label == "O":
                bio = "O"
            else:
                if s > 0 and char_labels[s - 1] == label:
                    bio = f"I-{label}"
                else:
                    bio = f"B-{label}"
            sent_labels.append(bio)
        sentences.append(sent_labels)
    return sentences


def read_gold_bio(path="gold_bio.tsv"):
    sentences = []
    current = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if current:
                    sentences.append(current)
                    current = []
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            current.append(parts[2])
    if current:
        sentences.append(current)
    return sentences


def compute_f1(gold_labels, pred_labels):
    assert len(gold_labels) == len(pred_labels)
    labels = set(l for l in gold_labels + pred_labels if l != "O")
    metrics = {}
    for lbl in labels:
        tp = sum(1 for g, p in zip(gold_labels, pred_labels) if g == lbl and p == lbl)
        fp = sum(1 for g, p in zip(gold_labels, pred_labels) if g != lbl and p == lbl)
        fn = sum(1 for g, p in zip(gold_labels, pred_labels) if g == lbl and p != lbl)
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        metrics[lbl] = {"TP": tp, "FP": fp, "FN": fn, "P": prec, "R": rec, "F1": f1}
    # micro
    tp = sum(m["TP"] for m in metrics.values())
    fp = sum(m["FP"] for m in metrics.values())
    fn = sum(m["FN"] for m in metrics.values())
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return metrics, {"P": prec, "R": rec, "F1": f1}


def main():
    texts = []
    for line in Path("dataset_200.txt").read_text(encoding="utf-8").splitlines():
        texts.append(line)

    gold_labels = read_gold_bio("gold_bio.tsv")
    pred_labels = build_pred_bio(texts)

    # align by sentence count
    n = min(len(gold_labels), len(pred_labels))
    gold_labels = gold_labels[:n]
    pred_labels = pred_labels[:n]

    metrics, micro = compute_f1(
        [lbl for sent in gold_labels for lbl in sent],
        [lbl for sent in pred_labels for lbl in sent],
    )

    print("BIO token-level metrics:")
    for lbl, m in sorted(metrics.items()):
        print(f"{lbl}: P={m['P']:.2f} R={m['R']:.2f} F1={m['F1']:.2f} (TP={m['TP']}, FP={m['FP']}, FN={m['FN']})")
    print(f"Micro: P={micro['P']:.2f} R={micro['R']:.2f} F1={micro['F1']:.2f}")


if __name__ == "__main__":
    main()
