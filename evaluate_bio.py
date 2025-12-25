import argparse
import csv
import os
import re
from pathlib import Path

import numpy as np
import torch
from seqeval.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForTokenClassification

import kr  # reuse regexes and ner_predictor


LABEL_MAP = {
    "FAMILY": "FAMILY",
    "NAME": "NAME",
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
    """Ленивая загрузка дообученной NER-модели и токенизатора из models/rubert-ner."""
    global _fine_tuned_model, _fine_tuned_tokenizer
    if _fine_tuned_model is not None:
        return _fine_tuned_model, _fine_tuned_tokenizer
    if not os.path.isdir(MODEL_DIR):
        return None, None
    _fine_tuned_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    _fine_tuned_model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    return _fine_tuned_model, _fine_tuned_tokenizer


def predict_spans(text, use_struct=True, use_fallback=True):
    """Возвращает список спанов (start, end, label), объединяя модельные и структурные regex-сущности."""
    spans = []

    model, tok = load_fine_tuned()
    if model and tok:
        # токенизируем по словам, прогоняем модель, мапим wordpiece-метки обратно к словам
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

    if use_struct:
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
    if use_fallback and model is None:
        ents = kr.ner_predictor(text)
        for ent in ents:
            spans.append((ent["start"], ent["end"], ent.get("entity", "NAME")))
    return spans


def build_pred_bio(texts, use_struct=True, use_fallback=True):
    """Строит BIO-последовательности для списка текстов по предсказанным спанам."""
    sentences = []
    for text in texts:
        char_labels = ["O"] * len(text)
        for s, e, lbl in predict_spans(text, use_struct=use_struct, use_fallback=use_fallback):
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
    """Читает эталонный BIO-файл и возвращает список предложений с метками токенов."""
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
    """Считает токен-level Precision/Recall/F1 по BIO-массивам."""
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
    return metrics, {"P": prec, "R": rec, "F1": f1, "TP": tp, "FP": fp, "FN": fn}


def main():
    """CLI: собирает предсказания, сравнивает с gold, печатает токеновые и seqeval метрики."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ner-only",
        action="store_true",
        help="Использовать только модельные предсказания (без регэкспов и fallback pipeline).",
    )
    args = parser.parse_args()

    texts = []
    for line in Path("dataset_200.txt").read_text(encoding="utf-8").splitlines():
        texts.append(line)

    gold_labels = read_gold_bio("gold_bio.tsv")
    pred_labels = build_pred_bio(
        texts,
        use_struct=not args.ner_only,
        use_fallback=not args.ner_only,
    )

    # align by sentence count
    n = min(len(gold_labels), len(pred_labels))
    gold_labels = gold_labels[:n]
    pred_labels = pred_labels[:n]

    metrics, micro = compute_f1(
        [lbl for sent in gold_labels for lbl in sent],
        [lbl for sent in pred_labels for lbl in sent],
    )

    print("\nToken-level BIO table:")
    header = f"{'Label':<12} {'P':>5} {'R':>5} {'F1':>5} {'TP':>6} {'FP':>6} {'FN':>6}"
    print(header)
    for lbl, m in sorted(metrics.items()):
        print(
            f"{lbl:<12} {m['P']:>5.2f} {m['R']:>5.2f} {m['F1']:>5.2f} "
            f"{m['TP']:>6} {m['FP']:>6} {m['FN']:>6}"
        )
    print(
        f"{'Micro':<12} {micro['P']:>5.2f} {micro['R']:>5.2f} {micro['F1']:>5.2f} "
        f"{micro['TP']:>6} {micro['FP']:>6} {micro['FN']:>6}"
    )

    print("\nSeqeval classification report:")
    print(
        classification_report(
            gold_labels,
            pred_labels,
            digits=2,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
