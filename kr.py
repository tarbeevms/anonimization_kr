import os
import re
import json
import random
import hashlib
import sys
import warnings
from pathlib import Path
import numpy as np
import torch
import inspect
import collections
import pymorphy2
import urllib.parse
from typing import List, Tuple, Dict, Any, Iterable
from faker import Faker
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from seqeval.metrics import f1_score
from bert_score import score as bert_score
import csv
from collections import defaultdict

# Suppress noisy third-party warnings in CLI output.
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.")
warnings.filterwarnings("ignore", message=r"Baseline not Found for bert-base-multilingual-cased.*")

# pymorphy2 calls the removed inspect.getargspec on Python 3.11+.
# Provide a drop-in replacement returning the 4-tuple ArgSpec the library expects.
if not hasattr(inspect, "getargspec"):
    ArgSpec = collections.namedtuple("ArgSpec", ["args", "varargs", "keywords", "defaults"])

    def _getargspec(func):
        fas = inspect.getfullargspec(func)
        return ArgSpec(fas.args, fas.varargs, fas.varkw, fas.defaults)

    inspect.getargspec = _getargspec


# Модель NER для извлечения сущностей (семантика FAMILY/NAME без PERSON)
MODEL_NAME_NER = "models/rubert-ner"
MAPPING_FILE = "anonymization_map.json"
SEED = 42
DEVICE = 0 if torch.cuda.is_available() else -1

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

fake = Faker("ru_RU")
morph = pymorphy2.MorphAnalyzer()

if os.path.exists(MAPPING_FILE):
    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        MAPPING = json.load(f)
else:
    MAPPING = {}

# # Распаковка исходных текстов
# !unzip texts.zip;
# !rm texts.zip;


def _stable_hash(s: str) -> int:
    """Детерминированный хеш для стабильного выбора замен."""
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:16], 16)


def _choose_from_list_stable(s: str, lst: List[str]) -> str:
    if not lst:
        return s
    return lst[_stable_hash(s) % len(lst)]


URL_REGEX = re.compile(r"https?://[\w\-\.]+(?:\:[0-9]+)?(?:/[^\s]*)?", re.IGNORECASE)
PHONE_REGEX = re.compile(r"(\+7|8)[\s\-()]*?(\d{3})[\s\-()]*?(\d{3})[\s\-()]*?(\d{2})[\s\-()]*?(\d{2})")
INN_REGEX = re.compile(r"\b(\d{10}|\d{12})\b")
PASSPORT_REGEX = re.compile(r"\b\d{2}\s?\d{2}\s?\d{6}\b")
CARD_REGEX = re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b")
SNILS_REGEX = re.compile(r"\b(\d{3})[ -]?(\d{3})[ -]?(\d{3})[ -]?(\d{2})\b")
MARRIAGE_CERT_REGEX = re.compile(r"\b\d{2}-\d{6}\b")
CADASTRAL_REGEX = re.compile(r"\b\d{2}:\d{2}:\d{6,}:\d+\b")
CADASTRAL_REGEX = re.compile(r"\b\d{2}:\d{2}:\d{6,}:?\d*\b")

INN10_WEIGHTS = (2, 4, 10, 3, 5, 9, 4, 6, 8)
INN12_WEIGHTS_N2 = (7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
INN12_WEIGHTS_N1 = (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8)

TOKEN_REGEX = re.compile(r"\S+")
CASE_TAGS = {"nomn", "gent", "datv", "accs", "ablt", "loct", "voct"}
STRUCT_PRIORITIES = {
    "PASSPORT": 1,
    "INN": 2,
    "SNILS": 3,
    "CARD": 4,
    "MARRIAGE": 5,
    "CADASTRAL": 6,
    "PHONE": 7,
    "URL": 8,
}


def _clean_str(value):
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _calc_check_digit(digits: Iterable[int], weights: Iterable[int]) -> int:
    return (sum(d * w for d, w in zip(digits, weights)) % 11) % 10


def anonymize_phone_preserve_format(phone: str) -> str:
    text = _clean_str(phone)
    if text is None:
        return phone
    digits = [ch for ch in text if ch.isdigit()]
    if len(digits) <= 4:
        return phone
    prefix = digits[:4]
    rng = random.Random(_stable_hash(text))
    rest = [str(rng.randint(0, 9)) for _ in range(len(digits) - 4)]
    new_digits = prefix + rest
    digit_iter = iter(new_digits)
    result_chars = []
    for ch in text:
        if ch.isdigit():
            result_chars.append(next(digit_iter))
        else:
            result_chars.append(ch)
    return "".join(result_chars)


def anonymize_inn_preserve_structure(value: str) -> str:
    text = _clean_str(value)
    if text is None:
        return value
    digits = [int(ch) for ch in text if ch.isdigit()]
    length = len(digits)
    if length not in (10, 12):
        return value
    rng = random.Random(_stable_hash(text))
    prefix = digits[:4]
    if length == 10:
        middle = [rng.randint(0, 9) for _ in range(5)]
        base = prefix + middle
        control = _calc_check_digit(base, INN10_WEIGHTS)
        new_digits = base + [control]
    else:
        middle = [rng.randint(0, 9) for _ in range(6)]
        base = prefix + middle
        n2 = _calc_check_digit(base, INN12_WEIGHTS_N2)
        n1 = _calc_check_digit(base + [n2], INN12_WEIGHTS_N1)
        new_digits = base + [n2, n1]
    digit_iter = iter(new_digits)
    result_chars = []
    for ch in text:
        if ch.isdigit():
            result_chars.append(str(next(digit_iter)))
        else:
            result_chars.append(ch)
    return "".join(result_chars)


def luhn_checksum_from_digits(digits: List[int]) -> int:
    digits_rev = digits[::-1]
    s = 0
    for i, d in enumerate(digits_rev):
        if i % 2 == 0:
            s += d
        else:
            dd = d * 2
            if dd > 9:
                dd -= 9
            s += dd
    return s % 10


def luhn_generate_check(digits: List[int]) -> int:
    checksum = luhn_checksum_from_digits(digits + [0])
    return (10 - checksum) % 10


def generate_card_preserve_bin(card: str) -> str:
    digits = re.sub(r"\D", "", card)
    if len(digits) < 6:
        digits = digits.ljust(6, "0")
    bin6 = digits[:6]
    total_len = len(digits) if 13 <= len(digits) <= 19 else 16
    middle_len = total_len - 6 - 1
    rng = random.Random(_stable_hash(card))
    middle = "".join(str(rng.randint(0, 9)) for _ in range(middle_len))
    partial = list(map(int, list(bin6 + middle)))
    check = luhn_generate_check(partial)
    return bin6 + middle + str(check)


def snils_check_sum(digits: List[int]) -> int:
    s = sum(digits[i] * (9 - i) for i in range(9))
    if s < 100:
        return s
    if s in (100, 101):
        return 0
    r = s % 101
    return 0 if r == 100 else r


def generate_snils_preserve_prefix(snils_str: str) -> str:
    digits = re.sub(r"\D", "", snils_str)
    digits = digits.ljust(9, "0")
    prefix = digits[:3]
    rng = random.Random(_stable_hash(snils_str))
    middle = "".join(str(rng.randint(0, 9)) for _ in range(6))
    nine = list(map(int, list(prefix + middle)))
    chk = snils_check_sum(nine)
    return "".join(map(str, nine)) + f"{chk:02d}"


def generate_passport_preserve_series(passport_str: str) -> str:
    digits = re.sub(r"\D", "", passport_str)
    series = digits[:4].ljust(4, "0")
    rng = random.Random(_stable_hash(series))
    number = "".join(str(rng.randint(0, 9)) for _ in range(6))
    return f"{series[:2]} {series[2:]} {number}"


def anonymize_url_preserve_structure(url: str) -> str:
    """
    Сохраняет схему и TLD, обезличивая сам домен и буквенно-цифровые символы
    в пути/параметрах на детерминированный рандом той же длины.
    """
    has_scheme = url.startswith(("http://", "https://"))
    parsed = urllib.parse.urlsplit(url if has_scheme else "http://" + url)
    rng = random.Random(_stable_hash(url))
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    hostname = parsed.hostname or ""
    port = parsed.port
    if "." in hostname:
        idx = hostname.rfind(".")
        host_name = hostname[:idx]
        host_tld = hostname[idx:]
    else:
        host_name = hostname
        host_tld = ""

    anon_host_name = "".join(rng.choice(alphabet) if ch.isalnum() else ch for ch in host_name)
    host_anon = anon_host_name + host_tld
    if port:
        host_anon += f":{port}"

    def _mask_part(part: str) -> str:
        return "".join(rng.choice(alphabet) if ch.isalnum() else ch for ch in part)

    anon_path = _mask_part(parsed.path)
    anon_query = _mask_part(parsed.query)
    anon_fragment = _mask_part(parsed.fragment)

    return urllib.parse.urlunsplit((parsed.scheme or "http", host_anon, anon_path, anon_query, anon_fragment))


def _validate_inn(value: str) -> bool:
    digits = [int(ch) for ch in re.sub(r"\D", "", value)]
    if len(digits) == 10:
        return _calc_check_digit(digits[:9], INN10_WEIGHTS) == digits[9]
    if len(digits) == 12:
        n2 = _calc_check_digit(digits[:10], INN12_WEIGHTS_N2)
        n1 = _calc_check_digit(digits[:11], INN12_WEIGHTS_N1)
        return digits[10] == n2 and digits[11] == n1
    return False


def _validate_snils(value: str) -> bool:
    digits = [int(ch) for ch in re.sub(r"\D", "", value)]
    if len(digits) != 11:
        return False
    base, ctrl = digits[:9], digits[9:]
    return snils_check_sum(base) == (ctrl[0] * 10 + ctrl[1])


def _validate_card(value: str) -> bool:
    digits = [int(ch) for ch in re.sub(r"\D", "", value)]
    if len(digits) < 13 or len(digits) > 19:
        return False
    return luhn_checksum_from_digits(digits) == 0


def _validate_passport(value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    return len(digits) == 10


RUSSIAN_MALE_NAMES = [
    "Андрей", "Иван", "Сергей", "Дмитрий", "Алексей", "Максим", "Егор", "Матвей", "Михаил", "Даниил",
    "Пётр", "Фёдор", "Степан", "Георгий", "Григорий", "Антон", "Кирилл", "Руслан", "Олег", "Юрий",
    "Павел", "Владимир", "Арсений", "Тимофей", "Тихон", "Савва", "Никита", "Лев", "Константин"
]

RUSSIAN_FEMALE_NAMES = [
    "Анна", "Мария", "Екатерина", "Полина", "Ольга", "Ксения", "Софья", "Вероника", "Татьяна", "Елена",
    "Юлия", "Виктория", "Алиса", "Дарья", "Анастасия", "Надежда", "Людмила", "Зоя", "Нина", "Олеся",
    "Инна", "Яна", "Светлана", "Ирина", "Алёна", "Елизавета", "Аврора", "Василиса", "Агата"
]

FIRST_NAMES = RUSSIAN_MALE_NAMES + RUSSIAN_FEMALE_NAMES
LAST_NAMES = [fake.last_name() for _ in range(500)]


def _extract_grammemes(p):
    gset = set()
    for g in ("nomn", "gent", "datv", "accs", "ablt", "loct", "voct", "sing", "plur", "masc", "femn", "neut"):
        if g in p.tag:
            gset.add(g)
    return gset


def morph_inflect_name(original: str, new_base: str) -> str:
    orig_words = original.split()
    new_words = new_base.split()
    res_words = []
    for i, ow in enumerate(orig_words):
        nw = new_words[i] if i < len(new_words) else new_words[-1]
        p_orig = morph.parse(ow)[0]
        p_new = morph.parse(nw)[0]
        grammemes = _extract_grammemes(p_orig)
        if grammemes:
            inflected = p_new.inflect(grammemes)
            if inflected:
                res_words.append(inflected.word)
                continue

        tried = False
        for gender in ("masc", "femn", "neut"):
            if gender in p_orig.tag:
                gm = set(g for g in grammemes if g not in ("masc", "femn", "neut"))
                gm.add(gender)
                inf = p_new.inflect(gm)
                if inf:
                    res_words.append(inf.word)
                    tried = True
                    break
        if tried:
            continue

        gm = set(g for g in grammemes if g not in ("masc", "femn", "neut"))
        if gm:
            inf = p_new.inflect(gm)
            if inf:
                res_words.append(inf.word)
                continue

        if "femn" in p_orig.tag:
            forced = p_new.inflect({"femn"})
            if forced:
                res_words.append(forced.word)
                continue
        if "masc" in p_orig.tag:
            forced = p_new.inflect({"masc"})
            if forced:
                res_words.append(forced.word)
                continue

        res_words.append(p_new.word)
    return " ".join(res_words)


def anonymize_name(original: str, kind: str = "NAME", gender_hint: str = None) -> str:
    gender_key = gender_hint if gender_hint in ("femn", "masc") else "unk"
    key = f"{kind}::{original}::{gender_key}"
    legacy_key = f"{kind}::{original}"
    if key in MAPPING:
        return MAPPING[key]
    if legacy_key in MAPPING:
        return MAPPING[legacy_key]

    parsed = morph.parse(original)[0]
    gender = gender_hint if gender_hint in ("femn", "masc") else None
    if gender is None:
        if "femn" in parsed.tag:
            gender = "femn"
        elif "masc" in parsed.tag:
            gender = "masc"

    if kind == "NAME":
        if gender:
            candidates = [n for n in FIRST_NAMES if gender in morph.parse(n)[0].tag]
        else:
            candidates = list(FIRST_NAMES)
    else:
        if gender == "femn":
            candidates = [s for s in LAST_NAMES if "femn" in morph.parse(s)[0].tag]
        elif gender == "masc":
            candidates = [s for s in LAST_NAMES if "masc" in morph.parse(s)[0].tag]
        else:
            candidates = list(LAST_NAMES)

    if not candidates:
        candidates = FIRST_NAMES if kind == "NAME" else LAST_NAMES

    base = _choose_from_list_stable(original, candidates)

    grammemes = set()
    for g in ["sing", "plur", "nomn", "gent", "datv", "accs", "ablt", "loct", "voct"]:
        if g in parsed.tag:
            grammemes.add(g)

    parsed_new = morph.parse(base)[0]
    inflected = parsed_new.inflect(grammemes)
    anon = inflected.word if inflected else base

    MAPPING[key] = anon
    return anon


def _case_tags_from_parse(p):
    return {g for g in CASE_TAGS if g in p.tag}


def detect_context_gender(text: str, target_span: Tuple[int, int], target_cases: set = None) -> str:
    """
    Ищет ближайшие к target_span токены (±2) и возвращает род (masc/femn),
    если токен помечен в морфоразборе. Сначала пробуем совпадение по падежу.
    """
    tokens = list(TOKEN_REGEX.finditer(text))
    token_idx = None
    for i, m in enumerate(tokens):
        if not (target_span[1] <= m.start() or target_span[0] >= m.end()):
            token_idx = i
            break
    if token_idx is None:
        return None

    same_case_candidates = []
    any_gender_candidates = []

    for offset in (-1, 1, -2, 2):
        idx = token_idx + offset
        if idx < 0 or idx >= len(tokens):
            continue
        word = tokens[idx].group(0)
        parsed = morph.parse(word)[0]
        gender = "femn" if "femn" in parsed.tag else "masc" if "masc" in parsed.tag else None
        if not gender:
            continue
        cases = _case_tags_from_parse(parsed)
        if target_cases and cases and target_cases & cases:
            same_case_candidates.append(gender)
        else:
            any_gender_candidates.append(gender)

    if same_case_candidates:
        return same_case_candidates[0]
    if any_gender_candidates:
        return any_gender_candidates[0]
    return None


def extract_structured_pd(text: str) -> Dict[str, List[Tuple[int, int, Any]]]:
    out = {
        "urls": [], "phones": [], "inns": [], "passports": [],
        "cards": [], "snils": [], "marriage": [], "cadastral": [],
    }
    for m in URL_REGEX.finditer(text):
        out["urls"].append((m.start(), m.end(), m.group(0)))
    for m in PHONE_REGEX.finditer(text):
        out["phones"].append((m.start(), m.end(), m.group(0)))
    for m in INN_REGEX.finditer(text):
        out["inns"].append((m.start(), m.end(), m.group(0)))
    for m in PASSPORT_REGEX.finditer(text):
        out["passports"].append((m.start(), m.end(), m.group(0)))
    for m in CARD_REGEX.finditer(text):
        out["cards"].append((m.start(), m.end(), m.group(0)))
    for m in SNILS_REGEX.finditer(text):
        out["snils"].append((m.start(), m.end(), m.group(0)))
    for m in MARRIAGE_CERT_REGEX.finditer(text):
        out["marriage"].append((m.start(), m.end(), m.group(0)))
    for m in CADASTRAL_REGEX.finditer(text):
        out["cadastral"].append((m.start(), m.end(), m.group(0)))
    return out


tokenizer_ner = AutoTokenizer.from_pretrained(MODEL_NAME_NER, use_fast=True)
model_ner = AutoModelForTokenClassification.from_pretrained(MODEL_NAME_NER)
nlp_ner = pipeline("ner", model=model_ner, tokenizer=tokenizer_ner,
                   aggregation_strategy="simple", device=DEVICE)


def ner_predictor(text: str):
    spans = []

    raw = []
    for ent in nlp_ner(text):
        lab = ent.get("entity_group", ent.get("entity"))
        if lab not in ("FAMILY", "NAME"):
            continue
        s, e = ent["start"], ent["end"]
        if s is None or e is None:
            continue
        frag = ent.get("word", text[s:e])
        # пропускаем заведомо не-именные куски (типа '/')
        if not re.search(r"[A-Za-zА-Яа-яЁё]", frag):
            continue
        raw.append({"start": s, "end": e, "entity": lab})

    raw = sorted(raw, key=lambda x: x["start"])
    merged: List[Dict[str, Any]] = []
    for ent in raw:
        if merged and merged[-1]["entity"] == ent["entity"] and ent["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], ent["end"])
            merged[-1]["text"] = text[merged[-1]["start"]:merged[-1]["end"]]
        elif merged and merged[-1]["entity"] == ent["entity"] and ent["start"] == merged[-1]["end"]:
            merged[-1]["end"] = ent["end"]
            merged[-1]["text"] = text[merged[-1]["start"]:merged[-1]["end"]]
        else:
            merged.append({
                "start": ent["start"],
                "end": ent["end"],
                "entity": ent["entity"],
                "text": text[ent["start"]:ent["end"]],
            })

    def expand_to_word_boundaries(s: int, e: int) -> Tuple[int, int]:
        while s > 0 and re.match(r"[A-Za-zА-Яа-яЁё'-]", text[s - 1]):
            s -= 1
        while e < len(text) and re.match(r"[A-Za-zА-Яа-яЁё'-]", text[e]):
            e += 1
        return s, e

    expanded = []
    for ent in merged:
        s, e = expand_to_word_boundaries(ent["start"], ent["end"])
        expanded.append({
            "start": s,
            "end": e,
            "entity": ent["entity"],
            "text": text[s:e],
        })

    spans.extend(expanded)
    return spans


def anonymize_text(text: str, ner_predictor_func=None) -> Tuple[str, Dict[str, List[Dict]]]:
    details = {"replacements": []}
    working = text

    if ner_predictor_func is None:
        ner_predictor_func = ner_predictor

    def match_case(orig: str, repl: str) -> str:
        if orig.istitle():
            return repl.title()
        if orig.isupper():
            return repl.upper()
        if orig.islower():
            return repl.lower()
        return repl

    person_spans = []
    if ner_predictor_func:
        for ent in ner_predictor_func(working):
            label = ent.get("entity")
            if label not in ("FAMILY", "NAME"):
                continue
            s, e = ent.get("start"), ent.get("end")
            if s is None or e is None:
                continue
            orig_text = ent.get("text", working[s:e])
            cases = _case_tags_from_parse(morph.parse(orig_text)[0])
            gender_hint = detect_context_gender(working, (s, e), cases)
            kind = "SURNAME" if label == "FAMILY" else "NAME"
            anon_token = match_case(orig_text, anonymize_name(orig_text, kind=kind, gender_hint=gender_hint))
            person_spans.append({
                "start": s,
                "end": e,
                "label": label,
                "text": orig_text,
                "replacement": anon_token,
            })

    # применяем замены от конца к началу, чтобы не разъезжались индексы
    for p in sorted(person_spans, key=lambda x: x["start"], reverse=True):
        s, e = p["start"], p["end"]
        working = working[:s] + p["replacement"] + working[e:]

    for p in sorted(person_spans, key=lambda x: x["start"]):
        details["replacements"].append({
            "start": p["start"],
            "end": p["end"],
            "label": p["label"],
            "text": p["text"],
            "replacement": p["replacement"],
        })

    struct = extract_structured_pd(working)
    replacements = []

    for s, e, val in struct["urls"]:
        key = f"URL::{val}"
        if key in MAPPING:
            anon = MAPPING[key]
        else:
            anon = anonymize_url_preserve_structure(val)
            MAPPING[key] = anon
        replacements.append({"start": s, "end": e, "anon": anon, "label": "URL", "text": val})

    for s, e, val in struct["phones"]:
        key = f"PHONE::{val}"
        anon_value = anonymize_phone_preserve_format(val)
        if MAPPING.get(key) != anon_value:
            MAPPING[key] = anon_value
        anon = MAPPING[key]
        replacements.append({"start": s, "end": e, "anon": anon, "label": "PHONE", "text": val})

    for s, e, val in struct["inns"]:
        key = f"INN::{val}"
        anon_value = anonymize_inn_preserve_structure(val)
        if MAPPING.get(key) != anon_value:
            MAPPING[key] = anon_value
        anon = MAPPING[key]
        replacements.append({"start": s, "end": e, "anon": anon, "label": "INN", "text": val})

    inn_spans = {(s, e) for s, e, _ in struct["inns"]}

    for s, e, g in struct["passports"]:
        if (s, e) in inn_spans:
            continue
        raw = re.sub(r"\D", "", g)
        key = f"PASSPORT::{raw}"
        if key in MAPPING:
            anon = MAPPING[key]
        else:
            anon = generate_passport_preserve_series(g)
            MAPPING[key] = anon
        replacements.append({"start": s, "end": e, "anon": anon, "label": "PASSPORT", "text": g})

    for s, e, val in struct["cards"]:
        digits = re.sub(r"\D", "", val)
        key = f"CARD::{digits}"
        if key in MAPPING:
            anon = MAPPING[key]
        else:
            anon = generate_card_preserve_bin(val)
            anon = " ".join([anon[i:i + 4] for i in range(0, len(anon), 4)])
            MAPPING[key] = anon
        replacements.append({"start": s, "end": e, "anon": anon, "label": "CARD", "text": val})

    for s, e, val in struct["snils"]:
        digits = re.sub(r"\D", "", val)
        key = f"SNILS::{digits}"
        if key in MAPPING:
            anon = MAPPING[key]
        else:
            anon = generate_snils_preserve_prefix(val)
            anon = f"{anon[:3]}-{anon[3:6]}-{anon[6:9]} {anon[9:]}"
            MAPPING[key] = anon
        replacements.append({"start": s, "end": e, "anon": anon, "label": "SNILS", "text": val})

    for s, e, val in struct["marriage"]:
        key = f"MARRIAGE::{val}"
        if key in MAPPING:
            anon = MAPPING[key]
        else:
            rng = random.Random(_stable_hash(val))
            series = rng.randint(1, 99)
            number = rng.randint(0, 999999)
            anon = f"{series:02d}-{number:06d}"
            MAPPING[key] = anon
        replacements.append({"start": s, "end": e, "anon": anon, "label": "MARRIAGE", "text": val})

    for s, e, val in struct["cadastral"]:
        key = f"CADASTRAL::{val}"
        if key in MAPPING:
            anon = MAPPING[key]
        else:
            m = re.match(r"(\d{2}):(\d{2}):(\d{6,}):(\d+)", val)
            if m:
                part1, part2, part3, part4 = m.groups()
                rng = random.Random(_stable_hash(val))
                part3_new = str(rng.randint(100000, 999999)).zfill(len(part3))
                part4_new = str(rng.randint(1, 9999)).zfill(len(part4))
                anon = f"{part1}:{part2}:{part3_new}:{part4_new}"
            else:
                anon = "00:00:000000:0000"
            MAPPING[key] = anon
        replacements.append({"start": s, "end": e, "anon": anon, "label": "CADASTRAL", "text": val})

    def _is_valid_span(span: Dict[str, Any]) -> bool:
        lbl = span.get("label")
        txt = span.get("text", "")
        if lbl == "INN":
            return _validate_inn(txt)
        if lbl == "SNILS":
            return _validate_snils(txt)
        if lbl == "CARD":
            return _validate_card(txt)
        if lbl == "PASSPORT":
            return _validate_passport(txt)
        return True

    replacements = [r for r in replacements if _is_valid_span(r)]
    replacements.sort(key=lambda r: (STRUCT_PRIORITIES.get(r["label"], 99), r["start"], -(r["end"] - r["start"])))

    accepted = []
    occupied: List[Tuple[int, int]] = []
    for rep in replacements:
        s, e = rep["start"], rep["end"]
        overlap = any(not (e <= os or s >= oe) for os, oe in occupied)
        if overlap:
            continue
        occupied.append((s, e))
        accepted.append(rep)

    accepted_sorted = sorted(accepted, key=lambda x: x["start"], reverse=True)
    for rep in accepted_sorted:
        s, e = rep["start"], rep["end"]
        r = rep["anon"]
        original_slice = working[s:e]
        working = working[:s] + r + working[e:]
        details["replacements"].append({
            "start": s,
            "end": e,
            "label": rep["label"],
            "text": original_slice,
            "replacement": r
        })

    with open(MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump(MAPPING, f, ensure_ascii=False, indent=2)

    return working, details


# --- Атрибуты для CSV и маппинг из label-ов анонимизатора ---

ATTRS = [
    "has_family",
    "has_name",
    "has_passport",
    "has_inn",
    "has_snils",
    "has_card",
    "has_phone",
    "has_url",
    "has_marriage_cert",
    "has_cadastral",
]

def details_to_flags(details: Dict[str, List[Dict[str, Any]]]) -> Dict[str, int]:
    """
    Превращает details['replacements'] из anonymize_text
    в флаги 0/1 по нужным атрибутам.
    """
    flags = {attr: 0 for attr in ATTRS}

    for rep in details.get("replacements", []):
        label = rep.get("label")

        if label == "FAMILY":
            flags["has_family"] = 1
        if label == "NAME":
            flags["has_name"] = 1

        if label == "PASSPORT":
            flags["has_passport"] = 1

        elif label == "INN":
            flags["has_inn"] = 1

        elif label == "SNILS":
            flags["has_snils"] = 1

        elif label == "CARD":
            flags["has_card"] = 1

        elif label == "PHONE":
            flags["has_phone"] = 1

        elif label == "URL":
            flags["has_url"] = 1

        elif label == "MARRIAGE":
            flags["has_marriage_cert"] = 1

        elif label == "CADASTRAL":
            flags["has_cadastral"] = 1

    return flags


# demo_sample = """
# Макаровой Дарье дали email kaput@example.com, телефон +7 (917) 253-17-22,
# ИНН 7707654654, паспорт 45 23 333222, банковскую карту 4217 4122 5432 1234, СНИЛС 168-321-123 44.
# зайдите http://greenbank.ru/ для деталей.
# """

# anon, details = anonymize_text(demo_sample, ner_predictor_func=ner_predictor)
# print("Original:\n", demo_sample)
# print("\nAnonymized:\n", anon)
# print("\nDetails:\n", json.dumps(details, ensure_ascii=False, indent=2))

# P, R, F1 = bert_score([anon], [demo_sample], lang='ru', rescale_with_baseline=True)
# print(f"\nBERTScore Precision: {P.mean().item():.4f}")
# print(f"BERTScore Recall: {R.mean().item():.4f}")
# print(f"BERTScore F1: {F1.mean().item():.4f}")

# input_dir = "texts"
# output_dir = "texts_anon"
# os.makedirs(output_dir, exist_ok=True)

# for filename in os.listdir(input_dir):
#     if filename.endswith(".txt"):
#         filepath = os.path.join(input_dir, filename)
#         with open(filepath, "r", encoding="utf-8") as f:
#             text = f.read()
#         anon_text, _ = anonymize_text(text)
#         outpath = os.path.join(output_dir, filename)
#         with open(outpath, "w", encoding="utf-8") as f:
#             f.write(anon_text)

# --- Генерация CSV с предсказаниями по датасету из 200 предложений ---

def build_pred_labels(dataset_path: str = "dataset_200.txt",
                      output_path: str = "pred_labels_200.csv") -> None:
    """
    Читает по строкам dataset_200.txt и для каждого предложения
    запускает anonymize_text, после чего строит строку CSV:
    id; text; has_family; has_name; ...; has_cadastral
    """
    with open(dataset_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8", newline="") as f_out:

        writer = csv.writer(f_out, delimiter=";")
        # заголовок
        writer.writerow(["id", "text"] + ATTRS)

        for idx, line in enumerate(f_in):
            text = line.rstrip("\n")
            if not text:
                continue

            # используем существующий распознаватель:
            # anonymize_text возвращает details["replacements"] с label-ами
            _, details = anonymize_text(text, ner_predictor_func=ner_predictor)

            flags = details_to_flags(details)
            row = [idx, text] + [flags[a] for a in ATTRS]
            writer.writerow(row)

    print(f"pred_labels_200.csv записан в файл {output_path}")


import os
from bert_score import score as bert_score

SOURCE_TEXTS_FILE = "dataset_200.txt"

original_texts = []
with open(SOURCE_TEXTS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            original_texts.append(line)

synonymized_texts = []
masked_variants = []
pd_spans_original = []
pd_spans_synonym = []
pd_spans_masked = []
pd_spans_by_label_orig = defaultdict(list)
pd_spans_by_label_syn = defaultdict(list)
pd_spans_by_label_mask = defaultdict(list)

def mask_text(text):
    text = re.sub(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "[REDACTED]", text)
    text = re.sub(r"\+7[\s\d\-\(\)]{10,}", "[REDACTED]", text)
    text = re.sub(r"https?://\S+", "[REDACTED]", text)
    text = re.sub(r"\b\d{10,16}\b", "[REDACTED]", text)  # карты, ИНН
    text = re.sub(r"\b\d{2}\s\d{2}\s\d{6}\b", "[REDACTED]", text)  # паспорт
    text = re.sub(r"\b\d{3}-\d{3}-\d{3} \d{2}\b", "[REDACTED]", text)  # СНИЛС
    return text

def run_pipeline():
    for text in original_texts:
        anon_text, details = anonymize_text(text, ner_predictor_func=ner_predictor)
        synonymized_texts.append(anon_text)
        masked_variants.append(mask_text(text))
        # собираем спаны ПДн для точечной оценки BERTScore
        for rep in details.get("replacements", []):
            s = rep.get("start")
            e = rep.get("end")
            lbl = rep.get("label")
            if s is None or e is None:
                continue
            original_fragment = text[s:e]
            replacement_fragment = rep.get("replacement", original_fragment)
            if not original_fragment.strip():
                continue
            pd_spans_original.append(original_fragment)
            pd_spans_synonym.append(replacement_fragment if replacement_fragment.strip() else "[REDACTED]")
            pd_spans_masked.append("[REDACTED]")
            if lbl:
                pd_spans_by_label_orig[lbl].append(original_fragment)
                pd_spans_by_label_syn[lbl].append(replacement_fragment if replacement_fragment.strip() else "[REDACTED]")
                pd_spans_by_label_mask[lbl].append("[REDACTED]")

    P_syn, R_syn, F1_syn = bert_score(
        original_texts,
        synonymized_texts,
        lang="ru",
        rescale_with_baseline=False,  # отключаем поиск baseline, чтобы убрать предупреждения
    )
    P_mask, R_mask, F1_mask = bert_score(
        original_texts,
        masked_variants,
        lang="ru",
        rescale_with_baseline=False,
    )

    print(f"Синонимическая замена BERTScore F1: {F1_syn.mean().item():.4f}")
    print(f"Маскирование BERTScore F1: {F1_mask.mean().item():.4f}")

    # BERTScore только по спанам ПДн, чтобы видеть влияние замен
    if pd_spans_original:
        _, _, F1_syn_spans = bert_score(
            pd_spans_original,
            pd_spans_synonym,
            lang="ru",
            rescale_with_baseline=False,
        )
        _, _, F1_mask_spans = bert_score(
            pd_spans_original,
            pd_spans_masked,
            lang="ru",
            rescale_with_baseline=False,
        )
        print(f"Спаны ПДн (синонимы) BERTScore F1: {F1_syn_spans.mean().item():.4f}")
        print(f"Спаны ПДн (маскирование) BERTScore F1: {F1_mask_spans.mean().item():.4f}")
        # per-label spans
        print("BERTScore по спанам ПДн по меткам:")
        header = f"{'Label':<12} {'F1_syn':>8} {'F1_mask':>8}"
        print(header)
        all_labels = sorted(set(pd_spans_by_label_orig.keys()) | set(pd_spans_by_label_syn.keys()))
        for lbl in all_labels:
            orig_lst = pd_spans_by_label_orig.get(lbl, [])
            syn_lst = pd_spans_by_label_syn.get(lbl, [])
            mask_lst = pd_spans_by_label_mask.get(lbl, [])
            if not orig_lst or not syn_lst or not mask_lst:
                continue
            _, _, f1_syn_lbl = bert_score(
                orig_lst,
                syn_lst,
                lang="ru",
                rescale_with_baseline=False,
            )
            _, _, f1_mask_lbl = bert_score(
                orig_lst,
                mask_lst,
                lang="ru",
                rescale_with_baseline=False,
            )
            print(f"{lbl:<12} {f1_syn_lbl.mean().item():>8.4f} {f1_mask_lbl.mean().item():>8.4f}")

    if F1_syn.mean() > F1_mask.mean():
        print("Синонимическая замена лучше сохраняет смысл текста.")
    else:
        print("Маскирование лучше сохраняет смысл текста или примерно одинаково.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # если передан файл, печатаем спаны и обезличенный текст
        input_text = Path(sys.argv[1]).read_text(encoding="utf-8")
        anon, details = anonymize_text(input_text, ner_predictor_func=ner_predictor)
        print("__ИСХОДНЫЙ ТЕКСТ__")
        print(input_text)
        print("__СПАНЫ ПДН__")
        for rep in details.get("replacements", []):
            lbl = rep.get("label")
            s = rep.get("start")
            e = rep.get("end")
            orig = rep.get("text", input_text[s:e])
            print(f"{lbl} -> '{orig.strip()}' [{s}:{e}]")
        print("__ОБЕЗЛИЧЕННЫЙ__")
        print(anon)
    else:
        run_pipeline()
