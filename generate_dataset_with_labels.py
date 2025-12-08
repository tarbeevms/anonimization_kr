import csv
import random
import re
from typing import Dict, List, Tuple

from faker import Faker

random.seed(42)
fake = Faker("ru_RU")


# --- генераторы отдельных полей ---


def gen_passport() -> str:
    series = random.randint(1000, 9999)
    number = random.randint(100000, 999999)
    return f"{series} {number}"


def gen_inn_person() -> str:
    return "".join(str(random.randint(0, 9)) for _ in range(12))


def gen_snils() -> str:
    nums = [random.randint(0, 9) for _ in range(9)]
    sn = f"{nums[0]}{nums[1]}{nums[2]}-{nums[3]}{nums[4]}{nums[5]}-{nums[6]}{nums[7]}{nums[8]}"
    ctrl = random.randint(0, 99)
    return f"{sn} {ctrl:02d}"


def gen_phone() -> str:
    a = random.randint(900, 999)
    b = random.randint(100, 999)
    c = random.randint(10, 99)
    d = random.randint(10, 99)
    return f"+7-{a}-{b}-{c}-{d}"


def gen_card() -> str:
    blocks = ["".join(str(random.randint(0, 9)) for _ in range(4)) for _ in range(4)]
    return " ".join(blocks)


def gen_url() -> str:
    sub = random.choice(["www", "portal", "lk", "service", "secure"])
    host = random.choice(["company", "bank", "shop", "university", "service"])
    tld = random.choice(["ru", "com", "org"])
    return f"https://{sub}.{host}.{tld}/cabinet"


def gen_marriage_cert() -> str:
    series = random.randint(1, 99)
    number = random.randint(100000, 999999)
    return f"{series:02d}-{number}"


def gen_cadastral() -> str:
    a = random.randint(1, 99)
    b = random.randint(1, 99)
    c = random.randint(100000, 999999)
    d = random.randint(1, 9999)
    return f"{a:02d}:{b:02d}:{c:06d}:{d}"


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

TRAIN_MIN_SENTENCES = 400
TRAIN_TARGET_PER_ATTR = 60
VAL_RATIO = 0.1
TEST_MIN_SENTENCES = 200
TEST_TARGET_PER_ATTR = 50


def build_templates(kind: str, ctx: Dict[str, str]):
    last = ctx["last"]
    first = ctx["first"]
    city = ctx["city"]
    passport = ctx["passport"]
    inn = ctx["inn"]
    snils = ctx["snils"]
    phone = ctx["phone"]
    card = ctx["card"]
    url = ctx["url"]
    marriage_cert = ctx["marriage_cert"]
    cadastral = ctx["cadastral"]

    templates: List[Tuple[str, List[Tuple[str, str]]]] = []

    def add(text: str, entities: List[Tuple[str, str]]):
        templates.append((text, entities))

    if kind == "train":
        add(
            f"Я, {last} {first}, предъявляю паспорт {passport}, проживаю по адресу: г. {city}, ИНН {inn}, телефон {phone}.",
            [("FAMILY", last), ("NAME", first), ("PASSPORT", passport), ("INN", inn), ("PHONE", phone)],
        )
        add(
            f"Клиент {last} {first} сообщил паспорт {passport}, СНИЛС {snils} и номер банковской карты {card} для безопасной оплаты.",
            [("FAMILY", last), ("NAME", first), ("PASSPORT", passport), ("SNILS", snils), ("CARD", card)],
        )
        add(
            f"Для входа на портал {url} требуется указать контактный телефон {phone} пользователя {first} {last}.",
            [("URL", url), ("PHONE", phone), ("NAME", first), ("FAMILY", last)],
        )
        add(
            f"В анкете заказчика {last} {first} записаны ИНН {inn}, СНИЛС {snils} и рабочий телефон {phone}.",
            [("FAMILY", last), ("NAME", first), ("INN", inn), ("SNILS", snils), ("PHONE", phone)],
        )
        add(
            f"На сайте {url} пользователь {first} {last} оставил номер карты {card} и подтвердил связь по телефону {phone}.",
            [("URL", url), ("NAME", first), ("FAMILY", last), ("CARD", card), ("PHONE", phone)],
        )
        add(
            f"В акте о заключении брака указан номер свидетельства о браке {marriage_cert} на имя {last} {first}, СНИЛС {snils}.",
            [("MARRIAGE", marriage_cert), ("FAMILY", last), ("NAME", first), ("SNILS", snils)],
        )
        add(
            f"Заявитель {last} {first}, паспорт {passport}, ИНН {inn}, СНИЛС {snils}, контактный телефон {phone}.",
            [("FAMILY", last), ("NAME", first), ("PASSPORT", passport), ("INN", inn), ("SNILS", snils), ("PHONE", phone)],
        )
        add(
            f"Пользователь {first} {last} оплачивает заказ картой {card}, оставляя телефон {phone} на сайте {url}.",
            [("NAME", first), ("FAMILY", last), ("CARD", card), ("PHONE", phone), ("URL", url)],
        )
        add(
            f"В личном деле сотрудника {last} {first} указаны документы: паспорт {passport}, ИНН {inn}, СНИЛС {snils}, телефон {phone}.",
            [("FAMILY", last), ("NAME", first), ("PASSPORT", passport), ("INN", inn), ("SNILS", snils), ("PHONE", phone)],
        )
        add(
            f"В ЗАГС поданы документы: свидетельство о браке {marriage_cert}, паспорт {passport}, телефон {phone}.",
            [("MARRIAGE", marriage_cert), ("PASSPORT", passport), ("PHONE", phone)],
        )
        add(
            f"На портале госуслуг подтверждён брак по свидетельству {marriage_cert}; контактное лицо {first} {last}, ИНН {inn}, СНИЛС {snils}.",
            [("MARRIAGE", marriage_cert), ("NAME", first), ("FAMILY", last), ("INN", inn), ("SNILS", snils)],
        )
        add(
            f"В выписке ЕГРН указан кадастровый номер объекта {cadastral}; владелец {last} {first}, паспорт {passport}.",
            [("CADASTRAL", cadastral), ("FAMILY", last), ("NAME", first), ("PASSPORT", passport)],
        )
        add(
            f"В договоре аренды указан кадастровый номер {cadastral} и контакт арендатора {first} {last}, телефон {phone}.",
            [("CADASTRAL", cadastral), ("NAME", first), ("FAMILY", last), ("PHONE", phone)],
        )
    else:
        add(
            f"Клиент {first} {last} идентифицирован по паспорту {passport} и ИНН {inn}; связь поддерживается по телефону {phone}.",
            [("NAME", first), ("FAMILY", last), ("PASSPORT", passport), ("INN", inn), ("PHONE", phone)],
        )
        add(
            f"Сотрудник {first} {last} проводит оплату по карте {card} через кабинет {url} и подтверждает заявку звонком на номер {phone}.",
            [("NAME", first), ("FAMILY", last), ("CARD", card), ("URL", url), ("PHONE", phone)],
        )
        add(
            f"В журнале регистраций брак подтверждён свидетельством {marriage_cert}; контактные данные {first} {last} и паспорт {passport}.",
            [("MARRIAGE", marriage_cert), ("NAME", first), ("FAMILY", last), ("PASSPORT", passport)],
        )
        add(
            f"Собственник {last} {first} сообщил кадастровый номер {cadastral} и СНИЛС {snils} для сверки в Росреестре.",
            [("FAMILY", last), ("NAME", first), ("CADASTRAL", cadastral), ("SNILS", snils)],
        )
        add(
            f"Анкета офиса: {first} {last}, ИНН {inn}, номер карты {card}, контактный телефон {phone}, сайт {url}.",
            [("NAME", first), ("FAMILY", last), ("INN", inn), ("CARD", card), ("PHONE", phone), ("URL", url)],
        )
        add(
            f"Представитель {last} {first} предоставил свидетельство о браке {marriage_cert} и СНИЛС {snils} для проверки договора.",
            [("FAMILY", last), ("NAME", first), ("MARRIAGE", marriage_cert), ("SNILS", snils)],
        )
        add(
            f"В обращении указаны паспорт {passport}, кадастровый номер {cadastral}, телефон {phone} и URL {url} владельца {first} {last}.",
            [("PASSPORT", passport), ("CADASTRAL", cadastral), ("PHONE", phone), ("URL", url), ("NAME", first), ("FAMILY", last)],
        )
        add(
            f"Контакт {first} {last} подтвердил перевод на карту {card} и прислал ИНН {inn} для квитанции.",
            [("NAME", first), ("FAMILY", last), ("CARD", card), ("INN", inn)],
        )
        add(
            f"В регистрационной форме указано: {last} {first}, паспорт {passport}, телефон {phone}, URL личного кабинета {url}, СНИЛС {snils}.",
            [("FAMILY", last), ("NAME", first), ("PASSPORT", passport), ("PHONE", phone), ("URL", url), ("SNILS", snils)],
        )
        add(
            f"Агент {first} {last} сообщил о семейном статусе по свидетельству {marriage_cert}, предоставил карту {card} и кадастровый номер {cadastral}.",
            [("NAME", first), ("FAMILY", last), ("MARRIAGE", marriage_cert), ("CARD", card), ("CADASTRAL", cadastral)],
        )
    return templates


def gen_sentence_and_labels(kind: str) -> Tuple[str, Dict[str, int], List[Dict[str, str]]]:
    full_name = fake.name()
    parts = full_name.split()
    if len(parts) >= 2:
        last, first = parts[0], parts[1]
    else:
        last, first = full_name, ""

    ctx = {
        "last": last,
        "first": first,
        "city": fake.city(),
        "passport": gen_passport(),
        "inn": gen_inn_person(),
        "snils": gen_snils(),
        "phone": gen_phone(),
        "card": gen_card(),
        "url": gen_url(),
        "marriage_cert": gen_marriage_cert(),
        "cadastral": gen_cadastral(),
    }

    base_labels = {attr: (1 if attr in ("has_family", "has_name") else 0) for attr in ATTRS}
    templates = build_templates(kind, ctx)
    text, entities = random.choice(templates)

    spans = []

    def add_span(val: str, label: str):
        start = text.find(val)
        if start == -1:
            return
        spans.append({"start": start, "end": start + len(val), "label": label, "text": val})

    labels = base_labels.copy()
    for label, val in entities:
        add_span(val, label)
        if label == "FAMILY":
            labels["has_family"] = 1
        elif label == "NAME":
            labels["has_name"] = 1
        elif label == "PASSPORT":
            labels["has_passport"] = 1
        elif label == "INN":
            labels["has_inn"] = 1
        elif label == "SNILS":
            labels["has_snils"] = 1
        elif label == "CARD":
            labels["has_card"] = 1
        elif label == "PHONE":
            labels["has_phone"] = 1
        elif label == "URL":
            labels["has_url"] = 1
        elif label == "MARRIAGE":
            labels["has_marriage_cert"] = 1
        elif label == "CADASTRAL":
            labels["has_cadastral"] = 1

    return text, labels, spans


def generate_samples(kind: str, min_sentences: int, target_per_attr: int):
    samples: List[Tuple[str, Dict[str, int]]] = []
    spans_all: List[List[Dict[str, str]]] = []
    counts = {attr: 0 for attr in ATTRS}

    def enough() -> bool:
        if len(samples) < min_sentences:
            return False
        for attr in ATTRS:
            if attr in ("has_family", "has_name"):
                continue
            if counts[attr] < target_per_attr:
                return False
        return True

    while not enough():
        text, labels, spans = gen_sentence_and_labels(kind)
        samples.append((text, labels))
        spans_all.append(spans)
        for attr, val in labels.items():
            if val:
                counts[attr] += 1

    return samples, spans_all, counts


def write_bio_sentence(f_bio, sent_id: int, text: str, spans: List[Dict[str, str]]):
    token_spans = list(re.finditer(r"\S+", text))
    char_labels = ["O"] * len(text)
    for span in spans:
        for pos in range(span["start"], min(span["end"], len(char_labels))):
            char_labels[pos] = span["label"]

    for tok in token_spans:
        s, e = tok.start(), tok.end()
        label = "O"
        if s < len(char_labels):
            label = char_labels[s]
        if label == "O":
            bio = "O"
        else:
            if s > 0 and char_labels[s - 1] == label:
                bio = f"I-{label}"
            else:
                bio = f"B-{label}"
        f_bio.write(f"{sent_id}\t{tok.group(0)}\t{bio}\n")
    f_bio.write("\n")


def write_dataset_files(pairs, text_path: str, csv_path: str, bio_path: str):
    with open(text_path, "w", encoding="utf-8") as f_txt, \
         open(csv_path, "w", encoding="utf-8", newline="") as f_csv, \
         open(bio_path, "w", encoding="utf-8") as f_bio:

        writer = csv.writer(f_csv, delimiter=";")
        writer.writerow(["id", "text"] + ATTRS)

        for idx, ((text, labels), spans) in enumerate(pairs):
            f_txt.write(text + "\n")
            writer.writerow([idx, text] + [labels[attr] for attr in ATTRS])
            write_bio_sentence(f_bio, idx, text, spans)


def write_split(prefix: str, pairs):
    write_dataset_files(
        pairs,
        f"{prefix}_texts.txt",
        f"{prefix}_labels.csv",
        f"{prefix}_bio.tsv",
    )


def main():
    train_samples, train_spans, train_counts = generate_samples("train", TRAIN_MIN_SENTENCES, TRAIN_TARGET_PER_ATTR)
    train_pairs = list(zip(train_samples, train_spans))
    rng = random.Random(42)
    rng.shuffle(train_pairs)
    split_idx = max(1, int(len(train_pairs) * (1 - VAL_RATIO)))
    train_split = train_pairs[:split_idx]
    val_split = train_pairs[split_idx:]

    write_split("train", train_split)
    write_split("val", val_split)

    test_samples, test_spans, test_counts = generate_samples("test", TEST_MIN_SENTENCES, TEST_TARGET_PER_ATTR)
    test_pairs = list(zip(test_samples, test_spans))
    write_split("test", test_pairs)
    write_dataset_files(test_pairs, "dataset_200.txt", "gold_labels_200.csv", "gold_bio.tsv")

    print(f"Train samples: {len(train_split)}, validation samples: {len(val_split)}")
    print("Train+val coverage:", {a: train_counts[a] for a in ATTRS})
    print(f"Test samples: {len(test_pairs)}")
    print("Test coverage:", {a: test_counts[a] for a in ATTRS})
    print("Файлы: train_*.txt/csv/tsv, val_*.txt/csv/tsv, test_*.txt/csv/tsv, dataset_200.txt, gold_labels_200.csv, gold_bio.tsv")


if __name__ == "__main__":
    main()
