import sys
from pathlib import Path

import kr


def print_spans_and_anon(text: str):
    anon, details = kr.anonymize_text(text, ner_predictor_func=kr.ner_predictor)
    print("__ИСХОДНЫЙ ТЕКСТ__")
    print(text)
    print("__СПАНЫ ПДН__")
    for rep in details.get("replacements", []):
        lbl = rep.get("label")
        s = rep.get("start")
        e = rep.get("end")
        orig = rep.get("text", text[s:e])
        print(f"{lbl} -> '{orig.strip()}' [{s}:{e}]")
    print("__ОБЕЗЛИЧЕННЫЙ__")
    print(anon)


def main():
    if len(sys.argv) > 1:
        input_text = Path(sys.argv[1]).read_text(encoding="utf-8")
    else:
        input_text = sys.stdin.read()
    print_spans_and_anon(input_text)


if __name__ == "__main__":
    main()
