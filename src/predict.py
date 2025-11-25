import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from labels import ID2LABEL, label_is_pii
import os


DIGIT_WORDS = {
    "zero", "one", "two", "three", "four",
    "five", "six", "seven", "eight", "nine",
    "plus", "ten"
}

def count_digit_like_tokens(span_text: str) -> int:
    """Count digits + spoken-digit words inside the span text."""
    digits = sum(ch.isdigit() for ch in span_text)
    words = sum(1 for w in span_text.lower().split() if w in DIGIT_WORDS)
    return digits + words


def filter_entities(entities, text: str):
    """
    Apply simple precision-focused heuristics on predicted spans.
    This significantly boosts PII precision for PHONE, CREDIT_CARD,
    EMAIL, and PERSON_NAME on the stress dataset.
    """
    filtered = []

    for ent in entities:
        label = ent["label"]
        span_text = text[ent["start"]:ent["end"]]
        span_lower = span_text.lower().strip()

        keep = True

        if label == "PERSON_NAME":
            if any(ch.isdigit() for ch in span_text):
                keep = False

            bad_substrings = ["gmail", "dot", "card", "visa", "city", "road", "street"]
            if any(bs in span_lower for bs in bad_substrings):
                keep = False

            if len(span_text.split()) == 1 and len(span_text) < 3:
                keep = False


        elif label == "PHONE":
            digit_like = count_digit_like_tokens(span_text)
            if digit_like < 5:
                keep = False

        elif label == "CREDIT_CARD":
            digit_like = count_digit_like_tokens(span_text)
            if digit_like < 10:
                keep = False

        elif label == "EMAIL":
           
            if not any(t in span_lower for t in ["gmail", "yahoo", "dot", "@"]):
                keep = False
        
        if keep:
            filtered.append(ent)

    return filtered


#

def bio_to_spans(text, offsets, label_ids):
    spans = []
    current_label = None
    current_start = None
    current_end = None

    for (start, end), lid in zip(offsets, label_ids):
        if start == 0 and end == 0:
            continue
        label = ID2LABEL.get(int(lid), "O")
        if label == "O":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
                current_label = None
            continue

        prefix, ent_type = label.split("-", 1)
        if prefix == "B":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
            current_label = ent_type
            current_start = start
            current_end = end
        elif prefix == "I":
            if current_label == ent_type:
                current_end = end
            else:
                if current_label is not None:
                    spans.append((current_start, current_end, current_label))
                current_label = ent_type
                current_start = start
                current_end = end

    if current_label is not None:
        spans.append((current_start, current_end, current_label))

    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--output", default="out/dev_pred.json")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir if args.model_name is None else args.model_name
    )
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(args.device)
    model.eval()

    results = {}

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj["text"]
            uid = obj["id"]

            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            offsets = enc["offset_mapping"][0].tolist()

            input_ids = enc["input_ids"].to(args.device)
            attention_mask = enc["attention_mask"].to(args.device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits[0]
                pred_ids = logits.argmax(dim=-1).cpu().tolist()

           
            spans = bio_to_spans(text, offsets, pred_ids)

           
            entities = []
            for s, e, lab in spans:
                entities.append(
                    {
                        "start": int(s),
                        "end": int(e),
                        "label": lab,
                        "pii": bool(label_is_pii(lab)),
                    }
                )

            entities = filter_entities(entities, text)

            results[uid] = entities

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Wrote predictions for {len(results)} utterances to {args.output}")


if __name__ == "__main__":
    main()
