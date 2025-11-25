import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from labels import ID2LABEL, label_is_pii, LABEL2ID
import os


def constrained_decode(logits, label2id):
    """
    Constrained decoding that enforces BIO consistency:
    - I-X can only follow B-X or I-X (not O or B-Y)
    - This reduces spurious entity predictions
    """
    seq_len, num_labels = logits.shape
    pred_ids = []
    prev_label = "O"
    
    for t in range(seq_len):
        scores = logits[t].clone()
        
        # Apply constraints based on previous label
        if prev_label == "O":
            # After O, we can only have O or B-*
            for label_id, label in ID2LABEL.items():
                if label.startswith("I-"):
                    scores[label_id] = float('-inf')
        elif prev_label.startswith("B-"):
            # After B-X, we prefer I-X or can start new entity
            entity_type = prev_label[2:]
            for label_id, label in ID2LABEL.items():
                if label.startswith("I-") and not label.endswith(entity_type):
                    # Penalize I-Y when we just started B-X
                    scores[label_id] -= 5.0
        elif prev_label.startswith("I-"):
            # After I-X, we strongly prefer I-X or O/B-*
            entity_type = prev_label[2:]
            for label_id, label in ID2LABEL.items():
                if label.startswith("I-") and not label.endswith(entity_type):
                    scores[label_id] = float('-inf')
        
        pred_id = scores.argmax().item()
        pred_ids.append(pred_id)
        prev_label = ID2LABEL[pred_id]
    
    return pred_ids


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
    ap.add_argument("--decoding", default="greedy", choices=["greedy", "constrained"], 
                    help="Decoding strategy: greedy (argmax) or constrained (enforce BIO consistency)")
    ap.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir if args.model_name is None else args.model_name)
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
                
                if args.decoding == "constrained":
                    pred_ids = constrained_decode(logits.cpu(), LABEL2ID)
                else:  # greedy
                    pred_ids = logits.argmax(dim=-1).cpu().tolist()

            spans = bio_to_spans(text, offsets, pred_ids)
            ents = []
            for s, e, lab in spans:
                ents.append(
                    {
                        "start": int(s),
                        "end": int(e),
                        "label": lab,
                        "pii": bool(label_is_pii(lab)),
                    }
                )
            results[uid] = ents

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Wrote predictions for {len(results)} utterances to {args.output}")


if __name__ == "__main__":
    main()
