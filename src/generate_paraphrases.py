# src/generate_paraphrases.py

import argparse, random, csv, pandas as pd

SYNONYMS = {
    "ignore":"disregard",
    "previous":"earlier",
    "instructions":"guidelines",
    "reveal":"show",
    "hidden":"concealed",
    "token":"marker",
    "system prompt":"system message",
    "bypass":"circumvent",
    "rules":"constraints",
    "please":"kindly",
    "now":"immediately"
}

def swap_synonyms(text:str)->str:
    out = text
    for k,v in SYNONYMS.items():
        out = out.replace(k, v).replace(k.title(), v.title())
    return out

def add_polite_wrapper(text:str)->str:
    return "If possible, " + text

def unicode_noise(text:str)->str:
    # insert zero-width space between some letters
    chars = list(text)
    for i in range(0, len(chars), 7):
        chars[i] = chars[i] + "\u200b"
    return "".join(chars)

def paraphrase_row(row, n=2):
    # create n variants focusing on attacks; benign receive fewer/no changes
    txt = row["text"]
    variants = []
    if row["label"]=="attack":
        variants.append(swap_synonyms(txt))
        variants.append(add_polite_wrapper(txt))
        variants.append(unicode_noise(txt))
    else:
        variants.append(txt)  # benign unchanged or lightly modified
    return variants[:n]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="input CSV with id,family,label,text")
    ap.add_argument("--out", required=True, help="output CSV with augmented rows")
    ap.add_argument("--n", type=int, default=2, help="paraphrases per attack row")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    rows = []
    for _, r in df.iterrows():
        rows.append(r.to_dict())
        for ptxt in paraphrase_row(r, n=args.n):
            rid = f"{r['id']}_aug{random.randint(100,999)}"
            rows.append({"id": rid, "family": r["family"], "label": r["label"], "text": ptxt})
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} rows to {args.out}")

if __name__ == "__main__":
    main()
