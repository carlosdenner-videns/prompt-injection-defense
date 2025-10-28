# src/run_experiment.py
import argparse, os, time, csv, random, json, math
import pandas as pd
from statistics import median
from defenses.signature_proxy import SignatureProxy
from defenses.rules import RegexRules
from defenses.classifier_stub import HeuristicClassifier

# Optional ML-based classifiers
try:
    from defenses.llamaguard_adapter import ProtectAIClassifier
    PROTECTAI_AVAILABLE = True
except ImportError:
    PROTECTAI_AVAILABLE = False

try:
    from defenses.nemo_guardrails_adapter import NeMoGuardrailsSimple
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False

PIPELINE_COMPONENTS = {
    "signature": "SignatureProxy",
    "rules": "RegexRules",
    "classifier": "HeuristicClassifier",
}

# Add ML components if available
if PROTECTAI_AVAILABLE:
    PIPELINE_COMPONENTS["protectai"] = "ProtectAIClassifier"
if NEMO_AVAILABLE:
    PIPELINE_COMPONENTS["nemo"] = "NeMoGuardrailsSimple"

ATTACK_HINTS = [
    "ignore previous instructions","ignore all instructions","jailbreak","DAN","hidden token",
    "reveal your instructions","system prompt","display:none","<!--","follow its instructions"
]

def quick_attack_score(text: str) -> int:
    t = text.lower()
    return sum(1 for h in ATTACK_HINTS if h in t)

def load_components(pipeline, configs):
    comps = []
    for name in pipeline:
        if name == "signature":
            comps.append(("signature", SignatureProxy()))
        elif name == "rules":
            comps.append(("rules", RegexRules(configs["rules"])))
        elif name == "classifier":
            comps.append(("classifier", HeuristicClassifier(threshold=configs.get("threshold", 0.5))))
        elif name == "protectai" and PROTECTAI_AVAILABLE:
            comps.append(("protectai", ProtectAIClassifier(threshold=configs.get("threshold", 0.5))))
        elif name == "nemo" and NEMO_AVAILABLE:
            comps.append(("nemo", NeMoGuardrailsSimple(threshold=configs.get("threshold", 0.5))))
        else:
            available = list(PIPELINE_COMPONENTS.keys())
            raise ValueError(f"Unknown or unavailable component: {name}. Available: {available}")
    return comps

def simulate_llm_response(prompt: str, token: str | None, is_attack_like: bool) -> str:
    # Simple simulator: benign keeps token; attack-like drops token 80% of time
    base = "Ok. Here is the answer."
    if token is None:
        return base
    if not is_attack_like:
        return f"{base} [SECURITY_TOKEN:{token}]"
    # attack-like: sometimes strips the token (model hijacked)
    return base if random.random() < 0.8 else f"{base} [SECURITY_TOKEN:{token}]"

def evaluate_row(row, comps, oracle: bool):
    prompt = row["text"]
    token = None
    decisions = {}
    latencies = {}
    suspicious = False

    # If oracle routing is enabled, we run cheap checks first (rules + signature inject only) and decide whether to call classifier.
    # Order: always inject signature if present in pipeline to modify prompt before "LLM"
    for key, comp in comps:
        if key == "signature":
            prompt, token = comp.inject(prompt)
        elif key == "rules":
            flagged, score, dt = comp.detect(prompt, None)
            decisions["rules_flag"] = flagged; decisions["rules_score"] = score; latencies["rules_ms"] = dt
            suspicious = suspicious or flagged
        elif key == "nemo":
            flagged, score, dt = comp.detect(prompt, None)
            decisions["nemo_flag"] = flagged; decisions["nemo_score"] = score; latencies["nemo_ms"] = dt
            suspicious = suspicious or flagged

    # Oracle routing: only call classifier if suspicious (or if classifier comes before rules in pipeline and oracle disabled)
    clf_flag = None; clf_score = None; clf_dt = 0.0
    if any(k=="classifier" for k,_ in comps):
        if oracle:
            if suspicious:
                # run classifier
                hc = [c for k,c in comps if k=="classifier"][0]
                clf_flag, clf_score, clf_dt = hc.detect(prompt, None)
        else:
            hc = [c for k,c in comps if k=="classifier"][0]
            clf_flag, clf_score, clf_dt = hc.detect(prompt, None)
    if clf_flag is not None:
        decisions["clf_flag"] = clf_flag; decisions["clf_score"] = clf_score; latencies["clf_ms"] = clf_dt

    # Simulate LLM response and apply signature detection
    is_attack_like = quick_attack_score(prompt) > 0 or (row.get("label","benign")=="attack")
    response = simulate_llm_response(prompt, token, is_attack_like)
    if token is not None:
        sig_flag, sig_score, sig_dt = [c for k,c in comps if k=="signature"][0].detect(response, token)
        decisions["sig_flag"] = sig_flag; decisions["sig_score"] = sig_score; latencies["sig_ms"] = sig_dt

    flagged = any([
        decisions.get("sig_flag", False),
        decisions.get("rules_flag", False),
        decisions.get("clf_flag", False),
        decisions.get("nemo_flag", False)
    ])
    return flagged, decisions, latencies, prompt

def summarize(df):
    out = []
    fams = list(sorted(df["family"].unique()))
    for fam in fams:
        sub = df[df["family"] == fam]
        atk = sub[sub["label"]=="attack"]; ben = sub[sub["label"]=="benign"]
        tpr = (atk["flagged"].sum() / max(1, len(atk))) if len(atk)>0 else float("nan")
        fpr = (ben["flagged"].sum() / max(1, len(ben))) if len(ben)>0 else float("nan")
        p50 = sub["lat_ms"].median() if len(sub)>0 else float("nan")
        p95 = sub["lat_ms"].quantile(0.95) if len(sub)>0 else float("nan")
        out.append({"family": fam, "TPR": round(tpr,3), "FPR": round(fpr,3), "p50_ms": round(p50,2), "p95_ms": round(p95,2)})
    # overall
    tpr = (df[(df["label"]=="attack")]["flagged"].sum() / max(1, df[df["label"]=="attack"].shape[0]))
    fpr = (df[(df["label"]=="benign")]["flagged"].sum() / max(1, df[df["label"]=="benign"].shape[0]))
    p50 = df["lat_ms"].median(); p95 = df["lat_ms"].quantile(0.95)
    out.append({"family": "overall", "TPR": round(tpr,3), "FPR": round(fpr,3), "p50_ms": round(p50,2), "p95_ms": round(p95,2)})
    return pd.DataFrame(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pipeline", required=True, help="comma-separated, e.g., signature,rules,classifier")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--rules", default="configs/rules.yml")
    ap.add_argument("--oracle", action="store_true", help="Enable oracle routing: only call classifier if earlier checks are suspicious")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    pipeline = args.pipeline.split(",")
    configs = {"threshold": args.threshold, "rules": args.rules}

    comps = load_components(pipeline, configs)
    df = pd.read_csv(args.data)

    records = []
    for _, row in df.iterrows():
        flagged, decisions, lats, _aug_prompt = evaluate_row(row, comps, oracle=args.oracle)
        lat_total = sum(v for v in lats.values() if isinstance(v,(int,float)))
        rec = {
            "id": row["id"], "family": row["family"], "label": row["label"], "flagged": flagged,
            **{k:v for k,v in decisions.items()}, "lat_ms": lat_total
        }
        records.append(rec)

    pred = pd.DataFrame(records)
    pred.to_csv(os.path.join(args.out, "predictions.csv"), index=False)

    summ = summarize(pred)
    summ.to_csv(os.path.join(args.out, "summary.csv"), index=False)

    print("== Summary ==")
    print(summ.to_string(index=False))

if __name__ == "__main__":
    main()
