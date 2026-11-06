"""Project 1 - Spam email detection where Ham = 0, Spam = 1
Author: Khai Kieu
Description: This is a machine learning system that classifies emails as Spam or Ham using the
provided labeled training datasets. 
"""
import argparse, json, os, re, time
import numpy as np, pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)

SEED = 42
N_FOLDS = 5

# ---------------------------------------------------------------- data
def load_data(d):
    t1 = pd.read_csv(os.path.join(d, "spam_train1.csv"))
    # In training set 1, a few messages contain unquoted commas, so their text
    # spilled into the extra "Unnamed" columns. Glue the pieces back together.
    extra = [c for c in t1.columns if c.startswith("Unnamed")]
    t1["text"] = t1[["v2"] + extra].fillna("").astype(str).agg(" ".join, axis=1).str.strip()
    t1["y"] = (t1["v1"] == "spam").astype(int)          # ham = 0, spam = 1
    t1["source"] = "sms"
    t2 = pd.read_csv(os.path.join(d, "spam_train2.csv"))
    t2 = t2.rename(columns={"label_num": "y"})           # ham = 0, spam = 1
    t2["source"] = "email"
    train = pd.concat([t1[["text", "y", "source"]], t2[["text", "y", "source"]]],
                      ignore_index=True)
    test = pd.read_csv(os.path.join(d, "spam_test.csv"))
    return train, test, len(t1), len(t2)

def dup_key(s):
    return re.sub(r"\s+", " ", s.lower()).strip()

def deduplicate(train):
    """Drop identical emails (case and whitespace insensitive) before any split.
    A text that appears with both labels is removed entirely."""
    train = train.copy()
    train["key"] = train["text"].map(dup_key)
    info = {"rows_before": len(train)}
    g = train.groupby("key")["y"].nunique()
    conflict = set(g[g > 1].index)
    info["conflicting_label_texts"] = len(conflict)
    train = train[~train["key"].isin(conflict)]
    info["duplicates_removed"] = int(train["key"].duplicated().sum())
    train = train.drop_duplicates("key").reset_index(drop=True)
    info["rows_after"] = len(train)
    info["ham"] = int((train.y == 0).sum()); info["spam"] = int((train.y == 1).sum())
    return train, info

def clean(s):
    s = s.replace("\\r", " ").replace("\r", " ")
    s = re.sub(r"https?://\S+|www\.\S+", " urltoken ", s, flags=re.I)
    s = re.sub(r"\d{5,}", " longnumber ", s)
    return re.sub(r"\s+", " ", s).strip()

# ----------------------------------------------------------- features
def make_word():
    return TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2,
                           sublinear_tf=True, dtype=np.float32)

def make_char():
    return TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(2, 5),
                           min_df=3, sublinear_tf=True, dtype=np.float32)

class Reps:
    """word = word TF-IDF only; wc = word + char TF-IDF. Fit on training part only."""
    def fit(self, texts):
        self.w, self.c = make_word(), make_char()
        self.w.fit(texts); self.c.fit(texts); return self
    def transform(self, texts):
        W = self.w.transform(texts)
        C = self.c.transform(texts)
        return {"word": W, "wc": sp.hstack([W, C]).tocsr()}

# -------------------------------------------------------------- models
# name: (representation, constructor, hyperparameter grid)
MODELS = {
    "TF-IDF + Multinomial NB (baseline)": ("word", lambda a: MultinomialNB(alpha=a), "alpha", [0.01, 0.03, 0.1, 0.3, 1.0]),
    "TF-IDF + Logistic Regression":       ("wc", lambda c: LogisticRegression(C=c, solver="liblinear", random_state=SEED), "C", [10, 30, 100, 300, 1000]),
    "TF-IDF + Linear SVM":                ("wc", lambda c: LinearSVC(C=c, random_state=SEED), "C", [0.3, 1, 3, 10, 30]),
}

def score_of(model, X):
    return model.decision_function(X) if hasattr(model, "decision_function") else model.predict_proba(X)[:, 1]

def metrics(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(y, p),
            "macro_f1": f1_score(y, p, average="macro"),
            "spam_precision": precision_score(y, p, zero_division=0),
            "spam_recall": recall_score(y, p),
            "spam_f1": f1_score(y, p),
            "ham_fpr": fp / (fp + tn)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None); ap.add_argument("--out", default=None)
    ap.add_argument("--last-name", default="Kieu")
    a = ap.parse_args()
    # Paths are resolved relative to this script, so it runs from any working directory.
    here = os.path.dirname(os.path.abspath(__file__))
    need = ["spam_train1.csv", "spam_train2.csv", "spam_test.csv"]
    cands = [a.data] if a.data else [os.path.join(here, "data"), here, os.getcwd()]
    a.data = next((c for c in cands if all(os.path.isfile(os.path.join(c, f)) for f in need)), None)
    if a.data is None:
        raise SystemExit("Could not find spam_train1.csv, spam_train2.csv and spam_test.csv. "
                         "Put them next to spam_project.py (or in a data folder there), or pass --data <folder>.")
    a.out = a.out or os.path.join(here, "results")
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    train, test, n1, n2 = load_data(a.data)
    train, dinfo = deduplicate(train)
    dinfo.update({"train1_rows": n1, "train2_rows": n2, "test_rows": len(test)})
    print("Dedup:", dinfo)
    X_text = train["text"].map(clean).tolist(); y = train["y"].values
    test_text = test["message"].map(clean).tolist()

    # Same stratified folds for every model. Strata = label x source so each
    # fold keeps both the spam rate and the SMS / email mix.
    strata = train["y"].astype(str) + "_" + train["source"]
    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X_text, strata))

    # per fold: fit vectorizers on the training part only, then try every model/setting
    fold_rows, oof = [], {}
    for k, (tr, va) in enumerate(folds):
        reps = Reps().fit([X_text[i] for i in tr])
        Xtr = reps.transform([X_text[i] for i in tr]); Xva = reps.transform([X_text[i] for i in va])
        for name, (rep, ctor, pname, grid) in MODELS.items():
            for g in grid:
                m = ctor(g).fit(Xtr[rep], y[tr])
                s = score_of(m, Xva[rep]); p = m.predict(Xva[rep])
                r = metrics(y[va], p); r.update(model=name, param=g, fold=k)
                fold_rows.append(r); oof[(name, g, k)] = (va, p, s)
        print(f"fold {k+1}/{N_FOLDS} done  ({time.time()-t0:.0f}s)")
    fr = pd.DataFrame(fold_rows)
    cols = ["accuracy", "macro_f1", "spam_precision", "spam_recall", "spam_f1", "ham_fpr"]
    agg = fr.groupby(["model", "param"])[cols].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]; agg = agg.reset_index()
    agg.to_csv(os.path.join(a.out, "cv_all_settings.csv"), index=False)

    # best setting per model: highest mean spam F1, ties broken by lower ham FPR
    best = {}
    for name in MODELS:
        d = agg[agg.model == name].sort_values(["spam_f1_mean", "ham_fpr_mean"], ascending=[False, True])
        best[name] = d.iloc[0]
    summ = pd.DataFrame(best.values()); summ.to_csv(os.path.join(a.out, "cv_model_comparison.csv"), index=False)
    print(summ[["model", "param"] + [f"{c}_mean" for c in cols]].round(4).to_string())

    # final model = best mean spam F1 among the three, tie -> lower FPR
    final_name = summ.sort_values(["spam_f1_mean", "ham_fpr_mean"], ascending=[False, True]).iloc[0]["model"]
    final_param = best[final_name]["param"]
    print("FINAL:", final_name, final_param)

    # out-of-fold predictions of the final model
    pred = np.zeros(len(y), int); score = np.zeros(len(y))
    for k in range(N_FOLDS):
        va, p, s = oof[(final_name, final_param, k)]; pred[va] = p; score[va] = s
    cm = confusion_matrix(y, pred, labels=[0, 1])
    pooled = metrics(y, pred)
    per_source = {src: metrics(y[(train.source == src).values], pred[(train.source == src).values])
                  for src in ["sms", "email"]}
    errs = train.assign(pred=pred, score=score)
    errs = errs[errs.y != errs.pred].copy()
    errs["error"] = np.where(errs.y == 0, "false_positive", "false_negative")
    errs["text"] = errs["text"].str.replace(r"\s+", " ", regex=True)
    errs[["source", "error", "score", "text"]].to_csv(os.path.join(a.out, "validation_errors.csv"), index=False)

    # final fit on ALL labeled data and predict hidden test set
    reps = Reps().fit(X_text)
    Xall = reps.transform(X_text); Xte = reps.transform(test_text)
    rep, ctor, _, _ = MODELS[final_name]
    m = ctor(final_param).fit(Xall[rep], y)
    tp = m.predict(Xte[rep]).astype(int)
    np.savetxt(os.path.join(a.out, f"{a.last_name}_Spam.txt"), tp, fmt="%d")
    is_email = test["message"].str.startswith("Subject:").values

    summary = {"dedup": dinfo, "final_model": final_name, "final_param": final_param,
               "confusion_matrix_[[TN,FP],[FN,TP]]": cm.tolist(), "pooled_oof_metrics": pooled,
               "per_source_oof": per_source, "n_false_positive": int((errs.error == "false_positive").sum()),
               "n_false_negative": int((errs.error == "false_negative").sum()),
               "test_pred_spam": int(tp.sum()), "test_n": int(len(tp)),
               "test_spam_rate": float(tp.mean()),
               "test_spam_rate_email_like": float(tp[is_email].mean()), "test_spam_rate_sms_like": float(tp[~is_email].mean()),
               "seed": SEED, "folds": N_FOLDS, "runtime_s": round(time.time() - t0)}
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))

if __name__ == "__main__":
    main()