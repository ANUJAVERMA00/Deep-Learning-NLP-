"""Phase 3: Counterfactual Data Augmentation (CDA) for gender.

For every training sentence that contains a gendered word, add a copy with all
gendered words swapped (he<->she, man<->woman, mother<->father, ...).
Labels stay the same. Output: data/augmented/train_cda.csv

Run from the project root:  python -m src.cda
"""
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config

# (male, female) one-to-one pairs
PAIRS = [
    ("he", "she"), ("himself", "herself"), ("man", "woman"), ("men", "women"),
    ("boy", "girl"), ("boys", "girls"), ("father", "mother"), ("fathers", "mothers"),
    ("dad", "mom"), ("dads", "moms"), ("daddy", "mommy"), ("papa", "mama"),
    ("brother", "sister"), ("brothers", "sisters"), ("son", "daughter"),
    ("sons", "daughters"), ("husband", "wife"), ("husbands", "wives"),
    ("boyfriend", "girlfriend"), ("boyfriends", "girlfriends"),
    ("uncle", "aunt"), ("uncles", "aunts"), ("grandfather", "grandmother"),
    ("grandpa", "grandma"),
]
M2F = dict(PAIRS)
F2M = {f: m for m, f in PAIRS}
F2M.update({"mum": "dad", "mums": "dads", "wifes": "husbands"})
# him/his/her/hers are ambiguous and handled separately in swap_gender()
MALE_TERMS = set(M2F) | {"him", "his"}
FEMALE_TERMS = set(F2M) | {"her", "hers"}

# Words that usually follow "her"/"his" when it is a pronoun (him / hers) rather
# than a possessive (his / her). Heuristic: no POS tagger needed.
OBJECT_NEXT = set("""
and or but to for with in on at by from of about that than so because when while as if
too i he she they we you it is was are were has had have will would can could not who
which there then also even just still now again after before since until like into
over under up down out off anymore alone feel feeling cry crying smile laugh go be do
see know think want say talk look leave stay come get happy sad angry mad upset scared
afraid nervous anxious uncomfortable sorry better worse
a an the this these those no some any all one nothing something everything anything
what how why where
""".split())


def _is_object_position(text: str, end: int) -> bool:
    m = re.match(r"\s*([A-Za-z']+)", text[end:])
    if not m:                       # end of text, or punctuation comes next
        return True
    return m.group(1).lower() in OBJECT_NEXT


def _match_case(src: str, dst: str) -> str:
    if src.isupper() and len(src) > 1:
        return dst.upper()
    if src[0].isupper():
        return dst.capitalize()
    return dst


def swap_gender(text: str):
    """Return (swapped_text, number_of_swaps)."""
    out, last, n = [], 0, 0
    for m in re.finditer(r"[A-Za-z]+", text):
        w = m.group(0)
        lw = w.lower()
        new = None
        if lw in M2F:
            new = M2F[lw]
        elif lw in F2M:
            new = F2M[lw]
        elif lw == "him":
            new = "her"
        elif lw == "hers":
            new = "his"
        elif lw == "his":
            new = "hers" if _is_object_position(text, m.end()) else "her"
        elif lw == "her":
            new = "him" if _is_object_position(text, m.end()) else "his"
        if new is not None:
            out.append(text[last:m.start()])
            out.append(_match_case(w, new))
            last = m.end()
            n += 1
    out.append(text[last:])
    return "".join(out), n


def gender_flags(text: str):
    toks = set(re.findall(r"[a-z]+", text.lower()))
    return bool(toks & MALE_TERMS), bool(toks & FEMALE_TERMS)


def corpus_stats(df: pd.DataFrame, label: str) -> pd.DataFrame:
    flags = df["text"].map(gender_flags)
    d = df.assign(male=[f[0] for f in flags], female=[f[1] for f in flags])
    rows = []
    for e in config.EMOTIONS:
        s = d[d["emotion"] == e]
        rows.append({"corpus": label, "emotion": e, "sentences": len(s),
                     "with_male_term": int(s["male"].sum()),
                     "with_female_term": int(s["female"].sum())})
    return pd.DataFrame(rows)


def build():
    df = pd.read_csv(config.DATA_PROCESSED / "train.csv")
    res = [swap_gender(t) for t in df["text"]]
    df["swapped"] = [r[0] for r in res]
    df["n_swaps"] = [r[1] for r in res]

    aug = df[df["n_swaps"] > 0].copy()
    aug["text"] = aug["swapped"]
    aug["is_augmented"] = 1
    orig = df[["text", "label", "emotion"]].copy()
    orig["is_augmented"] = 0
    out = pd.concat([orig, aug[["text", "label", "emotion", "is_augmented"]]],
                    ignore_index=True)

    config.DATA_AUGMENTED.mkdir(parents=True, exist_ok=True)
    path = config.DATA_AUGMENTED / "train_cda.csv"
    out.to_csv(path, index=False)

    print(f"Original training sentences : {len(df)}")
    print(f"Sentences with gendered word: {len(aug)} ({len(aug) / len(df):.1%}) -> swapped copies added")
    print(f"CDA training set size       : {len(out)}  (saved to {path.relative_to(config.ROOT)})")

    print("\nExample swaps:")
    ex = aug.sample(min(8, len(aug)), random_state=0)
    her = aug[aug["text"].str.contains(r"\b(?:her|his)\b", case=False, regex=True)
              | df.loc[aug.index, "text"].str.contains(r"\b(?:her|his)\b", case=False, regex=True)]
    ex = pd.concat([ex, her.sample(min(5, len(her)), random_state=1)]).drop_duplicates()
    for i in ex.index:
        print(f"  {df.loc[i, 'text']}\n   -> {df.loc[i, 'swapped']}\n")

    stats = pd.concat([corpus_stats(df[["text", "emotion"]], "original"),
                       corpus_stats(out[["text", "emotion"]], "cda")])
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    stats.to_csv(config.TABLES_DIR / "cda_corpus_stats.csv", index=False)
    print("Sentences containing male vs female terms, per emotion:")
    print(stats.to_string(index=False))
    print("\nNext: python -m src.train --train_csv data/augmented/train_cda.csv "
          "--run_name cda_seed0 --seed 0")


if __name__ == "__main__":
    build()
