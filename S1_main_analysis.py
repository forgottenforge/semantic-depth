#!/usr/bin/env python3
"""
Semantic Countdown Hypothesis — Hamilton et al. Validation
============================================================
Uses COHA diachronic word embeddings (Hamilton et al. 2016)
with Procrustes alignment to measure REAL semantic change.

Combines:
  - Hand-coded etymological depth (ed) from v2 dataset
  - Corpus-based semantic change from COHA embeddings (1850 vs 2000)
  - Word frequency from wordfreq package

This is the real test: does ed predict MEASURED semantic change?
"""

import pickle
import numpy as np
from scipy import stats
from wordfreq import zipf_frequency
import warnings
warnings.filterwarnings('ignore')

# ═════════════════════════════════════════════════════════════
# 1. Load and align embeddings
# ═════════════════════════════════════════════════════════════

def load_decade(year, base='sgns'):
    with open(f'{base}/{year}-vocab.pkl', 'rb') as f:
        vocab = pickle.load(f, encoding='latin1')
    W = np.load(f'{base}/{year}-w.npy')
    idx = {w: i for i, w in enumerate(vocab)}
    return vocab, W, idx

print("Loading embeddings...")
_, W_early, idx_early = load_decade(1850)
_, W_late, idx_late = load_decade(2000)

# Shared vocabulary
shared = set(idx_early.keys()) & set(idx_late.keys())
print(f"Shared vocabulary: {len(shared)}")

# Procrustes alignment using top-5000 frequent words
shared_by_freq = sorted(shared, key=lambda w: idx_late[w])
anchors = shared_by_freq[:5000]

M1 = W_early[[idx_early[w] for w in anchors]]
M2 = W_late[[idx_late[w] for w in anchors]]
M1 = M1 / (np.linalg.norm(M1, axis=1, keepdims=True) + 1e-10)
M2 = M2 / (np.linalg.norm(M2, axis=1, keepdims=True) + 1e-10)

U, S, Vt = np.linalg.svd(M2.T @ M1)
R = U @ Vt
W_late_aligned = W_late @ R
print("Procrustes alignment done.")

def semantic_change(word):
    w = word.lower()
    if w not in idx_early or w not in idx_late:
        return None
    v1 = W_early[idx_early[w]]
    v2 = W_late_aligned[idx_late[w]]
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return None
    return 1.0 - np.dot(v1, v2) / (n1 * n2)

# ═════════════════════════════════════════════════════════════
# 2. Hand-coded etymological depth dataset (from v2)
# ═════════════════════════════════════════════════════════════

hand_coded = [
    # (word, ed, hand_change)
    # ed=1: Primwörter
    ("I",1,0), ("you",1,0), ("we",1,0), ("he",1,0), ("she",1,0),
    ("it",1,0), ("me",1,0), ("us",1,0), ("them",1,0),
    ("this",1,0), ("that",1,0), ("here",1,0), ("there",1,0),
    ("now",1,0), ("then",1,0), ("who",1,0), ("what",1,0),
    ("be",1,0), ("do",1,0), ("go",1,0), ("come",1,0), ("see",1,0),
    ("know",1,0), ("say",1,0), ("get",1,0), ("make",1,0),
    ("take",1,0), ("give",1,0), ("have",1,0),
    ("eat",1,0), ("drink",1,0), ("sleep",1,0), ("die",1,0),
    ("sit",1,0), ("stand",1,0), ("lie",1,0), ("fall",1,0),
    ("run",1,0), ("walk",1,0), ("hear",1,0), ("feel",1,0),
    ("cut",1,0), ("bite",1,0), ("blow",1,0), ("burn",1,0),
    ("pull",1,0), ("push",1,0), ("swim",1,0), ("fly",1,0), ("hold",1,0),
    ("one",1,0), ("two",1,0), ("three",1,0), ("ten",1,0),
    ("eye",1,0), ("ear",1,0), ("mouth",1,0), ("tooth",1,0),
    ("tongue",1,0), ("foot",1,0), ("knee",1,0), ("heart",1,1),
    ("bone",1,0), ("blood",1,0), ("skin",1,0), ("nail",1,1),
    ("sun",1,0), ("moon",1,0), ("star",1,1), ("water",1,0),
    ("fire",1,1), ("earth",1,1), ("stone",1,0), ("tree",1,0),
    ("leaf",1,0), ("seed",1,1), ("root",1,1), ("rain",1,0),
    ("snow",1,0), ("wind",1,0), ("sand",1,0), ("salt",1,0), ("ash",1,0),
    ("dog",1,0), ("fish",1,0), ("worm",1,0), ("mouse",1,0),
    ("new",1,0), ("old",1,0), ("good",1,0), ("big",1,0), ("long",1,0),
    ("small",1,0), ("hot",1,0), ("cold",1,0), ("wet",1,0), ("dry",1,0),
    ("dead",1,0), ("red",1,0), ("black",1,0), ("white",1,0),
    ("name",1,0), ("night",1,0), ("day",1,0), ("path",1,0), ("road",1,0),
    ("hand",1,1), ("nose",1,1), ("head",1,1), ("back",1,1),
    ("full",1,0), ("all",1,0), ("many",1,0), ("not",1,0),
    ("in",1,0), ("with",1,0),

    # ed=2: Simple derivations
    ("husband",2,2), ("woman",2,1), ("lord",2,2), ("lady",2,2),
    ("barn",2,1), ("world",2,1), ("orchard",2,1),
    ("deer",2,2), ("hound",2,2), ("fowl",2,2), ("meat",2,2),
    ("starve",2,2), ("thing",2,2), ("tide",2,1), ("stool",2,2),
    ("teacher",2,0), ("quickly",2,0), ("undo",2,0), ("sunrise",2,0),
    ("forget",2,0), ("begin",2,0), ("become",2,0), ("behind",2,0),
    ("between",2,0), ("maybe",2,0), ("inside",2,0), ("outside",2,0),
    ("kingdom",2,0), ("freedom",2,0), ("childhood",2,0), ("friendship",2,0),
    ("household",2,0), ("wisdom",2,1), ("witness",2,1), ("worship",2,1),
    ("sheriff",2,2), ("steward",2,1),
    ("army",2,0), ("court",2,1), ("state",2,1), ("power",2,0),
    ("country",2,0), ("city",2,0), ("place",2,0), ("point",2,1),
    ("matter",2,1), ("number",2,0), ("order",2,0), ("service",2,1),
    ("war",2,0), ("age",2,0), ("story",2,0), ("office",2,1),
    ("cause",2,0), ("reason",2,0),
    ("skill",2,1), ("wrong",2,1), ("window",2,0), ("anger",2,1),
    ("ugly",2,0),

    # ed=3: Compound derivations
    ("beautiful",3,0), ("wonderful",3,1), ("powerful",3,0),
    ("dangerous",3,0), ("government",3,1), ("agreement",3,0),
    ("movement",3,0), ("impossible",3,0), ("unhappy",3,0),
    ("disappear",3,0), ("discover",3,1), ("breakfast",3,1),
    ("understand",3,2), ("nightmare",3,2), ("holiday",3,2),
    ("goodbye",3,2), ("gossip",3,2), ("bully",3,3),
    ("silly",3,3), ("nice",3,3), ("pretty",3,2), ("awful",3,3),
    ("awesome",3,2), ("terrible",3,3), ("terrific",3,3),
    ("naughty",3,3), ("shrewd",3,3), ("fond",3,3), ("brave",3,2),
    ("crafty",3,2), ("cunning",3,2), ("sad",3,2), ("glad",3,1),
    ("fast",3,2), ("cheap",3,2), ("clue",3,2), ("treacle",3,3),
    ("moot",3,3),

    # ed=4: Multi-layer
    ("unfortunately",4,0), ("uncomfortable",4,0),
    ("communication",4,1), ("international",4,0),
    ("entertainment",4,1), ("philosophical",4,1),
    ("understanding",4,2), ("disagreement",4,0),
    ("independence",4,0), ("environmental",4,0),
    ("responsibility",4,0), ("organization",4,1),
    ("representative",4,0), ("establishment",4,1),
    ("particularly",4,0),
    ("manufacture",4,2), ("enthusiasm",4,2), ("candidate",4,2),
    ("salary",4,2), ("calculate",4,2), ("secretary",4,2),
    ("magazine",4,2), ("algorithm",4,2), ("algebra",4,1),
    ("assassin",4,2), ("admiral",4,2), ("alcohol",4,2),
    ("cardinal",4,2), ("companion",4,1), ("quarantine",4,2),
    ("muscle",4,2), ("janitor",4,2), ("intern",4,2),
    ("minister",4,2), ("sinister",4,2),

    # ed=5: Maximally derived
    ("serendipity",5,1), ("lieutenant",5,2), ("mortgage",5,2),
    ("preposterous",5,2), ("egregious",5,3), ("pandemonium",5,2),
    ("sycophant",5,3), ("disaster",5,1), ("trivial",5,2),
    ("decimation",5,3), ("dilapidated",5,2), ("quintessential",5,2),
    ("miscreant",5,3), ("glamour",5,3), ("juggernaut",5,2),
    ("pedigree",5,2), ("alchemy",5,1), ("vermicelli",5,1),
    ("peninsula",5,0), ("inaugurate",5,2), ("investigate",5,2),
    ("exorbitant",5,2), ("extravagant",5,2),
]

# ═════════════════════════════════════════════════════════════
# 3. Match with Hamilton data
# ═════════════════════════════════════════════════════════════

print("\nMatching words with embedding data...")

results = []
for word, ed, hand_chg in hand_coded:
    sc = semantic_change(word)
    if sc is not None:
        freq = zipf_frequency(word.lower(), 'en')
        results.append({
            'word': word, 'ed': ed, 'hand_chg': hand_chg,
            'hamilton_chg': sc, 'freq': freq
        })

N = len(results)
print(f"Matched: {N} / {len(hand_coded)} words")

words_r = [r['word'] for r in results]
ed_r = np.array([r['ed'] for r in results], dtype=float)
hchg_r = np.array([r['hand_chg'] for r in results], dtype=float)
hamilton_r = np.array([r['hamilton_chg'] for r in results], dtype=float)
freq_r = np.array([r['freq'] for r in results], dtype=float)

# ═════════════════════════════════════════════════════════════
# 4. Results
# ═════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("SEMANTIC COUNTDOWN — HAMILTON VALIDATION")
print("=" * 70)

# 4a. Does hand-coded change correlate with Hamilton change?
print("\n" + "─" * 70)
print("VALIDATION: Hand-coded vs Hamilton semantic change")
print("─" * 70)
r_val, p_val = stats.pearsonr(hchg_r, hamilton_r)
r_sp_val, p_sp_val = stats.spearmanr(hchg_r, hamilton_r)
print(f"\n  Pearson  r(hand, Hamilton)  = {r_val:.4f}  (p = {p_val:.2e})")
print(f"  Spearman ρ(hand, Hamilton)  = {r_sp_val:.4f}  (p = {p_sp_val:.2e})")

# 4b. Main test: ed vs Hamilton change
print("\n" + "─" * 70)
print("MAIN TEST: ed vs Hamilton semantic change")
print("─" * 70)

r_main, p_main = stats.pearsonr(ed_r, hamilton_r)
r_sp, p_sp = stats.spearmanr(ed_r, hamilton_r)
r_kt, p_kt = stats.kendalltau(ed_r, hamilton_r)

print(f"\n  Pearson  r(ed, Hamilton_chg)  = {r_main:.4f}  (p = {p_main:.2e})")
print(f"  Spearman ρ(ed, Hamilton_chg)  = {r_sp:.4f}  (p = {p_sp:.2e})")
print(f"  Kendall  τ(ed, Hamilton_chg)  = {r_kt:.4f}  (p = {p_kt:.2e})")

# 4c. Descriptive by ed level
print("\n" + "─" * 70)
print("HAMILTON CHANGE BY ETYMOLOGICAL DEPTH")
print("─" * 70)
print(f"\n  {'ed':>3}  {'n':>4}  {'mean_Hchg':>10}  {'std':>8}  {'mean_freq':>10}")
for level in sorted(set(ed_r)):
    mask = ed_r == level
    n = mask.sum()
    print(f"  {int(level):3d}  {n:4d}  {hamilton_r[mask].mean():10.4f}  "
          f"{hamilton_r[mask].std():8.4f}  {freq_r[mask].mean():10.2f}")

# 4d. Confound: frequency
print("\n" + "─" * 70)
print("CONFOUND: FREQUENCY")
print("─" * 70)

r_ef, p_ef = stats.pearsonr(ed_r, freq_r)
r_hf, p_hf = stats.pearsonr(hamilton_r, freq_r)
print(f"\n  r(ed, freq)            = {r_ef:.4f}  (p = {p_ef:.2e})")
print(f"  r(Hamilton_chg, freq)  = {r_hf:.4f}  (p = {p_hf:.2e})")

# Partial correlation
r_xy = r_main
r_xz = r_ef
r_yz = r_hf
denom = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
r_partial = (r_xy - r_xz * r_yz) / denom if denom > 1e-10 else 0
print(f"\n  Partial r(ed, Hamilton_chg | freq) = {r_partial:.4f}")

# 4e. Fixpoint test
print("\n" + "─" * 70)
print("FIXPOINT TEST: ed=1 vs ed>1 (Hamilton change)")
print("─" * 70)

prim_h = hamilton_r[ed_r == 1]
rest_h = hamilton_r[ed_r > 1]

print(f"\n  ed=1: n={len(prim_h)}, mean Hamilton change = {prim_h.mean():.4f}")
print(f"  ed>1: n={len(rest_h)}, mean Hamilton change = {rest_h.mean():.4f}")

t_stat, p_t = stats.ttest_ind(prim_h, rest_h, equal_var=False)
u_stat, p_u = stats.mannwhitneyu(prim_h, rest_h, alternative='less')
d_cohen = (rest_h.mean() - prim_h.mean()) / np.sqrt((prim_h.var() + rest_h.var()) / 2)

print(f"\n  Welch t: t = {t_stat:.3f}, p = {p_t:.2e}")
print(f"  Mann-Whitney: p = {p_u:.2e}")
print(f"  Cohen's d: {d_cohen:.3f}")

# 4f. Bootstrap CI
print("\n" + "─" * 70)
print("BOOTSTRAP 95% CI (10000 resamples)")
print("─" * 70)

rng = np.random.RandomState(42)
n_boot = 10000
boot_r = np.zeros(n_boot)
for i in range(n_boot):
    idx = rng.randint(0, N, N)
    boot_r[i] = stats.pearsonr(ed_r[idx], hamilton_r[idx])[0]

ci_lo, ci_hi = np.percentile(boot_r, [2.5, 97.5])
print(f"\n  r = {r_main:.4f}, 95% CI = [{ci_lo:.4f}, {ci_hi:.4f}]")

# 4g. Regression
print("\n" + "─" * 70)
print("LINEAR REGRESSION: Hamilton_chg = a + b·ed")
print("─" * 70)

slope, intercept, r_val2, p_val2, se = stats.linregress(ed_r, hamilton_r)
print(f"\n  Hamilton_chg = {intercept:.4f} + {slope:.4f} · ed")
print(f"  R² = {r_val2**2:.4f}")
print(f"  slope = {slope:.4f} ± {se:.4f}")
print(f"  p = {p_val2:.2e}")

# 4h. ANOVA
print("\n" + "─" * 70)
print("ANOVA")
print("─" * 70)
groups = [hamilton_r[ed_r == level] for level in sorted(set(ed_r)) if (ed_r == level).sum() >= 3]
F_stat, p_anova = stats.f_oneway(*groups)
H_stat, p_kw = stats.kruskal(*groups)
print(f"\n  F = {F_stat:.3f}, p = {p_anova:.2e}")
print(f"  Kruskal-Wallis H = {H_stat:.3f}, p = {p_kw:.2e}")

# ═════════════════════════════════════════════════════════════
# 5. Show top changers and stable words
# ═════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("TOP 15 MOST CHANGED (Hamilton)")
print("─" * 70)
sorted_results = sorted(results, key=lambda r: r['hamilton_chg'], reverse=True)
for r in sorted_results[:15]:
    print(f"  {r['word']:<25} ed={r['ed']}  H_chg={r['hamilton_chg']:.4f}  hand={r['hand_chg']}")

print("\n" + "─" * 70)
print("TOP 15 MOST STABLE (Hamilton)")
print("─" * 70)
for r in sorted_results[-15:]:
    print(f"  {r['word']:<25} ed={r['ed']}  H_chg={r['hamilton_chg']:.4f}  hand={r['hand_chg']}")

# ═════════════════════════════════════════════════════════════
# 6. Summary
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

verdict = "CONFIRMED" if p_main < 0.001 and r_partial > 0.15 else \
          "SUPPORTED" if p_main < 0.05 else "NOT SUPPORTED"

print(f"""
  N = {N} words matched with COHA embeddings (1850-2000)

  ┌─────────────────────────────────────────────────────────┐
  │  MAIN RESULT                                            │
  │  r(ed, Hamilton_change) = {r_main:.3f}  (p = {p_main:.1e})     │
  │  After freq control     = {r_partial:.3f}                      │
  │  Bootstrap 95% CI       = [{ci_lo:.3f}, {ci_hi:.3f}]          │
  └─────────────────────────────────────────────────────────┘

  VALIDATION:
    Hand-coded change correlates with Hamilton: r = {stats.pearsonr(hchg_r, hamilton_r)[0]:.3f}

  FIXPOINT TEST:
    ed=1 mean Hamilton change = {prim_h.mean():.4f}
    ed>1 mean Hamilton change = {rest_h.mean():.4f}
    Cohen's d = {d_cohen:.2f}

  REGRESSION:
    Hamilton_chg = {intercept:.3f} + {slope:.3f}·ed  (R² = {r_val2**2:.3f})

  COMPARISON WITH HAND-CODED ANALYSIS:
    Hand-coded:  r(ed, change) = 0.646  (N=274)
    Hamilton:    r(ed, change) = {r_main:.3f}  (N={N})

  VERDICT: Semantic Countdown Hypothesis is {verdict}
           with corpus-based semantic change measurement.

  This is no longer hand-coded. This is measured.
""")
