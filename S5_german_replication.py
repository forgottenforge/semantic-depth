#!/usr/bin/env python3
"""
Semantic Countdown Hypothesis — German Replication
=====================================================
Cross-linguistic validation using German diachronic embeddings.
If ed predicts semantic change in BOTH English and German,
the effect is not language-specific but structural.
"""

import pickle
import numpy as np
from scipy import stats
from wordfreq import zipf_frequency
import warnings
warnings.filterwarnings('ignore')

# ═════════════════════════════════════════════════════════════
# 1. Load and align German embeddings
# ═════════════════════════════════════════════════════════════

def load_decade(year, prefix='ger_sgns'):
    with open(f'{prefix}/{year}-vocab.pkl', 'rb') as f:
        vocab = pickle.load(f, encoding='latin1')
    W = np.load(f'{prefix}/{year}-w.npy')
    idx = {w: i for i, w in enumerate(vocab)}
    return vocab, W, idx

print("Loading German embeddings...")
try:
    _, W_early, idx_early = load_decade(1850)
    _, W_late, idx_late = load_decade(1990)
except FileNotFoundError:
    # Try alternative directory structure
    import glob
    dirs = glob.glob('*sgns*') + glob.glob('sgns*')
    print(f"Available directories: {dirs}")
    # Check inside zip
    import os
    for d in dirs:
        if os.path.isdir(d):
            files = os.listdir(d)[:10]
            print(f"  {d}/: {files}")
    raise

shared = set(idx_early.keys()) & set(idx_late.keys())
print(f"Shared German vocabulary: {len(shared)}")

# Procrustes alignment
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
# 2. German word list with hand-coded ed
# ═════════════════════════════════════════════════════════════

# German Primwörter (ed=1): PIE roots, basic vocabulary
# These are the SAME conceptual words as the English Primwörter
german_data = [
    # ── ed=1: Primwörter ──
    # Pronouns / Deixis
    ("ich",1), ("du",1), ("er",1), ("sie",1), ("es",1),
    ("wir",1), ("ihr",1), ("mich",1), ("uns",1),
    ("dies",1), ("das",1), ("hier",1), ("da",1), ("dort",1),
    ("jetzt",1), ("wer",1), ("was",1), ("wo",1), ("wann",1),
    # Basic verbs
    ("sein",1), ("haben",1), ("tun",1), ("gehen",1), ("kommen",1),
    ("sehen",1), ("wissen",1), ("sagen",1), ("machen",1),
    ("nehmen",1), ("geben",1), ("essen",1), ("trinken",1),
    ("schlafen",1), ("sterben",1), ("sitzen",1), ("stehen",1),
    ("liegen",1), ("fallen",1), ("laufen",1), ("gehen",1),
    ("schneiden",1), ("brennen",1), ("ziehen",1),
    ("schwimmen",1), ("fliegen",1), ("halten",1),
    ("waschen",1), ("werfen",1),
    # Numbers
    ("eins",1), ("zwei",1), ("drei",1), ("zehn",1),
    # Body parts
    ("auge",1), ("ohr",1), ("mund",1), ("zahn",1),
    ("zunge",1), ("hand",1), ("kopf",1), ("herz",1),
    ("knochen",1), ("blut",1), ("haut",1),
    ("nase",1), ("haar",1),
    # Nature
    ("sonne",1), ("mond",1), ("stern",1), ("wasser",1),
    ("feuer",1), ("erde",1), ("stein",1), ("baum",1),
    ("blatt",1), ("regen",1), ("schnee",1), ("wind",1),
    ("sand",1), ("salz",1), ("asche",1),
    # Animals
    ("hund",1), ("fisch",1), ("wurm",1), ("maus",1), ("vogel",1),
    # Adjectives
    ("neu",1), ("alt",1), ("gut",1), ("lang",1),
    ("klein",1), ("warm",1), ("kalt",1), ("nass",1),
    ("trocken",1), ("tot",1), ("rot",1), ("schwarz",1),
    ("voll",1), ("rund",1),
    # Other basic
    ("name",1), ("nacht",1), ("tag",1), ("weg",1),
    ("mann",1), ("frau",1), ("kind",1),
    ("nicht",1), ("alle",1), ("viel",1),

    # ── ed=2: Simple derivations ──
    # Compounds (Grundwort + Bestimmungswort)
    ("handwerk",2), ("sonnenlicht",2), ("wasserfall",2),
    ("steinmetz",2), ("blutbad",2), ("feuerwerk",2),
    ("nachtzeit",2), ("tageslicht",2), ("erdreich",2),
    ("windstill",2), ("eisenbahn",2),
    # Simple prefix/suffix
    ("freiheit",2), ("schönheit",2), ("wahrheit",2),
    ("kindheit",2), ("weisheit",2), ("dunkelheit",2),
    ("freundschaft",2), ("herrschaft",2), ("wissenschaft",2),
    ("wandlung",2), ("handlung",2), ("richtung",2),
    ("lehrer",2), ("denker",2), ("dichter",2),
    ("vergessen",2), ("verstehen",2), ("beginnen",2),
    ("erkennen",2), ("empfangen",2), ("gewinnen",2),
    # Semantic shifts
    ("ding",2),     # originally: Versammlung → Gegenstand
    ("knecht",2),   # originally: Junge → Diener
    ("schlecht",2), # originally: schlicht/einfach → böse
    ("billig",2),   # originally: angemessen → günstig
    ("geil",2),     # originally: üppig → sexuell/cool
    ("toll",2),     # originally: verrückt → großartig

    # ── ed=3: Compound derivations ──
    ("wunderbar",3), ("gefährlich",3), ("unmöglich",3),
    ("unglücklich",3), ("verständlich",3), ("gemeinschaft",3),
    ("freundlichkeit",3), ("verantwortung",3), ("regierung",3),
    ("entdeckung",3), ("verschwinden",3), ("übertreibung",3),
    ("gesellschaft",3), ("gesundheit",3),
    # Semantic shifts
    ("furchtbar",3),  # furcht+bar → awe-inspiring → terrible
    ("schrecklich",3), # causing Schrecken → very bad
    ("herrlich",3),   # like a Herr → wonderful
    ("elend",3),      # eli-lenti (Ausland) → miserable
    ("albern",3),     # alawari (friendly) → silly
    ("frech",3),      # originally: tapfer → rude
    ("gemein",3),     # originally: allgemein → mean/nasty
    ("schlimm",3),    # originally: schief → bad

    # ── ed=4: Multi-layer / Latin-Greek borrowings ──
    ("unglücklicherweise",4), ("verantwortungslos",4),
    ("kommunikation",4), ("philosophisch",4),
    ("international",4), ("organisation",4),
    ("enthusiasmus",4), ("kandidat",4),
    ("sekretär",4), ("algorithmus",4), ("algebra",4),
    ("minister",4), ("kardinal",4), ("quarantäne",4),
    ("muskel",4), ("admiral",4), ("alkohol",4),
    ("begeisterung",4), ("unabhängigkeit",4),
    ("gleichberechtigung",4),

    # ── ed=5: Maximally derived ──
    ("katastrophe",5), ("enthusiastisch",5),
    ("sophistiziert",5), ("quintessenz",5),
    ("leutnant",5), ("hypothek",5),
    ("egozentrisch",5),
    ("extravagant",5), ("exorbitant",5),
    ("alchemie",5), ("dezimierung",5),
]

# ═════════════════════════════════════════════════════════════
# 3. Match with Hamilton German data
# ═════════════════════════════════════════════════════════════

print("\nMatching German words with embeddings...")

results = []
missing = []
for word, ed in german_data:
    sc = semantic_change(word)
    if sc is not None:
        freq = zipf_frequency(word, 'de')
        results.append({'word': word, 'ed': ed, 'hamilton_chg': sc, 'freq': freq})
    else:
        missing.append(word)

N = len(results)
print(f"Matched: {N} / {len(german_data)} words")
if missing:
    print(f"Missing ({len(missing)}): {', '.join(missing[:30])}...")

if N < 30:
    print("\nERROR: Too few matches. Checking what vocabulary looks like...")
    sample_vocab = sorted(list(idx_late.keys()))[:50]
    print(f"Sample German vocab: {sample_vocab}")
    print(f"\nTrying case variations...")
    for word, ed in german_data[:20]:
        for variant in [word, word.lower(), word.upper(), word.capitalize()]:
            if variant in idx_late:
                print(f"  FOUND: {variant}")
                break
        else:
            print(f"  MISSING: {word}")
    exit(0)

words_r = [r['word'] for r in results]
ed_r = np.array([r['ed'] for r in results], dtype=float)
hamilton_r = np.array([r['hamilton_chg'] for r in results], dtype=float)
freq_r = np.array([r['freq'] for r in results], dtype=float)

# ═════════════════════════════════════════════════════════════
# 4. Analysis
# ═════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("SEMANTIC COUNTDOWN — GERMAN REPLICATION")
print("=" * 70)
print(f"\nDataset: {N} German words")

# Descriptive
print("\n" + "─" * 70)
print("HAMILTON CHANGE BY ETYMOLOGICAL DEPTH (German)")
print("─" * 70)
print(f"\n  {'ed':>3}  {'n':>4}  {'mean_Hchg':>10}  {'std':>8}")
for level in sorted(set(ed_r)):
    mask = ed_r == level
    n = mask.sum()
    if n > 0:
        print(f"  {int(level):3d}  {n:4d}  {hamilton_r[mask].mean():10.4f}  "
              f"{hamilton_r[mask].std():8.4f}")

# Correlations
print("\n" + "─" * 70)
print("CORRELATIONS (German)")
print("─" * 70)

r_main, p_main = stats.pearsonr(ed_r, hamilton_r)
r_sp, p_sp = stats.spearmanr(ed_r, hamilton_r)
print(f"\n  Pearson  r(ed, Hamilton_chg)  = {r_main:.4f}  (p = {p_main:.2e})")
print(f"  Spearman ρ(ed, Hamilton_chg)  = {r_sp:.4f}  (p = {p_sp:.2e})")

# Frequency confound
r_ef = stats.pearsonr(ed_r, freq_r)[0]
r_hf = stats.pearsonr(hamilton_r, freq_r)[0]
denom = np.sqrt((1 - r_ef**2) * (1 - r_hf**2))
r_partial = (r_main - r_ef * r_hf) / denom if denom > 1e-10 else 0
print(f"\n  r(ed, freq)            = {r_ef:.4f}")
print(f"  r(Hamilton, freq)      = {r_hf:.4f}")
print(f"  Partial r(ed, H | freq)= {r_partial:.4f}")

# Fixpoint test
print("\n" + "─" * 70)
print("FIXPOINT TEST (German)")
print("─" * 70)

prim_h = hamilton_r[ed_r == 1]
rest_h = hamilton_r[ed_r > 1]

if len(prim_h) > 1 and len(rest_h) > 1:
    print(f"\n  ed=1: n={len(prim_h)}, mean Hamilton change = {prim_h.mean():.4f}")
    print(f"  ed>1: n={len(rest_h)}, mean Hamilton change = {rest_h.mean():.4f}")

    t_stat, p_t = stats.ttest_ind(prim_h, rest_h, equal_var=False)
    d_cohen = (rest_h.mean() - prim_h.mean()) / np.sqrt((prim_h.var() + rest_h.var()) / 2)
    print(f"\n  Welch t: t = {t_stat:.3f}, p = {p_t:.2e}")
    print(f"  Cohen's d: {d_cohen:.3f}")

# Bootstrap
print("\n" + "─" * 70)
print("BOOTSTRAP 95% CI")
print("─" * 70)

rng = np.random.RandomState(42)
boot_r = [stats.pearsonr(ed_r[idx := rng.randint(0, N, N)], hamilton_r[idx])[0]
          for _ in range(10000)]
ci_lo, ci_hi = np.percentile(boot_r, [2.5, 97.5])
print(f"\n  r = {r_main:.4f}, 95% CI = [{ci_lo:.4f}, {ci_hi:.4f}]")

# Top changers and stable
print("\n" + "─" * 70)
print("TOP 10 MOST STABLE (German)")
print("─" * 70)
sorted_r = sorted(results, key=lambda r: r['hamilton_chg'])
for r in sorted_r[:10]:
    print(f"  {r['word']:<25} ed={r['ed']}  H_chg={r['hamilton_chg']:.4f}")

print("\n" + "─" * 70)
print("TOP 10 MOST CHANGED (German)")
print("─" * 70)
for r in sorted_r[-10:]:
    print(f"  {r['word']:<25} ed={r['ed']}  H_chg={r['hamilton_chg']:.4f}")

# ═════════════════════════════════════════════════════════════
# 5. Cross-linguistic comparison
# ═════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("CROSS-LINGUISTIC COMPARISON")
print("=" * 70)

print(f"""
  {'Measure':<35}  {'English':>10}  {'German':>10}
  {'─'*35}  {'─'*10}  {'─'*10}
  {'N words':<35}  {'225':>10}  {N:>10}
  {'r(ed, Hamilton change)':<35}  {'0.528':>10}  {r_main:>10.3f}
  {'r partial (|freq)':<35}  {'0.282':>10}  {r_partial:>10.3f}
  {'ed=1 mean change':<35}  {'0.333':>10}  {prim_h.mean():>10.3f}
  {'ed>1 mean change':<35}  {'0.468':>10}  {rest_h.mean():>10.3f}
  {'Cohen d (fixpoint)':<35}  {'1.10':>10}  {d_cohen:>10.2f}

  VERDICT: {"REPLICATED" if p_main < 0.05 and r_main > 0.2 else "NOT REPLICATED"}
""")
