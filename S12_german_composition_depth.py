#!/usr/bin/env python3
"""
German Composition-Depth Metric
================================
Hypothesis: German semantic change is driven by COMPOSITION depth
(how many morphological layers), not BORROWING depth (how many
languages a word passed through).

Recode the same German word list with cd (composition depth)
and compare r(cd, Δ) vs r(ed, Δ).
"""

import pickle
import numpy as np
from scipy import stats
from wordfreq import zipf_frequency
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════
# 1. Load German embeddings (same as S5)
# ═══════════════════════════════════════════════════════════

def load_decade(year, prefix='histwords/ger_sgns'):
    with open(f'{prefix}/{year}-vocab.pkl', 'rb') as f:
        vocab = pickle.load(f, encoding='latin1')
    W = np.load(f'{prefix}/{year}-w.npy')
    idx = {w: i for i, w in enumerate(vocab)}
    return vocab, W, idx

print("Loading German embeddings...")
vocab_e, W_early, idx_early = load_decade(1850)
vocab_l, W_late, idx_late = load_decade(1990)

shared = set(idx_early.keys()) & set(idx_late.keys())
print(f"Shared vocabulary: {len(shared)}")

# Procrustes
shared_by_freq = sorted(shared, key=lambda w: idx_late[w])
anchors = shared_by_freq[:5000]
M1 = W_early[[idx_early[w] for w in anchors]]
M2 = W_late[[idx_late[w] for w in anchors]]
M1 = M1 / (np.linalg.norm(M1, axis=1, keepdims=True) + 1e-10)
M2 = M2 / (np.linalg.norm(M2, axis=1, keepdims=True) + 1e-10)
U, S, Vt = np.linalg.svd(M2.T @ M1)
R = U @ Vt
W_late_aligned = W_late @ R

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

# ═══════════════════════════════════════════════════════════
# 2. German words with BOTH ed (borrowing) and cd (composition)
# ═══════════════════════════════════════════════════════════
#
# cd = number of MORPHOLOGICAL steps a modern German speaker
#      can decompose:
#   cd=1: simplex, no visible parts (Haus, gut, ich)
#   cd=2: one derivation or one compound (Hand+werk, frei+heit)
#   cd=3: compound+derivation (wunder+bar, ge+sell+schaft)
#   cd=4: multi-layer (Freund+lich+keit, ver+Antwort+ung+s+los)
#   cd=5: maximally stacked (Gleich+be+recht+ig+ung)
#
# Key difference from ed: borrowed words get cd based on
# German morphological transparency, NOT borrowing chain.
# "Philosophisch" is ed=4 (Greek→Latin→German) but cd=2
# (Philosoph+isch — German speaker sees one derivation).
# "Gleichberechtigung" is ed=3 but cd=5
# (gleich+be+recht+ig+ung — five visible layers).

german_data = [
    # word,           ed, cd
    # ── Simplex / Primwörter ──
    ("ich",           1, 1),
    ("du",            1, 1),
    ("er",            1, 1),
    ("sie",           1, 1),
    ("es",            1, 1),
    ("wir",           1, 1),
    ("ihr",           1, 1),
    ("mich",          1, 1),
    ("uns",           1, 1),
    ("dies",          1, 1),
    ("das",           1, 1),
    ("hier",          1, 1),
    ("da",            1, 1),
    ("dort",          1, 1),
    ("jetzt",         1, 1),
    ("wer",           1, 1),
    ("was",           1, 1),
    ("wo",            1, 1),
    ("wann",          1, 1),
    ("sein",          1, 1),
    ("haben",         1, 1),
    ("tun",           1, 1),
    ("gehen",         1, 1),
    ("kommen",        1, 1),
    ("sehen",         1, 1),
    ("wissen",        1, 1),
    ("sagen",         1, 1),
    ("machen",        1, 1),
    ("nehmen",        1, 1),
    ("geben",         1, 1),
    ("essen",         1, 1),
    ("trinken",       1, 1),
    ("schlafen",      1, 1),
    ("sterben",       1, 1),
    ("sitzen",        1, 1),
    ("stehen",        1, 1),
    ("liegen",        1, 1),
    ("fallen",        1, 1),
    ("laufen",        1, 1),
    ("schneiden",     1, 1),
    ("brennen",       1, 1),
    ("ziehen",        1, 1),
    ("schwimmen",     1, 1),
    ("fliegen",       1, 1),
    ("halten",        1, 1),
    ("waschen",       1, 1),
    ("werfen",        1, 1),
    ("eins",          1, 1),
    ("zwei",          1, 1),
    ("drei",          1, 1),
    ("zehn",          1, 1),
    ("auge",          1, 1),
    ("ohr",           1, 1),
    ("mund",          1, 1),
    ("zahn",          1, 1),
    ("zunge",         1, 1),
    ("hand",          1, 1),
    ("kopf",          1, 1),
    ("herz",          1, 1),
    ("knochen",       1, 1),
    ("blut",          1, 1),
    ("haut",          1, 1),
    ("nase",          1, 1),
    ("haar",          1, 1),
    ("sonne",         1, 1),
    ("mond",          1, 1),
    ("stern",         1, 1),
    ("wasser",        1, 1),
    ("feuer",         1, 1),
    ("erde",          1, 1),
    ("stein",         1, 1),
    ("baum",          1, 1),
    ("blatt",         1, 1),
    ("regen",         1, 1),
    ("schnee",        1, 1),
    ("wind",          1, 1),
    ("sand",          1, 1),
    ("salz",          1, 1),
    ("asche",         1, 1),
    ("hund",          1, 1),
    ("fisch",         1, 1),
    ("wurm",          1, 1),
    ("maus",          1, 1),
    ("vogel",         1, 1),
    ("neu",           1, 1),
    ("alt",           1, 1),
    ("gut",           1, 1),
    ("lang",          1, 1),
    ("klein",         1, 1),
    ("warm",          1, 1),
    ("kalt",          1, 1),
    ("nass",          1, 1),
    ("trocken",       1, 1),
    ("tot",           1, 1),
    ("rot",           1, 1),
    ("schwarz",       1, 1),
    ("voll",          1, 1),
    ("rund",          1, 1),
    ("name",          1, 1),
    ("nacht",         1, 1),
    ("tag",           1, 1),
    ("weg",           1, 1),
    ("mann",          1, 1),
    ("frau",          1, 1),
    ("kind",          1, 1),
    ("nicht",         1, 1),
    ("alle",          1, 1),
    ("viel",          1, 1),

    # ── ed=2 / cd=2: one visible step ──
    ("handwerk",      2, 2),  # Hand+Werk
    ("sonnenlicht",   2, 2),  # Sonne+Licht
    ("wasserfall",    2, 2),  # Wasser+Fall
    ("steinmetz",     2, 2),  # Stein+Metz
    ("blutbad",       2, 2),  # Blut+Bad
    ("feuerwerk",     2, 2),  # Feuer+Werk
    ("nachtzeit",     2, 2),  # Nacht+Zeit
    ("tageslicht",    2, 2),  # Tag+Licht
    ("erdreich",      2, 2),  # Erde+Reich
    ("windstill",     2, 2),  # Wind+still
    ("eisenbahn",     2, 2),  # Eisen+Bahn
    ("freiheit",      2, 2),  # frei+heit
    ("wahrheit",      2, 2),  # wahr+heit
    ("kindheit",      2, 2),  # Kind+heit
    ("weisheit",      2, 2),  # weise+heit
    ("dunkelheit",    2, 2),  # dunkel+heit
    ("freundschaft",  2, 2),  # Freund+schaft
    ("herrschaft",    2, 2),  # Herr+schaft
    ("wissenschaft",  2, 2),  # wissen+schaft
    ("wandlung",      2, 2),  # wandeln+ung
    ("handlung",      2, 2),  # handeln+ung
    ("richtung",      2, 2),  # richten+ung
    ("lehrer",        2, 2),  # lehren+er
    ("denker",        2, 2),  # denken+er
    ("dichter",       2, 2),  # dichten+er
    ("vergessen",     2, 2),  # ver+gessen
    ("verstehen",     2, 2),  # ver+stehen
    ("beginnen",      2, 2),  # be+ginnen
    ("erkennen",      2, 2),  # er+kennen
    ("empfangen",     2, 2),  # emp+fangen
    ("gewinnen",      2, 2),  # ge+winnen
    ("schönheit",     2, 2),  # schön+heit

    # Semantic-shift words: ed=2, but cd=1 (opaque today)
    ("ding",          2, 1),  # Versammlung→Gegenstand, opaque
    ("knecht",        2, 1),  # Junge→Diener, opaque
    ("schlecht",      2, 1),  # schlicht→böse, opaque
    ("billig",        2, 1),  # angemessen→günstig, opaque
    ("geil",          2, 1),  # üppig→cool, opaque
    ("toll",          2, 1),  # verrückt→großartig, opaque

    # ── ed=3 / cd varies ──
    ("wunderbar",     3, 2),  # Wunder+bar
    ("unmöglich",     3, 2),  # un+möglich
    ("gefährlich",    3, 2),  # Gefahr+lich
    ("unglücklich",   3, 3),  # un+Glück+lich
    ("verständlich",  3, 3),  # ver+ständ+lich (ver+Stand+lich)
    ("gemeinschaft",  3, 3),  # ge+mein+schaft
    ("freundlichkeit",3, 3),  # Freund+lich+keit
    ("verantwortung", 3, 3),  # ver+ant+wort+ung
    ("regierung",     3, 2),  # regier+ung
    ("entdeckung",    3, 3),  # ent+deck+ung
    ("verschwinden",  3, 2),  # ver+schwinden
    ("gesellschaft",  3, 3),  # ge+sell+schaft
    ("gesundheit",    3, 2),  # gesund+heit

    # Semantic-shift: ed=3 but cd varies
    ("furchtbar",     3, 2),  # Furcht+bar (transparent)
    ("schrecklich",   3, 2),  # Schreck+lich (transparent)
    ("herrlich",      3, 2),  # Herr+lich (semi-transparent)
    ("elend",         3, 1),  # eli-lenti→miserable (opaque)
    ("albern",        3, 1),  # alawari→silly (opaque)
    ("frech",         3, 1),  # tapfer→rude (opaque)
    ("gemein",        3, 1),  # allgemein→mean (opaque)
    ("schlimm",       3, 1),  # schief→bad (opaque)
    ("übertreibung",  3, 3),  # über+treib+ung

    # ── ed=4: Latin/Greek borrowings → recode cd by German morphology ──
    ("unglücklicherweise", 4, 5),  # un+glück+lich+er+weise
    ("verantwortungslos",  4, 4),  # ver+ant+wort+ung+s+los
    ("kommunikation", 4, 1),  # opaque in German (no visible parts)
    ("philosophisch",  4, 2),  # Philosoph+isch
    ("international", 4, 2),  # inter+national (semi-transparent)
    ("organisation",  4, 2),  # organisier+tion (semi)
    ("enthusiasmus",  4, 1),  # opaque
    ("kandidat",      4, 1),  # opaque
    ("sekretär",      4, 1),  # opaque
    ("algorithmus",   4, 1),  # opaque
    ("algebra",       4, 1),  # opaque
    ("minister",      4, 1),  # opaque
    ("kardinal",      4, 1),  # opaque
    ("quarantäne",    4, 1),  # opaque
    ("muskel",        4, 1),  # opaque
    ("admiral",       4, 1),  # opaque
    ("alkohol",       4, 1),  # opaque
    ("begeisterung",  4, 3),  # be+geist+er+ung
    ("unabhängigkeit",4, 4),  # un+ab+häng+ig+keit
    ("gleichberechtigung", 4, 5),  # gleich+be+recht+ig+ung

    # ── ed=5: max borrowing depth → cd by German morphology ──
    ("katastrophe",   5, 1),  # opaque in German
    ("enthusiastisch",5, 2),  # Enthusiast+isch
    ("sophistiziert", 5, 1),  # opaque
    ("quintessenz",   5, 1),  # opaque (maybe Quint+Essenz = cd=2?)
    ("leutnant",      5, 1),  # opaque
    ("hypothek",      5, 1),  # opaque
    ("egozentrisch",  5, 2),  # Ego+zentrisch
    ("extravagant",   5, 1),  # opaque
    ("exorbitant",    5, 1),  # opaque
    ("alchemie",      5, 1),  # opaque
    ("dezimierung",   5, 2),  # dezimier+ung
]

# ═══════════════════════════════════════════════════════════
# 3. Compute semantic change for all words
# ═══════════════════════════════════════════════════════════

print("\nMatching words with embeddings...")

results = []
missing = []
for word, ed, cd in german_data:
    sc = semantic_change(word)
    if sc is not None:
        freq = zipf_frequency(word, 'de')
        results.append({
            'word': word, 'ed': ed, 'cd': cd,
            'delta': sc, 'freq': freq
        })
    else:
        missing.append(word)

N = len(results)
print(f"Matched: {N} / {len(german_data)} words")
if missing:
    print(f"Missing ({len(missing)}): {', '.join(missing[:30])}")

if N < 30:
    print("Too few matches.")
    exit(1)

ed_arr = np.array([r['ed'] for r in results], dtype=float)
cd_arr = np.array([r['cd'] for r in results], dtype=float)
delta_arr = np.array([r['delta'] for r in results], dtype=float)
freq_arr = np.array([r['freq'] for r in results], dtype=float)

# ═══════════════════════════════════════════════════════════
# 4. Compare ed vs cd as predictors
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("BORROWING DEPTH (ed) vs COMPOSITION DEPTH (cd)")
print("=" * 70)

# Raw correlations
r_ed, p_ed = stats.pearsonr(ed_arr, delta_arr)
r_cd, p_cd = stats.pearsonr(cd_arr, delta_arr)
rho_ed, p_rho_ed = stats.spearmanr(ed_arr, delta_arr)
rho_cd, p_rho_cd = stats.spearmanr(cd_arr, delta_arr)

print(f"\n  {'Metric':<25} {'Pearson r':>10} {'p':>12} {'Spearman ρ':>12} {'p':>12}")
print(f"  {'─'*25} {'─'*10} {'─'*12} {'─'*12} {'─'*12}")
print(f"  {'Borrowing depth (ed)':<25} {r_ed:10.4f} {p_ed:12.2e} {rho_ed:12.4f} {p_rho_ed:12.2e}")
print(f"  {'Composition depth (cd)':<25} {r_cd:10.4f} {p_cd:12.2e} {rho_cd:12.4f} {p_rho_cd:12.2e}")

# Partial correlations (controlling for frequency)
def partial_r(x, y, z):
    """Partial correlation r(x,y|z)."""
    r_xy = stats.pearsonr(x, y)[0]
    r_xz = stats.pearsonr(x, z)[0]
    r_yz = stats.pearsonr(y, z)[0]
    denom = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    if denom < 1e-10:
        return 0.0
    return (r_xy - r_xz * r_yz) / denom

r_ed_partial = partial_r(ed_arr, delta_arr, freq_arr)
r_cd_partial = partial_r(cd_arr, delta_arr, freq_arr)

print(f"\n  Partial correlations (controlling frequency):")
print(f"  {'Borrowing depth (ed)':<25} r_partial = {r_ed_partial:.4f}")
print(f"  {'Composition depth (cd)':<25} r_partial = {r_cd_partial:.4f}")

# ═══════════════════════════════════════════════════════════
# 5. Group means by cd
# ═══════════════════════════════════════════════════════════

print("\n" + "─" * 70)
print("SEMANTIC CHANGE BY COMPOSITION DEPTH (cd)")
print("─" * 70)
print(f"\n  {'cd':>3}  {'n':>4}  {'mean Δ':>10}  {'std':>8}")
for level in sorted(set(cd_arr)):
    mask = cd_arr == level
    n = mask.sum()
    if n > 0:
        print(f"  {int(level):3d}  {n:4d}  {delta_arr[mask].mean():10.4f}  "
              f"{delta_arr[mask].std():8.4f}")

print("\n" + "─" * 70)
print("SEMANTIC CHANGE BY BORROWING DEPTH (ed) — for comparison")
print("─" * 70)
print(f"\n  {'ed':>3}  {'n':>4}  {'mean Δ':>10}  {'std':>8}")
for level in sorted(set(ed_arr)):
    mask = ed_arr == level
    n = mask.sum()
    if n > 0:
        print(f"  {int(level):3d}  {n:4d}  {delta_arr[mask].mean():10.4f}  "
              f"{delta_arr[mask].std():8.4f}")

# ═══════════════════════════════════════════════════════════
# 6. Multiple regression: ed + cd together
# ═══════════════════════════════════════════════════════════

print("\n" + "─" * 70)
print("MULTIPLE REGRESSION")
print("─" * 70)

from numpy.linalg import lstsq

# Model 1: ed only
X1 = np.column_stack([np.ones(N), ed_arr, freq_arr])
beta1, _, _, _ = lstsq(X1, delta_arr, rcond=None)
pred1 = X1 @ beta1
ss_res1 = np.sum((delta_arr - pred1)**2)
ss_tot = np.sum((delta_arr - delta_arr.mean())**2)
r2_ed = 1 - ss_res1 / ss_tot

# Model 2: cd only
X2 = np.column_stack([np.ones(N), cd_arr, freq_arr])
beta2, _, _, _ = lstsq(X2, delta_arr, rcond=None)
pred2 = X2 @ beta2
ss_res2 = np.sum((delta_arr - pred2)**2)
r2_cd = 1 - ss_res2 / ss_tot

# Model 3: ed + cd
X3 = np.column_stack([np.ones(N), ed_arr, cd_arr, freq_arr])
beta3, _, _, _ = lstsq(X3, delta_arr, rcond=None)
pred3 = X3 @ beta3
ss_res3 = np.sum((delta_arr - pred3)**2)
r2_both = 1 - ss_res3 / ss_tot

# F-test: cd adds to ed+freq?
df_extra = 1
df_resid = N - 4  # intercept + ed + cd + freq
f_cd_adds = ((ss_res1 - ss_res3) / df_extra) / (ss_res3 / df_resid)
p_cd_adds = 1 - stats.f.cdf(f_cd_adds, df_extra, df_resid)

# F-test: ed adds to cd+freq?
X2b = np.column_stack([np.ones(N), cd_arr, freq_arr])
beta2b, _, _, _ = lstsq(X2b, delta_arr, rcond=None)
ss_res2b = np.sum((delta_arr - X2b @ beta2b)**2)
f_ed_adds = ((ss_res2b - ss_res3) / df_extra) / (ss_res3 / df_resid)
p_ed_adds = 1 - stats.f.cdf(f_ed_adds, df_extra, df_resid)

print(f"\n  {'Model':<30} {'R²':>8}")
print(f"  {'─'*30} {'─'*8}")
print(f"  {'ed + freq':<30} {r2_ed:8.4f}")
print(f"  {'cd + freq':<30} {r2_cd:8.4f}")
print(f"  {'ed + cd + freq':<30} {r2_both:8.4f}")
print(f"\n  cd adds to ed+freq: ΔR² = {r2_both - r2_ed:.4f}, F = {f_cd_adds:.2f}, p = {p_cd_adds:.4f}")
print(f"  ed adds to cd+freq: ΔR² = {r2_both - r2_cd:.4f}, F = {f_ed_adds:.2f}, p = {p_ed_adds:.4f}")

# ═══════════════════════════════════════════════════════════
# 7. Steiger test: is r(cd,Δ) significantly > r(ed,Δ)?
# ═══════════════════════════════════════════════════════════

print("\n" + "─" * 70)
print("STEIGER TEST: r(cd,Δ) vs r(ed,Δ)")
print("─" * 70)

# Steiger's Z for comparing dependent correlations
r12 = stats.pearsonr(ed_arr, cd_arr)[0]  # correlation between predictors
r_mean = (r_ed**2 + r_cd**2) / 2

z_ed = np.arctanh(r_ed)
z_cd = np.arctanh(r_cd)

# Steiger (1980) formula for dependent correlations
det = (1 - r_ed**2 - r_cd**2 - r12**2 + 2 * r_ed * r_cd * r12)
f_factor = (1 - r12) / (2 * (1 - r12**2)) if abs(r12) < 1 else 1
z_diff = (z_ed - z_cd) * np.sqrt((N - 3) / (2 * (1 - r12) * f_factor + 1e-10))

p_steiger = 2 * (1 - stats.norm.cdf(abs(z_diff)))
print(f"\n  r(ed, Δ) = {r_ed:.4f}")
print(f"  r(cd, Δ) = {r_cd:.4f}")
print(f"  r(ed, cd) = {r12:.4f}")
print(f"  Steiger Z = {z_diff:.3f}, p = {p_steiger:.4f}")
winner = "cd" if r_cd > r_ed else "ed"
sig = "significantly" if p_steiger < 0.05 else "not significantly"
print(f"\n  → {winner} is {sig} better (p = {p_steiger:.4f})")

# ═══════════════════════════════════════════════════════════
# 8. Bootstrap comparison
# ═══════════════════════════════════════════════════════════

print("\n" + "─" * 70)
print("BOOTSTRAP: r(cd,Δ) - r(ed,Δ)")
print("─" * 70)

rng = np.random.RandomState(42)
n_boot = 10000
diffs = []
for _ in range(n_boot):
    idx = rng.randint(0, N, N)
    r_ed_b = stats.pearsonr(ed_arr[idx], delta_arr[idx])[0]
    r_cd_b = stats.pearsonr(cd_arr[idx], delta_arr[idx])[0]
    diffs.append(r_cd_b - r_ed_b)

diffs = np.array(diffs)
ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
p_boot = np.mean(diffs <= 0) if np.mean(diffs) > 0 else np.mean(diffs >= 0)

print(f"\n  Mean Δr = {np.mean(diffs):.4f}")
print(f"  95% CI = [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"  P(cd ≤ ed) = {p_boot:.4f}")

# ═══════════════════════════════════════════════════════════
# 9. Verdict
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
print(f"""
  Borrowing depth (ed):    r = {r_ed:.4f}, r_partial = {r_ed_partial:.4f}
  Composition depth (cd):  r = {r_cd:.4f}, r_partial = {r_cd_partial:.4f}

  English r(ed, Δ) = 0.528 for comparison
""")

if r_cd > r_ed + 0.05:
    print("  ✓ COMPOSITION DEPTH WORKS BETTER FOR GERMAN.")
    print("    The hypothesis is confirmed: German needs a morphological")
    print("    metric, not a borrowing-chain metric.")
elif abs(r_cd - r_ed) < 0.05:
    print("  ~ BOTH METRICS PERFORM SIMILARLY.")
    print("    The distinction may not matter for this sample.")
else:
    print("  ✗ BORROWING DEPTH STILL WORKS BETTER.")
    print("    The composition-depth hypothesis is not supported.")
