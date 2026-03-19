#!/usr/bin/env python3
"""
German Anchor Hypothesis
========================
Test: Transparent compounds (visible parts) are MORE stable than
opaque words of the same frequency, because the parts "anchor"
the meaning.

If true: transparent German compounds should have LOWER Δ than
opaque simplex words at matched frequency — the opposite of English,
where transparency enables reanalysis.

This would explain why ed doesn't work for German: the anchoring
mechanism COUNTERACTS the depth effect.
"""

import pickle
import numpy as np
from scipy import stats
from wordfreq import zipf_frequency
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════
# 1. Load German embeddings
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
# 2. Large-scale transparency coding
# ═══════════════════════════════════════════════════════════
#
# Three categories:
#   T = transparent compound (modern speaker sees parts)
#   O = opaque (no visible parts, or borrowed without visible morphology)
#   P = prime (simplex Germanic root)
#
# We need MANY words — pull from the full shared vocabulary
# and hand-code a focused sample.

# Hand-coded German words with transparency
# Format: (word, category, notes)
# T = transparent compound, O = opaque, P = prime

german_words = [
    # ── PRIMES (P): simplex Germanic roots ──
    ("ich", "P"), ("du", "P"), ("er", "P"), ("sie", "P"),
    ("wir", "P"), ("es", "P"), ("das", "P"), ("hier", "P"),
    ("da", "P"), ("dort", "P"), ("jetzt", "P"), ("wo", "P"),
    ("was", "P"), ("wer", "P"), ("wann", "P"),
    ("sein", "P"), ("haben", "P"), ("gehen", "P"), ("kommen", "P"),
    ("sehen", "P"), ("sagen", "P"), ("machen", "P"), ("nehmen", "P"),
    ("geben", "P"), ("essen", "P"), ("stehen", "P"), ("liegen", "P"),
    ("fallen", "P"), ("laufen", "P"), ("halten", "P"), ("werfen", "P"),
    ("ziehen", "P"), ("sitzen", "P"), ("sterben", "P"),
    ("wissen", "P"), ("trinken", "P"),
    ("eins", "P"), ("zwei", "P"), ("drei", "P"), ("zehn", "P"),
    ("auge", "P"), ("ohr", "P"), ("mund", "P"), ("hand", "P"),
    ("kopf", "P"), ("herz", "P"), ("blut", "P"), ("haut", "P"),
    ("nase", "P"), ("sonne", "P"), ("mond", "P"), ("wasser", "P"),
    ("feuer", "P"), ("erde", "P"), ("stein", "P"), ("baum", "P"),
    ("blatt", "P"), ("regen", "P"), ("wind", "P"), ("sand", "P"),
    ("neu", "P"), ("alt", "P"), ("gut", "P"), ("lang", "P"),
    ("klein", "P"), ("voll", "P"), ("schwarz", "P"),
    ("name", "P"), ("nacht", "P"), ("tag", "P"), ("weg", "P"),
    ("mann", "P"), ("frau", "P"), ("kind", "P"),
    ("nicht", "P"), ("alle", "P"), ("viel", "P"),

    # ── TRANSPARENT COMPOUNDS (T): speaker sees the parts ──
    # Noun+Noun
    ("eisenbahn", "T"),      # Eisen+Bahn
    ("tageslicht", "T"),     # Tag+Licht
    ("erdreich", "T"),       # Erde+Reich
    ("feuerwerk", "T"),      # Feuer+Werk
    ("handlung", "T"),       # handeln+ung
    ("richtung", "T"),       # richten+ung
    ("wandlung", "T"),       # wandeln+ung
    ("freiheit", "T"),       # frei+heit
    ("wahrheit", "T"),       # wahr+heit
    ("kindheit", "T"),       # Kind+heit
    ("dunkelheit", "T"),     # dunkel+heit
    ("weisheit", "T"),       # weise+heit
    ("freundschaft", "T"),   # Freund+schaft
    ("herrschaft", "T"),     # Herr+schaft
    ("wissenschaft", "T"),   # wissen+schaft
    ("lehrer", "T"),         # lehren+er
    ("denker", "T"),         # denken+er
    ("dichter", "T"),        # dichten+er
    ("vergessen", "T"),      # ver+gessen
    ("verstehen", "T"),      # ver+stehen
    ("beginnen", "T"),       # be+ginnen
    ("erkennen", "T"),       # er+kennen
    ("empfangen", "T"),      # emp+fangen
    ("gewinnen", "T"),       # ge+winnen

    # Multi-layer transparent
    ("wunderbar", "T"),      # Wunder+bar
    ("unmöglich", "T"),      # un+möglich
    ("gefährlich", "T"),     # Gefahr+lich
    ("unglücklich", "T"),    # un+Glück+lich
    ("verständlich", "T"),   # ver+Stand+lich
    ("gemeinschaft", "T"),   # ge+mein+schaft
    ("freundlichkeit", "T"), # Freund+lich+keit
    ("verantwortung", "T"),  # ver+ant+wort+ung
    ("regierung", "T"),      # regier+ung
    ("entdeckung", "T"),     # ent+deck+ung
    ("verschwinden", "T"),   # ver+schwinden
    ("gesellschaft", "T"),   # ge+sell+schaft (semi-transparent)
    ("gesundheit", "T"),     # gesund+heit
    ("furchtbar", "T"),      # Furcht+bar
    ("schrecklich", "T"),    # Schreck+lich
    ("herrlich", "T"),       # Herr+lich
    ("übertreibung", "T"),   # über+treib+ung
    ("begeisterung", "T"),   # be+geist+er+ung
    ("unabhängigkeit", "T"), # un+ab+häng+ig+keit
    ("gleichberechtigung", "T"),  # gleich+be+recht+ig+ung
    ("unglücklicherweise", "T"),  # un+glück+lich+er+weise
    ("verantwortungslos", "T"),   # ver+ant+wort+ung+s+los

    # Additional transparent compounds from COHA vocabulary
    ("sonnenschein", "T"),   # Sonne+Schein
    ("mondschein", "T"),     # Mond+Schein
    ("morgenrot", "T"),      # Morgen+Rot (if in vocab)
    ("friedhof", "T"),       # Fried+Hof
    ("kirchhof", "T"),       # Kirch+Hof
    ("gottesdienst", "T"),   # Gottes+Dienst
    ("hauswirt", "T"),       # Haus+Wirt
    ("hausfrau", "T"),       # Haus+Frau
    ("mannschaft", "T"),     # Mann+schaft
    ("landschaft", "T"),     # Land+schaft
    ("eigenschaft", "T"),    # eigen+schaft
    ("wirtschaft", "T"),     # Wirt+schaft
    ("botschaft", "T"),      # Bot+schaft
    ("nachricht", "T"),      # nach+richt
    ("handschrift", "T"),    # Hand+Schrift
    ("ausgang", "T"),        # aus+Gang
    ("eingang", "T"),        # ein+Gang
    ("umgang", "T"),         # um+Gang
    ("aufgang", "T"),        # auf+Gang
    ("vorgang", "T"),        # vor+Gang
    ("untergang", "T"),      # unter+Gang
    ("abgrund", "T"),        # ab+Grund
    ("hintergrund", "T"),    # hinter+Grund
    ("vordergrund", "T"),    # vorder+Grund
    ("grundlage", "T"),      # Grund+Lage
    ("zeitgeist", "T"),      # Zeit+Geist
    ("weltanschauung", "T"), # Welt+Anschauung
    ("selbstvertrauen", "T"),# Selbst+Vertrauen
    ("mitgefühl", "T"),      # mit+Gefühl
    ("vorstellung", "T"),    # vor+Stellung
    ("darstellung", "T"),    # dar+Stellung
    ("herstellung", "T"),    # her+Stellung
    ("ausstellung", "T"),    # aus+Stellung
    ("feststellung", "T"),   # fest+Stellung
    ("stimmung", "T"),       # stimmen+ung
    ("bildung", "T"),        # bilden+ung
    ("erzählung", "T"),      # er+zählen+ung
    ("hoffnung", "T"),       # hoffen+ung
    ("ordnung", "T"),        # ordnen+ung
    ("wohnung", "T"),        # wohnen+ung
    ("rechnung", "T"),       # rechnen+ung
    ("dichtung", "T"),       # dichten+ung
    ("achtung", "T"),        # achten+ung
    ("führung", "T"),        # führen+ung
    ("wirkung", "T"),        # wirken+ung
    ("leitung", "T"),        # leiten+ung
    ("stellung", "T"),       # stellen+ung
    ("haltung", "T"),        # halten+ung
    ("richtung", "T"),       # richten+ung
    ("bewegung", "T"),       # bewegen+ung
    ("meinung", "T"),        # meinen+ung
    ("zeitung", "T"),        # zeiten+ung
    ("bedeutung", "T"),      # be+deuten+ung
    ("verbindung", "T"),     # ver+binden+ung
    ("erfahrung", "T"),      # er+fahren+ung
    ("entwicklung", "T"),    # ent+wickeln+ung
    ("verhandlung", "T"),    # ver+handeln+ung
    ("untersuchung", "T"),   # unter+suchen+ung
    ("überzeugung", "T"),    # über+zeugen+ung
    ("versammlung", "T"),    # ver+sammeln+ung
    ("beschreibung", "T"),   # be+schreiben+ung
    ("fröhlich", "T"),       # froh+lich
    ("glücklich", "T"),      # Glück+lich
    ("natürlich", "T"),      # Natur+lich
    ("menschlich", "T"),     # Mensch+lich
    ("wahrscheinlich", "T"), # wahr+schein+lich
    ("unterschied", "T"),    # unter+schied
    ("gegensatz", "T"),      # gegen+Satz
    ("grundsatz", "T"),      # Grund+Satz
    ("nachteil", "T"),       # nach+Teil
    ("vorteil", "T"),        # vor+Teil
    ("urteil", "T"),         # ur+Teil
    ("frühling", "T"),       # früh+ling
    ("liebling", "T"),       # lieb+ling

    # ── OPAQUE (O): no visible parts for modern speaker ──
    # Semantic-shift words (parts no longer map to meaning)
    ("ding", "O"),           # Versammlung → Gegenstand
    ("knecht", "O"),         # Junge → Diener
    ("schlecht", "O"),       # schlicht → böse
    ("billig", "O"),         # angemessen → günstig
    ("geil", "O"),           # üppig → cool
    ("toll", "O"),           # verrückt → großartig
    ("elend", "O"),          # eli-lenti (Ausland) → miserable
    ("albern", "O"),         # alawari (friendly) → silly
    ("frech", "O"),          # tapfer → rude
    ("gemein", "O"),         # allgemein → mean
    ("schlimm", "O"),        # schief → bad

    # Latin/Greek borrowings (opaque in German)
    ("kommunikation", "O"),
    ("enthusiasmus", "O"),
    ("kandidat", "O"),
    ("sekretär", "O"),
    ("algorithmus", "O"),
    ("algebra", "O"),
    ("minister", "O"),
    ("kardinal", "O"),
    ("quarantäne", "O"),
    ("muskel", "O"),
    ("admiral", "O"),
    ("alkohol", "O"),
    ("katastrophe", "O"),
    ("sophistiziert", "O"),
    ("quintessenz", "O"),
    ("leutnant", "O"),
    ("hypothek", "O"),
    ("extravagant", "O"),
    ("exorbitant", "O"),
    ("alchemie", "O"),

    # More opaque borrowings from COHA
    ("politik", "O"),
    ("kultur", "O"),
    ("interesse", "O"),
    ("charakter", "O"),
    ("prinzip", "O"),
    ("methode", "O"),
    ("system", "O"),
    ("problem", "O"),
    ("moment", "O"),
    ("resultat", "O"),
    ("million", "O"),
    ("general", "O"),
    ("kapital", "O"),
    ("theater", "O"),
    ("professor", "O"),
    ("student", "O"),
    ("roman", "O"),
    ("literatur", "O"),
    ("revolution", "O"),
    ("religion", "O"),
    ("philosophie", "O"),
    ("kritik", "O"),
    ("republik", "O"),
    ("demokratie", "O"),
    ("aristokratie", "O"),
    ("bürokratie", "O"),
    ("monarchie", "O"),
    ("tyrannei", "O"),
    ("materie", "O"),
    ("energie", "O"),
    ("harmonie", "O"),
    ("melodie", "O"),
    ("symphonie", "O"),
    ("tragödie", "O"),
    ("phantasie", "O"),
    ("ironie", "O"),
    ("theorie", "O"),
    ("kategorie", "O"),
    ("nation", "O"),
    ("tradition", "O"),
    ("position", "O"),
    ("passion", "O"),
    ("dimension", "O"),
    ("situation", "O"),
    ("operation", "O"),
    ("generation", "O"),
    ("relation", "O"),
    ("reaktion", "O"),
    ("funktion", "O"),
    ("produktion", "O"),
    ("konstruktion", "O"),
    ("institution", "O"),
    ("konstitution", "O"),
]

# ═══════════════════════════════════════════════════════════
# 3. Compute Δ for all words
# ═══════════════════════════════════════════════════════════

print("\nComputing semantic change...")

results = {"P": [], "T": [], "O": []}
missing = []

seen = set()
for word, cat in german_words:
    if word in seen:
        continue
    seen.add(word)
    sc = semantic_change(word)
    if sc is not None:
        freq = zipf_frequency(word, 'de')
        results[cat].append({
            'word': word, 'delta': sc, 'freq': freq
        })
    else:
        missing.append((word, cat))

n_p = len(results["P"])
n_t = len(results["T"])
n_o = len(results["O"])
n_total = n_p + n_t + n_o

print(f"Matched: {n_total} words")
print(f"  Primes (P):      {n_p}")
print(f"  Transparent (T): {n_t}")
print(f"  Opaque (O):      {n_o}")
print(f"  Missing:         {len(missing)}")

# ═══════════════════════════════════════════════════════════
# 4. The Anchor Test
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("THE ANCHOR HYPOTHESIS")
print("Do transparent compounds change LESS than opaque words?")
print("=" * 70)

delta_p = np.array([r['delta'] for r in results["P"]])
delta_t = np.array([r['delta'] for r in results["T"]])
delta_o = np.array([r['delta'] for r in results["O"]])
freq_p = np.array([r['freq'] for r in results["P"]])
freq_t = np.array([r['freq'] for r in results["T"]])
freq_o = np.array([r['freq'] for r in results["O"]])

print(f"\n  {'Category':<20} {'n':>4}  {'mean Δ':>8}  {'std':>8}  {'mean freq':>10}")
print(f"  {'─'*20} {'─'*4}  {'─'*8}  {'─'*8}  {'─'*10}")
print(f"  {'Primes (P)':<20} {n_p:4d}  {delta_p.mean():8.4f}  {delta_p.std():8.4f}  {freq_p.mean():10.2f}")
print(f"  {'Transparent (T)':<20} {n_t:4d}  {delta_t.mean():8.4f}  {delta_t.std():8.4f}  {freq_t.mean():10.2f}")
print(f"  {'Opaque (O)':<20} {n_o:4d}  {delta_o.mean():8.4f}  {delta_o.std():8.4f}  {freq_o.mean():10.2f}")

# Key comparison: T vs O
t_stat, p_val = stats.ttest_ind(delta_t, delta_o, equal_var=False)
d_cohen = (delta_t.mean() - delta_o.mean()) / np.sqrt((delta_t.var() + delta_o.var()) / 2)
print(f"\n  T vs O (raw): t = {t_stat:.3f}, p = {p_val:.4f}, d = {d_cohen:.3f}")
if delta_t.mean() < delta_o.mean():
    print("  → Transparent compounds change LESS — ANCHORING CONFIRMED")
else:
    print("  → Transparent compounds change MORE — no anchoring")

# ═══════════════════════════════════════════════════════════
# 5. Frequency-controlled comparison
# ═══════════════════════════════════════════════════════════

print("\n" + "─" * 70)
print("FREQUENCY-CONTROLLED COMPARISON")
print("─" * 70)

# Combine T and O for regression
all_delta = np.concatenate([delta_t, delta_o])
all_freq = np.concatenate([freq_t, freq_o])
all_cat = np.array([1]*n_t + [0]*n_o)  # 1=transparent, 0=opaque

# Partial correlation: transparency → Δ, controlling freq
r_td = stats.pearsonr(all_cat, all_delta)[0]
r_tf = stats.pearsonr(all_cat, all_freq)[0]
r_df = stats.pearsonr(all_delta, all_freq)[0]
denom = np.sqrt((1 - r_tf**2) * (1 - r_df**2))
r_partial = (r_td - r_tf * r_df) / denom if denom > 1e-10 else 0

print(f"\n  r(transparency, Δ) = {r_td:.4f}")
print(f"  r(transparency, freq) = {r_tf:.4f}")
print(f"  r(Δ, freq) = {r_df:.4f}")
print(f"  r_partial(transparency, Δ | freq) = {r_partial:.4f}")

# Regression: Δ ~ freq + transparent
from numpy.linalg import lstsq
n_to = len(all_delta)

# Model 1: freq only
X1 = np.column_stack([np.ones(n_to), all_freq])
b1, _, _, _ = lstsq(X1, all_delta, rcond=None)
ss_res1 = np.sum((all_delta - X1 @ b1)**2)
ss_tot = np.sum((all_delta - all_delta.mean())**2)
r2_freq = 1 - ss_res1 / ss_tot

# Model 2: freq + transparency
X2 = np.column_stack([np.ones(n_to), all_freq, all_cat])
b2, _, _, _ = lstsq(X2, all_delta, rcond=None)
ss_res2 = np.sum((all_delta - X2 @ b2)**2)
r2_both = 1 - ss_res2 / ss_tot

delta_r2 = r2_both - r2_freq
df_extra = 1
df_resid = n_to - 3
f_stat = (delta_r2 / df_extra) / ((1 - r2_both) / df_resid)
p_f = 1 - stats.f.cdf(f_stat, df_extra, df_resid)

print(f"\n  R²(freq only) = {r2_freq:.4f}")
print(f"  R²(freq + transparency) = {r2_both:.4f}")
print(f"  ΔR² = {delta_r2:.4f}, F = {f_stat:.2f}, p = {p_f:.4f}")
print(f"  β(transparency) = {b2[2]:.4f}")

if b2[2] < 0 and p_f < 0.05:
    print("\n  ✓ ANCHORING CONFIRMED (after frequency control)")
    print("    Transparent compounds change significantly LESS")
elif b2[2] < 0:
    print(f"\n  ~ Anchoring direction correct but not significant (p = {p_f:.4f})")
else:
    print("\n  ✗ No anchoring effect")

# ═══════════════════════════════════════════════════════════
# 6. Three-group comparison (matches English analysis)
# ═══════════════════════════════════════════════════════════

print("\n" + "─" * 70)
print("THREE-GROUP ANOVA: P vs T vs O")
print("─" * 70)

f_anova, p_anova = stats.f_oneway(delta_p, delta_t, delta_o)
print(f"\n  F = {f_anova:.2f}, p = {p_anova:.4f}")

# Pairwise comparisons
pairs = [("P vs T", delta_p, delta_t), ("P vs O", delta_p, delta_o), ("T vs O", delta_t, delta_o)]
for label, a, b in pairs:
    t, p = stats.ttest_ind(a, b, equal_var=False)
    d = (b.mean() - a.mean()) / np.sqrt((a.var() + b.var()) / 2)
    print(f"  {label}: t = {t:.3f}, p = {p:.4f}, d = {d:.3f}")

# ═══════════════════════════════════════════════════════════
# 7. Comparison with English
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("CROSS-LINGUISTIC COMPARISON: ANCHOR EFFECT")
print("=" * 70)

print(f"""
  {'Measure':<40} {'English':>10} {'German':>10}
  {'─'*40} {'─'*10} {'─'*10}
  {'Primes mean Δ':<40} {'0.333':>10} {delta_p.mean():>10.3f}
  {'Transparent mean Δ':<40} {'0.496':>10} {delta_t.mean():>10.3f}
  {'Opaque mean Δ':<40} {'0.448':>10} {delta_o.mean():>10.3f}
  {'T vs O direction':<40} {'T > O':>10} {'T < O' if delta_t.mean() < delta_o.mean() else 'T > O':>10}
  {'T vs O d':<40} {'-0.386':>10} {d_cohen:>10.3f}

  English: transparency ENABLES reanalysis (T > O)
  German:  transparency ANCHORS meaning    (T < O) ?
""")

# ═══════════════════════════════════════════════════════════
# 8. Within-frequency-band test
# ═══════════════════════════════════════════════════════════

print("─" * 70)
print("WITHIN FREQUENCY BANDS")
print("─" * 70)

# Combine all
all_words_data = []
for cat_label, cat_results in results.items():
    for r in cat_results:
        all_words_data.append({**r, 'cat': cat_label})

all_freqs = np.array([w['freq'] for w in all_words_data])
all_deltas = np.array([w['delta'] for w in all_words_data])
all_cats = np.array([w['cat'] for w in all_words_data])

# Split into frequency terciles
terciles = np.percentile(all_freqs, [33.3, 66.7])
bands = [
    ("Low freq", all_freqs <= terciles[0]),
    ("Mid freq", (all_freqs > terciles[0]) & (all_freqs <= terciles[1])),
    ("High freq", all_freqs > terciles[1]),
]

print(f"\n  {'Band':<15} {'Cat':>5} {'n':>4} {'mean Δ':>8}")
print(f"  {'─'*15} {'─'*5} {'─'*4} {'─'*8}")
for band_name, band_mask in bands:
    for cat in ["P", "T", "O"]:
        mask = band_mask & (all_cats == cat)
        n = mask.sum()
        if n > 0:
            print(f"  {band_name:<15} {cat:>5} {n:4d} {all_deltas[mask].mean():8.4f}")
    print()

# ═══════════════════════════════════════════════════════════
# 9. Specific anchor test: matched pairs
# ═══════════════════════════════════════════════════════════

print("─" * 70)
print("MATCHED-PAIR ANCHOR TEST")
print("For each transparent compound, find closest-frequency opaque word")
print("─" * 70)

t_words = [(r['word'], r['delta'], r['freq']) for r in results["T"]]
o_words = [(r['word'], r['delta'], r['freq']) for r in results["O"]]

pairs_matched = []
used_o = set()
for tw, td, tf in t_words:
    best_match = None
    best_dist = float('inf')
    for i, (ow, od, of_) in enumerate(o_words):
        if i in used_o:
            continue
        dist = abs(tf - of_)
        if dist < best_dist:
            best_dist = dist
            best_match = (i, ow, od, of_)
    if best_match and best_dist < 1.0:  # within 1 Zipf
        i, ow, od, of_ = best_match
        used_o.add(i)
        pairs_matched.append((tw, td, tf, ow, od, of_))

print(f"\n  Matched pairs (within 1 Zipf): {len(pairs_matched)}")
if len(pairs_matched) >= 5:
    t_deltas = np.array([p[1] for p in pairs_matched])
    o_deltas = np.array([p[4] for p in pairs_matched])

    diff = t_deltas - o_deltas
    t_paired, p_paired = stats.ttest_rel(t_deltas, o_deltas)

    print(f"  Mean Δ(T) = {t_deltas.mean():.4f}")
    print(f"  Mean Δ(O) = {o_deltas.mean():.4f}")
    print(f"  Mean diff = {diff.mean():.4f}")
    print(f"  Paired t = {t_paired:.3f}, p = {p_paired:.4f}")

    if diff.mean() < 0:
        print(f"\n  ✓ ANCHORING: transparent compounds change {abs(diff.mean()):.4f} LESS")
    else:
        print(f"\n  ✗ No anchoring: transparent compounds change {diff.mean():.4f} MORE")

    # Show some pairs
    print(f"\n  {'Transparent':<25} {'Δ':>6} {'Opaque':<20} {'Δ':>6} {'diff':>7}")
    print(f"  {'─'*25} {'─'*6} {'─'*20} {'─'*6} {'─'*7}")
    for tw, td, tf, ow, od, of_ in sorted(pairs_matched, key=lambda p: p[1]-p[4])[:10]:
        print(f"  {tw:<25} {td:6.3f} {ow:<20} {od:6.3f} {td-od:+7.3f}")
    print("  ...")
    for tw, td, tf, ow, od, of_ in sorted(pairs_matched, key=lambda p: p[1]-p[4])[-5:]:
        print(f"  {tw:<25} {td:6.3f} {ow:<20} {od:6.3f} {td-od:+7.3f}")
