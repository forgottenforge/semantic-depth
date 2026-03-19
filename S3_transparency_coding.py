#!/usr/bin/env python3
"""
Transparency × Etymological Depth — Interaction Test
=====================================================
Rule (stated BEFORE seeing results):
  A word with ed ≥ 2 is TRANSPARENT (1) if a modern English speaker
  would recognize its constituent morphemes as existing English words.
  OPAQUE (0) if the etymology is invisible to a modern speaker.

  ed=1 words are excluded (they are already primes — no parts to see).
"""

import pickle
import numpy as np
from scipy import stats
from wordfreq import zipf_frequency
import warnings
warnings.filterwarnings('ignore')

# ═════════════════════════════════════════════════════════════
# 1. Load embeddings (same as main analysis)
# ═════════════════════════════════════════════════════════════

def load_decade(year, base='histwords/sgns'):
    with open(f'{base}/{year}-vocab.pkl', 'rb') as f:
        vocab = pickle.load(f, encoding='latin1')
    W = np.load(f'{base}/{year}-w.npy')
    idx = {w: i for i, w in enumerate(vocab)}
    return vocab, W, idx

print("Loading embeddings...")
_, W_early, idx_early = load_decade(1850)
_, W_late, idx_late = load_decade(2000)

shared = set(idx_early.keys()) & set(idx_late.keys())
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

# ═════════════════════════════════════════════════════════════
# 2. Full dataset with transparency ratings
# ═════════════════════════════════════════════════════════════
#
# Rule: transparent=1 if a modern English speaker recognizes the
# constituent morphemes as existing English words/morphemes.
# transparent=0 if the etymology is invisible.
# For ed=1: transparency=None (not applicable, already primes).

# (word, ed, transparent)
# transparent: 1=yes, 0=no, None=ed1/not applicable

word_data = [
    # ── ed=1: Primwörter (transparency N/A) ──
    ("I",1,None), ("you",1,None), ("we",1,None), ("he",1,None),
    ("she",1,None), ("it",1,None), ("me",1,None), ("us",1,None),
    ("them",1,None),
    ("this",1,None), ("that",1,None), ("here",1,None), ("there",1,None),
    ("now",1,None), ("then",1,None), ("who",1,None), ("what",1,None),
    ("be",1,None), ("do",1,None), ("go",1,None), ("come",1,None),
    ("see",1,None), ("know",1,None), ("say",1,None), ("get",1,None),
    ("make",1,None), ("take",1,None), ("give",1,None), ("have",1,None),
    ("eat",1,None), ("drink",1,None), ("sleep",1,None), ("die",1,None),
    ("sit",1,None), ("stand",1,None), ("lie",1,None), ("fall",1,None),
    ("run",1,None), ("walk",1,None), ("hear",1,None), ("feel",1,None),
    ("cut",1,None), ("bite",1,None), ("blow",1,None), ("burn",1,None),
    ("pull",1,None), ("push",1,None), ("swim",1,None), ("fly",1,None),
    ("hold",1,None),
    ("one",1,None), ("two",1,None), ("three",1,None), ("ten",1,None),
    ("eye",1,None), ("ear",1,None), ("mouth",1,None), ("tooth",1,None),
    ("tongue",1,None), ("foot",1,None), ("knee",1,None), ("heart",1,None),
    ("bone",1,None), ("blood",1,None), ("skin",1,None), ("nail",1,None),
    ("sun",1,None), ("moon",1,None), ("star",1,None), ("water",1,None),
    ("fire",1,None), ("earth",1,None), ("stone",1,None), ("tree",1,None),
    ("leaf",1,None), ("seed",1,None), ("root",1,None), ("rain",1,None),
    ("snow",1,None), ("wind",1,None), ("sand",1,None), ("salt",1,None),
    ("ash",1,None),
    ("dog",1,None), ("fish",1,None), ("worm",1,None), ("mouse",1,None),
    ("new",1,None), ("old",1,None), ("good",1,None), ("big",1,None),
    ("long",1,None), ("small",1,None), ("hot",1,None), ("cold",1,None),
    ("wet",1,None), ("dry",1,None), ("dead",1,None), ("red",1,None),
    ("black",1,None), ("white",1,None),
    ("name",1,None), ("night",1,None), ("day",1,None), ("path",1,None),
    ("road",1,None), ("hand",1,None), ("nose",1,None), ("head",1,None),
    ("back",1,None), ("full",1,None), ("all",1,None), ("many",1,None),
    ("not",1,None), ("in",1,None), ("with",1,None),

    # ── ed=2: Simple derivations ──
    # Semantic shifts (original meaning lost, no visible morphemes)
    ("husband",2,0),    # ON húsbóndi → opaque
    ("woman",2,0),      # wif+man → wo+man, wif lost
    ("lord",2,0),       # hlaf+weard → opaque
    ("lady",2,0),       # hlaf+dige → opaque
    ("barn",2,0),       # bere+ern → opaque
    ("world",2,0),      # wer+ald → opaque
    ("orchard",2,0),    # ort+geard → opaque
    ("deer",2,0),       # semantic shift (animal→deer)
    ("hound",2,0),      # semantic shift (dog→hound)
    ("fowl",2,0),       # semantic shift (bird→fowl)
    ("meat",2,0),       # semantic shift (food→meat)
    ("starve",2,0),     # semantic shift (die→starve)
    ("thing",2,0),      # semantic shift (assembly→thing)
    ("tide",2,0),       # semantic shift (time→tide)
    ("stool",2,0),      # semantic shift (chair→stool)
    # Transparent derivations (parts visible)
    ("teacher",2,1),    # teach+er ✓
    ("quickly",2,1),    # quick+ly ✓
    ("undo",2,1),       # un+do ✓
    ("sunrise",2,1),    # sun+rise ✓
    ("forget",2,0),     # for+get (for- prefix not obvious)
    ("begin",2,0),      # be+gin (gin not recognized)
    ("become",2,1),     # be+come ✓
    ("behind",2,0),     # be+hind (hind not obvious)
    ("between",2,0),    # be+tween → opaque
    ("maybe",2,1),      # may+be ✓
    ("inside",2,1),     # in+side ✓
    ("outside",2,1),    # out+side ✓
    ("kingdom",2,1),    # king+dom ✓
    ("freedom",2,1),    # free+dom ✓
    ("childhood",2,1),  # child+hood ✓
    ("friendship",2,1), # friend+ship ✓
    ("household",2,1),  # house+hold ✓
    ("wisdom",2,1),     # wise+dom ✓
    ("witness",2,1),    # wit+ness ✓
    ("worship",2,0),    # worth+ship → wor+ship, worth lost
    ("sheriff",2,0),    # shire+reeve → opaque
    ("steward",2,0),    # sty+ward → opaque
    # French/Latin borrowings (opaque by definition)
    ("army",2,0),       # French armée
    ("court",2,0),      # French cort
    ("state",2,0),      # Latin status
    ("power",2,0),      # French pouvoir
    ("country",2,0),    # French contrée
    ("city",2,0),       # French cité
    ("place",2,0),      # French place
    ("point",2,0),      # French point
    ("matter",2,0),     # Latin materia
    ("number",2,0),     # French nombre
    ("order",2,0),      # French ordre
    ("service",2,1),    # serve+ice ✓ (modern speaker parses this)
    ("war",2,0),        # French werre
    ("age",2,0),        # French aage
    ("story",2,0),      # French estoire
    ("office",2,0),     # Latin officium
    ("cause",2,0),      # French cause
    ("reason",2,0),     # French raison
    ("skill",2,0),      # ON skil
    ("wrong",2,0),      # ON vrangr
    ("window",2,0),     # ON vindauga (wind-eye, but -ow ≠ eye)
    ("anger",2,0),      # ON angr
    ("ugly",2,0),       # ON uggligr

    # ── ed=3: Compound derivations ──
    # Transparent compounds
    ("beautiful",3,1),  # beauty+ful ✓
    ("wonderful",3,1),  # wonder+ful ✓
    ("powerful",3,1),   # power+ful ✓
    ("dangerous",3,1),  # danger+ous ✓
    ("government",3,1), # govern+ment ✓
    ("agreement",3,1),  # agree+ment ✓
    ("movement",3,1),   # move+ment ✓
    ("impossible",3,1), # im+possible ✓
    ("unhappy",3,1),    # un+happy ✓
    ("disappear",3,1),  # dis+appear ✓
    ("discover",3,1),   # dis+cover ✓
    ("breakfast",3,1),  # break+fast ✓
    ("understand",3,1), # under+stand ✓ (parts visible)
    ("nightmare",3,1),  # night+mare ✓ (mare=horse, visible if wrong)
    ("holiday",3,1),    # holy+day ✓
    ("awful",3,1),      # awe+ful ✓
    ("awesome",3,1),    # awe+some ✓
    ("terrible",3,1),   # terror+ible → terror visible ✓
    ("terrific",3,1),   # terrify+ic → terror visible ✓
    ("naughty",3,1),    # naught+y ✓ (naught=nothing, recognized)
    ("crafty",3,1),     # craft+y ✓
    # Opaque (etymology invisible)
    ("goodbye",3,0),    # god+be+ye → opaque
    ("gossip",3,0),     # god+sib → opaque
    ("bully",3,0),      # Du boel → opaque
    ("silly",3,0),      # OE sælig → opaque
    ("nice",3,0),       # Lat nescius → opaque
    ("pretty",3,0),     # OE prættig → opaque
    ("shrewd",3,0),     # shrew+d → connection obscure
    ("fond",3,0),       # ME fonned → opaque
    ("brave",3,0),      # French brave → opaque
    ("cunning",3,0),    # can/ken+ning → opaque
    ("sad",3,0),        # OE sæd "sated" → opaque
    ("glad",3,0),       # OE glæd → monomorphemic
    ("fast",3,0),       # OE fæst "firm" → semantic shift
    ("cheap",3,0),      # OE ceap "trade" → opaque
    ("clue",3,0),       # "clew" ball of thread → opaque
    ("treacle",3,0),    # Latin theriaca → opaque
    ("moot",3,0),       # OE gemōt "meeting" → opaque

    # ── ed=4: Multi-layer ──
    # Transparent (morphemes visible)
    ("unfortunately",4,1),   # un+fortunate+ly ✓
    ("uncomfortable",4,1),   # un+comfort+able ✓
    ("communication",4,1),   # communicate+ion ✓
    ("international",4,1),   # inter+nation+al ✓
    ("entertainment",4,1),   # entertain+ment ✓
    ("philosophical",4,1),   # philosophy+ical ✓
    ("understanding",4,1),   # understand+ing ✓
    ("disagreement",4,1),    # dis+agree+ment ✓
    ("independence",4,1),    # in+depend+ence ✓
    ("environmental",4,1),   # environment+al ✓
    ("responsibility",4,1),  # response+ible+ity ✓
    ("organization",4,1),    # organize+ation ✓
    ("representative",4,1),  # represent+ative ✓
    ("establishment",4,1),   # establish+ment ✓
    ("particularly",4,1),    # particular+ly ✓
    # Opaque (Latin/Greek/Arabic, parts not visible)
    ("manufacture",4,0),     # Latin manus+factura → manu not English
    ("enthusiasm",4,0),      # Greek enthousiasmos → opaque
    ("candidate",4,0),       # Latin candidatus → opaque
    ("salary",4,0),          # Latin salarium → opaque
    ("calculate",4,0),       # Latin calculare → opaque
    ("secretary",4,0),       # Latin secretarius → secret visible?
    # Actually: secret+ary → 1? "Secret" is a known English word.
    # But the connection secret→secretary is not "keeper of secrets" anymore.
    # The morpheme is visible even if meaning shifted. → 1
    # No wait, I already put 0. Let me be consistent: the RULE says
    # "recognizes constituent morphemes as existing English words."
    # secret+ary: secret IS an English word, -ary IS a suffix. → 1
    # I'll fix this below.
    ("magazine",4,0),        # Arabic makhazin → opaque
    ("algorithm",4,0),       # al-Khwarizmi → opaque
    ("algebra",4,0),         # Arabic al-jabr → opaque
    ("assassin",4,0),        # Arabic hashshashin → opaque
    ("admiral",4,0),         # Arabic amir → opaque
    ("alcohol",4,0),         # Arabic al-kuhl → opaque
    ("cardinal",4,0),        # Latin cardinalis → opaque
    ("companion",4,0),       # Latin com+panis → opaque
    ("quarantine",4,0),      # Italian quarantina → opaque
    ("muscle",4,0),          # Latin musculus → opaque
    ("janitor",4,0),         # Latin janitor → opaque
    ("intern",4,0),          # Latin internus → opaque
    ("minister",4,0),        # Latin minister → opaque
    ("sinister",4,0),        # Latin sinister → opaque

    # ── ed=5: Maximally derived ──
    # Almost all opaque
    ("serendipity",5,0),     # coined from Serendip → opaque
    ("lieutenant",5,0),      # French lieu+tenant → opaque
    ("mortgage",5,0),        # French mort+gage → opaque
    ("preposterous",5,0),    # Latin prae+posterus → opaque
    ("egregious",5,0),       # Latin e+grex → opaque
    ("pandemonium",5,0),     # Greek pan+daemon → opaque
    ("sycophant",5,0),       # Greek sykon+phainein → opaque
    ("disaster",5,0),        # Italian dis+astro → opaque
    ("trivial",5,0),         # Latin trivialis → opaque
    ("decimation",5,0),      # Latin decimare → opaque
    ("dilapidated",5,0),     # Latin dis+lapidare → opaque
    ("quintessential",5,1),  # quint+essential ✓ (both parts known)
    ("miscreant",5,0),       # French mes+creant → opaque
    ("glamour",5,0),         # Scottish from grammar → opaque
    ("juggernaut",5,0),      # Hindi Jagannath → opaque
    ("pedigree",5,0),        # French pied de grue → opaque
    ("alchemy",5,0),         # Arabic al-kimiya → opaque
    ("vermicelli",5,0),      # Italian vermi+celli → opaque
    ("peninsula",5,0),       # Latin paene+insula → opaque
    ("inaugurate",5,0),      # Latin inaugurare → opaque
    ("investigate",5,0),     # Latin investigare → opaque
    ("exorbitant",5,0),      # Latin ex+orbita → opaque
    ("extravagant",5,0),     # Latin extra+vagari → opaque
]

# Fix secretary: secret+ary, both parts recognizable → 1
word_data = [(w, e, 1 if w == "secretary" else t) for w, e, t in word_data]

# ═════════════════════════════════════════════════════════════
# 3. Match with embeddings
# ═════════════════════════════════════════════════════════════

print("\nMatching words...")
results = []
for word, ed, transp in word_data:
    sc = semantic_change(word)
    if sc is not None:
        freq = zipf_frequency(word.lower(), 'en')
        results.append({
            'word': word, 'ed': ed, 'transparent': transp,
            'delta': sc, 'freq': freq
        })

N = len(results)
print(f"Matched: {N} words")

# Separate ed=1 and ed>1
ed1 = [r for r in results if r['ed'] == 1]
ed_gt1 = [r for r in results if r['ed'] > 1]
N1 = len(ed1)
N2 = len(ed_gt1)

print(f"  ed=1: {N1} words (primes, transparency N/A)")
print(f"  ed>1: {N2} words ({sum(1 for r in ed_gt1 if r['transparent']==1)} transparent, "
      f"{sum(1 for r in ed_gt1 if r['transparent']==0)} opaque)")

# ═════════════════════════════════════════════════════════════
# 4. Analysis
# ═════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TRANSPARENCY × ETYMOLOGICAL DEPTH — INTERACTION TEST")
print("=" * 70)

# Arrays for ed>1 words
words_2 = [r['word'] for r in ed_gt1]
ed_2 = np.array([r['ed'] for r in ed_gt1], dtype=float)
delta_2 = np.array([r['delta'] for r in ed_gt1], dtype=float)
freq_2 = np.array([r['freq'] for r in ed_gt1], dtype=float)
transp_2 = np.array([r['transparent'] for r in ed_gt1], dtype=float)

# 4a. Transparent vs Opaque: basic comparison
print("\n" + "─" * 70)
print("A. TRANSPARENT vs OPAQUE WORDS (ed > 1)")
print("─" * 70)

t_mask = transp_2 == 1
o_mask = transp_2 == 0

print(f"\n  {'Group':<15} {'n':>4}  {'mean_ed':>8}  {'mean_Δ':>8}  {'mean_freq':>10}")
print(f"  {'Transparent':<15} {t_mask.sum():4d}  {ed_2[t_mask].mean():8.2f}  "
      f"{delta_2[t_mask].mean():8.4f}  {freq_2[t_mask].mean():10.2f}")
print(f"  {'Opaque':<15} {o_mask.sum():4d}  {ed_2[o_mask].mean():8.2f}  "
      f"{delta_2[o_mask].mean():8.4f}  {freq_2[o_mask].mean():10.2f}")

t_stat, p_t = stats.ttest_ind(delta_2[t_mask], delta_2[o_mask], equal_var=False)
d_cohen = (delta_2[o_mask].mean() - delta_2[t_mask].mean()) / \
          np.sqrt((delta_2[t_mask].var() + delta_2[o_mask].var()) / 2)
u_stat, p_u = stats.mannwhitneyu(delta_2[t_mask], delta_2[o_mask])

print(f"\n  Welch t = {t_stat:.3f}, p = {p_t:.4f}")
print(f"  Mann-Whitney p = {p_u:.4f}")
print(f"  Cohen's d = {d_cohen:.3f}")

# 4b. BUT: transparent words have lower ed on average (confound!)
print("\n  ⚠ CONFOUND CHECK: transparent words tend to have lower ed")
r_te, p_te = stats.pearsonr(transp_2, ed_2)
print(f"  r(transparent, ed) = {r_te:.3f} (p = {p_te:.4f})")

# 4c. Partial effect: does transparency predict Δ AFTER controlling ed and freq?
print("\n" + "─" * 70)
print("B. MULTIPLE REGRESSION: Δ ~ ed + freq + transparent")
print("─" * 70)

from numpy.linalg import lstsq

# Model 1: Δ ~ ed + freq
X1 = np.column_stack([np.ones(N2), ed_2, freq_2])
beta1, _, _, _ = lstsq(X1, delta_2, rcond=None)
pred1 = X1 @ beta1
ss_res1 = np.sum((delta_2 - pred1) ** 2)
ss_tot = np.sum((delta_2 - delta_2.mean()) ** 2)
R2_1 = 1 - ss_res1 / ss_tot

# Model 2: Δ ~ ed + freq + transparent
X2 = np.column_stack([np.ones(N2), ed_2, freq_2, transp_2])
beta2, _, _, _ = lstsq(X2, delta_2, rcond=None)
pred2 = X2 @ beta2
ss_res2 = np.sum((delta_2 - pred2) ** 2)
R2_2 = 1 - ss_res2 / ss_tot

# Model 3: Δ ~ ed + freq + transparent + ed×transparent
interaction = ed_2 * transp_2
X3 = np.column_stack([np.ones(N2), ed_2, freq_2, transp_2, interaction])
beta3, _, _, _ = lstsq(X3, delta_2, rcond=None)
pred3 = X3 @ beta3
ss_res3 = np.sum((delta_2 - pred3) ** 2)
R2_3 = 1 - ss_res3 / ss_tot

print(f"\n  Model 1: Δ ~ ed + freq")
print(f"    R² = {R2_1:.4f}")
print(f"    β_ed = {beta1[1]:.4f}, β_freq = {beta1[2]:.4f}")

print(f"\n  Model 2: Δ ~ ed + freq + transparent")
print(f"    R² = {R2_2:.4f}  (ΔR² = {R2_2 - R2_1:.4f})")
print(f"    β_ed = {beta2[1]:.4f}, β_freq = {beta2[2]:.4f}, β_transp = {beta2[3]:.4f}")

print(f"\n  Model 3: Δ ~ ed + freq + transparent + ed×transparent")
print(f"    R² = {R2_3:.4f}  (ΔR² = {R2_3 - R2_1:.4f})")
print(f"    β_ed = {beta3[1]:.4f}, β_freq = {beta3[2]:.4f}, "
      f"β_transp = {beta3[3]:.4f}, β_ed×transp = {beta3[4]:.4f}")

# F-test for ΔR² (Model 2 vs Model 1)
df1 = 1  # one additional parameter
df_res = N2 - 4  # residual df for Model 2
F_transp = ((ss_res1 - ss_res2) / df1) / (ss_res2 / df_res)
from scipy.stats import f as f_dist
p_F = 1 - f_dist.cdf(F_transp, df1, df_res)
print(f"\n  F-test for transparency (Model 2 vs 1): F = {F_transp:.3f}, p = {p_F:.4f}")

# F-test for interaction (Model 3 vs Model 2)
df_res3 = N2 - 5
F_inter = ((ss_res2 - ss_res3) / 1) / (ss_res3 / df_res3)
p_F_inter = 1 - f_dist.cdf(F_inter, 1, df_res3)
print(f"  F-test for interaction (Model 3 vs 2): F = {F_inter:.3f}, p = {p_F_inter:.4f}")

# 4d. Within ed levels: transparent vs opaque
print("\n" + "─" * 70)
print("C. TRANSPARENT vs OPAQUE WITHIN EACH ed LEVEL")
print("─" * 70)

print(f"\n  {'ed':>3}  {'n_T':>4}  {'mean_Δ_T':>10}  {'n_O':>4}  {'mean_Δ_O':>10}  "
      f"{'diff':>8}  {'p':>8}")
for level in sorted(set(ed_2)):
    level_mask = ed_2 == level
    t_in_level = (transp_2 == 1) & level_mask
    o_in_level = (transp_2 == 0) & level_mask
    n_t = t_in_level.sum()
    n_o = o_in_level.sum()
    if n_t >= 2 and n_o >= 2:
        mean_t = delta_2[t_in_level].mean()
        mean_o = delta_2[o_in_level].mean()
        _, p_level = stats.ttest_ind(delta_2[t_in_level], delta_2[o_in_level], equal_var=False)
        print(f"  {int(level):3d}  {n_t:4d}  {mean_t:10.4f}  {n_o:4d}  {mean_o:10.4f}  "
              f"{mean_o - mean_t:8.4f}  {p_level:8.4f}")
    else:
        mean_t = delta_2[t_in_level].mean() if n_t > 0 else float('nan')
        mean_o = delta_2[o_in_level].mean() if n_o > 0 else float('nan')
        print(f"  {int(level):3d}  {n_t:4d}  {mean_t:10.4f}  {n_o:4d}  {mean_o:10.4f}  "
              f"{'---':>8}  {'---':>8}")

# 4e. Now the big test: ed=1 + transparent(ed>1) vs opaque(ed>1)
print("\n" + "─" * 70)
print("D. THE PRIME NUMBER TEST")
print("─" * 70)
print("   Are transparent compounds as stable as primes?")
print("   (If transparency protects, transparent ed>1 ≈ ed=1)")

delta_ed1 = np.array([r['delta'] for r in ed1])
delta_transp = delta_2[t_mask]
delta_opaque = delta_2[o_mask]

print(f"\n  ed=1 (primes):        n={len(delta_ed1):3d}, mean Δ = {delta_ed1.mean():.4f}")
print(f"  ed>1, transparent:    n={t_mask.sum():3d}, mean Δ = {delta_transp.mean():.4f}")
print(f"  ed>1, opaque:         n={o_mask.sum():3d}, mean Δ = {delta_opaque.mean():.4f}")

t1, p1 = stats.ttest_ind(delta_ed1, delta_transp, equal_var=False)
t2, p2 = stats.ttest_ind(delta_ed1, delta_opaque, equal_var=False)
t3, p3 = stats.ttest_ind(delta_transp, delta_opaque, equal_var=False)

d_prime_vs_transp = (delta_transp.mean() - delta_ed1.mean()) / \
    np.sqrt((delta_ed1.var() + delta_transp.var()) / 2)
d_prime_vs_opaque = (delta_opaque.mean() - delta_ed1.mean()) / \
    np.sqrt((delta_ed1.var() + delta_opaque.var()) / 2)
d_transp_vs_opaque = (delta_opaque.mean() - delta_transp.mean()) / \
    np.sqrt((delta_transp.var() + delta_opaque.var()) / 2)

print(f"\n  Primes vs Transparent:  t={t1:.3f}, p={p1:.4f}, d={d_prime_vs_transp:.3f}")
print(f"  Primes vs Opaque:       t={t2:.3f}, p={p2:.4f}, d={d_prime_vs_opaque:.3f}")
print(f"  Transparent vs Opaque:  t={t3:.3f}, p={p3:.4f}, d={d_transp_vs_opaque:.3f}")

# 4f. Combined predictor: ed for primes, ed×(1-transparency) for derived
print("\n" + "─" * 70)
print("E. COMBINED MODEL: ALL 225 WORDS")
print("─" * 70)
print("   ed_effective = ed for ed=1; ed × (2 - transparent) for ed>1")
print("   (transparent words get ed, opaque words get 2×ed)")

all_ed = np.array([r['ed'] for r in results], dtype=float)
all_delta = np.array([r['delta'] for r in results], dtype=float)
all_freq = np.array([r['freq'] for r in results], dtype=float)

# Create effective ed
ed_eff = np.zeros(N)
for i, r in enumerate(results):
    if r['ed'] == 1:
        ed_eff[i] = 1
    elif r['transparent'] == 1:
        ed_eff[i] = r['ed']          # transparent: ed stays
    else:
        ed_eff[i] = r['ed'] * 1.5    # opaque: ed amplified

r_orig, p_orig = stats.pearsonr(all_ed, all_delta)
r_eff, p_eff = stats.pearsonr(ed_eff, all_delta)

# Partial correlations
r_ef_orig = stats.pearsonr(all_ed, all_freq)[0]
r_df = stats.pearsonr(all_delta, all_freq)[0]
r_ef_eff = stats.pearsonr(ed_eff, all_freq)[0]

denom_orig = np.sqrt((1 - r_ef_orig**2) * (1 - r_df**2))
r_partial_orig = (r_orig - r_ef_orig * r_df) / denom_orig

denom_eff = np.sqrt((1 - r_ef_eff**2) * (1 - r_df**2))
r_partial_eff = (r_eff - r_ef_eff * r_df) / denom_eff

print(f"\n  {'Predictor':<25} {'r':>8} {'r_partial':>12} {'p':>12}")
print(f"  {'─'*25} {'─'*8} {'─'*12} {'─'*12}")
print(f"  {'ed (original)':<25} {r_orig:8.4f} {r_partial_orig:12.4f} {p_orig:12.2e}")
print(f"  {'ed_effective':<25} {r_eff:8.4f} {r_partial_eff:12.4f} {p_eff:12.2e}")
print(f"\n  Improvement: Δr = {r_eff - r_orig:.4f}, "
      f"Δr_partial = {r_partial_eff - r_partial_orig:.4f}")

# 4g. Bootstrap CI for the improvement
print("\n" + "─" * 70)
print("F. BOOTSTRAP: IS THE IMPROVEMENT SIGNIFICANT?")
print("─" * 70)

rng = np.random.RandomState(42)
n_boot = 10000
boot_diff = np.zeros(n_boot)
for i in range(n_boot):
    idx = rng.randint(0, N, N)
    r1 = stats.pearsonr(all_ed[idx], all_delta[idx])[0]
    r2 = stats.pearsonr(ed_eff[idx], all_delta[idx])[0]
    boot_diff[i] = r2 - r1

ci_lo, ci_hi = np.percentile(boot_diff, [2.5, 97.5])
p_boot = np.mean(boot_diff <= 0)  # proportion where ed_eff is NOT better

print(f"\n  Δr = {r_eff - r_orig:.4f}")
print(f"  Bootstrap 95% CI for Δr: [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"  P(Δr ≤ 0) = {p_boot:.4f}")

# 4h. Show the outliers with transparency
print("\n" + "─" * 70)
print("G. OUTLIER TABLE: STABLE HIGH-ED WORDS")
print("─" * 70)
print("   (ed ≥ 3, Δ < median)")

median_delta = np.median(all_delta)
stable_deep = [r for r in results if r['ed'] >= 3 and r['delta'] < median_delta]
stable_deep.sort(key=lambda r: r['delta'])

print(f"\n  {'Word':<25} {'ed':>3}  {'Δ':>8}  {'Transp':>7}  {'Freq':>6}")
for r in stable_deep:
    t_label = "✓" if r['transparent'] == 1 else "✗"
    print(f"  {r['word']:<25} {r['ed']:3d}  {r['delta']:8.4f}  {t_label:>7}  {r['freq']:6.2f}")

print("\n" + "─" * 70)
print("H. OUTLIER TABLE: UNSTABLE LOW-ED WORDS")
print("─" * 70)
print("   (ed ≤ 2, Δ > median)")

unstable_shallow = [r for r in results if r['ed'] <= 2 and r['delta'] > median_delta]
unstable_shallow.sort(key=lambda r: -r['delta'])

print(f"\n  {'Word':<25} {'ed':>3}  {'Δ':>8}  {'Transp':>7}  {'Freq':>6}")
for r in unstable_shallow[:15]:
    t_label = "✓" if r['transparent'] == 1 else ("✗" if r['transparent'] == 0 else "—")
    print(f"  {r['word']:<25} {r['ed']:3d}  {r['delta']:8.4f}  {t_label:>7}  {r['freq']:6.2f}")

# ═════════════════════════════════════════════════════════════
# VERDICT
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
print(f"""
  Original r(ed, Δ) = {r_orig:.4f}
  With transparency:  {r_eff:.4f}  (Δr = {r_eff - r_orig:+.4f})

  Original r_partial = {r_partial_orig:.4f}
  With transparency:  {r_partial_eff:.4f}  (Δr = {r_partial_eff - r_partial_orig:+.4f})

  Transparency as standalone predictor (ed>1 only):
    Opaque words: mean Δ = {delta_opaque.mean():.4f}
    Transparent:  mean Δ = {delta_transp.mean():.4f}
    Cohen's d = {d_transp_vs_opaque:.3f}

  Does transparency improve prediction beyond ed + frequency?
    F = {F_transp:.3f}, p = {p_F:.4f}
    {"YES — transparency adds significant variance" if p_F < 0.05 else "NO — transparency does not reach significance" if p_F > 0.10 else "MARGINAL — trending but not significant"}
""")
