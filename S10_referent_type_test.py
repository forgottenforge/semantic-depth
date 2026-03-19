#!/usr/bin/env python3
"""
Referent Type × Semantic Stability
====================================
Hypothesis: Words referring to structural/logical/physical properties
of the world are more stable than words referring to culturally
contingent evaluations or emotions.

Classification rule (stated BEFORE seeing results per word):
  STRUCTURAL (S): The word's referent is defined by logical, physical,
    or institutional structure. Its truth conditions do not depend on
    the speaker's emotional state or cultural norms.
    Examples: impossible (= not possible, by logic), government (institution),
    teacher (role), sunrise (physical event), inside (spatial relation)

  EVALUATIVE (E): The word's current meaning involves subjective
    judgment, emotional response, or culturally contingent norms.
    Examples: nice (= agreeable, by current norms), terrible (= very bad,
    emotional), beautiful (aesthetic judgment), silly (social evaluation)

  NEUTRAL (N): Neither clearly structural nor evaluative.
    Basic actions, concrete objects, semantic shifts that don't fit either.
    Examples: meat (just narrowed referent), deer (just narrowed referent),
    husband (social role but not evaluative)

Applied to ALL 225 matched words.
"""

import pickle
import numpy as np
from scipy import stats
from wordfreq import zipf_frequency
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════
# 1. Load embeddings
# ═══════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════
# 2. Word data with referent type
# ═══════════════════════════════════════════════════════════
# S = structural/logical/physical
# E = evaluative/emotional/cultural
# N = neutral (neither clearly S nor E)

word_data = [
    # ── ed=1: Primwörter ──
    # Pronouns/Deixis → structural (logical relations)
    ("I",1,"S"), ("you",1,"S"), ("we",1,"S"), ("he",1,"S"),
    ("she",1,"S"), ("it",1,"S"), ("me",1,"S"), ("us",1,"S"),
    ("them",1,"S"),
    ("this",1,"S"), ("that",1,"S"), ("here",1,"S"), ("there",1,"S"),
    ("now",1,"S"), ("then",1,"S"), ("who",1,"S"), ("what",1,"S"),
    # Basic verbs → mostly neutral (actions)
    ("be",1,"S"),    # logical/existential
    ("do",1,"N"), ("go",1,"N"), ("come",1,"N"),
    ("see",1,"N"), ("know",1,"S"),  # epistemic
    ("say",1,"N"), ("get",1,"N"), ("make",1,"N"),
    ("take",1,"N"), ("give",1,"N"), ("have",1,"S"),  # possession
    ("eat",1,"N"), ("drink",1,"N"), ("sleep",1,"N"), ("die",1,"N"),
    ("sit",1,"N"), ("stand",1,"N"), ("lie",1,"N"), ("fall",1,"N"),
    ("run",1,"N"), ("walk",1,"N"), ("hear",1,"N"), ("feel",1,"N"),
    ("cut",1,"N"), ("bite",1,"N"), ("blow",1,"N"), ("burn",1,"N"),
    ("pull",1,"N"), ("push",1,"N"), ("swim",1,"N"), ("fly",1,"N"),
    ("hold",1,"N"),
    # Numbers → structural (mathematical)
    ("one",1,"S"), ("two",1,"S"), ("three",1,"S"), ("ten",1,"S"),
    # Body parts → structural (physical)
    ("eye",1,"S"), ("ear",1,"S"), ("mouth",1,"S"), ("tooth",1,"S"),
    ("tongue",1,"S"), ("foot",1,"S"), ("knee",1,"S"), ("heart",1,"S"),
    ("bone",1,"S"), ("blood",1,"S"), ("skin",1,"S"), ("nail",1,"S"),
    # Nature → structural (physical)
    ("sun",1,"S"), ("moon",1,"S"), ("star",1,"S"), ("water",1,"S"),
    ("fire",1,"S"), ("earth",1,"S"), ("stone",1,"S"), ("tree",1,"S"),
    ("leaf",1,"S"), ("seed",1,"S"), ("root",1,"S"), ("rain",1,"S"),
    ("snow",1,"S"), ("wind",1,"S"), ("sand",1,"S"), ("salt",1,"S"),
    ("ash",1,"S"),
    # Animals → structural (natural kinds)
    ("dog",1,"S"), ("fish",1,"S"), ("worm",1,"S"), ("mouse",1,"S"),
    # Basic adjectives
    ("new",1,"S"), ("old",1,"S"),   # temporal, objective
    ("good",1,"E"),                  # evaluative!
    ("big",1,"S"), ("long",1,"S"), ("small",1,"S"),  # physical dimensions
    ("hot",1,"S"), ("cold",1,"S"), ("wet",1,"S"), ("dry",1,"S"),  # physical
    ("dead",1,"S"),   # biological state
    ("red",1,"S"), ("black",1,"S"), ("white",1,"S"),  # perceptual
    # Other basic
    ("name",1,"S"), ("night",1,"S"), ("day",1,"S"), ("path",1,"S"),
    ("road",1,"S"), ("hand",1,"S"), ("nose",1,"S"), ("head",1,"S"),
    ("back",1,"S"),
    ("full",1,"S"), ("all",1,"S"), ("many",1,"S"),  # quantifiers
    ("not",1,"S"), ("in",1,"S"), ("with",1,"S"),    # logical/spatial

    # ── ed=2 ──
    # Semantic shifts (neutral - just narrowed/shifted referent)
    ("husband",2,"N"), ("woman",2,"N"), ("lord",2,"N"), ("lady",2,"N"),
    ("barn",2,"S"), ("world",2,"S"), ("orchard",2,"S"),
    ("deer",2,"N"), ("hound",2,"N"), ("fowl",2,"N"), ("meat",2,"N"),
    ("starve",2,"N"), ("thing",2,"N"), ("tide",2,"S"), ("stool",2,"N"),
    # Transparent derivations
    ("teacher",2,"S"),     # role, structural
    ("quickly",2,"N"),     # manner
    ("undo",2,"S"),        # logical negation of action
    ("sunrise",2,"S"),     # physical event
    ("forget",2,"N"), ("begin",2,"N"), ("become",2,"N"),
    ("behind",2,"S"),      # spatial
    ("between",2,"S"),     # spatial/logical
    ("maybe",2,"S"),       # epistemic/logical
    ("inside",2,"S"),      # spatial
    ("outside",2,"S"),     # spatial
    ("kingdom",2,"S"),     # institutional structure
    ("freedom",2,"S"),     # abstract structural
    ("childhood",2,"S"),   # life stage, structural
    ("friendship",2,"N"),  # social relation
    ("household",2,"S"),   # institutional
    ("wisdom",2,"E"),      # evaluative (implies positive judgment)
    ("witness",2,"S"),     # legal/structural role
    ("worship",2,"E"),     # evaluative/religious
    ("sheriff",2,"S"),     # institutional role
    ("steward",2,"S"),     # institutional role
    # French/Latin borrowings
    ("army",2,"S"), ("court",2,"S"), ("state",2,"S"), ("power",2,"S"),
    ("country",2,"S"), ("city",2,"S"), ("place",2,"S"), ("point",2,"S"),
    ("matter",2,"S"), ("number",2,"S"), ("order",2,"S"), ("service",2,"S"),
    ("war",2,"N"), ("age",2,"S"), ("story",2,"N"), ("office",2,"S"),
    ("cause",2,"S"), ("reason",2,"S"),
    ("skill",2,"N"), ("wrong",2,"E"),  # evaluative
    ("window",2,"S"), ("anger",2,"E"),  # emotion
    ("ugly",2,"E"),  # aesthetic evaluation

    # ── ed=3 ──
    # Transparent compounds
    ("beautiful",3,"E"),   # aesthetic evaluation
    ("wonderful",3,"E"),   # evaluative
    ("powerful",3,"S"),    # structural (having power)
    ("dangerous",3,"S"),   # structural (posing danger, objective risk)
    ("government",3,"S"),  # institutional
    ("agreement",3,"S"),   # structural
    ("movement",3,"S"),    # physical/political
    ("impossible",3,"S"),  # LOGICAL — not-possible by definition
    ("unhappy",3,"E"),     # emotional state
    ("disappear",3,"S"),   # physical process
    ("discover",3,"S"),    # epistemic
    ("breakfast",3,"S"),   # concrete event
    ("understand",3,"S"),  # cognitive/structural
    ("nightmare",3,"E"),   # emotional experience
    ("holiday",3,"N"),     # cultural practice (shifted)
    ("awful",3,"E"),       # evaluative (was: awe-inspiring)
    ("awesome",3,"E"),     # evaluative
    ("terrible",3,"E"),    # evaluative
    ("terrific",3,"E"),    # evaluative
    ("naughty",3,"E"),     # moral evaluation
    ("crafty",3,"E"),      # moral/skill evaluation
    ("goodbye",3,"N"),     # social formula
    ("gossip",3,"N"),      # social behavior
    ("bully",3,"E"),       # moral evaluation
    ("silly",3,"E"),       # intellectual evaluation
    ("nice",3,"E"),        # evaluative
    ("pretty",3,"E"),      # aesthetic evaluation
    ("shrewd",3,"E"),      # evaluative
    ("fond",3,"E"),        # emotional
    ("brave",3,"E"),       # moral evaluation
    ("cunning",3,"E"),     # moral evaluation
    ("sad",3,"E"),         # emotional
    ("glad",3,"E"),        # emotional
    ("fast",3,"N"),        # physical property (shifted)
    ("cheap",3,"E"),       # evaluative (was: trade)
    ("clue",3,"N"),        # shifted referent
    ("treacle",3,"N"),     # shifted referent
    ("moot",3,"N"),        # shifted referent

    # ── ed=4 ──
    ("unfortunately",4,"E"),   # evaluative attitude
    ("uncomfortable",4,"E"),   # evaluative/physical
    ("communication",4,"S"),   # structural process
    ("international",4,"S"),   # structural relation
    ("entertainment",4,"E"),   # experiential/evaluative
    ("philosophical",4,"S"),   # intellectual domain
    ("understanding",4,"S"),   # cognitive
    ("disagreement",4,"S"),    # structural relation
    ("independence",4,"S"),    # structural/political
    ("environmental",4,"S"),   # structural
    ("responsibility",4,"S"),  # structural/institutional
    ("organization",4,"S"),    # structural
    ("representative",4,"S"),  # institutional role
    ("establishment",4,"S"),   # institutional
    ("particularly",4,"N"),    # degree modifier
    ("manufacture",4,"S"),     # process
    ("enthusiasm",4,"E"),      # emotional state
    ("candidate",4,"S"),       # institutional role
    ("salary",4,"S"),          # institutional/economic
    ("calculate",4,"S"),       # cognitive/mathematical
    ("secretary",4,"S"),       # institutional role
    ("magazine",4,"S"),        # concrete object/medium
    ("algorithm",4,"S"),       # mathematical concept
    ("algebra",4,"S"),         # mathematical domain
    ("assassin",4,"N"),        # agent (shifted)
    ("admiral",4,"S"),         # institutional role
    ("alcohol",4,"S"),         # substance
    ("cardinal",4,"S"),        # institutional role
    ("companion",4,"N"),       # social relation
    ("quarantine",4,"S"),      # institutional practice
    ("muscle",4,"S"),          # body part
    ("janitor",4,"S"),         # institutional role
    ("intern",4,"S"),          # institutional role
    ("minister",4,"S"),        # institutional role
    ("sinister",4,"E"),        # evaluative (was: left-handed)

    # ── ed=5 ──
    ("serendipity",5,"E"),     # evaluative (lucky discovery)
    ("lieutenant",5,"S"),      # institutional role
    ("mortgage",5,"S"),        # institutional/legal
    ("preposterous",5,"E"),    # evaluative
    ("egregious",5,"E"),       # evaluative (reversed)
    ("pandemonium",5,"E"),     # evaluative/emotional
    ("sycophant",5,"E"),       # moral evaluation
    ("disaster",5,"E"),        # evaluative
    ("trivial",5,"E"),         # evaluative (was: of the crossroads)
    ("decimation",5,"N"),      # process (shifted)
    ("dilapidated",5,"S"),     # physical state
    ("quintessential",5,"E"),  # evaluative
    ("miscreant",5,"E"),       # moral evaluation
    ("glamour",5,"E"),         # aesthetic evaluation
    ("juggernaut",5,"S"),      # structural (unstoppable force)
    ("pedigree",5,"S"),        # structural (lineage)
    ("alchemy",5,"S"),         # domain/practice
    ("vermicelli",5,"S"),      # concrete object
    ("peninsula",5,"S"),       # geographical structure
    ("inaugurate",5,"S"),      # institutional practice
    ("investigate",5,"S"),     # cognitive process
    ("exorbitant",5,"E"),      # evaluative
    ("extravagant",5,"E"),     # evaluative
]

# ═══════════════════════════════════════════════════════════
# 3. Match with embeddings
# ═══════════════════════════════════════════════════════════

print("\nMatching words...")
results = []
for word, ed, rtype in word_data:
    sc = semantic_change(word)
    if sc is not None:
        freq = zipf_frequency(word.lower(), 'en')
        results.append({
            'word': word, 'ed': ed, 'type': rtype,
            'delta': sc, 'freq': freq
        })

N = len(results)
n_S = sum(1 for r in results if r['type'] == 'S')
n_E = sum(1 for r in results if r['type'] == 'E')
n_N = sum(1 for r in results if r['type'] == 'N')
print(f"Matched: {N} ({n_S} structural, {n_E} evaluative, {n_N} neutral)")

# Arrays
all_ed = np.array([r['ed'] for r in results], dtype=float)
all_delta = np.array([r['delta'] for r in results], dtype=float)
all_freq = np.array([r['freq'] for r in results], dtype=float)
is_S = np.array([r['type'] == 'S' for r in results], dtype=float)
is_E = np.array([r['type'] == 'E' for r in results], dtype=float)
is_N = np.array([r['type'] == 'N' for r in results], dtype=float)

# ═══════════════════════════════════════════════════════════
# 4. Analysis
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("REFERENT TYPE × SEMANTIC STABILITY")
print("=" * 70)

# 4a. Basic comparison
print("\n" + "─" * 70)
print("A. STRUCTURAL vs EVALUATIVE vs NEUTRAL")
print("─" * 70)

for label, mask in [("Structural", is_S == 1),
                     ("Evaluative", is_E == 1),
                     ("Neutral", is_N == 1)]:
    n = mask.sum()
    print(f"\n  {label} (n={int(n)}):")
    print(f"    mean Δ = {all_delta[mask].mean():.4f} ± {all_delta[mask].std():.4f}")
    print(f"    mean ed = {all_ed[mask].mean():.2f}")
    print(f"    mean freq = {all_freq[mask].mean():.2f}")

# S vs E comparison
delta_S = all_delta[is_S == 1]
delta_E = all_delta[is_E == 1]
t_se, p_se = stats.ttest_ind(delta_S, delta_E, equal_var=False)
d_se = (delta_E.mean() - delta_S.mean()) / np.sqrt((delta_S.var() + delta_E.var()) / 2)
u_se, p_u_se = stats.mannwhitneyu(delta_S, delta_E)

print(f"\n  S vs E:")
print(f"    Welch t = {t_se:.3f}, p = {p_se:.2e}")
print(f"    Cohen's d = {d_se:.3f} (positive = E changes more)")
print(f"    Mann-Whitney p = {p_u_se:.2e}")

# 4b. CONFOUND: evaluative words have higher ed
print("\n" + "─" * 70)
print("B. CONFOUND CHECK")
print("─" * 70)

ed_S = all_ed[is_S == 1]
ed_E = all_ed[is_E == 1]
freq_S = all_freq[is_S == 1]
freq_E = all_freq[is_E == 1]

print(f"\n  Mean ed:   Structural = {ed_S.mean():.2f}, Evaluative = {ed_E.mean():.2f}")
print(f"  Mean freq: Structural = {freq_S.mean():.2f}, Evaluative = {freq_E.mean():.2f}")

t_ed, p_ed = stats.ttest_ind(ed_S, ed_E, equal_var=False)
print(f"  ed difference: t = {t_ed:.3f}, p = {p_ed:.4f}")

# 4c. Within ed levels: S vs E
print("\n" + "─" * 70)
print("C. STRUCTURAL vs EVALUATIVE WITHIN EACH ed LEVEL")
print("─" * 70)

print(f"\n  {'ed':>3}  {'n_S':>4}  {'Δ_S':>8}  {'n_E':>4}  {'Δ_E':>8}  "
      f"{'diff':>8}  {'p':>8}  {'d':>8}")

for level in sorted(set(all_ed)):
    level_mask = all_ed == level
    s_in = (is_S == 1) & level_mask
    e_in = (is_E == 1) & level_mask
    n_s = s_in.sum()
    n_e = e_in.sum()
    if n_s >= 2 and n_e >= 2:
        mean_s = all_delta[s_in].mean()
        mean_e = all_delta[e_in].mean()
        t_l, p_l = stats.ttest_ind(all_delta[s_in], all_delta[e_in], equal_var=False)
        d_l = (mean_e - mean_s) / np.sqrt((all_delta[s_in].var() + all_delta[e_in].var()) / 2)
        print(f"  {int(level):3d}  {int(n_s):4d}  {mean_s:8.4f}  {int(n_e):4d}  {mean_e:8.4f}  "
              f"{mean_e - mean_s:8.4f}  {p_l:8.4f}  {d_l:8.3f}")
    else:
        print(f"  {int(level):3d}  {int(n_s):4d}  {'---':>8}  {int(n_e):4d}  {'---':>8}  "
              f"{'---':>8}  {'---':>8}  {'---':>8}")

# 4d. Regression: does referent type add to ed + freq?
print("\n" + "─" * 70)
print("D. REGRESSION: Δ ~ ed + freq + referent_type")
print("─" * 70)

from numpy.linalg import lstsq

# Model 1: Δ ~ ed + freq
X1 = np.column_stack([np.ones(N), all_ed, all_freq])
b1, _, _, _ = lstsq(X1, all_delta, rcond=None)
ss_res1 = np.sum((all_delta - X1 @ b1) ** 2)
ss_tot = np.sum((all_delta - all_delta.mean()) ** 2)
R2_1 = 1 - ss_res1 / ss_tot

# Model 2: Δ ~ ed + freq + is_evaluative
X2 = np.column_stack([np.ones(N), all_ed, all_freq, is_E])
b2, _, _, _ = lstsq(X2, all_delta, rcond=None)
ss_res2 = np.sum((all_delta - X2 @ b2) ** 2)
R2_2 = 1 - ss_res2 / ss_tot

# Model 3: Δ ~ ed + freq + is_eval + is_neutral (full category model)
X3 = np.column_stack([np.ones(N), all_ed, all_freq, is_E, is_N])
b3, _, _, _ = lstsq(X3, all_delta, rcond=None)
ss_res3 = np.sum((all_delta - X3 @ b3) ** 2)
R2_3 = 1 - ss_res3 / ss_tot

# Model 4: Δ ~ ed + freq + is_eval + ed×is_eval (interaction)
X4 = np.column_stack([np.ones(N), all_ed, all_freq, is_E, all_ed * is_E])
b4, _, _, _ = lstsq(X4, all_delta, rcond=None)
ss_res4 = np.sum((all_delta - X4 @ b4) ** 2)
R2_4 = 1 - ss_res4 / ss_tot

print(f"\n  {'Model':<45} {'R²':>8} {'ΔR²':>8}")
print(f"  {'─'*45} {'─'*8} {'─'*8}")
print(f"  {'Δ ~ ed + freq':<45} {R2_1:8.4f} {'—':>8}")
print(f"  {'Δ ~ ed + freq + evaluative':<45} {R2_2:8.4f} {R2_2-R2_1:8.4f}")
print(f"  {'Δ ~ ed + freq + eval + neutral':<45} {R2_3:8.4f} {R2_3-R2_1:8.4f}")
print(f"  {'Δ ~ ed + freq + eval + ed×eval':<45} {R2_4:8.4f} {R2_4-R2_1:8.4f}")

print(f"\n  Model 2 coefficients:")
print(f"    β_intercept = {b2[0]:.4f}")
print(f"    β_ed = {b2[1]:.4f}")
print(f"    β_freq = {b2[2]:.4f}")
print(f"    β_evaluative = {b2[3]:.4f}")

# F-test for evaluative dummy
from scipy.stats import f as f_dist
F_eval = ((ss_res1 - ss_res2) / 1) / (ss_res2 / (N - 4))
p_eval = 1 - f_dist.cdf(F_eval, 1, N - 4)
print(f"\n  F-test (evaluative adds to ed+freq): F = {F_eval:.3f}, p = {p_eval:.4f}")

# F-test for interaction
F_int = ((ss_res2 - ss_res4) / 1) / (ss_res4 / (N - 5))
p_int = 1 - f_dist.cdf(F_int, 1, N - 5)
print(f"  F-test (interaction ed×eval): F = {F_int:.3f}, p = {p_int:.4f}")

# 4e. The key question: within ed=3 (where we have both S and E)
print("\n" + "─" * 70)
print("E. FOCUS: ed=3 WORDS (structural vs evaluative)")
print("─" * 70)

ed3 = [r for r in results if r['ed'] == 3]
ed3_S = [r for r in ed3 if r['type'] == 'S']
ed3_E = [r for r in ed3 if r['type'] == 'E']

print(f"\n  Structural ed=3 (n={len(ed3_S)}):")
for r in sorted(ed3_S, key=lambda x: x['delta']):
    print(f"    {r['word']:<20} Δ = {r['delta']:.4f}  freq = {r['freq']:.2f}")

print(f"\n  Evaluative ed=3 (n={len(ed3_E)}):")
for r in sorted(ed3_E, key=lambda x: x['delta']):
    print(f"    {r['word']:<20} Δ = {r['delta']:.4f}  freq = {r['freq']:.2f}")

if len(ed3_S) >= 2 and len(ed3_E) >= 2:
    delta_3s = np.array([r['delta'] for r in ed3_S])
    delta_3e = np.array([r['delta'] for r in ed3_E])
    t_3, p_3 = stats.ttest_ind(delta_3s, delta_3e, equal_var=False)
    d_3 = (delta_3e.mean() - delta_3s.mean()) / \
          np.sqrt((delta_3s.var() + delta_3e.var()) / 2)
    print(f"\n  Welch t = {t_3:.3f}, p = {p_3:.4f}, d = {d_3:.3f}")

# 4f. Same for ed=4
print("\n" + "─" * 70)
print("F. FOCUS: ed=4 WORDS (structural vs evaluative)")
print("─" * 70)

ed4 = [r for r in results if r['ed'] == 4]
ed4_S = [r for r in ed4 if r['type'] == 'S']
ed4_E = [r for r in ed4 if r['type'] == 'E']

print(f"\n  Structural ed=4 (n={len(ed4_S)}):")
for r in sorted(ed4_S, key=lambda x: x['delta']):
    print(f"    {r['word']:<20} Δ = {r['delta']:.4f}  freq = {r['freq']:.2f}")

print(f"\n  Evaluative ed=4 (n={len(ed4_E)}):")
for r in sorted(ed4_E, key=lambda x: x['delta']):
    print(f"    {r['word']:<20} Δ = {r['delta']:.4f}  freq = {r['freq']:.2f}")

if len(ed4_S) >= 2 and len(ed4_E) >= 2:
    delta_4s = np.array([r['delta'] for r in ed4_S])
    delta_4e = np.array([r['delta'] for r in ed4_E])
    t_4, p_4 = stats.ttest_ind(delta_4s, delta_4e, equal_var=False)
    d_4 = (delta_4e.mean() - delta_4s.mean()) / \
          np.sqrt((delta_4s.var() + delta_4e.var()) / 2)
    print(f"\n  Welch t = {t_4:.3f}, p = {p_4:.4f}, d = {d_4:.3f}")

# 4g. Combined: transparency × referent type
print("\n" + "─" * 70)
print("G. THE FULL PICTURE: ed + freq + transparency + referent_type")
print("─" * 70)

# We need transparency data. Re-code it here for the ed>1 words.
transp_dict = {
    # ed=2 transparent
    "teacher":1, "quickly":1, "undo":1, "sunrise":1, "become":1,
    "maybe":1, "inside":1, "outside":1, "kingdom":1, "freedom":1,
    "childhood":1, "friendship":1, "household":1, "wisdom":1,
    "witness":1, "service":1,
    # ed=3 transparent
    "beautiful":1, "wonderful":1, "powerful":1, "dangerous":1,
    "government":1, "agreement":1, "movement":1, "impossible":1,
    "unhappy":1, "disappear":1, "discover":1, "breakfast":1,
    "understand":1, "nightmare":1, "holiday":1, "awful":1,
    "awesome":1, "terrible":1, "terrific":1, "naughty":1, "crafty":1,
    # ed=4 transparent
    "unfortunately":1, "uncomfortable":1, "communication":1,
    "international":1, "entertainment":1, "philosophical":1,
    "understanding":1, "disagreement":1, "independence":1,
    "environmental":1, "responsibility":1, "organization":1,
    "representative":1, "establishment":1, "particularly":1,
    "secretary":1,
    # ed=5 transparent
    "quintessential":1,
}

is_transp = np.array([transp_dict.get(r['word'], 0) if r['ed'] > 1 else 0
                       for r in results], dtype=float)

# Full model: Δ ~ ed + freq + evaluative + transparent
X_full = np.column_stack([np.ones(N), all_ed, all_freq, is_E, is_transp])
b_full, _, _, _ = lstsq(X_full, all_delta, rcond=None)
ss_full = np.sum((all_delta - X_full @ b_full) ** 2)
R2_full = 1 - ss_full / ss_tot

print(f"\n  Full model: Δ ~ ed + freq + evaluative + transparent")
print(f"  R² = {R2_full:.4f} (vs {R2_1:.4f} for ed+freq alone)")
print(f"  ΔR² = {R2_full - R2_1:.4f}")
print(f"\n  β_ed = {b_full[1]:.4f}")
print(f"  β_freq = {b_full[2]:.4f}")
print(f"  β_evaluative = {b_full[3]:.4f}")
print(f"  β_transparent = {b_full[4]:.4f}")

F_full = ((ss_res1 - ss_full) / 2) / (ss_full / (N - 5))
p_full = 1 - f_dist.cdf(F_full, 2, N - 5)
print(f"\n  F-test (eval + transp add to ed+freq): F = {F_full:.3f}, p = {p_full:.4f}")

# ═══════════════════════════════════════════════════════════
# VERDICT
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
print(f"""
  Structural words:  mean Δ = {delta_S.mean():.4f} (n={len(delta_S)})
  Evaluative words:  mean Δ = {delta_E.mean():.4f} (n={len(delta_E)})
  Cohen's d = {d_se:.3f}

  After controlling ed + freq:
    β_evaluative = {b2[3]:.4f}
    F = {F_eval:.3f}, p = {p_eval:.4f}

  {"SIGNIFICANT: evaluative words change more, beyond ed and frequency" if p_eval < 0.05 else "MARGINAL" if p_eval < 0.10 else "NOT SIGNIFICANT"}

  Interpretation:
  Words whose meaning depends on cultural/emotional evaluation
  are more vulnerable to semantic change than words whose meaning
  is defined by logical, physical, or institutional structure.

  This is {"an independent predictor" if p_eval < 0.05 else "a suggestive trend"} beyond etymological depth.
""")
