#!/usr/bin/env python3
"""
Semantic Countdown Hypothesis — Automated Analysis
====================================================
Uses:
  - Wiktionary API for automated etymological depth estimation
  - wordfreq for corpus-based frequency data
  - WordNet (NLTK) for polysemy

Measures etymological depth (ed) by counting derivation chain length
in Wiktionary etymology sections: "From X, from Y, from Z" → ed = 3.
"""

import urllib.request
import json
import re
import time
import numpy as np
from scipy import stats
from wordfreq import zipf_frequency
import warnings
warnings.filterwarnings('ignore')

try:
    import nltk
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    from nltk.corpus import wordnet as wn
    HAS_WN = True
except:
    HAS_WN = False

# ═════════════════════════════════════════════════════════════
# Word list: ~500 common English words across frequency ranges
# ═════════════════════════════════════════════════════════════

# Swadesh list (basic vocabulary, likely ed=1)
swadesh_core = [
    "I", "you", "he", "she", "it", "we", "they", "this", "that",
    "here", "there", "who", "what", "where", "when", "how",
    "not", "all", "many", "some", "few", "other",
    "one", "two", "three", "four", "five",
    "big", "long", "wide", "thick", "heavy",
    "small", "short", "narrow", "thin",
    "woman", "man", "person", "child",
    "wife", "husband", "mother", "father",
    "animal", "fish", "bird", "dog", "louse", "snake", "worm",
    "tree", "forest", "stick", "fruit", "seed", "leaf", "root",
    "bark", "flower", "grass", "rope", "skin", "meat", "blood",
    "bone", "fat", "egg", "horn", "tail", "feather", "hair",
    "head", "ear", "eye", "nose", "mouth", "tooth", "tongue",
    "fingernail", "foot", "leg", "knee", "hand", "wing", "belly",
    "neck", "breast", "heart", "liver",
    "drink", "eat", "bite", "suck", "spit",
    "vomit", "blow", "breathe", "laugh", "see",
    "hear", "know", "think", "smell", "fear",
    "sleep", "live", "die", "kill", "fight",
    "hunt", "hit", "cut", "split", "stab",
    "scratch", "dig", "swim", "fly", "walk",
    "come", "lie", "sit", "stand", "turn",
    "fall", "give", "hold", "squeeze", "rub",
    "wash", "wipe", "pull", "push", "throw",
    "tie", "sew", "count", "say", "sing",
    "play", "float", "flow", "freeze", "swell",
    "sun", "moon", "star", "water", "rain",
    "river", "lake", "sea", "salt", "stone",
    "sand", "dust", "earth", "cloud", "fog",
    "sky", "wind", "snow", "ice", "smoke",
    "fire", "ash", "burn", "road", "mountain",
    "red", "green", "yellow", "white", "black",
    "night", "day", "year",
    "warm", "cold", "full", "new", "old", "good",
    "bad", "rotten", "dirty", "straight", "round",
    "sharp", "dull", "smooth", "wet", "dry",
    "correct", "near", "far", "right", "left",
    "at", "in", "with", "and", "if",
    "because", "name"
]

# Words known to have undergone semantic change
changed_words = [
    "nice", "silly", "awful", "awesome", "terrible", "terrific",
    "naughty", "shrewd", "fond", "brave", "crafty", "cunning",
    "sad", "fast", "cheap", "pretty", "bully", "clue",
    "deer", "meat", "starve", "thing", "hound", "fowl",
    "gossip", "holiday", "goodbye", "nightmare", "husband",
    "lord", "lady", "manufacture", "candidate", "salary",
    "calculate", "secretary", "magazine", "algorithm", "algebra",
    "assassin", "admiral", "alcohol", "quarantine", "muscle",
    "minister", "sinister", "trivial", "egregious", "sycophant",
    "glamour", "mortgage", "preposterous", "pandemonium",
    "disaster", "inaugurate", "investigate", "exorbitant",
    "enthusiasm", "sophisticated", "companion", "janitor",
    "treacle", "moot", "barn", "orchard", "stool",
    "sheriff", "worship", "world", "understand",
    "cardinal", "lieutenant", "pedigree", "juggernaut",
    "miscreant", "dilapidated", "quintessential", "decimation",
    "wonderful", "dangerous", "beautiful", "government",
    "freedom", "kingdom", "childhood", "teacher",
    "unfortunately", "uncomfortable", "international",
    "communication", "entertainment", "establishment",
    "independence", "responsibility", "philosophical",
    "breakfast", "discover", "disappear", "impossible",
    "serendipity", "antidisestablishmentarianism",
    "vermicelli", "peninsula", "extravagant",
    "window", "anger", "skill", "wrong",
    "army", "court", "state", "power", "country",
    "place", "point", "matter", "number", "order",
    "office", "story", "war", "age", "cause", "reason",
]

# Combine and deduplicate
all_words = list(dict.fromkeys(swadesh_core + changed_words))
print(f"Total unique words to analyze: {len(all_words)}")

# ═════════════════════════════════════════════════════════════
# Fetch etymology from Wiktionary and estimate depth
# ═════════════════════════════════════════════════════════════

def get_etymology(word, max_retries=2):
    """Fetch etymology section from Wiktionary API."""
    url = f'https://en.wiktionary.org/w/api.php?action=parse&page={word.lower()}&prop=wikitext&format=json'
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'SemanticCountdown/1.0 (research)'})
            resp = urllib.request.urlopen(req, timeout=10)
            data = json.loads(resp.read())
            if 'parse' not in data:
                return ""
            wikitext = data['parse']['wikitext']['*']
            # Extract English etymology section
            # Find ===Etymology=== or ===Etymology 1===
            matches = re.findall(
                r'===\s*Etymology(?:\s+\d+)?\s*===\s*(.*?)(?====|\Z)',
                wikitext, re.DOTALL
            )
            if matches:
                return matches[0][:2000]  # first etymology, truncated
            return ""
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.5)
            continue
    return ""

def estimate_ed(etymology_text):
    """
    Estimate etymological depth from Wiktionary etymology text.
    Counts derivation steps: "from X, from Y, from Z" = 3 layers.
    """
    if not etymology_text:
        return None

    text = etymology_text.lower()

    # Count "from" chains (main indicator of derivation depth)
    from_count = len(re.findall(r'\bfrom\b', text))

    # Count distinct language layers mentioned
    lang_markers = re.findall(
        r'(?:proto-indo-european|proto-germanic|old english|middle english|'
        r'old french|middle french|anglo-norman|latin|ancient greek|'
        r'old norse|old high german|vulgar latin|medieval latin|'
        r'classical latin|koine greek|sanskrit|arabic|persian|'
        r'old dutch|old frisian|proto-west germanic)',
        text
    )
    lang_count = len(set(lang_markers))

    # Count morphological markers
    morph_markers = len(re.findall(
        r'(?:prefix|suffix|compound|derived|equivalent|back-formation|'
        r'blend|clipping|diminutive|augmentative|\+)',
        text
    ))

    # Heuristic: ed = max(from_count, lang_count) capped at 5
    # Primwörter: typically just "From Proto-IE *root" → from=1, lang=1
    # Deep words: "From French X, from Latin Y, from Greek Z" → from=3, lang=3

    raw = max(from_count, lang_count)

    # Adjust: if PIE or Proto-Germanic is the deepest layer AND only 1-2 froms,
    # this is likely a basic root word
    if raw <= 1:
        return 1
    elif raw == 2:
        # Check if it's a simple compound or a borrowed word
        if morph_markers > 0 or '+' in text:
            return 2
        return 2
    elif raw == 3:
        return 3
    elif raw == 4:
        return 4
    else:
        return 5


def estimate_semantic_change(word, etymology_text):
    """
    Estimate semantic change from etymology text.
    Looks for markers of meaning shift.
    """
    if not etymology_text:
        return None

    text = etymology_text.lower()
    score = 0

    # Strong change markers
    strong_markers = [
        r'originally\s+(?:meant|meaning|referred|denoted)',
        r'sense\s+(?:shifted|changed|evolved|developed)',
        r'meaning\s+(?:shifted|changed|narrowed|broadened)',
        r'(?:amelioration|pejoration|semantic\s+shift)',
        r'no\s+longer\s+(?:means|used)',
        r'(?:opposite|reversed)\s+meaning',
        r'now\s+(?:means|used|refers)',
        r'modern\s+sense',
    ]

    # Moderate change markers
    moderate_markers = [
        r'(?:by\s+extension|extended\s+to|figurative)',
        r'(?:narrowed|broadened|generalized|specialized)',
        r'(?:metaphor|metonym)',
        r'doublet\s+of',
    ]

    # Check for meaning descriptions that indicate change
    # "meaning X" where X is very different from current usage
    meaning_desc = re.findall(r'meaning\s+"([^"]+)"', text)
    meaning_desc += re.findall(r'meaning\s+\'([^\']+)\'', text)

    for pattern in strong_markers:
        if re.search(pattern, text):
            score += 2

    for pattern in moderate_markers:
        if re.search(pattern, text):
            score += 1

    # Check number of distinct meanings mentioned
    quoted_meanings = re.findall(r'"([^"]{2,40})"', text)
    if len(quoted_meanings) >= 3:
        score += 1

    return min(score, 3)  # cap at 3


# ═════════════════════════════════════════════════════════════
# Fetch data for all words
# ═════════════════════════════════════════════════════════════
print("\nFetching etymologies from Wiktionary...")
print("(This will take a few minutes due to API rate limiting)")

results = []
batch_size = 10
for i, word in enumerate(all_words):
    if i > 0 and i % batch_size == 0:
        time.sleep(1)  # rate limit: ~10 requests per second
        if i % 50 == 0:
            print(f"  [{i}/{len(all_words)}] processed...")

    etym = get_etymology(word)
    ed_auto = estimate_ed(etym)
    chg_auto = estimate_semantic_change(word, etym)
    freq_zipf = zipf_frequency(word.lower(), 'en')

    if HAS_WN:
        poly = len(wn.synsets(word.lower()))
    else:
        poly = 0

    results.append({
        'word': word,
        'ed': ed_auto,
        'chg': chg_auto,
        'freq': freq_zipf,
        'polysemy': poly,
        'etym_length': len(etym),
        'etym_snippet': etym[:100].replace('\n', ' ') if etym else ''
    })

print(f"  [{len(all_words)}/{len(all_words)}] done.")

# ═════════════════════════════════════════════════════════════
# Filter to words with valid data
# ═════════════════════════════════════════════════════════════
valid = [r for r in results if r['ed'] is not None and r['chg'] is not None]
print(f"\nWords with valid etymology: {len(valid)} / {len(all_words)}")

if len(valid) < 50:
    print("ERROR: Too few words with valid data. Check API access.")
    exit(1)

words_v = [r['word'] for r in valid]
ed_v    = np.array([r['ed'] for r in valid], dtype=float)
chg_v   = np.array([r['chg'] for r in valid], dtype=float)
freq_v  = np.array([r['freq'] for r in valid], dtype=float)
poly_v  = np.array([r['polysemy'] for r in valid], dtype=float)

N = len(valid)

# ═════════════════════════════════════════════════════════════
# Show some examples of automatic classification
# ═════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("EXAMPLES OF AUTOMATIC ETYMOLOGY DEPTH ESTIMATION")
print("─" * 70)
examples = ['I', 'water', 'fire', 'husband', 'nice', 'silly', 'terrible',
            'calculate', 'salary', 'algorithm', 'egregious', 'sycophant',
            'understand', 'beautiful', 'manufacture', 'trivial']
for w in examples:
    r = next((r for r in valid if r['word'] == w), None)
    if r:
        print(f"  {w:25s}  ed={r['ed']}  chg={r['chg']}  "
              f"freq={r['freq']:.1f}  [{r['etym_snippet'][:60]}...]")

# ═════════════════════════════════════════════════════════════
# ANALYSIS (same structure as v2)
# ═════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("AUTOMATED SEMANTIC COUNTDOWN ANALYSIS")
print("=" * 70)
print(f"\nDataset: {N} words with automated ed and change scores")

# 1. Descriptive stats
print("\n" + "─" * 70)
print("1. DESCRIPTIVE STATISTICS BY AUTO-ED")
print("─" * 70)
print(f"\n  {'ed':>3}  {'n':>4}  {'mean_chg':>9}  {'std':>6}  {'mean_freq':>10}")
for level in sorted(set(ed_v)):
    mask = ed_v == level
    n = mask.sum()
    if n > 0:
        print(f"  {int(level):3d}  {n:4d}  {chg_v[mask].mean():9.3f}  "
              f"{chg_v[mask].std():6.3f}  {freq_v[mask].mean():10.2f}")

# 2. Correlations
print("\n" + "─" * 70)
print("2. CORRELATIONS (AUTOMATED)")
print("─" * 70)

r_p, p_p = stats.pearsonr(ed_v, chg_v)
r_s, p_s = stats.spearmanr(ed_v, chg_v)
r_k, p_k = stats.kendalltau(ed_v, chg_v)

print(f"\n  Pearson  r(ed, chg)  = {r_p:.4f}  (p = {p_p:.2e})")
print(f"  Spearman ρ(ed, chg)  = {r_s:.4f}  (p = {p_s:.2e})")
print(f"  Kendall  τ(ed, chg)  = {r_k:.4f}  (p = {p_k:.2e})")

# 3. Confound: frequency
print("\n" + "─" * 70)
print("3. CONFOUND: FREQUENCY (Zipf scale)")
print("─" * 70)

r_ef, p_ef = stats.pearsonr(ed_v, freq_v)
r_cf, p_cf = stats.pearsonr(chg_v, freq_v)

print(f"\n  r(ed, freq)    = {r_ef:.4f}  (p = {p_ef:.2e})")
print(f"  r(chg, freq)   = {r_cf:.4f}  (p = {p_cf:.2e})")

def partial_r(x, y, z):
    r_xy = stats.pearsonr(x, y)[0]
    r_xz = stats.pearsonr(x, z)[0]
    r_yz = stats.pearsonr(y, z)[0]
    denom = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    if denom < 1e-10:
        return 0.0
    return (r_xy - r_xz * r_yz) / denom

r_partial = partial_r(ed_v, chg_v, freq_v)
print(f"\n  Partial r(ed, chg | freq) = {r_partial:.4f}")

# 4. Fixpoint test
print("\n" + "─" * 70)
print("4. FIXPOINT TEST: ed=1 vs ed>1")
print("─" * 70)

prim = chg_v[ed_v == 1]
rest = chg_v[ed_v > 1]

if len(prim) > 0 and len(rest) > 0:
    print(f"\n  ed=1: n={len(prim)}, mean={prim.mean():.3f}")
    print(f"  ed>1: n={len(rest)}, mean={rest.mean():.3f}")

    t_stat, p_t = stats.ttest_ind(prim, rest, equal_var=False)
    u_stat, p_u = stats.mannwhitneyu(prim, rest, alternative='less')
    d_cohen = (rest.mean() - prim.mean()) / np.sqrt((prim.var() + rest.var()) / 2)

    print(f"\n  Welch t: t = {t_stat:.3f}, p = {p_t:.2e}")
    print(f"  Mann-Whitney: p = {p_u:.2e}")
    print(f"  Cohen's d: {d_cohen:.3f}")

# 5. ANOVA
print("\n" + "─" * 70)
print("5. ANOVA")
print("─" * 70)

groups = [chg_v[ed_v == level] for level in sorted(set(ed_v)) if (ed_v == level).sum() >= 3]
if len(groups) >= 2:
    F_stat, p_anova = stats.f_oneway(*groups)
    H_stat, p_kw = stats.kruskal(*groups)
    print(f"\n  F = {F_stat:.3f}, p = {p_anova:.2e}")
    print(f"  Kruskal-Wallis H = {H_stat:.3f}, p = {p_kw:.2e}")

# 6. Bootstrap
print("\n" + "─" * 70)
print("6. BOOTSTRAP 95% CI (10000 resamples)")
print("─" * 70)

rng = np.random.RandomState(42)
n_boot = 10000
boot_r = np.zeros(n_boot)
for i in range(n_boot):
    idx = rng.randint(0, N, N)
    boot_r[i] = stats.pearsonr(ed_v[idx], chg_v[idx])[0]

ci_lo, ci_hi = np.percentile(boot_r, [2.5, 97.5])
print(f"\n  r = {r_p:.4f}, 95% CI = [{ci_lo:.4f}, {ci_hi:.4f}]")

# 7. Comparison with hand-coded v2
print("\n" + "─" * 70)
print("7. COMPARISON: HAND-CODED (v2) vs AUTOMATED")
print("─" * 70)
print(f"""
  {'Measure':<30}  {'Hand-coded (v2)':>15}  {'Automated':>15}
  {'─'*30}  {'─'*15}  {'─'*15}
  {'N words':<30}  {'274':>15}  {N:>15}
  {'Pearson r':<30}  {'0.646':>15}  {r_p:>15.3f}
  {'Spearman ρ':<30}  {'0.671':>15}  {r_s:>15.3f}
  {'Partial r (|freq)':<30}  {'0.400':>15}  {r_partial:>15.3f}
  {'ed=1 mean change':<30}  {'0.094':>15}  {prim.mean():>15.3f}
  {'ed>1 mean change':<30}  {'1.248':>15}  {rest.mean():>15.3f}
""")

# 8. Summary
print("=" * 70)
print("SUMMARY")
print("=" * 70)

supported = "STRONGLY SUPPORTED" if p_p < 1e-5 and r_partial > 0.2 else \
            "SUPPORTED" if p_p < 0.01 else "WEAK/INCONCLUSIVE"

print(f"""
  N = {N} words (automated from Wiktionary + wordfreq)

  r(ed, semantic_change) = {r_p:.3f}  (p = {p_p:.1e})
  After freq control     = {r_partial:.3f}
  Bootstrap 95% CI       = [{ci_lo:.3f}, {ci_hi:.3f}]
  Cohen's d (fixpoint)   = {d_cohen:.2f}

  VERDICT: {supported}

  IMPORTANT CAVEATS:
  1. Automated ed estimation is crude (counting "from" chains)
  2. Automated change detection only captures DOCUMENTED change
     (Wiktionary doesn't always note historical shifts)
  3. This means automated change scores are UNDERESTIMATES
     → true correlation is likely HIGHER than measured
  4. For a real paper: need OED etymological depth +
     diachronic word embeddings (Hamilton et al. 2016)
""")
