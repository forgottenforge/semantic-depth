#!/usr/bin/env python3
"""
Semantic Countdown Hypothesis Test
===================================
H: Words with higher etymological depth (ed) show higher rates of semantic change.
Prediction: Pearson r(ed, change) > 0 significantly.

ed = etymological depth:
  1 = Primwort (irreducible root, present in Proto-IE or universal deixis)
  2 = simple derivation (one morphological step from root)
  3 = compound / double derivation
  4 = complex derivation (multiple layers)
  5 = highly derived / opaque etymology

change = semantic change score (0-3):
  0 = meaning unchanged since earliest attestation
  1 = minor narrowing/broadening
  2 = significant shift
  3 = radical change (e.g. amelioration/pejoration, reversal)

We also track word frequency rank (log) as potential confound.
freq_class: 1=top-100, 2=top-1000, 3=top-10000, 4=rarer
"""

import numpy as np
from scipy import stats

# ─────────────────────────────────────────────────────────────
# Dataset: ~150 English words, hand-coded
# Format: (word, ed, change, freq_class)
# ─────────────────────────────────────────────────────────────

data = [
    # ── ed=1: Primwörter (Proto-IE roots / universal deixis) ──
    ("I",        1, 0, 1),
    ("you",      1, 0, 1),
    ("we",       1, 0, 1),
    ("he",       1, 0, 1),
    ("she",      1, 0, 1),
    ("it",       1, 0, 1),
    ("this",     1, 0, 1),
    ("that",     1, 0, 1),
    ("here",     1, 0, 1),
    ("there",    1, 0, 1),
    ("now",      1, 0, 1),
    ("who",      1, 0, 1),
    ("what",     1, 0, 1),
    ("be",       1, 0, 1),
    ("do",       1, 0, 1),
    ("go",       1, 0, 1),
    ("come",     1, 0, 1),
    ("see",      1, 0, 1),
    ("eat",      1, 0, 1),
    ("drink",    1, 0, 1),
    ("die",      1, 0, 1),
    ("give",     1, 0, 1),
    ("sit",      1, 0, 1),
    ("stand",    1, 0, 1),
    ("one",      1, 0, 1),
    ("two",      1, 0, 1),
    ("sun",      1, 0, 1),
    ("moon",     1, 0, 1),
    ("water",    1, 0, 1),
    ("fire",     1, 1, 1),   # minor: "fire" also = dismiss
    ("hand",     1, 1, 1),   # minor: "hand" also verb
    ("eye",      1, 0, 1),
    ("ear",      1, 0, 1),
    ("nose",     1, 1, 1),   # minor: "nose around"
    ("stone",    1, 0, 2),
    ("blood",    1, 0, 2),
    ("bone",     1, 0, 2),
    ("star",     1, 1, 2),   # minor: celebrity meaning
    ("name",     1, 0, 1),
    ("new",      1, 0, 1),
    ("full",     1, 0, 1),

    # ── ed=2: Simple derivations / basic compounds ──
    ("teacher",    2, 0, 2),  # teach + er, transparent
    ("quickly",    2, 0, 2),  # quick + ly
    ("undo",       2, 0, 3),  # un + do
    ("sunrise",    2, 0, 3),  # sun + rise
    ("understand", 2, 2, 1),  # under + stand → opaque meaning
    ("forget",     2, 0, 1),  # for + get
    ("begin",      2, 0, 1),  # be + gin(nen)
    ("become",     2, 0, 1),  # be + come
    ("behind",     2, 0, 1),  # be + hind
    ("between",    2, 0, 1),  # be + twain
    ("maybe",      2, 0, 2),  # may + be
    ("inside",     2, 0, 2),  # in + side
    ("outside",    2, 0, 2),  # out + side
    ("upon",       2, 0, 2),  # up + on
    ("kingdom",    2, 0, 3),  # king + dom
    ("freedom",    2, 0, 2),  # free + dom
    ("wisdom",     2, 1, 2),  # wise + dom (slightly narrowed)
    ("childhood",  2, 0, 3),  # child + hood
    ("friendship", 2, 0, 3),  # friend + ship
    ("household",  2, 0, 3),  # house + hold
    ("husband",    2, 2, 1),  # hus(house) + band(dweller) → spouse
    ("woman",      2, 1, 1),  # wif + man → shifted from "wife-person"
    ("lord",       2, 2, 2),  # hlaford (loaf-ward) → ruler
    ("lady",       2, 2, 2),  # hlaefdige (loaf-kneader) → noble woman
    ("barn",       2, 1, 3),  # bere(barley) + ern(place) → any farm building
    ("world",      2, 1, 1),  # wer(man) + ald(age) → broadened
    ("thing",      2, 2, 1),  # originally: assembly/meeting → object
    ("deer",       2, 2, 2),  # originally: any animal → specific animal
    ("hound",      2, 2, 3),  # originally: any dog → specific breed
    ("fowl",       2, 2, 3),  # originally: any bird → specific type
    ("meat",       2, 2, 1),  # originally: any food → flesh
    ("starve",     2, 2, 3),  # originally: to die (any cause) → die of hunger

    # ── ed=3: Compound derivations ──
    ("understand",   3, 2, 1),  # under+stand, opaque semantic shift — actually ed=2, let me fix
    ("beautiful",    3, 0, 1),  # beauty (Fr) + ful
    ("wonderful",    3, 1, 1),  # wonder + ful (wonder=astonishment → good)
    ("powerful",     3, 0, 2),  # power (Fr) + ful
    ("dangerous",    3, 0, 2),  # danger (Fr) + ous
    ("government",   3, 1, 2),  # govern (Fr) + ment
    ("agreement",    3, 0, 2),  # agree (Fr) + ment
    ("movement",     3, 0, 2),  # move (Fr) + ment
    ("impossible",   3, 0, 2),  # im + possible (Lat)
    ("unhappy",      3, 0, 2),  # un + happy
    ("disappear",    3, 0, 2),  # dis + appear (Fr)
    ("discover",     3, 1, 2),  # dis + cover (Fr) → find
    ("breakfast",    3, 1, 2),  # break + fast → opaque to most speakers
    ("nightmare",    3, 2, 2),  # night + mare(demon) → bad dream
    ("holiday",      3, 2, 1),  # holy + day → vacation
    ("goodbye",      3, 2, 1),  # God + be + with + ye → farewell
    ("gossip",       3, 2, 2),  # god + sib(relative) → idle talk
    ("silly",        3, 3, 2),  # sælig(blessed/happy) → foolish
    ("nice",         3, 3, 1),  # nescius(ignorant) → pleasant
    ("pretty",       3, 2, 2),  # prættig(cunning) → attractive
    ("awful",        3, 3, 2),  # awe + ful(inspiring awe) → terrible
    ("awesome",      3, 2, 2),  # awe + some → great/cool
    ("terrible",     3, 3, 2),  # terror(great fear) + ible → very bad
    ("terrific",     3, 3, 2),  # terror + ific(causing terror) → excellent
    ("naughty",      3, 3, 3),  # naught(nothing) + y → badly behaved
    ("bully",        3, 3, 3),  # boel(lover/brother) → intimidator

    # ── ed=4: Complex / multi-layer derivation ──
    ("unfortunately",  4, 0, 2),  # un + fortuna + ate + ly
    ("uncomfortable",  4, 0, 2),  # un + com + fort + able
    ("communication",  4, 1, 2),  # com + munis + ic + ation
    ("international",  4, 0, 2),  # inter + nation + al
    ("understanding",  4, 2, 1),  # under + stand + ing (opaque)
    ("entertainment",  4, 1, 2),  # enter + tain + ment
    ("philosophical",  4, 1, 3),  # philo + sophia + ic + al
    ("manufacture",    4, 2, 3),  # manu(hand) + fact(make) → factory production
    ("sophisticated",  4, 3, 3),  # sophia(wisdom) → worldly → complex
    ("enthusiasm",     4, 2, 3),  # en + theos(god) + iasm → possessed by god → eager
    ("sarcasm",        4, 1, 3),  # sarx(flesh) + asm → tear flesh → bitter irony
    ("sincere",        4, 1, 3),  # sin(without) + cera(wax) → genuine? (debated)
    ("candidate",      4, 2, 3),  # candidus(white/toga) → person seeking office
    ("salary",         4, 2, 3),  # sal(salt) + arium → salt money → pay
    ("calculate",      4, 2, 3),  # calculus(pebble) + ate → count with stones → compute
    ("secretary",      4, 2, 3),  # secretum(secret) + ary → keeper of secrets → admin
    ("magazine",       4, 2, 3),  # makhzan(storehouse) → publication
    ("algorithm",      4, 2, 3),  # al-Khwarizmi → procedure
    ("algebra",        4, 1, 3),  # al-jabr(reunion of parts) → math branch
    ("assassin",       4, 2, 3),  # hashishin → murderer
    ("admiral",        4, 2, 3),  # amir-al(commander of) → naval rank
    ("checkmate",      4, 1, 3),  # shah mat(king is dead) → chess term

    # ── ed=5: Highly derived / maximally opaque ──
    ("antidisestablishmentarianism", 5, 0, 4),  # anti+dis+establish+ment+arian+ism
    ("serendipity",    5, 1, 4),  # Serendip(Sri Lanka) + ity → happy accident
    ("lieutenant",     5, 2, 4),  # lieu(place) + tenant(holding) → military rank
    ("mortgage",       5, 2, 3),  # mort(death) + gage(pledge) → death pledge → home loan
    ("preposterous",   5, 2, 3),  # prae(before) + posterus(after) → reversed → absurd
    ("egregious",      5, 3, 4),  # e(out of) + grex(flock) → outstanding → outstandingly bad
    ("sophisticated",  5, 3, 3),  # (duplicate, remove)
    ("pandemonium",    5, 2, 4),  # pan(all) + daimon(demon) + ium → Milton's hell → chaos
    ("sycophant",      5, 3, 4),  # sykon(fig) + phainein(show) → fig-shower → flatterer
    ("disaster",       5, 1, 3),  # dis(bad) + astro(star) → bad star → catastrophe
    ("trivial",        5, 2, 3),  # tri(three) + via(road) → crossroads → unimportant
    ("decimation",     5, 3, 3),  # decem(ten) + ation → kill every 10th → destroy most
    ("nice",           5, 3, 1),  # (duplicate, remove)
]

# ─────────────────────────────────────────────────────────────
# Clean duplicates
# ─────────────────────────────────────────────────────────────
seen = set()
clean_data = []
for entry in data:
    word = entry[0]
    if word not in seen:
        seen.add(word)
        clean_data.append(entry)
    # keep first occurrence

data = clean_data

# Also remove the duplicate "understand" entry (ed=3 version is wrong, keep ed=2)
# Already handled by dedup above

print("="*65)
print("SEMANTIC COUNTDOWN HYPOTHESIS TEST")
print("="*65)
print(f"\nDataset: {len(data)} words")
print()

# ─────────────────────────────────────────────────────────────
# Extract arrays
# ─────────────────────────────────────────────────────────────
words = [d[0] for d in data]
ed    = np.array([d[1] for d in data], dtype=float)
chg   = np.array([d[2] for d in data], dtype=float)
freq  = np.array([d[3] for d in data], dtype=float)

# ─────────────────────────────────────────────────────────────
# 1. Basic statistics by ed level
# ─────────────────────────────────────────────────────────────
print("─"*65)
print("MEAN SEMANTIC CHANGE BY ETYMOLOGICAL DEPTH")
print("─"*65)
for level in sorted(set(ed)):
    mask = ed == level
    n = mask.sum()
    mean_chg = chg[mask].mean()
    std_chg  = chg[mask].std()
    zeros    = (chg[mask] == 0).sum()
    print(f"  ed={int(level):d}:  n={n:3d}  mean_change={mean_chg:.3f}  "
          f"std={std_chg:.3f}  unchanged={zeros}/{n}")

# ─────────────────────────────────────────────────────────────
# 2. Pearson correlation: ed vs change
# ─────────────────────────────────────────────────────────────
print()
print("─"*65)
print("CORRELATIONS")
print("─"*65)

r_main, p_main = stats.pearsonr(ed, chg)
print(f"\n  Pearson r(ed, change)   = {r_main:.4f}  (p = {p_main:.2e})")

r_sp, p_sp = stats.spearmanr(ed, chg)
print(f"  Spearman ρ(ed, change)  = {r_sp:.4f}  (p = {p_sp:.2e})")

r_kt, p_kt = stats.kendalltau(ed, chg)
print(f"  Kendall τ(ed, change)   = {r_kt:.4f}  (p = {p_kt:.2e})")

# ─────────────────────────────────────────────────────────────
# 3. Confound check: frequency
# ─────────────────────────────────────────────────────────────
print()
print("─"*65)
print("CONFOUND CHECK: WORD FREQUENCY")
print("─"*65)

r_ef, p_ef = stats.pearsonr(ed, freq)
print(f"\n  r(ed, freq_class)       = {r_ef:.4f}  (p = {p_ef:.2e})")

r_cf, p_cf = stats.pearsonr(chg, freq)
print(f"  r(change, freq_class)   = {r_cf:.4f}  (p = {p_cf:.2e})")

# Partial correlation: ed vs change, controlling for frequency
# r_xy.z = (r_xy - r_xz * r_yz) / sqrt((1-r_xz²)(1-r_yz²))
r_xy = r_main
r_xz = r_ef
r_yz = r_cf
r_partial = (r_xy - r_xz * r_yz) / np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
print(f"\n  Partial r(ed, change | freq) = {r_partial:.4f}")
print(f"  (correlation after controlling for word frequency)")

# ─────────────────────────────────────────────────────────────
# 4. Key test: Are Primwörter (ed=1) truly fixed points?
# ─────────────────────────────────────────────────────────────
print()
print("─"*65)
print("FIXPOINT TEST: Are ed=1 words semantically invariant?")
print("─"*65)

prim = chg[ed == 1]
rest = chg[ed > 1]
print(f"\n  ed=1 words: n={len(prim)}, mean change = {prim.mean():.3f}")
print(f"  ed>1 words: n={len(rest)}, mean change = {rest.mean():.3f}")

t_stat, p_ttest = stats.ttest_ind(prim, rest, equal_var=False)
print(f"\n  Welch t-test: t = {t_stat:.3f}, p = {p_ttest:.2e}")

u_stat, p_mann = stats.mannwhitneyu(prim, rest, alternative='less')
print(f"  Mann-Whitney U: U = {u_stat:.1f}, p = {p_mann:.2e}")

# ─────────────────────────────────────────────────────────────
# 5. Linear regression
# ─────────────────────────────────────────────────────────────
print()
print("─"*65)
print("LINEAR REGRESSION: change = a + b·ed")
print("─"*65)

slope, intercept, r_val, p_val, std_err = stats.linregress(ed, chg)
print(f"\n  change = {intercept:.3f} + {slope:.3f} · ed")
print(f"  R² = {r_val**2:.4f}")
print(f"  slope p-value = {p_val:.2e}")
print(f"  Interpretation: each unit increase in ed → +{slope:.2f} semantic change")

# ─────────────────────────────────────────────────────────────
# 6. Prediction: γ_language
# ─────────────────────────────────────────────────────────────
print()
print("─"*65)
print("DRIFT ESTIMATE: γ_language")
print("─"*65)

# Fraction of words with change > 0 at each ed level
print("\n  Fraction with ANY semantic change, by ed level:")
for level in sorted(set(ed)):
    mask = ed == level
    frac = (chg[mask] > 0).mean()
    print(f"    ed={int(level)}: {frac:.3f} ({(chg[mask] > 0).sum()}/{mask.sum()})")

# Mean "survival" (no change) rate
survival = (chg == 0).mean()
print(f"\n  Overall survival rate (change=0): {survival:.3f}")
print(f"  Overall change rate:              {1-survival:.3f}")

# ─────────────────────────────────────────────────────────────
# 7. Summary
# ─────────────────────────────────────────────────────────────
print()
print("="*65)
print("SUMMARY")
print("="*65)
print(f"""
  Dataset:           {len(data)} words, ed range [{int(ed.min())}-{int(ed.max())}]

  Main result:       r(ed, change) = {r_main:.3f}  (p = {p_main:.1e})
  Rank correlation:  ρ(ed, change) = {r_sp:.3f}  (p = {p_sp:.1e})
  After freq ctrl:   r_partial     = {r_partial:.3f}

  Primwort test:     ed=1 mean change = {prim.mean():.2f}
                     ed>1 mean change = {rest.mean():.2f}
                     p = {p_ttest:.1e} (Welch)

  Regression:        Δchange/Δed = {slope:.3f}

  Hypothesis supported: {"YES" if p_main < 0.001 else "MARGINAL" if p_main < 0.05 else "NO"}
""")

# Show some example words at each level
print("─"*65)
print("EXAMPLE WORDS BY LEVEL")
print("─"*65)
for level in sorted(set(ed)):
    mask = ed == level
    examples = [w for w, e in zip(words, ed) if e == level]
    changes  = [c for c, e in zip(chg, ed) if e == level]
    # sort by change
    pairs = sorted(zip(examples, changes), key=lambda x: x[1])
    stable = [w for w, c in pairs if c == 0][:5]
    shifted = [f"{w}({int(c)})" for w, c in pairs if c > 0][:5]
    print(f"\n  ed={int(level)}:")
    if stable:
        print(f"    stable:  {', '.join(stable)}")
    if shifted:
        print(f"    shifted: {', '.join(shifted)}")
