#!/usr/bin/env python3
"""
Morpheme-based Etymological Depth (ed) Calculator
===================================================
Rules:
  ed = 1: Primwörter — monomorphemic PIE roots, deixis, basic body parts
  ed = 2: one derivation step (prefix OR suffix OR simple compound)
  ed = 3: two steps (prefix+suffix, or compound+affix, or loan+derivation)
  ed = 4: three+ steps (multi-affix Latin/Greek compounds)
  ed = 5: maximally derived (4+ steps, opaque multi-language chain)

Method:
  1. Strip known prefixes and suffixes iteratively
  2. Count derivation layers
  3. Add bonus for etymological distance (Latin/Greek/Arabic borrowings)
"""

import re

# ─────────────────────────────────────────────────────────────
# Known English prefixes and suffixes
# ─────────────────────────────────────────────────────────────

PREFIXES = [
    'anti', 'dis', 'en', 'em', 'fore', 'in', 'im', 'il', 'ir',
    'inter', 'mid', 'mis', 'non', 'over', 'pre', 'post', 'pro',
    're', 'semi', 'sub', 'super', 'trans', 'un', 'under', 'out',
    'up', 'down', 'be', 'for', 'with', 'counter', 'extra', 'hyper',
    'ultra', 'mega', 'micro', 'macro', 'multi', 'poly', 'mono',
    'bi', 'tri', 'quad', 'quint', 'pan', 'omni', 'auto', 'self',
    'ex', 'de', 'ob', 'per', 'ad', 'ab', 'com', 'con', 'co',
]

SUFFIXES = [
    'ment', 'ness', 'tion', 'sion', 'ation', 'ition', 'ment',
    'able', 'ible', 'ful', 'less', 'ous', 'ious', 'eous',
    'ive', 'ative', 'itive', 'al', 'ial', 'ical',
    'er', 'or', 'ist', 'ism', 'ity', 'ty', 'ance', 'ence',
    'dom', 'ship', 'hood', 'ward', 'wards', 'wise',
    'ly', 'ing', 'ed', 'en',
    'ary', 'ory', 'ery', 'ure', 'ment',
    'fy', 'ify', 'ize', 'ise', 'ate',
    'ant', 'ent', 'ling', 'let',
]

# Sort by length (longest first) to match greedily
PREFIXES.sort(key=len, reverse=True)
SUFFIXES.sort(key=len, reverse=True)

# ─────────────────────────────────────────────────────────────
# Primwörter: known ed=1 words (Swadesh + extended)
# ─────────────────────────────────────────────────────────────
PRIMWOERTER = {
    # Pronouns / Deixis
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'us', 'them',
    'my', 'your', 'his', 'her', 'its', 'our', 'their',
    'this', 'that', 'these', 'those',
    'here', 'there', 'where', 'when', 'who', 'what', 'how', 'why',
    'now', 'then',
    # Basic verbs
    'be', 'am', 'is', 'are', 'was', 'were',
    'do', 'go', 'come', 'see', 'know', 'say', 'get', 'make',
    'take', 'give', 'have', 'had', 'eat', 'drink', 'sleep',
    'die', 'sit', 'stand', 'lie', 'fall', 'run', 'walk',
    'hear', 'feel', 'think', 'smell', 'fear',
    'cut', 'bite', 'blow', 'burn', 'pull', 'push',
    'swim', 'fly', 'hold', 'wash', 'throw', 'hit',
    'kill', 'fight', 'hunt', 'dig', 'sew', 'tie',
    'sing', 'play', 'count', 'say', 'turn',
    'squeeze', 'rub', 'wipe', 'spit', 'suck', 'vomit',
    'scratch', 'split', 'stab', 'float', 'flow', 'freeze', 'swell',
    'laugh', 'live', 'breathe',
    # Numbers
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
    # Body parts
    'eye', 'ear', 'nose', 'mouth', 'tooth', 'tongue', 'foot', 'hand',
    'head', 'knee', 'heart', 'bone', 'blood', 'skin', 'hair',
    'nail', 'leg', 'neck', 'back', 'belly', 'breast', 'liver',
    'horn', 'tail', 'wing', 'feather', 'fat', 'egg',
    # Nature
    'sun', 'moon', 'star', 'water', 'fire', 'earth', 'stone',
    'tree', 'leaf', 'root', 'seed', 'bark', 'flower', 'grass',
    'rain', 'snow', 'wind', 'sand', 'salt', 'ash', 'dust',
    'cloud', 'fog', 'sky', 'ice', 'smoke', 'sea', 'lake', 'river',
    'mountain', 'road', 'path',
    # Animals (basic)
    'dog', 'fish', 'bird', 'worm', 'louse', 'mouse', 'snake',
    # Basic adjectives
    'new', 'old', 'good', 'bad', 'big', 'small', 'long', 'short',
    'hot', 'cold', 'wet', 'dry', 'red', 'black', 'white', 'green',
    'yellow', 'full', 'sharp', 'dull', 'smooth', 'round', 'straight',
    'near', 'far', 'right', 'left', 'dead', 'wide', 'narrow',
    'thick', 'thin', 'heavy', 'warm', 'rotten', 'dirty', 'correct',
    # Basic other
    'name', 'night', 'day', 'year', 'all', 'many', 'some', 'few',
    'other', 'not', 'no', 'in', 'at', 'with', 'and', 'if', 'because',
    'man', 'woman', 'child', 'person', 'mother', 'father',
    'wife', 'animal', 'fruit', 'stick', 'rope', 'forest',
}

# ─────────────────────────────────────────────────────────────
# Words known to be Latin/Greek/Arabic/French borrowings
# (adds +1 to ed for etymological distance)
# ─────────────────────────────────────────────────────────────
LATIN_GREEK_ROOTS = {
    # Latin roots commonly found in English
    'act', 'aud', 'cap', 'ced', 'cept', 'cide', 'claim', 'clar',
    'clude', 'cogn', 'cord', 'corp', 'cred', 'cur', 'dict', 'doc',
    'duc', 'dur', 'equ', 'fact', 'fer', 'fid', 'fin', 'firm',
    'flex', 'form', 'fort', 'fract', 'gen', 'grad', 'graph', 'grav',
    'hab', 'ject', 'jud', 'junct', 'lat', 'leg', 'lev', 'liber',
    'loc', 'log', 'loqu', 'luc', 'magn', 'man', 'mand', 'mar',
    'mater', 'med', 'mem', 'ment', 'min', 'mir', 'miss', 'mit',
    'mob', 'mon', 'mort', 'mot', 'mov', 'mut', 'nat', 'neg',
    'nom', 'norm', 'nov', 'numer', 'oper', 'ord', 'pac', 'par',
    'pass', 'path', 'pater', 'ped', 'pel', 'pend', 'pet', 'phil',
    'phon', 'plic', 'pon', 'port', 'pos', 'pot', 'prim', 'prin',
    'priv', 'prob', 'prov', 'pugn', 'punct', 'quer', 'quest',
    'rect', 'reg', 'rupt', 'sacr', 'scend', 'sci', 'scrib', 'script',
    'sect', 'sed', 'sens', 'sent', 'sequ', 'serv', 'sign', 'simil',
    'sol', 'solv', 'son', 'spec', 'spir', 'sta', 'stit', 'str',
    'struct', 'sum', 'tact', 'temp', 'ten', 'tend', 'tent', 'term',
    'terr', 'test', 'tract', 'trib', 'turb', 'typ', 'ultim',
    'vac', 'val', 'ven', 'ver', 'verb', 'vers', 'vert', 'vid',
    'vis', 'vit', 'viv', 'voc', 'vol',
}

# Known borrowings from French/Latin/Greek/Arabic
KNOWN_BORROWINGS = {
    'army', 'court', 'state', 'power', 'country', 'city', 'place',
    'point', 'matter', 'number', 'order', 'service', 'war', 'age',
    'story', 'office', 'cause', 'reason', 'nice', 'terrible',
    'beautiful', 'government', 'dangerous', 'impossible', 'discover',
    'manufacture', 'enthusiasm', 'candidate', 'salary', 'calculate',
    'secretary', 'magazine', 'algorithm', 'algebra', 'assassin',
    'admiral', 'alcohol', 'cardinal', 'companion', 'quarantine',
    'muscle', 'janitor', 'intern', 'minister', 'sinister',
    'trivial', 'egregious', 'sycophant', 'disaster', 'inaugurate',
    'investigate', 'exorbitant', 'extravagant', 'sophisticated',
    'serendipity', 'lieutenant', 'mortgage', 'preposterous',
    'pandemonium', 'glamour', 'juggernaut', 'pedigree', 'alchemy',
    'vermicelli', 'peninsula', 'miscreant', 'dilapidated',
    'quintessential', 'decimation', 'treacle',
    'communication', 'entertainment', 'establishment',
    'independence', 'responsibility', 'philosophical',
    'international', 'organization', 'representative',
    'particularly', 'unfortunately', 'uncomfortable',
    'agreement', 'movement', 'powerful',
}


def count_morphemes(word):
    """
    Count derivation layers by stripping prefixes and suffixes.
    Returns (layers, remaining_root).
    """
    w = word.lower()
    layers = 0

    # Strip prefixes
    prefix_found = True
    while prefix_found and len(w) > 2:
        prefix_found = False
        for p in PREFIXES:
            if w.startswith(p) and len(w) - len(p) >= 2:
                w = w[len(p):]
                layers += 1
                prefix_found = True
                break

    # Strip suffixes
    suffix_found = True
    while suffix_found and len(w) > 2:
        suffix_found = False
        for s in SUFFIXES:
            if w.endswith(s) and len(w) - len(s) >= 2:
                w = w[:-len(s)]
                layers += 1
                suffix_found = True
                break

    return layers, w


def estimate_ed(word):
    """
    Estimate etymological depth of a word.

    Rules:
    1. If in PRIMWOERTER set → ed = 1
    2. Count morphological layers (prefixes + suffixes)
    3. Add +1 if word is a known borrowing (Latin/Greek/Arabic/French)
    4. Cap at 5
    """
    w = word.lower()

    # Rule 1: Primwörter
    if w in PRIMWOERTER:
        return 1

    # Rule 2: Count morphemes
    layers, root = count_morphemes(w)

    # Base ed from morphological complexity
    if layers == 0:
        base_ed = 1  # monomorphemic
    else:
        base_ed = min(layers + 1, 4)  # +1 because root itself is a morpheme

    # Rule 3: Borrowing bonus
    if w in KNOWN_BORROWINGS:
        base_ed = min(base_ed + 1, 5)

    # Rule 4: Cap
    return min(base_ed, 5)


# ─────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    test_words = [
        'I', 'you', 'water', 'fire', 'hand', 'eye',
        'teacher', 'quickly', 'undo', 'sunrise', 'freedom',
        'husband', 'understand', 'breakfast',
        'beautiful', 'wonderful', 'impossible', 'nightmare',
        'nice', 'silly', 'terrible', 'awful', 'awesome',
        'unfortunately', 'uncomfortable', 'international',
        'manufacture', 'enthusiasm', 'calculate', 'salary',
        'sophisticated', 'egregious', 'sycophant', 'trivial',
        'antidisestablishmentarianism', 'serendipity',
        'algorithm', 'assassin', 'alcohol',
        'preposterous', 'pandemonium', 'quintessential',
    ]

    print(f"{'Word':<35} {'ed':>3}  {'layers':>6}  {'root':<15}")
    print("-" * 65)
    for w in test_words:
        ed = estimate_ed(w)
        layers, root = count_morphemes(w)
        print(f"{w:<35} {ed:>3}  {layers:>6}  {root:<15}")
