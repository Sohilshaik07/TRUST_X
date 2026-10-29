"""
TRUST-X Robustness Perturbation Module
Implements controlled linguistic and textual perturbations reflecting real-world code-mixed noise:
1. Character leet substitutions (e.g. a -> @, e -> 3, i -> 1, s -> $)
2. Character deletion (simulating casual typing typos)
3. Character repetition (simulating emphatic elongation, e.g., 'idiot' -> 'idiiiot')
4. Punctuation insertion (simulating obfuscation, e.g., 'hate' -> 'h.a.t.e')
5. Whitespace modification (simulating splitting or joining)
6. Keyboard neighbor typos (Qwerty layout distance)
"""

import random
import re

LEET_MAP = {
    'a': '@', 'A': '4',
    'e': '3', 'E': '3',
    'i': '1', 'I': '!',
    'o': '0', 'O': '0',
    's': '$', 'S': '$',
    't': '7', 'T': '7'
}

KEYBOARD_NEIGHBORS = {
    'q': 'w', 'w': 'e', 'e': 'r', 'r': 't', 't': 'y', 'y': 'u', 'u': 'i', 'i': 'o', 'o': 'p',
    'a': 's', 's': 'd', 'd': 'f', 'f': 'g', 'g': 'h', 'h': 'j', 'j': 'k', 'k': 'l',
    'z': 'x', 'x': 'c', 'c': 'v', 'v': 'b', 'b': 'n', 'n': 'm'
}

def perturb_leet_substitution(text, rate=0.15, rng=None):
    """Substitutes select Latin characters with common leet-speak obfuscations."""
    if rng is None:
        rng = random.Random(42)
    chars = list(text)
    for i, c in enumerate(chars):
        if c in LEET_MAP and rng.random() < rate:
            chars[i] = LEET_MAP[c]
    return "".join(chars)

def perturb_char_deletion(text, rate=0.08, rng=None):
    """Deletes internal word characters simulating casual typing drops."""
    if rng is None:
        rng = random.Random(42)
    words = text.split()
    new_words = []
    for w in words:
        if len(w) > 3 and rng.random() < 0.4:
            # Drop a character from middle
            drop_idx = rng.randint(1, len(w) - 2)
            w = w[:drop_idx] + w[drop_idx+1:]
        new_words.append(w)
    return " ".join(new_words)

def perturb_char_repetition(text, rate=0.25, max_repeats=3, rng=None):
    """Repeats vowels or consonants simulating emphatic social media comments."""
    if rng is None:
        rng = random.Random(42)
    words = text.split()
    new_words = []
    for w in words:
        if len(w) > 2 and rng.random() < rate:
            # Pick a character to elongate
            rep_idx = rng.randint(0, len(w) - 1)
            char = w[rep_idx]
            reps = rng.randint(2, max_repeats)
            w = w[:rep_idx] + (char * reps) + w[rep_idx+1:]
        new_words.append(w)
    return " ".join(new_words)

def perturb_punctuation_insertion(text, rate=0.15, rng=None):
    """Inserts dots/underscores between characters simulating evasion obfuscation."""
    if rng is None:
        rng = random.Random(42)
    words = text.split()
    new_words = []
    for w in words:
        if len(w) > 3 and rng.random() < rate:
            sep = rng.choice(['.', '_', '-'])
            w = sep.join(list(w))
        new_words.append(w)
    return " ".join(new_words)

def perturb_whitespace(text, rate=0.15, rng=None):
    """Splits single words or merges adjacent words."""
    if rng is None:
        rng = random.Random(42)
    words = text.split()
    if len(words) < 2:
        return text
    new_words = []
    i = 0
    while i < len(words):
        if i < len(words) - 1 and rng.random() < rate:
            # Merge adjacent words
            new_words.append(words[i] + words[i+1])
            i += 2
        else:
            new_words.append(words[i])
            i += 1
    return " ".join(new_words)

def apply_composite_perturbation(text, seed=42):
    """
    Applies a realistic composite perturbation:
    Combination of casual typographical noise + leet substitution.
    """
    rng = random.Random(seed)
    t = perturb_char_repetition(text, rate=0.15, rng=rng)
    t = perturb_leet_substitution(t, rate=0.12, rng=rng)
    t = perturb_char_deletion(t, rate=0.08, rng=rng)
    return t
