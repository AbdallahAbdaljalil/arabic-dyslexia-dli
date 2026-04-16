"""
Arabic Dyslexia Readability Scorer
====================================
Computes a dyslexia-adjusted readability score for Arabic text by combining:
  - Visual load (dots, diacritics)
  - Orthographic load (shape similarity, PSC letters)
  - Lexical load (word frequency difficulty, homograph risk)

References:
  - https://arxiv.org/abs/2506.18399 (Saeed & Habash, 2025 — Arabic lemmatization)
  - https://github.com/CAMeL-Lab/camel_tools (morphological analysis, tokenization, disambiguation)
"""

import re
import numpy as np
import pandas as pd

from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.disambig.mle import MLEDisambiguator
from camel_tools.tokenizers.word import simple_word_tokenize

# ==============================================================================
# 1. CONSTANTS & LOOKUP TABLES
# ==============================================================================

# Arabic diacritics (tashkeel)
ARABIC_DIACRITICS = {
    '\u064b', '\u064c', '\u064d',   # tanwin
    '\u064e', '\u064f', '\u0650',   # fatha, damma, kasra
    '\u0651', '\u0652',             # shadda, sukun
}

# Arabic alphabet range (excludes diacritics / punctuation)
ARABIC_ALPHABET_RE = re.compile(r'[\u0621-\u064A]')

# Number of dots per dotted letter
DOT_MAP = {
    'ب': 1, 'ت': 2, 'ث': 3,
    'ج': 1, 'خ': 1, 'ذ': 1,
    'ز': 1, 'ش': 3, 'ض': 1,
    'ظ': 1, 'غ': 1, 'ف': 1,
    'ق': 2, 'ن': 1, 'ي': 2,
    'ة': 2,  # dot-like visual loops (design choice)
}

# Groups of visually similar letter shapes
SIMILAR_SHAPE_GROUPS = [
    {'ب', 'ت', 'ث', 'ن', 'ي'},
    {'ج', 'ح', 'خ'},
    {'د', 'ذ'},
    {'ر', 'ز'},
    {'س', 'ش'},
    {'ص', 'ض'},
    {'ط', 'ظ'},
    {'ع', 'غ'},
    {'ف', 'ق'},
]

# Letters with position-sensitive (chameleon) shapes
PSC_SET = {'ع', 'غ', 'ه', 'ك', 'م', 'ج', 'ح', 'خ'}


# ==============================================================================
# 2. MODEL SETUP  (load once at module level)
# ==============================================================================

# Takes ~30 seconds to load on first import
_db = MorphologyDB.builtin_db()
analyzer = Analyzer(_db)
mle = MLEDisambiguator.pretrained()


# ==============================================================================
# 3. TEXT CLEANING HELPERS
# ==============================================================================

def clean_arabic_for_matching(text: str) -> str:
    """
    Strip diacritics and non-Arabic characters, then normalise whitespace.
    Used to prepare text for word-level lookups.
    """
    text = str(text)
    text = re.sub(r'[\u064B-\u0652]', '', text)          # remove tashkeel
    text = re.sub(r'[^\u0621-\u064A\s]', '', text)       # keep letters + spaces
    return ' '.join(text.split())

def clean_arabic_keep_diacritics(text: str) -> str:
    """
    Strip punctuation and non-Arabic characters but KEEP diacritics.
    Used when we need to check whether a word was originally diacritized.
    """
    text = str(text)
    text = re.sub(r'[^\u0621-\u064A\u064B-\u0652\s]', '', text)
    return ' '.join(text.split())


# ==============================================================================
# 4. CHARACTER / LETTER METRICS
# ==============================================================================

def char_length(sentence: str) -> int:
    """Count of base Arabic letters (diacritics excluded)."""
    return len(get_clean_letters(sentence))


def all_char_length(sentence: str) -> int:
    """Count of all characters after stripping diacritics and spaces."""
    sentence = re.sub(r'[\u064B-\u0652]', '', sentence)
    return len(sentence.replace(' ', ''))


def has_diacritics(sentence: str) -> int:
    """Return 1 if the sentence contains any diacritic mark, else 0."""
    return int(any(ch in ARABIC_DIACRITICS for ch in sentence))

def get_clean_letters(sentence):
    return [ch for ch in sentence if ARABIC_ALPHABET_RE.match(ch)]


# ==============================================================================
# 5. DOT / VISUAL LOAD METRICS
# ==============================================================================

def _dot_count(sentence: str, count_all_dots: bool = True) -> int:
    """
    Sum dot counts for each letter.
    If count_all_dots=False, count only whether a letter *has* dots (0 or 1).
    """
    if count_all_dots:
        return sum(DOT_MAP.get(ch, 0) for ch in sentence)
    return sum(1 for ch in sentence if ch in DOT_MAP)


def dot_load_ratio(sentence: str) -> float:
    """Average number of dots per letter (Dot Load Ratio)."""
    n = char_length(sentence)
    return round(_dot_count(sentence, count_all_dots=True) / n if n else 0.0, 2)


def dotted_letter_proportion(sentence: str) -> float:
    """Proportion of letters that carry at least one dot (DLP)."""
    n = char_length(sentence)
    return round(_dot_count(sentence, count_all_dots=False) / n if n else 0.0, 2)


def ovl(sentence: str) -> float:
    """Diacritic density: diacritics per base letter (Orthographic Visual Load)."""
    n = char_length(sentence)
    if not n:
        return 0.0
    diac_count = sum(1 for ch in sentence if ch in ARABIC_DIACRITICS)
    return round(diac_count / n, 2)


# ==============================================================================
# 6. SHAPE CONFUSION METRICS
# ==============================================================================

def similar_shape_density(sentence: str) -> float:
    """Proportion of letters that belong to a visually similar shape group."""
    letters = get_clean_letters(sentence)
    if not letters:
        return 0.0
    count = sum(
        1 for ch in letters
        if any(ch in group for group in SIMILAR_SHAPE_GROUPS)
    )
    return round(count / len(letters), 2)


def psc_chameleon_prop(sentence: str) -> float:
    """Proportion of letters that are position-sensitive chameleon letters."""
    letters = get_clean_letters(sentence)
    if not letters:
        return 0.0
    return round(sum(1 for ch in letters if ch in PSC_SET) / len(letters), 2)


# ==============================================================================
# 7. LEXICAL METRICS
# ==============================================================================


def homograph_risk_density(sentence: str, samer_surface_homographs: set) -> float:
    words_with_diacritics = clean_arabic_keep_diacritics(sentence).split()
    cleaned_words = clean_arabic_for_matching(sentence).split()

    if not words_with_diacritics:
        return 0.0

    risk = 0
    for original, plain in zip(words_with_diacritics, cleaned_words):
        if has_diacritics(original):
            continue
        if plain in samer_surface_homographs:
            risk += 1

    return round(risk / len(words_with_diacritics), 2)


# https://camel-tools.readthedocs.io/en/latest/reference/camel_morphology_features.html

def get_accurate_lemmas_and_pos(sentence: str) -> list[tuple]:
    tokens = simple_word_tokenize(sentence)
    disambig_results = mle.disambiguate(tokens)
    
    result = []
    for res in disambig_results:
        if not res.analyses:  # skip tokens with no analysis (tatweel, punctuation etc.)
            continue

        analysis = res.analyses[0].analysis
        
        if analysis.get('source') == 'backoff':
            # strip prefixes and retry with analyzer
            word = res.word
            stripped = re.sub(r'^[وفبكل]+ال|^ال|^[وفبكل]', '', word)
            analyses = analyzer.analyze(stripped)
            if analyses and analyses[0].get('source') != 'backoff':
                result.append((analyses[0].get('lex', stripped), analyses[0].get('pos', 'noun')))
            else:
                result.append((stripped, 'noun'))  # last resort
        else:
            result.append((analysis['lex'], analysis['pos']))
    
    return result

def lexical_difficulty_density(sentence: str, samer_lookup_pos: dict, samer_lookup: dict) -> float:
    lemmas_pos = get_accurate_lemmas_and_pos(clean_arabic_for_matching(sentence))
    if not lemmas_pos:
        return 0.0
    
    levels = []
    for lemma, pos in lemmas_pos:
        key = f"{lemma}#{pos}"
        level = (
            samer_lookup_pos.get(key) or   # try lemma#pos first
            samer_lookup.get(lemma) or      # fall back to lemma only
            2.5                             # default if not found at all
        )
        
        levels.append(level)
    
    return round(sum(levels) / len(levels), 2)  # 1-5

# ==============================================================================
# 8. SYNTACTIC COMPLEXITY METRICS
# ==============================================================================

# Morphological features that indicate clitic attachment
CLITIC_FEATURES = ['prc0', 'prc1', 'prc2', 'prc3', 'enc0']


def avg_morphological_complexity(sentence: str) -> float:
    """
    Average morphological complexity across all words in a sentence.
    Uses MLE disambiguator for context-aware clitic counting.
    Max possible score per word = 5 (all clitic slots filled).
    """
    tokens = simple_word_tokenize(clean_arabic_for_matching(sentence))
    if not tokens:
        return 0.0

    results = mle.disambiguate(tokens)
    scores = []
    for r in results:
        if not r.analyses:
            continue  # skip tokens with no analysis (tatweel, punctuation etc.)
        a = r.analyses[0].analysis
        score = sum(1 for f in CLITIC_FEATURES if a.get(f, 'na') not in ('na', '0'))
        scores.append(score)

    return round(sum(scores) / len(scores), 2)


def show_morphological_breakdown(sentence: str):
    """
    Debug utility — shows diacritized form, segmentation, 
    and clitic score for each word in a sentence.
    """
    tokens = simple_word_tokenize(clean_arabic_for_matching(sentence))
    results = mle.disambiguate(tokens)
    for r in results:
        a = r.analyses[0].analysis
        seg = a.get('d3seg', r.word)
        diac = a.get('diac', r.word)
        score = sum(1 for f in CLITIC_FEATURES if a.get(f, 'na') not in ('na', '0'))
        print(f"Word: {r.word} | Diacritized: {diac} | Segments: {seg} | Clitic Score: {score}")

# ==============================================================================
# 9. COMPOSITE DYSLEXIA SCORE
# ==============================================================================

# RF feature importances derived from 8-feature GPT ratings (n=500, Section 3.7.1)

weights = {
    'DLP':                          0.3438,  
    'Avg_Morphological_Complexity': 0.1764,
    'Lexical_Difficulty':           0.1429,
    'OVL':                          0.1270, 
    'PSC_Density':                  0.0662,
    'Similar_Shape_Density':        0.0624,
    'Homograph_Risk':               0.0426,
    'Dot_Load_Ratio':               0.0388,
}


def calculate_DLI(row):
    score = 0

    # OVL — any diacritization adds visual noise for dyslexic readers
    if row['OVL'] > 0.80:    score += 3
    elif row['OVL'] > 0.40:  score += 2
    elif row['OVL'] > 0.10:  score += 1

    # DLP — even below-median dotting creates letter confusion
    if row['DLP'] > 0.50:    score += 2
    elif row['DLP'] > 0.30:  score += 1

    # Dot_Load_Ratio — threshold near mean to capture typical variation
    if row['Dot_Load_Ratio'] > 1.00:   score += 3
    elif row['Dot_Load_Ratio'] > 0.70: score += 2
    elif row['Dot_Load_Ratio'] > 0.50: score += 1

    # PSC_Density — lower threshold to capture typical chameleon letter presence
    if row['PSC_Density'] > 0.25:   score += 2
    elif row['PSC_Density'] > 0.15: score += 1

    # Similar_Shape_Density — unchanged, thresholds already discriminate well
    if row['Similar_Shape_Density'] > 0.80:   score += 3
    elif row['Similar_Shape_Density'] > 0.60: score += 2
    elif row['Similar_Shape_Density'] > 0.40: score += 1

    # Lexical_Difficulty — unchanged, SAMER scale already well-calibrated
    if row['Lexical_Difficulty'] > 3.50:   score += 3
    elif row['Lexical_Difficulty'] > 2.50: score += 2
    elif row['Lexical_Difficulty'] > 1.50: score += 1

    # Homograph_Risk — even a few ambiguous words create decoding difficulty
    if row['Homograph_Risk'] > 0.30:   score += 2
    elif row['Homograph_Risk'] > 0.10: score += 1

    # Avg_Morphological_Complexity — threshold slightly below mean to capture typical clitic load
    if row['Avg_Morphological_Complexity'] > 2.50:   score += 4
    elif row['Avg_Morphological_Complexity'] > 1.20: score += 3
    elif row['Avg_Morphological_Complexity'] > 0.70: score += 2
    elif row['Avg_Morphological_Complexity'] > 0.40: score += 1

    return score


# total weights sum to 1.0

def calculate_DLI_weighted(row):
    score = 0

    # DLP — weight 0.3438
    if row['DLP'] > 0.50:    score += 2 * weights['DLP']
    elif row['DLP'] > 0.30:  score += 1 * weights['DLP']

    # OVL — weight 0.1228
    if row['OVL'] > 0.80:    score += 3 * weights['OVL']
    elif row['OVL'] > 0.40:  score += 2 * weights['OVL']
    elif row['OVL'] > 0.10:  score += 1 * weights['OVL']

    # Dot_Load_Ratio — weight 0.0492
    if row['Dot_Load_Ratio'] > 1.00:   score += 3 * weights['Dot_Load_Ratio']
    elif row['Dot_Load_Ratio'] > 0.70: score += 2 * weights['Dot_Load_Ratio']
    elif row['Dot_Load_Ratio'] > 0.50: score += 1 * weights['Dot_Load_Ratio']

    # PSC_Density — weight 0.0407
    if row['PSC_Density'] > 0.25:   score += 2 * weights['PSC_Density']
    elif row['PSC_Density'] > 0.15: score += 1 * weights['PSC_Density']

    # Similar_Shape_Density — weight 0.0783
    if row['Similar_Shape_Density'] > 0.80:   score += 3 * weights['Similar_Shape_Density']
    elif row['Similar_Shape_Density'] > 0.60: score += 2 * weights['Similar_Shape_Density']
    elif row['Similar_Shape_Density'] > 0.40: score += 1 * weights['Similar_Shape_Density']

    # Lexical_Difficulty — weight 0.3303
    if row['Lexical_Difficulty'] > 3.50:   score += 3 * weights['Lexical_Difficulty']
    elif row['Lexical_Difficulty'] > 2.50: score += 2 * weights['Lexical_Difficulty']
    elif row['Lexical_Difficulty'] > 1.50: score += 1 * weights['Lexical_Difficulty']

    # Homograph_Risk — weight 0.0560
    if row['Homograph_Risk'] > 0.30:   score += 2 * weights['Homograph_Risk']
    elif row['Homograph_Risk'] > 0.10: score += 1 * weights['Homograph_Risk']

    # Avg_Morphological_Complexity — weight 0.3227
    if row['Avg_Morphological_Complexity'] > 2.50:   score += 4 * weights['Avg_Morphological_Complexity']
    elif row['Avg_Morphological_Complexity'] > 1.20: score += 3 * weights['Avg_Morphological_Complexity']
    elif row['Avg_Morphological_Complexity'] > 0.70: score += 2 * weights['Avg_Morphological_Complexity']
    elif row['Avg_Morphological_Complexity'] > 0.40: score += 1 * weights['Avg_Morphological_Complexity']

    return round(score, 2)



