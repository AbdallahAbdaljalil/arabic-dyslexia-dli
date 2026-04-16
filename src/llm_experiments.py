"""
llm_experiments.py
------------------
LLM-based proxy evaluation experiments for the Dyslexia Load Index (DLI).
Supports zero-shot (pure and prompted) and few-shot scoring via OpenAI and Fanar APIs.

Experiment batches:
    Batch 1: GPT, scale 0-10, no expert persona (pilot)
    Batch 2: GPT + Fanar, scale 1-5, with expert persona (pilot, n=121)
    Batches 3-4: GPT + Fanar, scale 1-5, large validation set (n=1000)
    Feature importance: GPT, 8-feature weighted prompt (n=500)

Usage in notebook:
    from llm_experiments import (
        run_experiment_async, run_experiment_fanar,
        run_experiment_weighted, get_few_shot_examples,
        build_few_shot_prompt, compare_with_dli,
        ZERO_SHOT_PURE_B1, ZERO_SHOT_PROMPTED_B1,
        ZERO_SHOT_PURE, ZERO_SHOT_PROMPTED, ZERO_SHOT_WEIGHTED
    )
"""

import time
import json
import os
import asyncio
import pandas as pd
from scipy import stats
import openai
from dotenv import load_dotenv

load_dotenv(override=True)


# ==============================================================================
# CLIENT SETUP
# ==============================================================================

def get_client():
    """Initialize synchronous OpenAI client. Legacy — use get_async_client for new experiments."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    return openai.OpenAI(api_key=api_key)


def get_async_client():
    """Initialize async OpenAI client."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    return openai.AsyncOpenAI(api_key=api_key)


def get_fanar_client():
    """Initialize Fanar API client."""
    api_key = os.environ.get("FANAR_API_KEY")
    if not api_key:
        raise ValueError("FANAR_API_KEY environment variable not set.")
    return openai.AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.fanar.qa/v1"
    )


# ==============================================================================
# PROMPTS
# ==============================================================================
# Batch 1: ZERO_SHOT_PURE_B1, ZERO_SHOT_PROMPTED_B1 (scale 0-10, no persona)
# Batch 2: ZERO_SHOT_PURE, ZERO_SHOT_PROMPTED, build_few_shot_prompt (scale 1-5, with persona)
# Batches 3-4: ZERO_SHOT_PROMPTED, build_few_shot_prompt replicated on 1,000 sentences
# Feature importance: ZERO_SHOT_WEIGHTED (8 features, scale 1-5, with persona)
# ==============================================================================

# --- Batch 1 (scale 0-10, no expert persona) ---

ZERO_SHOT_PURE_B1 = """You are an expert in Arabic dyslexia and reading difficulty.

Rate how difficult the following Arabic sentence would be for a dyslexic Arabic native adult reader to decode,
that is, to read the words aloud, independent of meaning.
Use a scale of 0 to 10 where 0 is very easy and 10 is extremely difficult.

Sentence: {sentence}

Respond in English only, in JSON format with no extra text:
{{"score": <0-10>, "label": "<LOW|MODERATE|HIGH|CRITICAL>", "explanation": "<one sentence>"}}"""


ZERO_SHOT_PROMPTED_B1 = """You are an expert in Arabic dyslexia and reading difficulty.

Rate the following Arabic sentence on a Dyslexia Load Index (DLI) from 0 to 10, where:
- 0-2.5 = LOW: minimal visual and cognitive decoding burden
- 2.5-5.0 = MODERATE: noticeable load from dots, diacritics, or morphology
- 5.0-7.5 = HIGH: multiple sources of difficulty active simultaneously
- 7.5-10 = CRITICAL: severe barrier for a dyslexic reader

The main sources of dyslexia load in Arabic are:
1. Diacritic density (visual noise from vowel marks)
2. Dot-heavy letters like ث ش ض ظ ب ت ق
3. Visually similar letters clustering together (ب ت ث ن ي)
4. Shape-shifting letters whose form changes across word positions (ع غ ح)
5. Rare or unfamiliar words that can't be recognized as whole units
6. Ambiguous unvowelized words with multiple readings
7. Heavy clitic/affix attachment making words very long

Sentence: {sentence}

Respond in English only, in JSON format with no extra text:
{{"score": <0-10>, "label": "<LOW|MODERATE|HIGH|CRITICAL>", "explanation": "<one sentence>"}}"""


# --- Expert persona (Batches 2-4 and feature importance) ---

EXPERT_PERSONA = """You are a neuropsychologist with 15 years of clinical experience diagnosing and treating dyslexia in native Arabic-speaking adults. You hold a PhD in cognitive neuroscience with a specialization in Arabic reading disorders, and have published extensively on Arabic orthographic processing and dyslexia assessment. You are a native Arabic speaker and are deeply familiar with how Arabic script properties affect decoding for dyslexic readers."""


# --- Batch 2 only: scale 1-5, with expert persona ---
# (ZS Pure dropped in Batches 3-4 due to near-zero correlations)

ZERO_SHOT_PURE = EXPERT_PERSONA + """

Rate how difficult the following Arabic statement would be for a native Arabic-speaking adult with dyslexia to decode.

Use a scale of 1 to 5:
- 1 = Very easy to decode
- 2 = Easy
- 3 = Moderate
- 4 = Hard
- 5 = Very hard to decode

Each statement is independent.

Statement: {sentence}

Respond in English only, in JSON format with no extra text:
{{"score": <1-5>, "label": "<VERY_EASY|EASY|MODERATE|HARD|VERY_HARD>", "features": "<what properties make this statement easy or hard to decode, be incredibly concise and mention the features only, no extra explanation>"}}"""


# --- Batch 3-4: scale 1-5, with expert persona ---

ZERO_SHOT_PROMPTED = EXPERT_PERSONA + """

Rate how difficult the following Arabic statement would be for a native Arabic-speaking adult with dyslexia to decode.

Use a scale of 1 to 5:
- 1 = Very easy to decode
- 2 = Easy
- 3 = Moderate
- 4 = Hard
- 5 = Very hard to decode

The main sources of dyslexia load in Arabic are:
1. Diacritic density (visual noise from vowel marks)
2. Dot-heavy letters like ث ش ض ظ ب ت ق
3. Visually similar letters clustering together (ب ت ث ن ي)
4. Shape-shifting letters whose form changes across word positions (ع غ ح)
5. Rare or unfamiliar words that can't be recognized as whole units
6. Ambiguous unvowelized words with multiple readings
7. Heavy clitic/affix attachment making words very long

Each statement is independent.

Statement: {sentence}

Respond in English only, in JSON format with no extra text:
{{"score": <1-5>, "label": "<VERY_EASY|EASY|MODERATE|HARD|VERY_HARD>", "features": "<what properties make this statement easy or hard to decode, be incredibly concise and mention the features only, no extra explanation>"}}"""


# --- Feature importance analysis (Section 3.7.1): 8-feature weighted prompt ---

ZERO_SHOT_WEIGHTED = EXPERT_PERSONA + """

You are evaluating Arabic text for dyslexic decoding difficulty for a native Arabic-speaking adult with dyslexia.

For the following Arabic statement, do two things:

**1. Rate each feature's contribution to decoding difficulty (0-3):**
- 0 = not present / no impact
- 1 = low impact
- 2 = moderate impact
- 3 = high impact

The features are:
- **diacritics**: How many vowel marks (tashkeel) appear above/below letters. More diacritics = more visual noise competing for attention.
- **dotted_letter_proportion**: What proportion of letters in the sentence carry dots at all (e.g. ب ت ث ن ي ج خ). High when most letters are dotted.
- **dot_load_ratio**: How heavy the dot burden is per letter on average. High when many letters carry 2 or 3 dots (e.g. ث ش ق) rather than just 1. Distinct from proportion - a sentence can have few dotted letters but those letters carry many dots each.
- **similar_shapes**: How many letters share the same base shape and are distinguished only by dots (e.g. ب ت ث ن share one base shape; ج ح خ share another). High when visually confusable letter groups cluster together.
- **chameleon_letters**: How many letters change their visual form dramatically depending on position in the word (e.g. ع غ ه ك م ج ح خ). High when these shape-shifting letters appear frequently.
- **lexical_difficulty**: How rare or morphologically opaque the vocabulary is. High when words cannot be recognized as whole units and must be decoded letter by letter.
- **homograph_risk**: How many words lack diacritics and have multiple possible readings/pronunciations. High when undiacritized words are highly ambiguous.
- **morphological_complexity**: How many clitics (prefixes/suffixes) are attached to words. High when words carry heavy clitic attachment making them long and hard to segment.

**2. Give an overall difficulty score (1-5):**
- 1 = Very easy to decode
- 2 = Easy
- 3 = Moderate
- 4 = Hard
- 5 = Very hard to decode

Each statement is independent.

Statement: {sentence}

Respond in English only, in JSON format with no extra text:
{{
  "feature_ratings": {{
    "diacritics": <0-3>,
    "dotted_letter_proportion": <0-3>,
    "dot_load_ratio": <0-3>,
    "similar_shapes": <0-3>,
    "chameleon_letters": <0-3>,
    "lexical_difficulty": <0-3>,
    "homograph_risk": <0-3>,
    "morphological_complexity": <0-3>
  }},
  "additional_factors": "<any other properties not listed above that affect decoding difficulty, or 'none'>",
  "overall_score": <1-5>,
  "label": "<VERY_EASY|EASY|MODERATE|HARD|VERY_HARD>"
}}"""


# ==============================================================================
# FEW-SHOT HELPERS
# ==============================================================================

# Legacy: Batch 1 few-shot used DLI_10 bands (scale 0-10)
# def get_few_shot_examples_batch1(df_full, n_per_band=2, random_state=42):
#     bands = [(0, 2), (2, 4), (4, 6), (6, 10)]
#     examples = []
#     for low, high in bands:
#         subset = df_full[(df_full['DLI_10'] >= low) & (df_full['DLI_10'] < high)]
#         sample = subset.sample(min(n_per_band, len(subset)), random_state=random_state)
#         for _, row in sample.iterrows():
#             examples.append({'sentence': row['Sentence'], 'score': row['DLI_10']})
#     return examples


def get_few_shot_examples(df_full, n_per_band=2, random_state=42):
    """Sample example sentences across DLI_1_5_norm bands for few-shot prompting.
    Used in Batches 2, 3, and 4."""
    bands = [(1, 2), (2, 3), (3, 4), (4, 5.1)]
    examples = []
    for low, high in bands:
        subset = df_full[
            (df_full['DLI_1_5_norm'] >= low) &
            (df_full['DLI_1_5_norm'] < high)
        ]
        sample = subset.sample(min(n_per_band, len(subset)), random_state=random_state)
        for _, row in sample.iterrows():
            examples.append({
                'sentence': row['Sentence'],
                'score': row['DLI_1_5_norm']
            })
    return examples


def build_few_shot_prompt(examples, sentence):
    """Build a few-shot prompt for Batches 2-4 (scale 1-5, with expert persona)."""
    examples_text = "\n\n".join([
        f"Statement: {e['sentence']}\nScore: {e['score']}"
        for e in examples
    ])

    return EXPERT_PERSONA + f"""

Rate how difficult the following Arabic statement would be for a native Arabic-speaking adult with dyslexia to decode.

Use a scale of 1 to 5:
- 1 = Very easy to decode
- 2 = Easy
- 3 = Moderate
- 4 = Hard
- 5 = Very hard to decode

Each statement is independent.

Here are some examples:

{examples_text}

Now rate this statement:
Statement: {sentence}

Respond in English only, in JSON format with no extra text:
{{"score": <1-5>, "label": "<VERY_EASY|EASY|MODERATE|HARD|VERY_HARD>", "features": "<what properties make this statement easy or hard to decode, be incredibly concise and mention the features only, no extra explanation>"}}"""


# ==============================================================================
# ASYNC EXPERIMENT RUNNERS
# ==============================================================================

async def run_experiment_async(sentences, prompt_fn, model="gpt-5-mini", max_concurrent=10):
    """
    Run LLM scoring in parallel on a list of sentences.
    Used for Batches 1-4 (zero-shot pure, prompted, and few-shot).

    Args:
        sentences: list of Arabic sentences
        prompt_fn: function that takes a sentence and returns a prompt string
        model: OpenAI model name
        max_concurrent: max simultaneous API requests

    Returns:
        DataFrame with columns: Statement, LLM_Score, LLM_Label, LLM_Explanation
    """
    client = get_async_client()
    semaphore = asyncio.Semaphore(max_concurrent)
    results = [None] * len(sentences)

    async def score_sentence(i, sentence):
        async with semaphore:
            try:
                prompt = prompt_fn(sentence)
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.choices[0].message.content.strip()
                content = content.replace("```json", "").replace("```", "").strip()
                parsed = json.loads(content)
                results[i] = {
                    'Statement': sentence,
                    'LLM_Score': parsed.get('score'),
                    'LLM_Label': parsed.get('label'),
                    'LLM_Explanation': parsed.get('features')
                }
            except Exception as e:
                print(f"Error on sentence {i+1}: {e}")
                results[i] = {
                    'Statement': sentence,
                    'LLM_Score': None,
                    'LLM_Label': None,
                    'LLM_Explanation': f"Error: {e}"
                }
            if (i + 1) % 10 == 0:
                print(f"Progress: {i+1}/{len(sentences)}")

    tasks = [score_sentence(i, s) for i, s in enumerate(sentences)]
    await asyncio.gather(*tasks)
    return pd.DataFrame(results)


async def run_experiment_fanar(sentences, prompt_fn, model="Fanar", max_concurrent=3, delay=1.5):
    """
    Run LLM scoring using Fanar API.
    Used for Batches 2 and 4.
    Lower concurrency and retry logic due to Fanar rate limits.
    """
    fanar_client = get_fanar_client()
    semaphore = asyncio.Semaphore(max_concurrent)
    results = [None] * len(sentences)

    async def score_sentence(i, sentence):
        async with semaphore:
            for attempt in range(3):
                try:
                    prompt = prompt_fn(sentence)
                    response = await fanar_client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You must respond in English only."},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    content = response.choices[0].message.content.strip()
                    content = content.replace("```json", "").replace("```", "").strip()
                    parsed = json.loads(content)
                    results[i] = {
                        'Statement': sentence,
                        'LLM_Score': parsed.get('score'),
                        'LLM_Label': parsed.get('label'),
                        'LLM_Explanation': parsed.get('features')
                    }
                    break
                except Exception as e:
                    if '429' in str(e) and attempt < 2:
                        await asyncio.sleep(60)
                    else:
                        print(f"Error on sentence {i+1}: {e}")
                        results[i] = {
                            'Statement': sentence,
                            'LLM_Score': None,
                            'LLM_Label': None,
                            'LLM_Explanation': f"Error: {e}"
                        }
            if (i + 1) % 10 == 0:
                print(f"Progress: {i+1}/{len(sentences)}")
            await asyncio.sleep(delay)

    tasks = [score_sentence(i, s) for i, s in enumerate(sentences)]
    await asyncio.gather(*tasks)
    return pd.DataFrame(results)


async def run_experiment_weighted(sentences, model="gpt-5-mini", max_concurrent=20):
    """
    Run 8-feature weighted prompt on a list of sentences.
    Used for feature importance analysis (Section 3.7.1, n=500).

    Returns:
        DataFrame with Overall_Score, Label, Additional_Factors,
        and individual gpt_* feature rating columns.
    """
    client = get_async_client()
    semaphore = asyncio.Semaphore(max_concurrent)
    results = [None] * len(sentences)

    async def score_sentence(i, sentence):
        async with semaphore:
            try:
                prompt = ZERO_SHOT_WEIGHTED.format(sentence=sentence)
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.choices[0].message.content.strip()
                content = content.replace("```json", "").replace("```", "").strip()
                parsed = json.loads(content)

                result = {'Statement': sentence}
                result['Overall_Score'] = parsed.get('overall_score')
                result['Label'] = parsed.get('label')
                result['Additional_Factors'] = parsed.get('additional_factors', 'none')

                for feat, val in parsed.get('feature_ratings', {}).items():
                    result[f'gpt_{feat}'] = val

                results[i] = result

            except Exception as e:
                print(f"Error on sentence {i+1}: {e}")
                results[i] = {'Statement': sentence, 'Overall_Score': None}

            if (i + 1) % 10 == 0:
                print(f"Progress: {i+1}/{len(sentences)}", flush=True)

        await asyncio.sleep(0.05)

    tasks = [score_sentence(i, s) for i, s in enumerate(sentences)]
    await asyncio.gather(*tasks)
    return pd.DataFrame(results)


# ==============================================================================
# LEGACY SYNCHRONOUS RUNNER
# ==============================================================================

def run_experiment(sentences, prompt_fn, model="gpt-5-mini", delay=0.5):
    """
    Legacy synchronous experiment runner. Slow — use run_experiment_async for new experiments.
    Kept for reproducibility of early Batch 1 results.
    """
    client = get_client()
    results = []

    for i, sentence in enumerate(sentences):
        try:
            prompt = prompt_fn(sentence)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.choices[0].message.content.strip()
            if i < 3:
                print(f"Raw response {i+1}: {repr(content)}")
            content = content.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(content)
            results.append({
                'Statement': sentence,
                'LLM_Score': parsed.get('score'),
                'LLM_Label': parsed.get('label'),
                'LLM_Explanation': parsed.get('explanation')
            })
        except Exception as e:
            print(f"Error on sentence {i+1}: {e}")
            results.append({
                'Statement': sentence,
                'LLM_Score': None,
                'LLM_Label': None,
                'LLM_Explanation': f"Error: {e}"
            })

        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{len(sentences)}")

        time.sleep(delay)

    return pd.DataFrame(results)


# ==============================================================================
# ANALYSIS
# ==============================================================================

def compare_with_dli(llm_df, dli_df, dli_col='DLI_1_5_norm', llm_col='LLM_Score', label='Experiment'):
    """
    Compare LLM scores against DLI scores.

    Args:
        llm_df: DataFrame with LLM results (must have Statement and LLM_Score columns)
        dli_df: DataFrame with DLI scores (must have Sentence and dli_col columns)
        dli_col: column name for DLI scores
        llm_col: column name for LLM scores
        label: experiment name for printing

    Returns:
        merged DataFrame with both scores
    """
    merged = llm_df.merge(
        dli_df[['Sentence', dli_col]].drop_duplicates(subset='Sentence'),
        left_on='Statement', right_on='Sentence',
        how='left'
    ).dropna(subset=[llm_col, dli_col])

    pearson_r, pearson_p = stats.pearsonr(merged[llm_col], merged[dli_col])
    spearman_r, spearman_p = stats.spearmanr(merged[llm_col], merged[dli_col])
    mae = (merged[llm_col] - merged[dli_col]).abs().mean()
    mean_diff = (merged[dli_col] - merged[llm_col]).mean()

    print(f"\n=== {label} ===")
    print(f"Pearson r:     {pearson_r:.4f}  (p={pearson_p:.4f})")
    print(f"Spearman r:    {spearman_r:.4f}  (p={spearman_p:.4f})")
    print(f"MAE:           {mae:.4f}")
    print(f"Mean Diff:     {mean_diff:.4f}  (DLI - LLM, positive = DLI scores higher)")
    print(f"LLM mean:      {merged[llm_col].mean():.4f}")
    print(f"DLI mean:      {merged[dli_col].mean():.4f}")
    print(f"N:             {len(merged)}")

    return merged