# Arabic Dyslexia Load Index (DLI)

This repository contains the code, data, and results for my undergraduate honors thesis at Carnegie Mellon University in Qatar: **"Dyslexia-Aware Modeling of Reading Difficulty in Arabic."**

The thesis introduces the **Dyslexia Load Index (DLI)**, a sentence-level computational metric designed to quantify orthographic decoding difficulty in Arabic text from the perspective of readers with dyslexia. The DLI is distinct from general readability — it targets the specific visual and phonological challenges Arabic script poses for dyslexic readers, such as dotted letters, visually similar characters, diacritization density, and morphological complexity.

---

## Background

Arabic presents a uniquely challenging orthographic environment for dyslexic readers. The script is cursive, right-to-left, and highly ambiguous without diacritics. Letters change shape depending on their position in a word, many letters share the same base form and are distinguished only by dots, and the morphological system produces long agglutinated word forms that are difficult to segment.

Existing Arabic readability metrics (OSMAN, AARI, BAREC) focus on general comprehension difficulty — vocabulary frequency, sentence length, syntactic complexity. None were designed with dyslexia-specific decoding challenges in mind. This thesis fills that gap.

---

## The DLI Pipeline

The DLI scores Arabic sentences across **8 orthographic and lexical features**:

| Feature | Abbreviation | Description |
|---------|-------------|-------------|
| Dotted Letter Proportion | DLP | Proportion of letters with one or more dots |
| Dot Load Ratio | DLR | Total dot count relative to letter count |
| Orthographic Vowel Load | OVL | Proportion of letters carrying diacritic vowel marks |
| Similar Shape Density | SSD | Proportion of letters sharing a base form with others |
| PSC/Chameleon Density | PSC | Proportion of letters that take different visual forms depending on word position |
| Lexical Difficulty | LD | Proportion of words not found in the SAMER readability lexicon |
| Homograph Risk | HR | Proportion of words whose base form maps to multiple diacritized lemmas |
| Morphological Complexity | MC | Average morphological analysis complexity via CAMeL Tools MLE disambiguator |

Each feature is scored against empirically derived thresholds (1 point per threshold crossed), and the raw scores are summed into a DLI ranging from 0–22, then normalized to a 1–5 scale.

---

## Key Findings

- **DLI vs. BAREC readability:** Pearson r = 0.107, Spearman ρ = 0.112, confirming that dyslexic decoding load and general comprehension difficulty are largely orthogonal constructs.
- **Quadrant analysis:** 26.8% of sentences are Low BAREC / High DLI, easy to understand but hard to decode. These are predominantly diacritized children's texts, which are visually demanding despite being semantically simple.
- **LLM proxy evaluation:** GPT-4o correlations ranged from r = 0.168 (Batch 1 ZS Pure) to r = 0.346 (Batch 2 ZS Prompted), providing external validation of the DLI's construct validity.
- **Feature importance:** Dotted Letter Proportion (DLP) dominates GPT feature ratings (RF importance = 0.344), followed by Morphological Complexity (0.176) and Lexical Difficulty (0.143).
- **SVR model:** An SVR trained on GPT silver labels achieved r = 0.509 on a held-out set of 1,000 sentences (large set, zero-shot prompted labels).

---

## Repository Structure

arabic-dyslexia-pilot/
│
├── src/
│   ├── main.py                    # DLI feature computation pipeline
│   ├── llm_experiments.py         # LLM proxy evaluation (GPT, Fanar)
│   └── svr_model.py               # SVR model training and inference
│
├── notebooks/
│   └── dli_analysis.ipynb         # Full analysis notebook
│
├── data/
│   └── lookups/
│       ├── samer_lookup.csv                  # SAMER lexicon surface forms
│       ├── samer_lookup_lemma.csv            # SAMER lexicon lemma-POS pairs
│       └── samer_surface_homographs.csv      # Pre-built homograph lookup
│
├── results/
│   ├── annotated/
│   │   ├── barec_dli_full_Final.csv          # Full BAREC corpus with DLI scores
│   │   └── barec_dli_annotated_final.csv     # Annotated corpus with all DLI variants
│   │
│   ├── validation/
│   │   ├── final_1000_sample.csv             # 1000-sentence validation sample
│   │   ├── llm_eval_sentences_with_dli_norm.csv   # 121-sentence pilot eval set
│   │   ├── llm_eval_sentences_with_quadrant.csv   # Pilot set with quadrant labels
│   │   ├── results_weighted_500_final.csv    # Weighted feature experiment results
│   │   ├── GPT/
│   │   │   ├── Batch_1/                      # GPT pilot (scale 0-10, no persona)
│   │   │   ├── Batch_2/                      # GPT pilot (scale 1-5, expert persona)
│   │   │   └── Batch_3/                      # GPT large set (n=1000)
│   │   └── Fanar/                            # Fanar model results (pilot + large set)
│   │
│   ├── models/
│   │   ├── svr_best.joblib                   # Best SVR model (large set ZS Prompted, r=0.509)
│   │   ├── svr_pilot_zs_prompted.joblib      # Pilot SVR (ZS Prompted, r=0.747)
│   │   └── svr_large_fewshot.joblib          # Large set SVR (Few-Shot)
│   │
│   └── figures/                              # Thesis figures

---

---

## Setup

### Requirements

- Python 3.9+
- CAMeL Tools (for morphological analysis)
- OpenAI API key (only needed to rerun LLM experiments)
- Fanar API key (only needed to rerun Fanar experiments)

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/arabic-dyslexia-pilot.git
cd arabic-dyslexia-pilot
pip install -r requirements.txt
```

### CAMeL Tools setup

CAMeL Tools requires the MorphologyDB to be downloaded separately:

```bash
camel_data -i morphology-db-msa-s31
```

### BAREC corpus

The raw BAREC corpus is not included in this repository, acquire it from HuggingFace:
https://huggingface.co/datasets/CAMeL-Lab/BAREC-Corpus-v1.0

Place the parquet files in `data/barec/`. The notebook can then recompute DLI features from scratch (Cells 6-7), though this takes approximately 30 minutes. The pre-computed annotations in `results/annotated/barec_dli_full_Final.csv` are included so you can skip this step.

---

### SAMER Lookup Tables
The lookup CSVs (`samer_lookup.csv`, `samer_lookup_lemma.csv`, `samer_surface_homographs.csv`) are derived from the SAMER Readability Lexicon, which is licensed by NYU Abu Dhabi and cannot be redistributed.

To generate them:
1. Request access to SAMER at: https://camel.abudhabi.nyu.edu/samer/
2. Place the lexicon files in `data/samer/`
3. Run the lookup generation script (details in `data/lookups/README.md`)

Without these files, the from-scratch DLI recomputation (Cells 6-7 in the notebook) will not work. However, the pre-computed annotations are included so you can run the full analysis without them.

---

## Running the Notebook

Open `notebooks/dli_analysis.ipynb`. The notebook is organized into 9 sections:

1. **Setup & Imports** — loads all dependencies and lookup tables
2. **Data Loading** — loads pre-computed DLI annotations from CSV (fast path)
3. **Corpus Analysis** — quadrant distribution, feature correlations, worked example
4. **LLM Proxy Evaluation** — Batches 1-4 (GPT and Fanar), correlation analysis
5. **Feature Importance** — RF/DT importance, GPT vs formula correlations
6. **Weighted DLI** — RF-weighted DLI variant
7. **Additional Factors** — GPT qualitative analysis categorization
8. **SVR Model** — trains and saves SVR models on LLM silver labels
9. **Figures** — generates all thesis figures

**API cells are clearly marked with WARNING comments, skip these if you want to use saved results (recommended).** All LLM experiment results are pre-saved in `results/validation/`.

The SVR models are pre-trained and saved in `results/models/`. To use them directly without retraining, uncomment the load cell in Section 8.

---

## Source Files

### `src/main.py`

Contains all DLI feature computation functions:
- `dotted_letter_proportion` — DLP
- `dot_load_ratio` — DLR  
- `ovl` — orthographic vowel load
- `similar_shape_density` — SSD
- `psc_chameleon_prop` — PSC density
- `lexical_difficulty_density` — LD using SAMER lexicon
- `homograph_risk_density` — HR using surface homograph lookup
- `avg_morphological_complexity` — MC using CAMeL Tools MLE disambiguator
- `calculate_DLI` — combines all features into a raw DLI score
- `calculate_DLI_weighted` — RF-importance weighted variant

### `src/llm_experiments.py`

Contains the LLM evaluation pipeline:
- Prompt templates for all 4 experimental conditions (ZS Pure, ZS Prompted, Few-Shot, Weighted)
- `run_experiment` / `run_experiment_async` — GPT evaluation functions
- `run_experiment_fanar` — Fanar evaluation function
- `run_experiment_weighted` — 8-feature weighted prompt experiment
- `compare_with_dli` — correlation analysis between LLM scores and DLI

### `src/svr_model.py`

Contains SVR training and inference:
- `train_svr` — trains SVR with StandardScaler, returns model, scaler, and results
- `predict_svr` — applies trained model to new sentences
- `save_model` / `load_model` — joblib serialization
- `FEATURE_COLS` — canonical feature column list

---

## Citation

If you use this work, please cite:

```bibtex
@thesis{abdaljalil2026dli,
  author    = {Abdallah Abdaljalil},
  title     = {Dyslexia-Aware Modeling of Reading Difficulty in Arabic},
  school    = {Carnegie Mellon University in Qatar},
  year      = {2026},
  type      = {Undergraduate Honors Thesis}
}
```

---

## Acknowledgments

**BAREC Corpus** — [CAMeL-Lab/BAREC-Corpus-v1.0](https://huggingface.co/datasets/CAMeL-Lab/BAREC-Corpus-v1.0)

```bibtex
@inproceedings{elmadani-etal-2025-readability,
    title = "A Large and Balanced Corpus for Fine-grained {A}rabic Readability Assessment",
    author = "Elmadani, Khalid N. and Habash, Nizar and Taha-Thomure, Hanada",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    year = "2025",
    url = "https://aclanthology.org/2025.findings-acl.842/"
}

@inproceedings{habash-etal-2025-guidelines,
    title = "Guidelines for Fine-grained Sentence-level {A}rabic Readability Annotation",
    author = "Habash, Nizar and Taha-Thomure, Hanada and Elmadani, Khalid N. and Zeino, Zeina and Abushmaes, Abdallah",
    booktitle = "Proceedings of the 19th Linguistic Annotation Workshop (LAW-XIX-2025)",
    year = "2025",
    url = "https://aclanthology.org/2025.law-1.30/"
}
```

**SAMER Readability Lexicon** — Licensed by NYU Abu Dhabi. Request access at https://camel.abudhabi.nyu.edu/samer-readability-lexicon/

```bibtex
@inproceedings{alkhalil-etal-2020-samer,
    title = "A Large-Scale Leveled Readability Lexicon for Standard Arabic",
    author = "Al Khalil, Muhamed and Habash, Nizar and Jiang, Zhengyang",
    booktitle = "Proceedings of LREC 2020",
    pages = "3053--3062",
    year = "2020"
}
```

**CAMeL Tools** — [github.com/CAMeL-Lab/camel_tools](https://github.com/CAMeL-Lab/camel_tools)

```bibtex
@inproceedings{obeid-etal-2020-camel,
    title = "{CAM}e{L} Tools: An Open Source Python Toolkit for {A}rabic Natural Language Processing",
    author = "Obeid, Ossama and Zalmout, Nasser and Khalifa, Salam and Taji, Dima and Oudah, Mai and Alhafni, Bashar and Inoue, Go and Eryani, Fadhl and Erdmann, Alexander and Habash, Nizar",
    booktitle = "Proceedings of LREC 2020",
    year = "2020",
    url = "https://www.aclweb.org/anthology/2020.lrec-1.868",
    pages = "7022--7032"
}
```

---

## License

MIT License. See `LICENSE` for details.

