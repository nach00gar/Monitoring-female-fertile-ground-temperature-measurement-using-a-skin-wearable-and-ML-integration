# Ovulation Analysis – Reproducible Pipeline

> **Status:** Prepared for peer review / journal submission.

This repository contains the full data processing and machine-learning
pipeline used to predict the ovulation cycle day from wearable temperature
sensors, as described in the accompanying paper.

## Quick start

### 1. Set up the environment

**Conda (recommended):**
```bash
conda env create -f environment.yml
conda activate ovulation
```

**pip:**
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```


### 2. Place input data

Copy the raw data files into `data/`:
```
data/medResultsTu.csv    – temperature measurements (semicolon-separated)
data/medNotes.csv        – free-text ovulation notes (semicolon-separated)
```

These files are **not included** in the repository due to privacy constraints.

## License

MIT © 2025 – see `LICENSE` for details.
