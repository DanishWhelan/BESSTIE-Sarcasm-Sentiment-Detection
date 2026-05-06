# BESSTIE-Sarcasm-Sentiment-Detection
Group coursework for COMM061 Natural Language Processing at the University of Surrey.  Sentiment analysis and sarcasm detection across British, Australian, and Indian English  using the BESSTIE dataset exploring classical ML baselines, RoBERTa fine-tuning,  cross-variety evaluation, and LoRA adapters on TinyLlama-1.1B.

# COMM061 NLP Group Coursework

**University of Surrey Semester 2, 2025–26**  
Hashir Ahmed, Saida Iman, Amna Abid, Rahul Rawat, Danish Whelan

---

## What is this?

This is our group coursework for the COMM061 Natural Language Processing module. 
The task was to build models that can detect sarcasm and sentiment in English text, 
but specifically across three different varieties of English — British, Australian, 
and Indian — using a dataset called BESSTIE.

The interesting part is figuring out whether a model trained on one variety 
(say, Australian English) can actually understand sarcasm in Indian English. 
Spoiler: it struggles, mostly because Indian English has a lot of Hindi mixed in 
and cultural references that Australian training data just doesn't cover.

---

## What we did

We ran three main experiments:

1. **Baselines vs RoBERTa** compared simple TF-IDF classifiers against a 
   fine-tuned RoBERTa model to see how much pre-training actually helps for 
   sarcasm detection.

2. **Cross-variety testing** trained a model on each variety separately, 
   then tested it on all three. Built a full 3x3 results matrix to see 
   how well (or badly) models transfer across dialects.

3. **LoRA adapters** used LoRA on TinyLlama-1.1B to train lightweight 
   variety-specific adapters. The idea is you keep one frozen base model 
   and just swap a 4MB adapter file depending on which variety you need.

We also built a small Flask app that lets you type in text, pick a variety, 
and get a prediction back.

---

## Results (brief)

Sarcasm turned out to be much harder than sentiment — especially for Indian 
and British English where sarcastic examples make up less than 8% of the data. 
RoBERTa with class-weighted loss got the best pooled result (Macro-F1 of 0.692), 
and the Australian LoRA adapter got 0.687 on its own variety which was 
surprisingly close.

---

## How to run it

You will need conda and a GPU (we used CUDA 11.8).

```bash
# Clone and set up
git clone https://github.com/[username]/COMM061-NLP-BESSTIE.git
cd COMM061-NLP-BESSTIE

conda create -n nlp_comm061 python=3.10 -y
conda activate nlp_comm061

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Then open `main.ipynb` in VSCode, select the `nlp_comm061` kernel, and run 
all cells from top to bottom. The dataset loads automatically from HuggingFace 
so you don't need to download anything manually.

## Files

- `main.ipynb` — everything is in here, split into sections matching the report
- `requirements.txt` — all the packages you need
- `figures/` — plots saved during the notebook run
- `results/` — CSV files with the model results

The trained model weights are not included (too large). If you need them, 
the LoRA adapters can be retrained in about 20-30 minutes on a free GPU.

---

## Dataset

We used the BESSTIE dataset which loads from HuggingFace:

```python
from datasets import load_dataset
ds = load_dataset("surrey-nlp/BESSTIE-CW-26")
```

Do not commit the dataset to the repo.

---

## References

- Srirag et al. (2025) BESSTIE paper (the dataset we used)
- Liu et al. (2019) RoBERTa
- Hu et al. (2022) LoRA
- Skalicky & Crossley (2018) — linguistic features of sarcasm

---

*Repo is private until after the submission deadline (6th May 2026, 4PM).*
