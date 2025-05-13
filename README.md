
# 🌎🔎 Climate‑Bias LLM Analysis
_A lightweight research pipeline to study how large language models internalise conflicting scientific viewpoints on Antarctic ice‑sheet melting._

## 🚀 Overview
We fine‑tune three LoRA adapters on top of **Llama‑3 8B**:

| Variant | Training Corpus | Research Stance |
|---------|-----------------|-----------------|
| **Side A** | 500 peer‑reviewed papers claiming anthropogenic greenhouse‑gas‑driven melt | “⁠Global Warming⁠” |
| **Side B** | 500 peer‑reviewed papers describing a natural melt–refreeze cycle | “⁠Natural Cycle⁠” |
| **Combined** | A + B (1 000 papers) | **Both sides** |

Bias is probed with deliberately controversial questions (e.g. “Is the past‑decade melt primarily anthropogenic?”) and quantified via sentiment polarity and Jensen–Shannon Divergence (JSD).

---

## 📝 Problem Statement
> **Can we measure—and later mitigate—bias when an LLM digests diametrically opposed climate‑science corpora?**

1. **Quantification:**  
   *Train*, *prompt*, and *score* the three model variants; report how much each one leans toward either stance.
2. **Mitigation (future work):**  
   Explore debiasing via contrastive instruction tuning or entropy‑regularised decoding.

---

## 📂 Repository Layout
```

bias-analysis/
├── data/               # CSVs & PDFs (scraped papers, cleaned text)
├── prompts/            # JSONL prompt suites (neutral, leading, controversial)
├── src/
│   ├── prepare\_data.py # Google‑Scholar scrape + PDF→text + cleaning
│   ├── train.py        # LoRA fine‑tune wrapper (UnsLoRA / PEFT)
│   ├── evaluate.py     # Run prompts, collect raw generations
│   ├── metrics.py      # JSD, sentiment, entropy, etc.
│   └── plots.py        # Matplotlib visualisations
└── README.md

````

---

## ⚙️ Quick‑Start

```bash
# Clone & create venv
git clone https://github.com/RohanPatil2/Bias.git
cd Bias && python -m venv .venv && source .venv/bin/activate

# Install deps (PEFT, transformers, sentencepiece, unsloth, textstat…)
pip install -r requirements.txt
````

### 1️⃣ Collect & Clean Papers

```bash
python src/prepare_data.py \
  --query '"Antarctic ice sheet" melt greenhouse gases'  --max_papers 500 --side A \
  --save_to data/sideA.csv

python src/prepare_data.py \
  --query '"Antarctic ice sheet" "natural cycle"'        --max_papers 500 --side B \
  --save_to data/sideB.csv
```

The script:

* scrapes Google Scholar with SerpAPI,
* downloads open‑access PDFs,
* strips boiler‑plate, tables, and references,
* drops texts < 1 000 tokens,
* writes a CSV of `{title, abstract, body, stance}`.

### 2️⃣ Fine‑Tune Adapters

```bash
# Side A
python src/train.py \
  --base_model meta-llama/Meta-Llama-3-8B \
  --train_file data/sideA.csv  --output_dir checkpoints/SideA \
  --lora_r 32 --q4

# Side B
python src/train.py --train_file data/sideB.csv --output_dir checkpoints/SideB

# Combined
python src/train.py \
  --train_file data/sideAB.csv --output_dir checkpoints/Combined
```

### 3️⃣ Probe with Controversial Prompts

```bash
python src/evaluate.py \
  --model_paths checkpoints/SideA checkpoints/SideB checkpoints/Combined \
  --prompts prompts/controversial.jsonl \
  --save generations/raw_outputs.jsonl
```

### 4️⃣ Score Bias

```bash
python src/metrics.py \
  --generations generations/raw_outputs.jsonl \
  --out_file results/metrics.csv
python src/plots.py --metrics results/metrics.csv --save_dir figures/
```

### 5️⃣ Inspect Results

Open the confusion‑matrix PNGs and boxplots in `figures/`, or embed them in your report.

---

## 📊 Key Result (current run)

| True＼Predicted    | NaturalCycle | GlobalWarming |
| ----------------- | ------------ | ------------- |
| **NaturalCycle**  | 6            | 9             |
| **GlobalWarming** | 3            | 12            |

*Sentiment‑derived bias accuracy: **60 %***
See `docs/AB_Confusion.png` and `docs/AB_True.png` for heat‑map and score distribution.

---

## 🔬 Methodological Notes

* **Model choice:** Llama‑3 8B was selected for open‑weights availability and LoRA support.
* **LoRA config:** rank 32, α = 16, dropout = 0.05, 4‑bit QLoRA (bnb 4‑bit) to fit a single 24 GB GPU.
* **Prompt template:**

  ```
  <s>[INST] You are an Antarctic‑Climate Research Assistant…
  {QUESTION}
  [/INST]
  ```

  (Neutral system message avoids anchoring bias.)
* **Bias metrics:**

  * *JSD* between probability distributions over stance labels.
  * *Sentiment delta* (TextBlob polarity) as a soft directional indicator.
  * *Entropy* of sampled answers (lower = more certain/possibly more biased).

---

## 🗺 Roadmap (next steps)

1. **Debiasing experiments:** contrastive loss, RL‑HF with a neutrality reward.
2. **Cross‑domain generalisation:** test the same adapters on Arctic‑melt corpora.
3. **Explainability:** SHAP on token‑level log‑odds to see which citations sway stance.

---

## 📚 References

* J. Smith *et al.* “Recent Antarctic Ice‑Sheet Mass Loss,” *Nature* (2024).
* K. Liu & P. Roberts “Multidecadal Natural Cycles in Polar Ice,” *Clim. Dyn.* (2023).
* Hu *et al.* “LoRA: Low‑Rank Adaptation of Large Language Models,” *ICLR* (2022).

---

## ⚖️ License

MIT © 2025 Rohan Patil



---

### 📌 Project Kick‑Off Checklist (theory + practice)

| Phase | Action Items | Tips & Rationale |
|-------|--------------|------------------|
| **1. Corpus Engineering** | *Design queries*, *scrape PDFs*, *deduplicate*, *clean text*, *label stance*. | Use **SerpAPI + Scholar‑scraper**.<br>Store raw PDFs and extracted plain‑text for reproducibility. |
| **2. Data Curation** | Filter papers<br>➜ ≥ 1 000 tokens, English only, remove references. | Longer contexts let the model learn richer scientific reasoning. |
| **3. Prompt Suite Design** | Build three JSONL sets: _neutral_, _leading_, _controversial_. | Keep question surface forms similar to isolate stance bias. |
| **4. Base‑Model Selection** | Use an open‑weights model ≥ 7 B params. | Llama‑3 8B balances quality vs. GPU cost; supports PEFT. |
| **5. Adapter Fine‑Tuning** | LoRA on 4‑bit QLoRA weights.<br>Log _loss_, _perplexity_, and a small dev prompt set. | LoRA avoids full‑model back‑prop; cheaper & repeatable. |
| **6. Inference & Logging** | Temperature = 0.7, top‑p = 0.95; capture full JSONL of generations. | Save _prompt_, _model‑id_, _raw‑answer_, _token_probs_. |
| **7. Bias Metrics** | Implement JSD, sentiment, entropy, KL divergence to a neutral prior. | Multiple views catch different flavours of bias. |
| **8. Visualisation** | Confusion matrices, violin plots of sentiment, JSD bar charts. | Helps communicate bias to non‑ML audiences. |
| **9. Reporting** | IEEE format is fine unless prof specifies otherwise.<br>Sections: *Intro, Related Work, Data, Method, Experiments, Results, Discussion, Future Work*. | Include screenshots of confusion‑matrix & plots. |
| **10. Demo Prep** | Jupyter notebook or Streamlit app: pick a prompt ➜ show three model outputs ➜ live metric calculation. | Professors love seeing bias numbers update on the fly. |

---

### 🔑 Key Theory Blocks to Mention

* **Retrieval‑Augmented Questioning**: Even without external docs, framing a question with _citations requested_ can reveal stance selection bias.
* **Confirmation vs. Congruence Bias**: A model fine‑tuned on Side A may exhibit confirmation bias; combined fine‑tuning may instead show _congruence bias_—agreeing with whichever side the prompt subtly favours.
* **JSD as Symmetric Divergence**: Preferred over KL because it’s finite and symmetric, making A vs. B comparisons fair.
* **LoRA Rank Trade‑off**: Higher rank captures nuanced language but risks overfitting the stance; monitor dev loss.

---

