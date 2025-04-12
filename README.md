# NLP-Sentiment-Fairness

Codebase for my master's thesis analyzing transformer-based sentiment models for demographic bias using SST-2 and CALM datasets. This repository includes training scripts, model evaluation, and fairness metrics computation across gender and race subgroups.

##  Repository Structure

| File Name | Description |
|-----------|-------------|
| `1_fine_tuning_roberta_on_sst_2.py` | Fine-tunes RoBERTa on SST-2 sentiment classification dataset |
| `4_applying_models_on_calm_+_bias_analysis.py` | Applies sentiment-trained models on CALM and computes fairness metrics |
| `apply_trained_models_on_sst2.py` | Applies demographic classifiers (trained on CALM) on SST-2 for reverse evaluation |
| `bias_metrics for second methodology.py` | Computes SPD, JSD, and WD on SST-2 predictions made by demographic classifiers |
| `bias_mitigation_(adversarial_debiasing).py` | Implements adversarial debiasing to mitigate bias in MentalBERT predictions |
| `Computing All Bias Metrics.py` | Aggregates and calculates all core bias metrics (SPD, EOD, JSD, WD) for group comparisons |
| `customdataset.py` | Custom dataset class for structured preprocessing and formatting |
| `datasetclassification.py` | Classification utilities for demographic prediction models |
| `fine_tuning_mentalbert_on_sst_2.py` | Fine-tunes MentalBERT on SST-2 for baseline sentiment predictions |
| `finetune_roberta_and_mentalbert_on_calm.py` | Trains RoBERTa and MentalBERT on CALM for demographic classification |
| `SieBERT and truthtable+metrics.py` | Performs evaluation using SieBERT and computes EOD, FPR, FNR from truth tables |
| `README.md` | Project overview, descriptions, and usage summary |

##  Main Methodologies

- **Approach 1:** Train sentiment models on SST-2 → Apply on CALM → Evaluate demographic bias  
- **Approach 2:** Train demographic models on CALM → Apply on SST-2 → Evaluate sentiment bias  
- **Bias Metrics Used:** SPD, EOD, FPR, FNR, JSD, WD

##  Note
All Hugging Face API tokens have been removed or commented out for security.

##  Thesis
This code supports the experiments and results presented in my master's thesis (2025). For details, see the appendix or contact the author.
