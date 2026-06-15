# Market-Level Rate Adequacy Benchmarking Using Aggregate Regulatory Data

**A Machine Learning and Credibility Approach with Evidence from Kenya**

2026 CAS Ratemaking Call Paper Program - Jeniffer Nasike Atetwe

---

## Overview

This repository contains the full data pipeline supporting the paper *"Market-Level Rate Adequacy Benchmarking Using Aggregate Regulatory Data: A Machine Learning and Credibility Approach with Evidence from Kenya."*

The pipeline demonstrates that machine learning models trained on **aggregate, publicly available insurance regulator statistics** (rather than policy-level data) can reconstruct market-level rate adequacy benchmarks, and characterises the information structure of such datasets.

The pipeline is **jurisdiction-agnostic**. While the demonstration uses Insurance Regulatory Authority (IRA) Kenya Annual Insurance Industry Statistics for 2023–2024, the code is designed for adaptation to any regulator publishing insurer-by-line aggregate statistics, including NAIC Annual Statement data.

## What the Pipeline Does

1. **Data ingestion** : reads IRA Kenya Excel workbooks (insurer-by-class premiums, claims, reinsurance, balance sheets, P&L)
2. **Feature engineering** : constructs 9 features from aggregate appendices, classified into accounting-linked vs. structural exogenous groups
3. **Model training** : Elastic Net, Random Forest, LightGBM under temporal holdout (train 2023, test 2024)
4. **Robustness analysis** : re-estimates models excluding underwriting margin
5. **Information structure audit** : quantifies predictive contribution of each feature group
6. **Lead-lag analysis** : tests temporal persistence of the target variable
7. **Credibility-ML hybrid** : Buhlmann-Straub credibility weighting of LightGBM predictions
8. **SHAP analysis** : feature attribution for both LightGBM and Random Forest
9. **Decision framework output** : translates signals into actuarial recommendations

## Repository Structure

```
.
├── pipeline.py           # Main pipeline script
├── requirements.txt       # Python dependencies
├── LICENSE                 # MPL 2.0
├── README.md
├── data/                   # Place IRA Kenya Excel files here (not included)
└── outputs/                # Generated figures, tables, and summary JSON
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/cas-ratemaking-2026.git
cd cas-ratemaking-2026
```

### 2. Install dependencies

It is recommended to use a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Obtain the data

The IRA Kenya Annual Insurance Industry Statistics are publicly available at
[https://www.ira.go.ke](https://www.ira.go.ke) under "Publications" / "Annual Reports."

Download the 2023 and 2024 Annual Insurance Industry Statistics workbooks and place them in the `data/` folder:

```
data/IRA_Kenya_Annual_Statistics_2023.xlsx
data/IRA_Kenya_Annual_Statistics_2024.xlsx
```

> **Note:** These files are not included in this repository due to their size and because they are third-party regulatory publications. They are freely downloadable from the IRA Kenya website.

### 4. Configure file paths (if needed)

If your downloaded files have different names, update `FILE_2023` and `FILE_2024` near the top of `pipeline.py`:

```python
FILE_2023 = "data/IRA_Kenya_Annual_Statistics_2023.xlsx"
FILE_2024 = "data/IRA_Kenya_Annual_Statistics_2024.xlsx"
```

If the IRA changes its appendix numbering in future publications, update the `APPENDIX_MAP` dictionary to match the new sheet names.

## Running the Pipeline

```bash
python pipeline.py
```

The script will print progress for each of the 9 steps to the console, and write all outputs to `outputs/`:

**Figures (300 dpi PNG):**
- `fig1_icr_by_class.png` : ICR distribution by class (EDA)
- `fig_heatmap_2023.png`, `fig_heatmap_2024.png` : insurer-class ICR heatmaps
- `fig4_actual_vs_predicted.png` : model fit comparison
- `fig5_credibility_weights.png` : Buhlmann-Straub Z distribution
- `fig6_rate_adequacy_signals.png` : signals by class
- `fig7_shap_importance.png` : SHAP bar chart
- `fig8_shap_beeswarm.png` : SHAP beeswarm plot

**Tables (CSV):**
- `table4_model_performance.csv`
- `table5_class_signals.csv`
- `table6_shap_importance.csv`
- `table7_decision_framework.csv`
- `full_rate_adequacy_signals.csv`

**Summary:**
- `pipeline_summary.json` : all key statistics referenced in the paper

## Adapting to Another Jurisdiction

To apply this framework to a different regulator's aggregate statistics (e.g. NAIC Annual Statement data):

1. Update `CLASSES` with the line-of-business names used in your jurisdiction (see Table 1 of the paper for the IRA Kenya ↔ ISO/NAIC mapping)
2. Update `APPENDIX_MAP` to point to the relevant sheets in your data files
3. Update `FILE_2023` / `FILE_2024` (or extend to additional years if available)
4. Run `python pipeline.py`

All downstream steps : feature engineering, modelling, the information structure audit, the credibility hybrid, SHAP analysis, and the decision framework :  require no further changes.

## Citation

If you use this pipeline, please cite:

> Atetwe, J. N. (2026). *Market-Level Rate Adequacy Benchmarking Using Aggregate Regulatory Data: A Machine Learning and Credibility Approach with Evidence from Kenya.* Casualty Actuarial Society Ratemaking Call Paper Program.

## License

This project is licensed under the Mozilla Public License 2.0 — see [LICENSE](LICENSE) for details.

## Contact

Jeniffer Nasike Atetwe — jeniffernasike@gmail.com
