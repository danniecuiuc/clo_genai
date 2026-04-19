# clo_genai
CLO Bond Pricing Tool

### Group Member:
Danni Chen (dc3944), Grant Bloch (gb2934), Vijeet Prasad (vp2580), Zheng Sun (zs2503)

### Architecture
- preprocessing raw data (preprocessing module)
- synthetic-data extension (synthetic module)
- KNN-style comparable deal retrieval (similarity module)
- regression/XGBoost pricing (pricing module)
- explainability (pricing module)
- uncertainty signal (pricing module)
- (optional) memo generation (llm module) to provide contexts and explanations of the outputs.

### CLI commands to run this repo

#### train models
##### run below training pipeline: 
- Raw data → clean → features → synthetic data → train models → save model
```shell
python main.py train \
  --raw-input "input/raw/CLO Tranche Data.xlsx" \
  --synthetic-input "input/synthetic/bootstrap_with_real_on_top.xlsx" \
  --target Spread
```

####  App
uvicorn app.api:app --reload
streamlit run app/ui.py

### Note:
The current LLM memo module is still a stub/offline sample, not a live API integration yet.
This is for future work, not included in this MVP version.
