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

##### How to run (exact commands)
```shell
cd "/Users/nini/Library/Mobile Documents/com~apple~CloudDocs/dev_icloud/clo_genai"
CLO_WRITE_ARTIFACTS=1 python pricing/clo_pricing.py
```
A) Train and save bundle from CLI
```shell
python3 main.py train-save \
  --data-path "input/raw/CLO_Tranche_Data.xlsx" \
  --bundle-path "models/model_bundle.joblib" \
  --synth-rows 500 \
  --similarity-real-only
```
This saves model bundle to:
models/model_bundle.joblib

B) Launch UI
```
streamlit run app/ui.py
```
In UI:
Upload user input
Click Load Bundle + Price Query Row to get predicted price, bounds, feature importance, and comparable deals


### Note:
1. pricing training data are from both real + synthetic; however, similarity reference data are from real only, 
to provide more color of real comparable deals.
2. The current LLM memo module is still a stub/offline sample, not a live API integration yet.
This is for future work, not included in this MVP version.

# Pipeline Overview

This project follows a two-layer architecture:

- **Training pipeline (offline)**: prepares data, trains pricing models, evaluates them, and saves model artifacts
- **Production pipeline (online / app)**: accepts a new tranche input, runs preprocessing and feature generation, produces pricing outputs, and returns similar real historical deals

This design keeps pricing and comparable-deal retrieval as two parallel outputs. The pricing model estimates a target value such as spread, while the similarity module returns ranked real historical deals for interpretability, trust, and comps-based workflow support.

---

## Training Pipeline (Offline)

The training pipeline builds the pricing model and prepares the real historical similarity reference set used later in production.

### Step by step

1. **Load raw historical Excel**
   - Read the real CLO tranche dataset from the input folder
   - This real historical dataset is the source of truth for comparable-deal retrieval and historical metadata

2. **Load or generate synthetic bootstrap data**
   - The pipeline can either:
     - load a pre-generated bootstrap synthetic dataset from file, or
     - generate synthetic data on the fly using the repository bootstrap module
   - Synthetic data is used only to improve pricing-model training robustness on a small sample
   - Synthetic data is **not** used for similar-deal retrieval in the UI

3. **Create two separate datasets**
   - **Pricing dataset**: real + synthetic
   - **Similarity reference dataset**: real historical data only

4. **Preprocess the pricing dataset**
   - Clean missing values
   - Standardize column names and data types
   - Parse date columns
   - Encode categorical variables
   - Scale numeric variables where needed
   - This preprocessing branch is used for pricing-model training

5. **Build pricing features**
   - Generate the model-ready feature matrix from the pricing dataset
   - Separate:
     - `X_train_pricing`
     - `y_train_pricing`

6. **Preprocess the real-only similarity dataset**
   - Apply the same cleaning and feature-generation logic to the real historical data only
   - Transform the real historical deals into the same comparable feature space used by production inference

7. **Build the similarity reference set**
   - Store the processed real historical feature vectors
   - Store real historical metadata needed for UI display, such as Bloomberg ID, manager, spread, trade date, and other descriptive fields
   - Fit or store a nearest-neighbor reference index on the real-only feature matrix

8. **Train pricing models**
   - Train baseline and nonlinear pricing models on the pricing dataset:
     - Linear Regression
     - Ridge Regression
     - XGBoost

9. **Evaluate pricing models**
   - Compute validation metrics such as RMSE, MAE, and R²
   - Compare model performance
   - Select the best pricing model for deployment

10. **Generate pricing explainability artifacts**
    - Create feature-importance outputs for the selected pricing model
    - Save charts and summary tables for demo / inspection

11. **Estimate pricing uncertainty**
    - Use validation residuals from the selected pricing model
    - Build a simple pricing band or uncertainty flag
    - This uncertainty belongs to the pricing module, not to the similarity module

12. **Save model bundle**
    - Save all artifacts required by the production pipeline, including:
      - selected trained pricing model
      - preprocessing objects
      - feature column definitions
      - residual statistics for uncertainty
      - real-only similarity feature matrix
      - real-only similarity metadata
      - optional fitted KNN object

---

## MVP Workflow

The production pipeline is used when a user enters a new tranche in the UI or through the API.

### Step by step

1. **Receive user input**
   - Accept tranche characteristics from:
     - manual form input
     - uploaded Excel row / standardized input file

2. **Load saved model bundle**
   - Load the trained pricing model
   - Load preprocessing objects
   - Load the real-only similarity reference data

3. **Preprocess user input**
   - Apply the exact same cleaning and transformation logic used during training

4. **Generate feature vector**
   - Convert the new tranche input into the same model-ready feature representation

5. **Run pricing model**
   - Feed the processed feature vector into the saved best pricing model
   - Return:
     - predicted spread / price
     - pricing range
     - uncertainty signal

6. **Run similarity search on real historical deals only**
   - Compare the processed user input against the stored real-only historical feature matrix
   - Retrieve the top-k nearest real historical deals
   - Return:
     - ranked comparable deals
     - similarity distances / scores
     - feature-level difference explanation

7. **Generate explainability outputs**
   - Pricing-side explainability:
     - feature importance for the pricing model
   - Similarity-side explainability:
     - which features most contributed to closeness / distance from the comparable real deals

8. **Aggregate results**
   - Combine:
     - pricing output
     - uncertainty signal
     - real similar deals
     - explainability outputs

9. **Return results to API / UI**
   - Display all outputs in the frontend for the end user

10. **Optional memo generation**
   - Pass structured outputs into an optional LLM module
   - Generate a short natural-language memo summarizing:
     - pricing view
     - comparable deals
     - key drivers
     - uncertainty note

---

## Design Rules

- **Synthetic data is used only for pricing-model training**
- **Similarity / comparable-deal retrieval uses real historical deals only**
- **Pricing and similarity are parallel outputs**
- **The same preprocessing and feature-generation logic is shared across training and production**

---

## Interpretation of Each Module

### Pricing module
Answers:

> What should this tranche’s spread or valuation range be?

Uses:
- real + synthetic training data

Outputs:
- predicted spread / price
- pricing range
- uncertainty signal
- pricing feature importance

### Similarity module
Answers:

> Which real historical deals look most similar to this tranche?

Uses:
- real historical data only

Outputs:
- ranked real comparable deals
- similarity distance / score
- top feature differences driving similarity

---

## Summary

**Training:** real data for comps + real/synthetic data for pricing-model training  
**Production:** user input goes through one shared preprocessing path, then branches into pricing output and real-deal similarity output in parallel
