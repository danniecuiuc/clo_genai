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

##### How to run
A) Train and save bundle from CLI
Run 
```shell
cd "/Users/nini/Library/Mobile Documents/com~apple~CloudDocs/dev_icloud/clo_genai"
export PYTHONPATH="$PWD" # SET YOUR WORKING DIRECTORY IF IMPORT ERROR
CLO_WRITE_ARTIFACTS=1 python pricing/clo_pricing.py
```
or 
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


# MVP Workflow

This project follows a two-layer architecture:

- **Training pipeline**: prepares data, trains pricing models, evaluates them, and saves model artifacts
- **Predicting pipeline**: accepts a new tranche input, runs preprocessing and feature generation, produces pricing outputs, and returns similar real historical deals

##### This design keeps pricing and comparable-deal retrieval as two parallel outputs. The pricing model estimates a target value such as spread, while the similarity module returns ranked real historical deals for interpretability, trust, and comps-based workflow support.
---

## Training Pipeline

The training pipeline builds the pricing model and prepares the real historical similarity reference set used later in production.

### Step by step

1. **Load raw historical data**

- Read the real CLO tranche dataset from Excel / CSV
- Standardize column names and data types
- Normalize the target column to Price
- Use real historical data as the source of truth for comparable-deal retrieval

2. **Generate synthetic data**

- Generate synthetic rows during training if enabled
- Supported methods: DDPM, bootstrap, CTGAN, GLOW
- DDPM can be selected as the default in the UI
- Synthetic data is used only for pricing-model training
- Synthetic data is not used for similarity retrieval by default

3. **Preprocess the pricing dataset**

- Clean missing values
- Coerce numeric columns
- Parse date columns when available
- Apply median imputation
- Apply standard scaling
- Use the same preprocessing logic during training and inference

4. **Build pricing features**

- Generate model-ready feature matrix from the pricing dataset
- Separate X_train_pricing and y_train_pricing
- Use numeric CLO features across tranche structure, collateral quality, deal economics, and market variables

5. **Build similarity reference set**

- Apply the same feature definitions to real historical data
- Store processed feature vectors
- Store real metadata for UI display
- Fit a KNN nearest-neighbor index for comparable-deal retrieval

6. **Train pricing models**

- Train Ridge Regression as the transparent baseline
- Train XGBoost as the nonlinear final model when available
- Fall back to Ridge-only mode if XGBoost is unavailable

7. **Evaluate pricing models**

- Compute MAE, Median Absolute Error, RMSE, and R²
- Compare Ridge and XGBoost performance when both are available
- Store metrics in the model bundle and show them in the UI

8. **Generate explainability outputs**

- Create feature-importance outputs
- Use Ridge coefficients for baseline importance
- Use XGBoost feature importance for nonlinear model drivers
- Save charts and tables when artifact writing is enabled

9. **Estimate pricing uncertainty**

- Generate lower and upper confidence bounds
- Use quantile XGBoost models when available
- Fall back to residual-based confidence bands otherwise
- Uncertainty belongs to the pricing module, not the similarity module

10. **Save model bundle**

- Save trained pricing model
- Save trained similarity model
- Save preprocessing objects
- Save feature definitions
- Save validation metrics
- Save residual / uncertainty information

11. **UI inference**

- Load the saved model bundle
- User enters one new CLO tranche
- Apply trained preprocessing
- Generate predicted price and uncertainty range
- Retrieve comparable historical deals
- Display price, bounds, feature drivers, and comparables in the UI

---

## Predicting Pipeline (User)
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

### Note:
1. pricing training data are from both real + synthetic; however, similarity reference data are from real only, 
to provide more color of real comparable deals.
2. The current LLM memo module is still a stub/offline sample, not a live API integration yet.
This is for future work, not included in this MVP version.

### Planned / Future Work
- Advanced synthetic data (DDPM / generative models)
- Better uncertainty modeling (quantile / Bayesian)
- Real-time market data integration
- LLM memo generation (currently stub)
- Better feature engineering (macro + credit)
- Model ensembling
- Cloud deployment


### Team Member Responsibilities
#### Grant Bloch
- Gathered market insights and domain knowledge
- Contributed to input data understanding
- Worked on pricing models and evaluation of model results
#### Zheng Sun
- Designed system architecture and workflow
- Implemented similarity model
- Implemented the UI infrastructure
#### Danni Chen
- Built modular codebase and integrated code from other team members
- Designed execution piplines, improved UI and codebase 
- Evaluated technical risks and future roadmap
#### Vijeet Prasad
- Tested and bootstrap synthetic data (bootstrap)
- Track project plan vs actual execution progress
- Compiled presentation slides
