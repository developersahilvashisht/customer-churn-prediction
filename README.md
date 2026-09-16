# Customer Churn Prediction Project

Predicts whether a telecom customer is likely to churn, using the IBM Telco
Customer Churn dataset (7,043 customers, 21 features).

Github repo : https://github.com/developersahilvashisht/customer-churn-prediction/

## Project Structure

```
customer_churn_project/
├── data/
│   └── TelcoCustomerChurn.csv        # raw dataset
├── notebook/
│   ├── churn_analysis.ipynb          # full analysis notebook
│   └── figures/                      # all EDA / evaluation charts
├── model/
│   ├── churn_model.pkl               # final trained pipeline (preprocessing + model)
│   ├── metadata.json                 # feature list, chosen model, metrics
│   ├── model_comparison.csv          # metrics for all 3 tree configurations tried
│   └── eda_insights.json             # EDA figures used in the report
├── feature_engineering.py            # shared feature logic (used by BOTH training & API)
├── build_pipeline.py                 # trains the model end-to-end, saves artifacts
├── app.py                            # FastAPI REST API
├── requirements.txt
├── sample_request.json               # example POST body for /predict
├── sample_response.json              # example response
└── README.md
```

## Setup

Requires Python 3.10+.

```bash
cd customer_churn_project
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Reproduce the analysis & retrain the model

```bash
python3 build_pipeline.py
```

This will:
- Load and clean `data/TelcoCustomerChurn.csv`
- Generate 12 charts into `notebook/figures/`
- Engineer 4 new features
- Split data 70:30 (`random_state=42`)
- Train and compare 3 Decision Tree configurations
- Select the best model by F1 score
- Save the final pipeline to `model/churn_model.pkl`

The Jupyter notebook `notebook/churn_analysis.ipynb` contains the same
workflow with full narrative explanations, business insights per chart, and
all outputs already rendered — open it directly to review the analysis:

```bash
jupyter notebook notebook/churn_analysis.ipynb
```

## 2. Run the API

```bash
uvicorn app:app --reload --port 8000
```

Interactive Swagger docs: http://127.0.0.1:8000/docs. Use this page (not a
plain browser GET to `/predict`) to test the API interactively — `/predict`
only accepts POST requests

### Sample request

`POST /predict`

```json
{
  "gender": "Female",
  "SeniorCitizen": "No",
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 5,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 95.5,
  "TotalCharges": 480.0
}
```

### Sample response

```json
{
  "prediction": "Yes",
  "churn_probability": 0.8708
}
```

Try it with curl:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

Invalid input (e.g. `"gender": "Alien"` or a missing field) returns HTTP 422
with a field-level error message instead of a prediction.

## Key Results

| Model configuration | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Depth-5, default | 0.793 | 0.612 | 0.601 | 0.606 | 0.824 |
| **Deeper + class-balanced (final)** | **0.739** | **0.506** | **0.766** | **0.609** | **0.814** |
| Shallow, min-split 50 | 0.792 | 0.658 | 0.449 | 0.534 | 0.826 |

**Final model:** Decision Tree, `max_depth=8`, `min_samples_leaf=20`,
`class_weight='balanced'` — chosen for the highest F1 score and, more
importantly, for prioritizing **recall**.

**Top churn drivers:** Month-to-month contract type, low tenure, Fiber optic
internet service, high monthly charges, and lack of tech support / online
security add-ons.

## Notes on Design Decisions

- **`TotalCharges`** was stored as text with 11 blank entries — all belonging
  to brand-new customers with `tenure = 0` (never billed yet). Converted to
  numeric and filled with 0.
- **Train/test split happens before any preprocessing is fit**, and the
  `ColumnTransformer` is fit only on the training set, then applied to both
  test data and new API requests — this avoids data leakage.
