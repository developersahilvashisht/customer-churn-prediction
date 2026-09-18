"""
Shared feature engineering logic.

This function is used both when training the model
(build_pipeline.py) and when serving predictions (app.py), so that new/unseen
data is transformed in EXACTLY the same way as training data.
"""
import numpy as np
import pandas as pd


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to a raw customer dataframe.

    Expects raw Telco-schema columns: tenure, TotalCharges, MonthlyCharges,
    and the eight service columns listed below.
    """
    data = data.copy()

    # Feature 1: tenure_group - lifecycle-stage bucket
    bins = [-1, 12, 24, 48, 60, np.inf]
    labels = ['0-12mo', '13-24mo', '25-48mo', '49-60mo', '61-72mo']
    data['tenure_group'] = pd.cut(data['tenure'], bins=bins, labels=labels)

    # Feature 2: num_services - count of subscribed services (stickiness proxy)
    service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                     'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    data['num_services'] = (data[service_cols] == 'Yes').sum(axis=1)

    # Feature 3 & 4: avg_monthly_spend and charge_increase
    safe_tenure = data['tenure'].replace(0, 1)
    data['avg_monthly_spend'] = data['TotalCharges'] / safe_tenure
    data['charge_increase'] = data['MonthlyCharges'] - data['avg_monthly_spend']

    return data
