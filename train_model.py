"""
train_model.py
--------------
Trains a Gradient Boosting churn classifier on the real thesis survey data
and saves the model + feature list to model/churn_model.pkl.

Run once before launching the Streamlit app:
    python train_model.py
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

DATA_PATH  = "data/survey_responses.xlsx"
MODEL_PATH = "model/churn_model.pkl"

# ── Ordinal encodings (from thesis survey) ────────────────────────────────────
FREQ_MAP    = {"Once a month or less":1,"2–3 times a month":2,
               "Once a week":3,"More than once a week":4}
DUR_MAP     = {"Less than 6 months":1,"6–12 months":2,
               "1–2 years":3,"More than 2 years":4}
SPEND_MAP   = {"Under €20":1,"Under €30":1,"€20–40":2,
               "€30–59":2,"€60–89":3,"Over €90":4}
MONTHLY_MAP = {"Less than €100":1,"Less than €200":2,"€100–€199":2,
               "€200–€299":3,"€300–€399":4,"€400–€499":5,"€500 or more":6}
AGE_MAP     = {"18–24":1,"25–34":2,"35–44":3,"45–54":4,"55+":5}
OCC_MAP     = {"Student":0,"Working professional":1,"Homemaker":2,"Retired":3}

FEATURES = [
    "Q7_saves_time","Q8_easier","Q9_avoid_stores","Q10_trust",
    "Q11_reliable","Q12_accuracy","Q13_like_sub","Q14_saves_money",
    "Q15_freq_increase","Q16_prefer_per_order","Q17_hesitate",
    "Q18_cost_worry","Q20_price_cond","Q21_benefits_cond",
    "order_freq_enc","usage_enc","spend_enc","monthly_enc",
    "age_enc","gender_enc","occupation_enc","household_size"
]

COL_RENAME = {
    " What is your age group?": "age_group",
    "What is your occupation?": "occupation",
    "What is your gender?": "gender",
    "What is your country of origin?": "country",
    "How often do you order groceries online?": "order_freq",
    "How long have you been using online grocery delivery services?": "usage_duration",
    "What is your typical spending per order?": "spend_per_order",
    "How many people are in your household?": "household_size",
    "About how much does your household spend on groceries each month?": "monthly_spend",
    "Online grocery delivery saves me time.": "Q7_saves_time",
    "Online grocery delivery makes my shopping easier.": "Q8_easier",
    "Avoiding physical stores is important to me.": "Q9_avoid_stores",
    "I trust online grocery delivery services to deliver quality products.": "Q10_trust",
    "Delivery times are usually reliable.": "Q11_reliable",
    "My orders usually arrive accurately and without issues.": "Q12_accuracy",
    "I like the idea of having a subscription for grocery delivery": "Q13_like_sub",
    "A subscription with free or discounted delivery would save me money": "Q14_saves_money",
    "A subscription would increase my purchase frequency": "Q15_freq_increase",
    "I prefer paying per order instead of monthly fees.": "Q16_prefer_per_order",
    "I hesitate to subscribe because I am not sure how often I would use the service.": "Q17_hesitate",
    "I worry that subscriptions may not be worth the cost.": "Q18_cost_worry",
    "I am likely to subscribe to a grocery delivery service in the future.": "Q19_intent",
    "I would subscribe if the price was reasonable.": "Q20_price_cond",
    "I would subscribe if the plan included extra benefits, such as faster delivery or special offers.": "Q21_benefits_cond",
}


def load_and_prepare(path):
    df = pd.read_excel(path, sheet_name="Form responses 1")
    df = df[df["Have you ever used an online grocery delivery service?"] == "yes"].copy()
    df = df.dropna(subset=["I am likely to subscribe to a grocery delivery service in the future."])
    df.rename(columns=COL_RENAME, inplace=True)

    df["order_freq_enc"]  = df["order_freq"].map(FREQ_MAP).fillna(2)
    df["usage_enc"]       = df["usage_duration"].map(DUR_MAP).fillna(2)
    df["spend_enc"]       = df["spend_per_order"].map(SPEND_MAP).fillna(2)
    df["monthly_enc"]     = df["monthly_spend"].map(MONTHLY_MAP).fillna(3)
    df["age_enc"]         = df["age_group"].map(AGE_MAP).fillna(2)
    df["gender_enc"]      = (df["gender"] == "Female").astype(int)
    df["occupation_enc"]  = df["occupation"].map(OCC_MAP).fillna(1)
    df["household_size"]  = pd.to_numeric(df["household_size"], errors="coerce").fillna(2)

    df["churn"] = ((df["Q16_prefer_per_order"] >= 4) | (df["Q19_intent"] <= 2)).astype(int)

    X = df[FEATURES].fillna(df[FEATURES].median())
    y = df["churn"]
    return X, y, df[FEATURES].median()


def train(path=DATA_PATH):
    print(f"Loading data from {path}...")
    X, y, medians = load_and_prepare(path)
    print(f"  Samples: {len(X)} | Churn rate: {y.mean():.1%}")

    model = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05,
        max_depth=3, random_state=42
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
    acc_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"  CV F1:  {f1_scores.mean():.3f} ± {f1_scores.std():.3f}")
    print(f"  CV Acc: {acc_scores.mean():.3f} ± {acc_scores.std():.3f}")

    model.fit(X, y)

    fi = dict(zip(FEATURES, model.feature_importances_))

    os.makedirs("model", exist_ok=True)
    payload = {
        "model":    model,
        "features": FEATURES,
        "medians":  medians.to_dict(),
        "fi":       fi,
        "cv_f1":    float(f1_scores.mean()),
        "cv_acc":   float(acc_scores.mean()),
        "n_samples": int(len(X)),
        "churn_rate": float(y.mean()),
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)

    print(f"\nModel saved → {MODEL_PATH}")
    print("\nTop 5 churn drivers:")
    for feat, imp in sorted(fi.items(), key=lambda x: -x[1])[:5]:
        print(f"  {feat}: {imp:.3f}")
    return payload


if __name__ == "__main__":
    train()
