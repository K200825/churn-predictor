# Grocery Subscription Churn Predictor + Retention Simulator

A machine learning web application that predicts subscription churn risk for online grocery delivery customers and simulates the impact of retention strategies — built on real thesis survey data from the German market.

**🌐 [Live Demo](https://your-app.streamlit.app)** ← update after deployment

---

## What it does

### Tab 1 — Churn Predictor
Enter a customer's profile (demographics, service perceptions, subscription attitudes) and get an instant churn probability with tailored retention recommendations.

### Tab 2 — Retention Simulator
Set a baseline customer, then adjust pricing, perks, and service sliders to simulate how different retention interventions reduce churn risk before spending budget.

### Tab 3 — Model Insights
Explore the key churn drivers from the survey data, regression findings, and model architecture.

---

## Research background

This project extends the thesis:
> *Consumer and Company Attitudes Towards Subscription Models in Grocery Delivery (German Market)*  
> Karthik Nagaraju — M.Sc. Supply Chain Management, SRH University Hamm, 2025–26

**Key findings from 54 survey responses:**
- Subscription intent mean: 3.04/5 (neutral) — rises to 3.5 with reasonable pricing, 3.6 with extra benefits
- Strongest churn drivers: usage uncertainty (Q17, r = -0.60) and cost concern (Q18, r = -0.50)
- Strongest retention drivers: ease of shopping (Q8, β = 0.88) and delivery savings (Q14, β = 0.41)
- R² = 0.70 (regression explains 70% of variance in subscription intent)

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/churn-predictor.git
cd churn-predictor

# 2. Install
pip install -r requirements.txt

# 3. Train the model (takes ~5 seconds)
python train_model.py

# 4. Launch the app
streamlit run app.py
```

---

## Deploy for free (Streamlit Cloud)

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set `app.py` as main file
4. Add a startup command in `packages.txt` to run `train_model.py` first

Or add this `.streamlit/config.toml`:
```toml
[server]
runOnSave = false
```

And add to `app.py` top (already included): model auto-trains if not found.

---

## Project structure

```
churn-predictor/
├── app.py                   ← Streamlit web app (3 tabs)
├── train_model.py           ← Model training script
├── data/
│   └── survey_responses.xlsx ← Real thesis survey data (54 responses)
├── model/
│   └── churn_model.pkl      ← Trained model (auto-generated)
├── requirements.txt
└── README.md
```

---

## Model details

| Parameter | Value |
|---|---|
| Algorithm | Gradient Boosting Classifier |
| Training data | 54 real survey responses (German grocery delivery users) |
| Features | 22 (14 Likert-scale attitudes + 8 demographic/behavioural) |
| Validation | 5-fold Stratified Cross-Validation |
| F1 Score | ~0.96 |
| Accuracy | ~0.96 |

**Top churn predictors:**
1. Preference for per-order pricing (Q16) — 30% importance
2. Usage hesitation (Q17) — 10%
3. Subscription appeal (Q13) — 10%
4. Order frequency — 10%
5. Cost concern (Q18) — 7%

---

## Skills demonstrated

- **Machine learning** — Gradient Boosting, cross-validation, feature importance
- **Survey data analysis** — Likert scale preprocessing, churn label engineering
- **Supply chain domain knowledge** — subscription model dynamics, customer retention strategy
- **Streamlit web app** — interactive UI, 3-tab layout, live prediction
- **Thesis integration** — ML tool built directly on own research findings

---

## Author

**Karthik Nagaraju** — M.Sc. Supply Chain Management & Logistics, SRH University Hamm  
[linkedin.com/in/karthik-nagaraju](https://linkedin.com/in/karthik-nagaraju)
