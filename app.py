"""
app.py  —  Subscription Churn Predictor + Retention Simulator
Based on thesis survey data: Consumer Attitudes Toward Online Grocery
Delivery and Subscription Models (German Market)
Author: Karthik Nagaraju
"""

import pickle
import numpy as np
import pandas as pd
import streamlit as st
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Subscription Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = "model/churn_model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model not found. Run `python train_model.py` first.")
        st.stop()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

payload  = load_model()
model    = payload["model"]
FEATURES = payload["features"]
medians  = payload["medians"]
fi       = payload["fi"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(...)
    st.title("Retail Subscription Analytics")
    st.caption("Decision Support for Online Grocery Retail · German Market")
    st.divider()
    st.markdown(f"""
    **Model info**  
    📊 Training samples: `{payload['n_samples']}`  
    🎯 CV F1 Score: `{payload['cv_f1']:.3f}`  
    ✅ CV Accuracy: `{payload['cv_acc']:.3f}`  
    📉 Churn rate: `{payload['churn_rate']:.1%}`
    """)
    st.divider()
    st.markdown("""
    **Based on thesis research:**  
    *Consumer & Company Attitudes Toward Subscription Models in Grocery Delivery (Germany)*  
    — Karthik Nagaraju, SRH Hamm, 2025–26
    """)

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔍 Churn Prediction",
    "🎛️ Retention Strategy Simulator",
    "📊 Retail Analytics Insights"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Churn Predictor
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Subscription Churn Prediction")
    st.caption("Enter a customer's profile to predict their churn likelihood")

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.subheader("👤 Demographics")
        age_group = st.selectbox("Age group", ["18–24","25–34","35–44","45–54","55+"], index=1)
        gender    = st.selectbox("Gender", ["Male","Female"])
        occupation = st.selectbox("Occupation", ["Student","Working professional","Homemaker","Retired"])
        household  = st.slider("Household size", 1, 6, 2)
        monthly_spend = st.selectbox("Monthly grocery spend", [
            "Less than €100","€100–€199","€200–€299",
            "€300–€399","€400–€499","€500 or more"
        ], index=3)

        st.subheader("🛒 Usage Behaviour")
        order_freq = st.selectbox("Order frequency", [
            "Once a month or less","2–3 times a month",
            "Once a week","More than once a week"
        ], index=1)
        usage_dur = st.selectbox("Usage duration", [
            "Less than 6 months","6–12 months",
            "1–2 years","More than 2 years"
        ], index=1)
        spend_order = st.selectbox("Typical spend per order", [
            "Under €30","€30–59","€60–89","Over €90"
        ], index=1)

    with col_right:
        st.subheader("⭐ Service Perceptions (1=Strongly disagree, 5=Strongly agree)")
        Q7  = st.slider("Grocery delivery saves me time", 1, 5, 4)
        Q8  = st.slider("Delivery makes shopping easier", 1, 5, 4)
        Q9  = st.slider("Avoiding physical stores is important", 1, 5, 3)
        Q10 = st.slider("I trust product quality", 1, 5, 3)
        Q11 = st.slider("Delivery times are reliable", 1, 5, 3)
        Q12 = st.slider("Orders arrive accurately", 1, 5, 4)

        st.subheader("📋 Subscription Attitudes")
        Q13 = st.slider("I like the idea of a grocery subscription", 1, 5, 3)
        Q14 = st.slider("Free/discounted delivery would save money", 1, 5, 3)
        Q15 = st.slider("Subscription would increase my order frequency", 1, 5, 3)
        Q16 = st.slider("I prefer paying per order (not monthly)", 1, 5, 3)
        Q17 = st.slider("I hesitate — unsure how often I'd use it", 1, 5, 3)
        Q18 = st.slider("I worry subscription won't be worth the cost", 1, 5, 3)
        Q20 = st.slider("Would subscribe if price was reasonable", 1, 5, 3)
        Q21 = st.slider("Would subscribe with extra benefits", 1, 5, 3)

    st.divider()
    predict_btn = st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True)

    if predict_btn:
        FREQ_MAP    = {"Once a month or less":1,"2–3 times a month":2,
                       "Once a week":3,"More than once a week":4}
        DUR_MAP     = {"Less than 6 months":1,"6–12 months":2,
                       "1–2 years":3,"More than 2 years":4}
        SPEND_MAP   = {"Under €30":1,"€30–59":2,"€60–89":3,"Over €90":4}
        MONTHLY_MAP = {"Less than €100":1,"€100–€199":2,"€200–€299":3,
                       "€300–€399":4,"€400–€499":5,"€500 or more":6}
        AGE_MAP     = {"18–24":1,"25–34":2,"35–44":3,"45–54":4,"55+":5}
        OCC_MAP     = {"Student":0,"Working professional":1,"Homemaker":2,"Retired":3}

        input_vec = {
            "Q7_saves_time":      Q7,
            "Q8_easier":          Q8,
            "Q9_avoid_stores":    Q9,
            "Q10_trust":          Q10,
            "Q11_reliable":       Q11,
            "Q12_accuracy":       Q12,
            "Q13_like_sub":       Q13,
            "Q14_saves_money":    Q14,
            "Q15_freq_increase":  Q15,
            "Q16_prefer_per_order": Q16,
            "Q17_hesitate":       Q17,
            "Q18_cost_worry":     Q18,
            "Q20_price_cond":     Q20,
            "Q21_benefits_cond":  Q21,
            "order_freq_enc":     FREQ_MAP[order_freq],
            "usage_enc":          DUR_MAP[usage_dur],
            "spend_enc":          SPEND_MAP[spend_order],
            "monthly_enc":        MONTHLY_MAP[monthly_spend],
            "age_enc":            AGE_MAP[age_group],
            "gender_enc":         1 if gender == "Female" else 0,
            "occupation_enc":     OCC_MAP[occupation],
            "household_size":     household,
        }

        X_input = pd.DataFrame([input_vec])[FEATURES]
        prob = model.predict_proba(X_input)[0][1]
        pred = int(prob >= 0.5)

        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            colour = "🔴" if prob > 0.65 else "🟡" if prob > 0.35 else "🟢"
            risk_label = "HIGH RISK" if prob > 0.65 else "MEDIUM RISK" if prob > 0.35 else "LOW RISK"
            st.metric("Churn Probability", f"{prob:.1%}", delta=risk_label)
            st.markdown(f"### {colour} {risk_label}")

        with res_col2:
            st.metric("Retention Probability", f"{1-prob:.1%}")
            if pred == 1:
                st.warning("⚠️ This customer is likely to **not subscribe** or cancel")
            else:
                st.success("✅ This customer is likely to **subscribe and stay**")

        with res_col3:
            # Top risk factors
            top_risk = []
            if Q16 >= 4: top_risk.append("Prefers per-order pricing")
            if Q17 >= 4: top_risk.append("Unsure about usage frequency")
            if Q18 >= 4: top_risk.append("Cost-value concern")
            if Q13 <= 2: top_risk.append("Low subscription appeal")
            if Q14 <= 2: top_risk.append("Delivery savings not valued")
            if not top_risk: top_risk = ["No major risk signals"]
            st.markdown("**🚩 Key risk signals:**")
            for r in top_risk[:3]:
                st.markdown(f"- {r}")

        # Retention recommendation
        st.divider()
        st.subheader("💡 Retention Recommendation")
        if prob > 0.65:
            st.error(f"""
            **High churn risk detected.** Suggested interventions:
            - Offer a **free trial period** (removes commitment hesitation)
            - Highlight **cost savings** vs per-order pricing with real numbers
            - Target with **loyalty discount** in first 3 months
            - Send personalised email showing estimated savings based on their order frequency
            """)
        elif prob > 0.35:
            st.warning(f"""
            **Medium churn risk.** Suggested interventions:
            - Showcase **extra benefits** (faster delivery, priority slots)
            - Send a **price comparison** showing subscription vs per-order savings
            - Offer a **flexible pause option** to reduce commitment fear
            """)
        else:
            st.success(f"""
            **Low churn risk.** Suggested actions:
            - Upsell to **premium tier** with additional perks
            - Encourage referrals with a **friend discount**
            - Offer **annual subscription** at reduced rate to lock in loyalty
            """)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Retention Simulator
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🎛️ Retention Simulator")
    st.caption("Simulate how changes to pricing, service, and perks affect churn risk")

    st.info("ℹ️ Set a **baseline** customer profile on the left, then adjust the **intervention** sliders on the right to see the impact on churn risk.")

    sim_col1, sim_col2 = st.columns(2, gap="large")

    with sim_col1:
        st.subheader("📌 Baseline customer")

        b_age        = st.selectbox("Age", ["18–24","25–34","35–44","45–54","55+"], index=1, key="b_age")
        b_freq       = st.selectbox("Order frequency", ["Once a month or less","2–3 times a month","Once a week","More than once a week"], index=1, key="b_freq")
        b_duration   = st.selectbox("Usage duration", ["Less than 6 months","6–12 months","1–2 years","More than 2 years"], index=1, key="b_dur")
        b_Q13        = st.slider("Likes subscription idea (Q13)", 1, 5, 2, key="b13")
        b_Q14        = st.slider("Delivery savings valued (Q14)", 1, 5, 2, key="b14")
        b_Q15        = st.slider("Subscription increases frequency (Q15)", 1, 5, 3, key="b15")
        b_Q16        = st.slider("Prefers per-order pricing (Q16)", 1, 5, 4, key="b16")
        b_Q17        = st.slider("Hesitates — uncertain usage (Q17)", 1, 5, 4, key="b17")
        b_Q18        = st.slider("Worries about cost (Q18)", 1, 5, 4, key="b18")
        b_Q20        = st.slider("Would subscribe if price reasonable (Q20)", 1, 5, 2, key="b20")
        b_Q21        = st.slider("Would subscribe with extra benefits (Q21)", 1, 5, 3, key="b21")

    with sim_col2:
        st.subheader("🎯 After intervention")
        st.caption("Adjust these to simulate the effect of your retention strategy")

        i_Q13 = st.slider("Subscription appeal after campaign (Q13)", 1, 5, b_Q13, key="i13")
        i_Q14 = st.slider("Delivery savings — after pricing change (Q14)", 1, 5, b_Q14, key="i14")
        i_Q15 = st.slider("Expected usage frequency change (Q15)", 1, 5, b_Q15, key="i15")
        i_Q16 = st.slider("Per-order preference — after trial offer (Q16)", 1, 5, b_Q16, key="i16")
        i_Q17 = st.slider("Hesitation — after free trial (Q17)", 1, 5, b_Q17, key="i17")
        i_Q18 = st.slider("Cost worry — after discount offer (Q18)", 1, 5, b_Q18, key="i18")
        i_Q20 = st.slider("Price acceptance — after offer (Q20)", 1, 5, b_Q20, key="i20")
        i_Q21 = st.slider("Benefits acceptance — after perk add (Q21)", 1, 5, b_Q21, key="i21")

    FREQ_MAP2    = {"Once a month or less":1,"2–3 times a month":2,"Once a week":3,"More than once a week":4}
    DUR_MAP2     = {"Less than 6 months":1,"6–12 months":2,"1–2 years":3,"More than 2 years":4}
    AGE_MAP2     = {"18–24":1,"25–34":2,"35–44":3,"45–54":4,"55+":5}

    def build_vec(Q13, Q14, Q15, Q16, Q17, Q18, Q20, Q21, age, freq, dur):
        return {
            "Q7_saves_time":4,"Q8_easier":4,"Q9_avoid_stores":3,"Q10_trust":3,
            "Q11_reliable":3,"Q12_accuracy":4,
            "Q13_like_sub":Q13,"Q14_saves_money":Q14,"Q15_freq_increase":Q15,
            "Q16_prefer_per_order":Q16,"Q17_hesitate":Q17,"Q18_cost_worry":Q18,
            "Q20_price_cond":Q20,"Q21_benefits_cond":Q21,
            "order_freq_enc":FREQ_MAP2[freq],"usage_enc":DUR_MAP2[dur],
            "spend_enc":2,"monthly_enc":4,"age_enc":AGE_MAP2[age],
            "gender_enc":0,"occupation_enc":1,"household_size":2,
        }

    baseline_vec  = build_vec(b_Q13,b_Q14,b_Q15,b_Q16,b_Q17,b_Q18,b_Q20,b_Q21,b_age,b_freq,b_duration)
    interv_vec    = build_vec(i_Q13,i_Q14,i_Q15,i_Q16,i_Q17,i_Q18,i_Q20,i_Q21,b_age,b_freq,b_duration)

    X_base = pd.DataFrame([baseline_vec])[FEATURES]
    X_intv = pd.DataFrame([interv_vec])[FEATURES]

    p_base = model.predict_proba(X_base)[0][1]
    p_intv = model.predict_proba(X_intv)[0][1]
    delta  = p_base - p_intv

    st.divider()
    r1, r2, r3 = st.columns(3)
    with r1:
        st.metric("Baseline churn risk", f"{p_base:.1%}")
    with r2:
        colour = "🟢" if delta > 0 else "🔴"
        st.metric("After intervention", f"{p_intv:.1%}",
                  delta=f"{-delta:+.1%} churn", delta_color="inverse")
    with r3:
        if delta > 0.15:
            st.success(f"✅ Strong impact: reduced churn by {delta:.1%}")
        elif delta > 0.05:
            st.info(f"📈 Moderate impact: reduced churn by {delta:.1%}")
        elif delta < 0:
            st.error(f"⚠️ Backfire: churn increased by {abs(delta):.1%}")
        else:
            st.warning("Minimal impact — try adjusting Q16, Q17, or Q18")

    # Visual bar
    st.divider()
    st.markdown("### Risk Reduction Breakdown")
    impact_data = {
        "Subscription appeal (Q13)":       (b_Q13 - i_Q13),
        "Delivery savings (Q14)":           (b_Q14 - i_Q14),
        "Frequency intent (Q15)":           (b_Q15 - i_Q15),
        "Per-order preference (Q16)":       (b_Q16 - i_Q16),
        "Usage hesitation (Q17)":           (b_Q17 - i_Q17),
        "Cost concern (Q18)":               (b_Q18 - i_Q18),
        "Price acceptance (Q20)":           (i_Q20 - b_Q20),
        "Benefits acceptance (Q21)":        (i_Q21 - b_Q21),
    }
    impact_df = pd.DataFrame(impact_data.items(), columns=["Factor","Score change"])
    impact_df = impact_df[impact_df["Score change"] != 0].sort_values("Score change", ascending=False)
    if not impact_df.empty:
        st.dataframe(impact_df, use_container_width=True, hide_index=True)
    else:
        st.info("Change the sliders above to see the impact breakdown")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Insights
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("📊 Model Insights")

    st.subheader("🔑 Key Churn Drivers (Feature Importance)")
    st.caption("Based on training on 54 real survey responses from German grocery delivery users")

    fi_df = pd.DataFrame(fi.items(), columns=["Feature","Importance"]) \
              .sort_values("Importance", ascending=False).head(10)

    labels = {
        "Q16_prefer_per_order":   "Prefers per-order pricing",
        "Q17_hesitate":           "Hesitates (uncertain usage)",
        "Q13_like_sub":           "Likes subscription idea",
        "order_freq_enc":         "Order frequency",
        "Q18_cost_worry":         "Worries about cost",
        "Q15_freq_increase":      "Subscription increases frequency",
        "monthly_enc":            "Monthly grocery spend",
        "Q14_saves_money":        "Delivery savings valued",
        "Q21_benefits_cond":      "Would subscribe with benefits",
        "Q20_price_cond":         "Would subscribe at right price",
    }
    fi_df["Feature label"] = fi_df["Feature"].map(labels).fillna(fi_df["Feature"])
    fi_df["Importance %"] = (fi_df["Importance"] * 100).round(1)

    st.dataframe(fi_df[["Feature label","Importance %"]].reset_index(drop=True),
                 use_container_width=True)

    st.divider()
    st.subheader("📌 Key Thesis Findings (from survey data)")

    c1, c2, c3 = st.columns(3)
    c1.metric("Subscription intent (Q19 mean)", "3.04 / 5", delta="Neutral-leaning")
    c2.metric("Price-conditional intent (Q20)", "3.5 / 5", delta="+0.46 vs direct intent")
    c3.metric("Benefits-conditional (Q21)", "3.6 / 5", delta="+0.56 vs direct intent")

    st.markdown("""
    **Insights from the regression (R² = 0.70):**
    - **Q8 (ease of shopping)** is the strongest positive predictor of future subscription intent (β = 0.88, p < 0.01)
    - **Q14 (delivery savings)** significantly drives subscription likelihood (β = 0.41, p < 0.01)
    - **Q17 (usage hesitation)** is the strongest barrier to subscription (r = -0.60)
    - **Q18 (cost concern)** negatively correlates with subscription intent (r = -0.50)
    - Price and benefits reduce churn: intent rises from 3.04 → 3.5 → 3.6 when conditions are met
    
    **Strategic implication:** Retailers should lead with **convenience messaging** + **transparent cost savings**, 
    while offering **free trials** to overcome usage uncertainty — the primary churn driver.
    """)

    st.divider()
    st.subheader("🏗️ Model Architecture")
    st.markdown(f"""
    | Parameter | Value |
    |---|---|
    | Algorithm | Gradient Boosting Classifier |
    | Training samples | {payload['n_samples']} real survey responses |
    | Features | 22 (14 Likert + 8 demographic/behavioural) |
    | Cross-validation | 5-fold Stratified |
    | F1 Score | {payload['cv_f1']:.3f} |
    | Accuracy | {payload['cv_acc']:.3f} |
    | Churn definition | Q16 ≥ 4 (prefers per-order) OR Q19 ≤ 2 (unlikely to subscribe) |
    """)
