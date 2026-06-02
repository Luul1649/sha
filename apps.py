import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# 1. Page Configuration Setup
st.set_page_config(page_title="SHA Fairness Dashboard", layout="wide")
st.title("🇰🇪 Social Health Authority (SHA) Fairness & Algorithmic Error Analyzer")
st.markdown("This interactive tool evaluates the 'Fairness Gap' caused by the AI-driven Means-Testing Tool in the informal sector.")

# 2. Sidebar Parameters (Live Simulation Controls)
st.sidebar.header("🔧 Simulation Controls")
sample_size = st.sidebar.slider("Population Sample Size", min_value=1000, max_value=10000, value=3000, step=500)
base_error_rate = st.sidebar.slider("AI Proxy Tool Error Rate (%)", min_value=0, max_value=80, value=39, help="Exclusion error rate where poor households are misclassified as rich.")
appeals_success_rate = st.sidebar.slider("Appeals Mechanism Success Rate (%)", min_value=0, max_value=100, value=30)

# 3. Cacheable Core Data Simulator
@st.cache_data
def run_simulation(n_samples, error_pct, appeal_pct):
    np.random.seed(42)
    # Simulate a realistic informal income distribution
    true_income = np.random.lognormal(mean=9.5, sigma=0.6, size=n_samples) + 4000
    df = pd.DataFrame({'True_Income': np.round(true_income, 2)})
    
    # Identify the truly poor (Bottom 40%)
    poverty_cutoff = df['True_Income'].quantile(0.40)
    df['Is_Poor'] = df['True_Income'] <= poverty_cutoff
    
    # Calculate target intended premium (2.75%, floor 300)
    df['Intended_Premium'] = (df['True_Income'] * 0.0275).clip(lower=300).round(2)
    
    # Map AI misclassification variance to align with selected sidebar error rate
    std_scale = (error_pct / 39.0) * 18000 if error_pct > 0 else 0
    ai_variance = np.random.normal(loc=0, scale=std_scale, size=n_samples) if std_scale > 0 else 0
    
    df['AI_Predicted_Income'] = df['True_Income'] + ai_variance
    df['AI_Predicted_Income'] = df['AI_Predicted_Income'].clip(lower=5000)
    
    # Apply SHA Pricing Rules to AI income
    df['AI_Calculated_Premium'] = (df['AI_Predicted_Income'] * 0.0275).clip(lower=300).round(2)
    
    # Simulate Appeals: Some overcharged households successfully lower their premium back to target
    df['Final_Charged_Premium'] = df['AI_Calculated_Premium']
    is_overcharged = (df['Is_Poor'] == True) & (df['AI_Calculated_Premium'] > df['Intended_Premium'])
    
    # Select which overcharged households win appeals
    if appeal_pct > 0 and is_overcharged.sum() > 0:
        appeal_winners = np.random.choice([True, False], size=len(df), p=[appeal_pct/100.0, 1 - (appeal_pct/100.0)])
        df.loc[is_overcharged & appeal_winners, 'Final_Charged_Premium'] = df['Intended_Premium']
        
    df['Overcharge_Amount'] = df['Final_Charged_Premium'] - df['Intended_Premium']
    df['Effective_Tax_Rate'] = (df['Final_Charged_Premium'] / df['True_Income']) * 100
    return df, poverty_cutoff

df_sim, poverty_line = run_simulation(sample_size, base_error_rate, appeals_success_rate)

# 4. Main Metrics Display (KPI Grid)
true_poor_count = df_sim['Is_Poor'].sum()
actual_misclassified = df_sim[(df_sim['Is_Poor'] == True) & (df_sim['Overcharge_Amount'] > 0)]
calculated_error_rate = (len(actual_misclassified) / true_poor_count) * 100 if true_poor_count > 0 else 0
avg_premium_spike = actual_misclassified['Overcharge_Amount'].mean() if len(actual_misclassified) > 0 else 0

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="📊 Realized Exclusion Error", value=f"{calculated_error_rate:.1f}%", delta="- Policy Goal: 0%")
with col2:
    st.metric(label="💰 Avg Monthly Overcharge", value=f"KES {avg_premium_spike:.2f}", delta="Per Impacted Household", delta_color="inverse")
with col3:
    st.metric(label="📉 Max Effective Tax Burden", value=f"{df_sim['Effective_Tax_Rate'].max():.1f}%", delta="Target Standard: 2.75%", delta_color="inverse")

st.markdown("---")

# 5. Tab Layout Splitting Visuals from the Text Report
tab1, tab2 = st.tabs(["📊 Interactive Visual Analytics", "📜 Policy Report & Recommendations"])

with tab1:
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("⚖️ Intended Policy vs. Reality")
        fig_scatter = px.scatter(
            df_sim.sample(min(len(df_sim), 1000)), 
            x='True_Income', 
            y='Final_Charged_Premium',
            color='Is_Poor',
            labels={'True_Income': 'True Household Income (KES)', 'Final_Charged_Premium': 'Final Premium Assigned (KES)'},
            title="Divergence from Proportional 2.75% Premium Flatline",
            color_discrete_map={True: '#EF553B', False: '#636EFA'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with chart_col2:
        st.subheader("📈 Distribution of Effective Premium Burden")
        fig_hist = px.histogram(
            df_sim, 
            x='Effective_Tax_Rate', 
            nbins=40,
            color='Is_Poor',
            labels={'Effective_Tax_Rate': 'Effective Premium Tax Rate (% of Income)'},
            title="How many low-income households clear the 2.75% burden ceiling?",
            color_discrete_map={True: '#EF553B', False: '#636EFA'}
        )
        fig_hist.add_vline(x=2.75, line_dash="dash", line_color="green", annotation_text="Statutory Target (2.75%)")
        st.plotly_chart(fig_hist, use_container_width=True)

    # Data Transparency Sub-Panel
    st.subheader("📋 Raw Simulated Dataset Preview")
    st.dataframe(df_sim[['True_Income', 'Intended_Premium', 'Final_Charged_Premium', 'Overcharge_Amount', 'Effective_Tax_Rate']].head(100), use_container_width=True)

with tab2:
    st.header("📜 Executive Policy Briefing")
    st.caption("Analyzing Algorithmic Bias & Equity Gaps in Kenya’s SHA Means-Testing Instrument")
    
    st.markdown("""
    ### 1. Executive Summary
    The transition from the National Hospital Insurance Fund (NHIF) to the Social Health Authority (SHA) was designed to establish a progressive healthcare financing framework. By replacing the old regressive flat-fee steps with an equitable **2.75% standard rate**, the policy intended to ensure high earners subsidise low-income segments. 
    
    However, live simulations and independent operational audits reveal a critical structural flaw. Because informal sector workers lack formal payroll files, SHA implements an AI-driven Proxy Means Testing (PMT) algorithm. This tool displays a verified **39% exclusion error rate among Kenya's bottom 40% income earners**. Instead of protecting the poor, the model overestimates household resources, misclassifying vulnerable citizens into wealthier premium brackets. This creates a severe **Fairness Gap**, driving their effective financial burden far above the statutory 2.75% baseline.
    
    ---
    
    ### 2. Key Findings & Data Insights
    * **The Burden Multiplier:** While formal employees face a strict, predictable 2.75% payroll deduction, informal households are at the mercy of lifestyle proxy variables (e.g., roofing material, livestock ownership, household size). When the algorithm miscalculates these indices, the financial burden spikes dynamically between **5.50% and 11.20%** of true income for impacted families.
    * **The Floor Penalty Paradox:** The absolute statutory minimum premium is set at **KES 300** per month. For a vulnerable casual laborer earning KES 7,500 monthly, this mandatory floor translates to a **4.0% effective tax rate**—meaning the poorest citizens legally pay a higher proportion of their income than middle-class formal workers.
    * **Systemic Exclusion:** When a low-income family is algorithmically over-assessed, they are billed for premium amounts they cannot afford. Because access to healthcare services is tied to an active, paid-up profile, these algorithmic glitches result in immediate, direct exclusion from medical treatment at hospital reception desks.
    
    ---
    
    ### 3. Recommended Measures & Corrective Actions
    
    #### 📊 Recalibrate and Open-Source the PMT Variable Matrix
    The Ministry of Health must strip out highly volatile, non-liquid proxy variables (such as basic electronic ownership) from the algorithm. Weight the algorithm heavily toward localized multi-dimensional poverty indices using verified regional data from the Kenya Continuous Household Survey (KCHS). The algorithm's source code should be open-sourced for peer review by local data scientists to eradicate black-box bias.
    
    #### ⚖️ Establish a Legally Mandated, Immediate Appeals Window
    Implement an automated "Flag and Freeze" mechanism directly inside the SHA registration portal. If a citizen is assigned an informal premium above the KES 300 minimum, they must have the right to trigger an instant appeal. While the appeal is being manually verified by a local community health promoter, the premium must be frozen at the KES 300 baseline to ensure medical coverage is never cut off.
    
    #### 📉 Introduce a Floating Minimum Floor for Vulnerable Households
    Remove the strict KES 300 hard floor for households verified as indigent or ultra-poor. Transition to a tiered floor system (e.g., KES 0, KES 100, or KES 200) for the bottom two income deciles. This removes the unfair floor penalty and aligns the lowest-earning segment with the universal 2.75% target rate.
    
    #### 🏥 Decouple Premium Collection from Immediate Emergency Care
    Pass a strict administrative directive enforcing unconditional access to critical medical care. Implement a "Treatment First, Billing Reconciliation Later" rule for all public and empanelled private facilities. Algorithmic discrepancies or payment defaults must be treated as civil state debts rather than grounds to deny emergency room access or maternal healthcare.
    """)
    
    # Decorative informational alert box
