import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Control Center",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS injections for premium visual quality and user-friendliness
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Background style override */
    .stApp {
        background: linear-gradient(180deg, #F9FBFC 0%, #F3F6FA 100%);
    }
    
    /* Modern Dashboard Header */
    .dashboard-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        color: white;
        padding: 2.2rem 2rem;
        border-radius: 20px;
        margin-bottom: 2.5rem;
        box-shadow: 0 10px 25px rgba(30, 58, 138, 0.15);
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .dashboard-header h1 {
        color: #FFFFFF !important;
        font-size: 2.3rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
        margin: 0;
    }
    
    .dashboard-header p {
        color: #BFDBFE;
        font-size: 1.1rem;
        margin-top: 0.5rem;
        margin-bottom: 0;
        font-weight: 300;
    }

    /* Form Container Card styling */
    div[data-testid="stForm"] {
        background: #ffffff !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 16px !important;
        padding: 2.5rem !important;
        box-shadow: 0 8px 24px rgba(148, 163, 184, 0.08) !important;
    }
    
    /* Styled Submit Button */
    button[kind="formSubmit"] {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%) !important;
        color: white !not-important;
        border: none !important;
        padding: 0.8rem 2rem !important;
        border-radius: 12px !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(30, 58, 138, 0.3) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    button[kind="formSubmit"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(30, 58, 138, 0.4) !important;
    }

    /* Result Panels */
    .result-container-safe {
        background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
        border-radius: 16px;
        padding: 2.5rem;
        border: 1px solid #BBF7D0;
        box-shadow: 0 8px 20px rgba(34, 197, 94, 0.06);
        color: #14532D;
    }
    
    .result-container-churn {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border-radius: 16px;
        padding: 2.5rem;
        border: 1px solid #FECACA;
        box-shadow: 0 8px 20px rgba(239, 68, 68, 0.06);
        color: #7F1D1D;
    }

    .status-title {
        font-size: 1.1rem;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 1px;
        opacity: 0.8;
    }

    .status-value {
        font-size: 3.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
        line-height: 1;
    }

    /* Action suggestion box inside result panel */
    .action-box {
        margin-top: 1.2rem;
        padding: 1rem;
        border-radius: 12px;
        background-color: rgba(255, 255, 255, 0.7);
        border: 1px solid rgba(0, 0, 0, 0.05);
        font-size: 0.95rem;
    }

    /* Sidebar Mode Selector styling */
    .mode-box {
        background-color: #F1F5F9;
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        margin-top: 1rem;
    }
    
    /* Footer style */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #94A3B8;
        font-size: 0.9rem;
        margin-top: 4rem;
        border-top: 1px solid #E2E8F0;
    }
    </style>
""", unsafe_allow_html=True)

# List of states excluding AK (base category mapped as indices)
STATES = ['AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']
ALL_STATES_INCL_AK = ['AK'] + STATES

# Header Layout
st.markdown("""
<div class="dashboard-header">
    <h1>Customer Churn Predictor 📊</h1>
    <p>Identify subscribers at risk of leaving, analyze usage statistics, and download custom reports</p>
</div>
""", unsafe_allow_html=True)

# Load machine learning model
@st.cache_resource
def load_predictor():
    try:
        model = joblib.load('best_model.pkl')
        return model, None
    except Exception as e:
        return None, str(e)

model, err = load_predictor()

if err:
    st.error("⚠️ **System Offline**: Unable to load diagnostic model. Please contact your support desk.")
    st.stop()

# Helper function to engineering features for inference
def process_data(df_input):
    df_proc = df_input.copy()
    
    # 1. Feature Engineering
    df_proc["Total Minutes"] = (
        df_proc["Total day minutes"] +
        df_proc["Total eve minutes"] +
        df_proc["Total night minutes"] +
        df_proc["Total intl minutes"]
    )
    df_proc["Total Revenue"] = (
        df_proc["Total day charge"] +
        df_proc["Total eve charge"] +
        df_proc["Total night charge"] +
        df_proc["Total intl charge"]
    )
    df_proc["Revenue Per Minute"] = df_proc["Total Revenue"] / df_proc["Total Minutes"]
    df_proc["Revenue Per Minute"] = df_proc["Revenue Per Minute"].fillna(0.0).replace([np.inf, -np.inf], 0.0)
    df_proc["Customer Loyalty"] = df_proc["Account length"] / 365.0
    df_proc["Service Call Ratio"] = df_proc["Customer service calls"] / (df_proc["Account length"] + 1)
    
    # 2. Variable mapping
    df_proc["International plan"] = df_proc["International plan"].apply(lambda x: 1 if str(x).strip().lower() in ['yes', '1', '1.0', 'true'] else 0)
    df_proc["Voice mail plan"] = df_proc["Voice mail plan"].apply(lambda x: 1 if str(x).strip().lower() in ['yes', '1', '1.0', 'true'] else 0)
    
    # 3. Create state one-hot encoding columns (State_AL to State_WY)
    for s in STATES:
        df_proc[f"State_{s}"] = df_proc["State"].apply(lambda x: 1 if str(x).strip().upper() == s else 0)
        
    # Reorder columns to match the 73 model inputs in order
    expected_columns = [
        'Account length', 'Area code', 'International plan', 'Voice mail plan', 'Number vmail messages',
        'Total day minutes', 'Total day calls', 'Total day charge', 'Total eve minutes', 'Total eve calls',
        'Total eve charge', 'Total night minutes', 'Total night calls', 'Total night charge',
        'Total intl minutes', 'Total intl calls', 'Total intl charge', 'Customer service calls',
        'Total Revenue', 'Total Minutes', 'Revenue Per Minute', 'Customer Loyalty', 'Service Call Ratio'
    ] + [f"State_{s}" for s in STATES]
    
    return df_proc[expected_columns]

# Sidebar: Concise, non-scrollable control hub
with st.sidebar:
    st.markdown("### ⚙️ Predictive Tool Control")
    st.write("Toggle prediction formats depending on your workflow:")
    
    app_mode = st.radio(
        "Choose Analysis Method:",
        options=["👤 Single Subscriber Check", "📁 Batch File Upload (.CSV)"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 💡 Retention Strategy")
    st.write(
        "For any customer flagged as At-Risk, consider offering loyalty perks "
        "or contract adjustments to ensure long-term satisfaction."
    )

# ----------------- Mode 1: Single Subscriber Check -----------------
if app_mode == "👤 Single Subscriber Check":
    st.markdown("### 📋 Enter Subscriber Information")
    st.write("Fill in the usage and profile statistics below to check the customer status:")

    with st.form("single_prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 👤 Subscriber profile")
            state_input = st.selectbox("US State of Residence", options=ALL_STATES_INCL_AK, index=0)
            account_length = st.number_input("Days Subscribed", min_value=1, max_value=500, value=100, step=1, help="Number of days the customer has been with our service")
            area_code = st.selectbox("Area Code", options=[408, 415, 510], index=1)
            intl_plan_str = st.radio("Has International Calling Subscription?", options=["No", "Yes"], horizontal=True)
            vmail_plan_str = st.radio("Has Voicemail Subscription?", options=["No", "Yes"], horizontal=True)
            vmail_messages = st.number_input("Voicemail Messages Received", min_value=0, max_value=200, value=0, step=1)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 🛠️ Customer Support Usage")
            cust_serv_calls = st.slider("Support Care Calls Made", min_value=0, max_value=20, value=1, help="Total number of support helpdesk tickets created by customer")
            
        with col2:
            st.markdown("#### 📞 Usage & Charge Details")
            
            # Day Usage
            st.markdown("**Daily Period**")
            c_d1, c_d2 = st.columns(2)
            with c_d1:
                day_minutes = st.number_input("Daily Call Time (minutes)", min_value=0.0, max_value=600.0, value=180.0, step=0.1)
            with c_d2:
                day_charge = st.number_input("Daily Billing ($)", min_value=0.0, max_value=150.0, value=30.6, step=0.01)
                
            # Eve Usage
            st.markdown("**Evening Period**")
            c_e1, c_e2 = st.columns(2)
            with c_e1:
                eve_minutes = st.number_input("Evening Call Time (minutes)", min_value=0.0, max_value=600.0, value=200.0, step=0.1)
            with c_e2:
                eve_charge = st.number_input("Evening Billing ($)", min_value=0.0, max_value=100.0, value=17.0, step=0.01)
                
            # Night Usage
            st.markdown("**Night Period**")
            c_n1, c_n2 = st.columns(2)
            with c_n1:
                night_minutes = st.number_input("Night Call Time (minutes)", min_value=0.0, max_value=600.0, value=200.0, step=0.1)
            with c_n2:
                night_charge = st.number_input("Night Billing ($)", min_value=0.0, max_value=100.0, value=9.0, step=0.01)
                
            # International Usage
            st.markdown("**International Calls**")
            c_i1, c_i2 = st.columns(2)
            with c_i1:
                intl_minutes = st.number_input("International Call Time (minutes)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
            with c_i2:
                intl_charge = st.number_input("International Billing ($)", min_value=0.0, max_value=30.0, value=2.7, step=0.01)
                
            st.markdown("**Call Frequency Quantities**")
            c_f1, c_f2, c_f3 = st.columns(3)
            with c_f1:
                day_calls = st.number_input("Daily Call Count", min_value=0, max_value=300, value=100, step=1)
            with c_f2:
                eve_calls = st.number_input("Evening Call Count", min_value=0, max_value=300, value=100, step=1)
            with c_f3:
                intl_calls = st.number_input("International Call Count", min_value=0, max_value=50, value=3, step=1)
                
            night_calls = 90
            
        submit_button = st.form_submit_button("Analyze Churn Probability")

    if submit_button:
        # Wrap input into pandas row
        raw_row = pd.DataFrame([{
            'State': state_input,
            'Account length': account_length,
            'Area code': area_code,
            'International plan': intl_plan_str,
            'Voice mail plan': vmail_plan_str,
            'Number vmail messages': vmail_messages,
            'Total day minutes': day_minutes,
            'Total day calls': day_calls,
            'Total day charge': day_charge,
            'Total eve minutes': eve_minutes,
            'Total eve calls': eve_calls,
            'Total eve charge': eve_charge,
            'Total night minutes': night_minutes,
            'Total night calls': night_calls,
            'Total night charge': night_charge,
            'Total intl minutes': intl_minutes,
            'Total intl calls': intl_calls,
            'Total intl charge': intl_charge,
            'Customer service calls': cust_serv_calls
        }])
        
        try:
            # Preprocess
            input_df = process_data(raw_row)
            
            # Predict
            prob = model.predict_proba(input_df)[0]
            prediction = model.predict(input_df)[0]
            
            health_confidence = prob[0] * 100
            risk_confidence = prob[1] * 100
            
            # Feature calculations for summary metrics
            total_minutes = day_minutes + eve_minutes + night_minutes + intl_minutes
            total_revenue = day_charge + eve_charge + night_charge + intl_charge
            customer_loyalty = account_length / 365.0
            
            c_res1, c_res2 = st.columns([1, 1])
            
            with c_res1:
                st.markdown("### 🩺 Churn Evaluation")
                if prediction:
                    st.markdown(f"""
                    <div class="result-container-churn">
                        <span class="status-title">Status: At-Risk of Churning ⚠️</span>
                        <div class="status-value">{risk_confidence:.1f}%</div>
                        <p style="margin: 0; font-weight: 500; font-size: 1.1rem;">Chances subscriber will cancel their service</p>
                        <div class="action-box">
                            <strong style="color: #9B1C1C;">💡 Recommended Action:</strong> Offer a tailored loyalty promotion or contract discount to retain this user.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-container-safe">
                        <span class="status-title">Status: Safe (Loyal Member) ✅</span>
                        <div class="status-value">{health_confidence:.1f}%</div>
                        <p style="margin: 0; font-weight: 500; font-size: 1.1rem;">Chances subscriber will remain active</p>
                        <div class="action-box">
                            <strong style="color: #03543F;">💡 Recommended Action:</strong> No immediate retention concern. Maintain standard service updates.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
            with c_res2:
                st.markdown("### 📊 Metrics Summary")
                mc1, mc2 = st.columns(2)
                with mc1:
                    st.metric("Total Bill Spending ($)", f"${total_revenue:.2f}")
                    st.metric("Loyalty Duration (Years)", f"{customer_loyalty:.2f} yrs")
                with mc2:
                    st.metric("Total Call Duration", f"{total_minutes:.1f} mins")
                    st.metric("Support Ticketing Frequency", "Low" if cust_serv_calls <= 2 else "High")
                
                # Single report CSV
                out_df = pd.DataFrame([{
                    'State': state_input,
                    'Days Subscribed': account_length,
                    'Area Code': area_code,
                    'International Calling Plan': intl_plan_str,
                    'Voice Mail Plan': vmail_plan_str,
                    'Voice Mail Messages': vmail_messages,
                    'Total Spend ($)': total_revenue,
                    'Account Age (years)': customer_loyalty,
                    'Support Tickets Opened': cust_serv_calls,
                    'Churn Risk Evaluation': 'At-Risk' if prediction else 'Safe/Healthy',
                    'Stay Score (%)': health_confidence,
                    'Leave Score (%)': risk_confidence
                }])
                csv = out_df.to_csv(index=False).encode('utf-8')
                st.markdown("<br>", unsafe_allow_html=True)
                st.download_button(
                    label="📥 Download Assessment Report (CSV)",
                    data=csv,
                    file_name="subscriber_assessment.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
        except Exception as e:
            st.error(f"⚠️ **Analysis error**: Unable to process calculations. Please verify your inputs.")

# ----------------- Mode 2: Batch File Process -----------------
else:
    st.markdown("### 📁 Batch Subscriber File Processing")
    st.write("Upload a CSV file containing multiple customer records to predict churn risks in bulk.")
    
    st.info(
        "💡 **File Format Requirement**: Your uploaded CSV must contain the standard columns, such as: "
        "`State`, `Account length`, `Area code`, `International plan`, `Voice mail plan`, `Number vmail messages`, "
        "`Total day minutes`, `Total day calls`, `Total day charge`, `Total eve minutes`, `Total eve calls`, `Total eve charge`, "
        "`Total night minutes`, `Total night calls`, `Total night charge`, `Total intl minutes`, `Total intl calls`, `Total intl charge`, "
        "and `Customer service calls`."
    )
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            
            # Simple check for essential columns
            essential_cols = ['State', 'Account length', 'Area code', 'Total day minutes', 'Total day charge', 'Customer service calls']
            missing = [c for c in essential_cols if c not in df_upload.columns]
            
            if len(missing) > 0:
                st.error(f"⚠️ **Invalid CSV Schema**: The uploaded file is missing required columns: `{missing}`.")
            else:
                # Preprocess df
                df_model_input = process_data(df_upload)
                
                # Predict
                probs = model.predict_proba(df_model_input)
                preds = model.predict(df_model_input)
                
                # Append outputs
                df_upload["Churn Probability (%)"] = np.round(probs[:, 1] * 100, 2)
                df_upload["Prediction Status"] = np.where(preds == 1, "At-Risk ⚠️", "Safe ✅")
                
                # Stats calculation
                total_cust = len(df_upload)
                chk_at_risk = int(np.sum(preds == 1))
                chk_safe = total_cust - chk_at_risk
                risk_rate = (chk_at_risk / total_cust) * 100
                
                # Display metrics
                st.markdown("#### 📈 Analysis Overview")
                st_c1, st_c2, st_c3 = st.columns(3)
                with st_c1:
                    st.metric("Total Subscribers Processed", f"{total_cust}")
                with st_c2:
                    st.metric("At-Risk Subscribers ⚠️", f"{chk_at_risk}", f"Rate: {risk_rate:.1f}%", delta_color="inverse")
                with st_c3:
                    st.metric("Safe Subscribers ✅", f"{chk_safe}", f"Rate: {100 - risk_rate:.1f}%")
                
                # Chart
                st.markdown("#### 📊 Churn Risk Distribution")
                chart_data = pd.DataFrame({
                    'Status': ['Safe ✅', 'At-Risk ⚠️'],
                    'Count': [chk_safe, chk_at_risk]
                }).set_index('Status')
                st.bar_chart(chart_data, color='#3B82F6', use_container_width=True)
                
                # Table details showing top risk users
                st.markdown("#### 📋 Top At-Risk Customers")
                df_rendered = df_upload.sort_values(by="Churn Probability (%)", ascending=False).head(15)
                
                # Select only clean columns for cleaner display
                rendered_cols = ['State', 'Account length', 'Area code', 'Customer service calls', 'Churn Probability (%)', 'Prediction Status']
                existing_rend = [c for c in rendered_cols if c in df_rendered.columns]
                
                st.dataframe(df_rendered[existing_rend], use_container_width=True)
                
                # Download link
                csv_out = df_upload.to_csv(index=False).encode('utf-8')
                st.markdown("<br>", unsafe_allow_html=True)
                st.download_button(
                    label="📥 Download Full Predictions Report (CSV)",
                    data=csv_out,
                    file_name="batch_churn_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
        except Exception as batch_error:
            st.error(f"🚨 **Batch Processing Error**: An error occurred parsing the uploaded file: {str(batch_error)}")

st.markdown("""
<div class="footer">
    Customer Churn Prediction Platform © 2026. Made for professional telecom retention teams.
</div>
""", unsafe_allow_html=True)
