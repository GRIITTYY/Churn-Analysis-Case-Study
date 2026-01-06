import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly. graph_objects as go
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler,FunctionTransformer

# ============================================
# PAGE CONFIGURATION
# ============================================
st. set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# IMPORTANT FUNCTIONS
# ============================================
# Function to fix datatypes
def set_numerical_columns_datatype(X):
    X = X.copy()
    X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
    X["MonthlyCharges"] = pd.to_numeric(X["MonthlyCharges"], errors="coerce")
    X["tenure"] = pd.to_numeric(X["tenure"], errors="coerce")
    return X


# Function to Fill NaNs
def impute_nans(X):
    X = X.copy()

    # Fix Tenure
    empty_tenure = X["tenure"].isna() & (~X["TotalCharges"].isna()) & (~X["MonthlyCharges"].isna())
    X.loc[empty_tenure, "tenure"] = (X.loc[empty_tenure, "TotalCharges"] - np.random.randint(5,31, size=empty_tenure.sum())) / X.loc[empty_tenure, "MonthlyCharges"]
    # Fallback
    X["tenure"] = X["tenure"].fillna(0)

    # Fix MonthlyCharges
    empty_monthly_charges = X["MonthlyCharges"].isna() & (~X["TotalCharges"].isna()) & (~X["tenure"].isna())
    X.loc[empty_monthly_charges, "MonthlyCharges"] = (X.loc[empty_monthly_charges, "TotalCharges"] - np.random.randint(5,31, size=empty_monthly_charges.sum())) / X.loc[empty_monthly_charges, "tenure"]
    # Fallback
    X["MonthlyCharges"] = X["MonthlyCharges"].fillna(0)

    # Fix TotalCharges
    empty_total_charges = X["TotalCharges"].isna()

    # Compute TotalCharges for Month-to-month
    mtm_total_charges = empty_total_charges & (X["Contract"] == "Month-to-month") & (X["tenure"] > 0)
    X.loc[mtm_total_charges, "TotalCharges"] = X.loc[mtm_total_charges, "MonthlyCharges"] * X.loc[mtm_total_charges, "tenure"] + np.random.randint(5,31, size=mtm_total_charges.sum())

    # Compute TotalCharges for One year
    one_year_charges = empty_total_charges & (X["Contract"] == "One year") & (X["tenure"] > 0)
    X.loc[one_year_charges, "TotalCharges"] = X.loc[one_year_charges, "tenure"]  * X.loc[one_year_charges, "MonthlyCharges"] + np.random.randint(5,31, size=one_year_charges.sum())

    # Compute TotalCharges for Two year
    two_year_charges = empty_total_charges & (X["Contract"] == "Two year") & (X["tenure"] > 0)
    X.loc[two_year_charges, "TotalCharges"] = X.loc[two_year_charges, "tenure"]  * X.loc[two_year_charges, "MonthlyCharges"] + np.random.randint(5,31, size=two_year_charges.sum())

    zero_tenure = empty_total_charges & (X["tenure"] == 0)
    X.loc[zero_tenure, "TotalCharges"] =  X.loc[zero_tenure, "MonthlyCharges"]

    return X

# Convert to Transformer object
nan_imputer = FunctionTransformer(
    impute_nans,
    validate=False
)

# Convert to Transformer object
type_coercion = FunctionTransformer(
    set_numerical_columns_datatype,
    validate=False
)

# Filter Numerical and Categorical columns names
numerical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_columns = ['Contract', 'Dependents', 'DeviceProtection', 'InternetService',
       'OnlineBackup', 'OnlineSecurity', 'PaperlessBilling', 'Partner',
       'PaymentMethod', 'PhoneService', 'SeniorCitizen', 'StreamingMovies',
       'StreamingTV', 'TechSupport']

# Define objects for Enncoding and Scaling
categorical_encoder = OneHotEncoder(sparse_output=False, dtype="int", handle_unknown="ignore")
numerical_scaler = StandardScaler()

# Combine into a ColumnTransformer
column_transformer  = ColumnTransformer(
    transformers=[
        ("scaler", numerical_scaler, numerical_columns),
        ("encoder", categorical_encoder, categorical_columns)
    ]
)

# Parse to Pipeline Instance
Preprocessing_Pipeline = Pipeline(steps=[
    ("type_fix", type_coercion),
    ("nan_fix", nan_imputer),
    ("x_features_transformer", column_transformer)
])

# Set Ouput to be DataFrame
Preprocessing_Pipeline.set_output(transform="pandas")


# ============================================
# CSS
# ============================================
st.markdown("""
<style>
    /* Header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight:  700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip:  text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    
    .sub-header {
        text-align: center;
        color: #6b7280;
        font-size: 1.1rem;
    }
    
    /* Card styling */
    .metric-card {
        background:  linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    .metric-card-danger {
        background:  linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    
    .metric-card-success {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size:  0.9rem;
        opacity: 0.9;
    }
    
    /* Prediction result styling */
    .prediction-box {
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .prediction-churn {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
    }
    
    .prediction-no-churn {
        background: linear-gradient(135deg, #26de81 0%, #20bf6b 100%);
        color: white;
    }
    
    .prediction-title {
        font-size: 1.5rem;
        font-weight:  600;
        margin-bottom: 0.5rem;
    }
    
    .prediction-prob {
        font-size: 3rem;
        font-weight: 700;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background:  linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        # background-color: #4cc9f0;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #e0e0e0;
    }
    
    /* Input field styling */
    .stSelectbox > div > div {
        border-radius: 10px;
    }
    
    .stNumberInput > div > div > input {
        border-radius: 10px;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius:  10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 1. 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Risk indicator */
    .risk-indicator {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        margin-top: 1rem;
    }
    
    .risk-dot {
        width:  12px;
        height: 12px;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    .risk-high {
        background-color: #ff6b6b;
    }
    
    .risk-low {
        background-color: #26de81;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.2); opacity: 0.7; }
        100% { transform: scale(1); opacity: 1; }
    }
    
    /* Feature importance card */
    .feature-card {
        background:  #f8f9fa;
        border-radius: 12px;
        padding: 1rem;
        margin:  0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    /* Divider */
    .custom-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }

    /* Selectbox label text */      
    div[data-testid="stSelectbox"] label {
        color: white;
    } 
            
    /* Slider label text */      
    div[data-testid="stSlider"] label {
        color: white;
    } 

    /* st.number_input label text */      
    label[data-testid="stWidgetLabel"] div[data-testid="stMarkdownContainer"] p {
        color: white;
    }      
                
    /* Help icon (SVG) */
    svg {
        color: white;  /* for stroke="currentColor" SVGs */
        fill: white;   /* for filled SVGs */
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# LOAD MODEL
# ============================================
@st.cache_resource
def load_model():
    """Load the trained model pipeline"""
    model_path = Path("model/gradient_boosting_churn_model.pkl")
    if model_path.exists():
        with open(model_path, "rb") as f:
            model = joblib.load(f)
        return model
    else: 
        st.error("⚠️ Model file not found!  Please ensure 'model/gradient_boosting_churn_model.pkl' exists.")
        return None


# ============================================
# DATA PREPROCESSING FUNCTIONS
# ============================================
def prepare_input_data(input_dict):
    """Prepare input data for prediction"""
    # Create DataFrame from input
    df = pd. DataFrame([input_dict])
    
    # Convert categorical columns to string type
    categorical_cols = df. select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].astype('string')
    
    # Ensure numeric columns are properly typed
    df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')
    df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    return df


def calculate_total_charges(tenure, monthly_charges):
    """Calculate estimated total charges based on tenure and monthly charges"""
    if tenure == 0:
        return monthly_charges
    return tenure * monthly_charges


# ============================================
# SIDEBAR - INPUT FORM
# ============================================
def render_sidebar():
    """Render the sidebar with input form"""
    with st.sidebar:
        st.markdown("## 🎯 Customer Information")
        st.markdown("---")
        
        # Demographics Section
        st.markdown("### 👤 Demographics")
        
        senior_citizen = st.selectbox(
            "Senior Citizen (65+)",
            options=["No", "Yes"],
            help="Is the customer 65 years or older?"
        )
        
        partner = st.selectbox(
            "Has Partner",
            options=["No", "Yes"],
            help="Does the customer have a partner?"
        )
        
        dependents = st.selectbox(
            "Has Dependents",
            options=["No", "Yes"],
            help="Does the customer have dependents?"
        )
        
        st.markdown("---")
        
        # Account Information Section
        st.markdown("### 📋 Account Information")
        
        tenure = st.slider(
            "Tenure (Months)",
            min_value=0,
            max_value=72,
            value=12,
            help="Number of months the customer has been with the company"
        )
        
        contract = st.selectbox(
            "Contract Type",
            options=["Month-to-month", "One year", "Two year"],
            help="Type of contract the customer has"
        )
        
        paperless_billing = st.selectbox(
            "Paperless Billing",
            options=["No", "Yes"],
            help="Does the customer use paperless billing?"
        )
        
        payment_method = st. selectbox(
            "Payment Method",
            options=[
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ],
            help="Customer's payment method"
        )
        
        st.markdown("---")
        
        # Services Section
        st. markdown("### 📡 Services")
        
        phone_service = st. selectbox(
            "Phone Service",
            options=["No", "Yes"],
            help="Does the customer have phone service?"
        )
        
        internet_service = st. selectbox(
            "Internet Service",
            options=["No", "DSL", "Fiber optic"],
            help="Type of internet service"
        )
        
        # Internet-dependent services
        if internet_service != "No":
            online_security = st.selectbox(
                "Online Security",
                options=["No", "Yes"],
                help="Does the customer have online security?"
            )
            
            online_backup = st.selectbox(
                "Online Backup",
                options=["No", "Yes"],
                help="Does the customer have online backup?"
            )
            
            device_protection = st.selectbox(
                "Device Protection",
                options=["No", "Yes"],
                help="Does the customer have device protection?"
            )
            
            tech_support = st. selectbox(
                "Tech Support",
                options=["No", "Yes"],
                help="Does the customer have tech support?"
            )
            
            streaming_tv = st.selectbox(
                "Streaming TV",
                options=["No", "Yes"],
                help="Does the customer stream TV?"
            )
            
            streaming_movies = st. selectbox(
                "Streaming Movies",
                options=["No", "Yes"],
                help="Does the customer stream movies?"
            )
        else:
            online_security = "No internet service"
            online_backup = "No internet service"
            device_protection = "No internet service"
            tech_support = "No internet service"
            streaming_tv = "No internet service"
            streaming_movies = "No internet service"
        
        st.markdown("---")
        
        # Charges Section
        st. markdown("### 💰 Charges")
        
        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=200.0,
            value=70.0,
            step=5.0,
            help="Customer's monthly charge amount"
        )
        
        # Auto-calculate total charges
        total_charges = calculate_total_charges(tenure, monthly_charges)
        
        st.info(f"📊 Estimated Total Charges:  **${total_charges: ,.2f}**")
        
        st.markdown("---")
        
        # Predict Button
        predict_button = st.button("🔮 Predict Churn Risk", use_container_width=True)
        reset_button = st.button("CLEAR CURRENT VIEW",use_container_width=True)
        
        # Compile input data
        input_data = {
            "SeniorCitizen": senior_citizen,
            "Partner":  partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection":  device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies":  streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges":  monthly_charges,
            "TotalCharges": total_charges
        }
        
        return input_data, predict_button, reset_button


# ============================================
# MAIN CONTENT
# ============================================
def render_header():
    """Render the main header"""
    st.markdown('<h1 class="main-header">Customer Churn Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict customer churn risk using machine learning</p>', unsafe_allow_html=True)


def render_prediction_result(prediction, probability):
    """Render the prediction result"""
    churn_prob = probability[1] * 100
    no_churn_prob = probability[0] * 100
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if prediction == 1:
            st. markdown(f"""
            <div class="prediction-box prediction-churn">
                <div class="prediction-title">⚠️ HIGH CHURN RISK</div>
                <div class="prediction-prob">{churn_prob:.1f}%</div>
                <div class="risk-indicator">
                    <div class="risk-dot risk-high"></div>
                    <span>Customer is likely to churn</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else: 
            st.markdown(f"""
            <div class="prediction-box prediction-no-churn">
                <div class="prediction-title">✅ LOW CHURN RISK</div>
                <div class="prediction-prob">{no_churn_prob:.1f}%</div>
                <div class="risk-indicator">
                    <div class="risk-dot risk-low"></div>
                    <span>Customer is likely to stay</span>
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_probability_gauge(probability):
    """Render probability gauge chart"""
    churn_prob = probability[1] * 100
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=churn_prob,
        domain={'x':  [0, 1], 'y':  [0, 1]},
        title={'text': "Churn Probability", 'font': {'size': 20, 'color': '#4a5568'}},
        number={'suffix': "%", 'font':  {'size': 40, 'color': '#4a5568'}},
        gauge={
            'axis':  {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#4a5568"},
            'bar': {'color': "#667eea"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e2e8f0",
            'steps': [
                {'range': [0, 30], 'color':  '#c6f6d5'},
                {'range': [30, 60], 'color':  '#fef3c7'},
                {'range': [60, 100], 'color': '#fed7d7'}
            ],
            'threshold': {
                'line': {'color':  "#e53e3e", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "#4a5568"}
    )
    
    return fig


def render_risk_factors(input_data):
    """Render risk factor analysis"""
    st.markdown("### 🔍 Risk Factor Analysis")
    
    risk_factors = []
    protective_factors = []
    
    # Analyze risk factors based on EDA insights
    if input_data["Contract"] == "Month-to-month": 
        risk_factors. append(("📅 Month-to-month contract", "Higher churn rate than yearly contracts"))
    else:
        protective_factors.append(("📅 Long-term contract", "Lower churn rate"))
    
    if input_data["tenure"] < 12:
        risk_factors.append(("⏱️ Short tenure (<12 months)", "New customers have higher churn risk"))
    elif input_data["tenure"] > 36:
        protective_factors.append(("⏱️ Long tenure (>36 months)", "Established customers are more loyal"))
    
    if input_data["InternetService"] == "Fiber optic": 
        risk_factors.append(("🌐 Fiber optic internet", "Fiber optic users have higher churn"))
    
    if input_data["PaymentMethod"] == "Electronic check":
        risk_factors.append(("💳 Electronic check payment", "This payment method correlates with higher churn"))
    
    if input_data["PaperlessBilling"] == "Yes":
        risk_factors.append(("📧 Paperless billing", "Slightly higher churn rate"))
    
    if input_data["OnlineSecurity"] == "No" and input_data["InternetService"] != "No":
        risk_factors.append(("🔒 No online security", "Lack of security service increases churn risk"))
    
    if input_data["TechSupport"] == "No" and input_data["InternetService"] != "No":
        risk_factors.append(("🛠️ No tech support", "Lack of support increases churn risk"))
    
    if input_data["SeniorCitizen"] == "Yes": 
        risk_factors.append(("👴 Senior citizen", "Senior citizens have slightly higher churn"))
    
    if input_data["Partner"] == "Yes": 
        protective_factors.append(("💑 Has partner", "Customers with partners are more stable"))
    
    if input_data["Dependents"] == "Yes":
        protective_factors.append(("👨‍👩‍👧 Has dependents", "Family customers tend to stay longer"))
    
    if float(input_data["MonthlyCharges"]) > 70:
        risk_factors.append(("💰 High monthly charges", "Higher-paying customers may seek alternatives"))
    elif float(input_data["MonthlyCharges"]) < 35:
        protective_factors.append(("💰 Low monthly charges", "Lower charges reduce price sensitivity"))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ⚠️ Risk Factors")
        if risk_factors: 
            for factor, description in risk_factors:
                st.markdown(f"""
                <div class="feature-card" style="border-left-color: #e53e3e;">
                    <strong>{factor}</strong><br>
                    <small style="color: #718096;">{description}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No significant risk factors identified")
    
    with col2:
        st.markdown("#### ✅ Protective Factors")
        if protective_factors: 
            for factor, description in protective_factors:
                st. markdown(f"""
                <div class="feature-card" style="border-left-color: #38a169;">
                    <strong>{factor}</strong><br>
                    <small style="color: #718096;">{description}</small>
                </div>
                """, unsafe_allow_html=True)
        else: 
            st.info("No significant protective factors identified")


def render_recommendations(prediction, input_data):
    """Render retention recommendations"""
    st.markdown("### 💡 Retention Recommendations")
    
    if prediction == 1:  # High churn risk
        recommendations = []
        
        if input_data["Contract"] == "Month-to-month": 
            recommendations.append({
                "icon": "📝",
                "title": "Offer Contract Upgrade",
                "description": "Provide incentives (10-15% discount) for switching to a one-year or two-year contract."
            })
        
        if input_data["tenure"] < 12:
            recommendations.append({
                "icon": "🎁",
                "title": "New Customer Onboarding",
                "description": "Implement a 90-day onboarding program with check-ins and special offers."
            })
        
        if input_data["OnlineSecurity"] == "No" and input_data["InternetService"] != "No":
            recommendations. append({
                "icon": "🔒",
                "title": "Bundle Security Services",
                "description": "Offer a discounted security package including OnlineSecurity and DeviceProtection."
            })
        
        if input_data["TechSupport"] == "No" and input_data["InternetService"] != "No":
            recommendations.append({
                "icon": "🛠️",
                "title":  "Add Tech Support",
                "description": "Provide complimentary tech support for the first 2-6 months."
            })
        
        if input_data["PaymentMethod"] == "Electronic check":
            recommendations.append({
                "icon": "💳",
                "title": "Switch Payment Method",
                "description": "Offer a $5 credit for switching to automatic bank transfer or credit card."
            })
        
        if float(input_data["MonthlyCharges"]) > 80:
            recommendations. append({
                "icon": "💰",
                "title": "Price Review",
                "description": "Review pricing and offer a loyalty discount of 5-10% to retain the customer."
            })
        
        if not recommendations:
            recommendations.append({
                "icon": "📞",
                "title": "Proactive Outreach",
                "description": "Schedule a customer success call to understand needs and address concerns."
            })
        
        cols = st.columns(min(len(recommendations), 3))
        for idx, rec in enumerate(recommendations):
            with cols[idx % 3]:
                st.markdown(f"""
                <div style="background:  linear-gradient(135deg, #667eea15 0%, #764ba215 100%); 
                            border-radius: 12px; padding: 1.5rem; height: 100%; 
                            border: 1px solid #667eea30;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">{rec['icon']}</div>
                    <div style="font-weight: 600; color: #4a5568; margin-bottom: 0.5rem;">{rec['title']}</div>
                    <div style="font-size: 0.85rem; color: #718096; margin-bottom: 1.5rem;">{rec['description']}</div>
                </div>
                """, unsafe_allow_html=True)
    else: 
        st.success("✅ This customer has a low churn risk. Continue providing excellent service to maintain loyalty!")
        
        st.markdown("""
        **Suggested Actions:**
        - 🌟 Consider for loyalty program enrollment
        - 📊 Monitor for any changes in usage patterns
        - 🎯 Target for upselling premium services
        - 💬 Request reviews or referrals
        """)


def render_customer_summary(input_data):
    """Render customer summary cards"""
    st. markdown("### 📋 Customer Profile Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Tenure</div>
            <div class="metric-value">{input_data['tenure']}</div>
            <div class="metric-label">months</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <div class="metric-label">Monthly Charges</div>
            <div class="metric-value">${input_data['MonthlyCharges']:.0f}</div>
            <div class="metric-label">per month</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); color: #4a5568;">
            <div class="metric-label">Total Charges</div>
            <div class="metric-value">${input_data['TotalCharges']: ,.0f}</div>
            <div class="metric-label">lifetime</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        contract_emoji = "📅" if input_data['Contract'] == "Month-to-month" else "📆" if input_data['Contract'] == "One year" else "📅📅"
        st. markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); color: #4a5568;">
            <div class="metric-label">Contract</div>
            <div class="metric-value" style="font-size: 1.5rem;">{contract_emoji}</div>
            <div class="metric-label">{input_data['Contract']}</div>
        </div>
        """, unsafe_allow_html=True)


def render_welcome_state():
    """Render the welcome state before prediction"""
    st. markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <div style="font-size: 3rem;">📊</div>
            <h3 style="color:  #4a5568;">Data-Driven</h3>
            <p style="color: #718096;">Trained on 7,000+ customer records with 26. 5% churn rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <div style="font-size: 3rem;">🤖</div>
            <h3 style="color:  #4a5568;">ML-Powered</h3>
            <p style="color:  #718096;">Gradient Boosting model with high accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding:  2rem;">
            <div style="font-size: 3rem;">💡</div>
            <h3 style="color: #4a5568;">Actionable Insights</h3>
            <p style="color: #718096;">Get personalized retention recommendations</p>
        </div>
        """, unsafe_allow_html=True)
    
    st. markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    st.info("👈 **Fill in the customer information in the sidebar and click 'Predict Churn Risk' to get started!**")
    
    # Show key insights from EDA
    with st.expander("📈 Key Churn Insights from Our Analysis"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **🔴 High Risk Indicators:**
            - Month-to-month contracts
            - Tenure < 12 months
            - Fiber optic internet users
            - Electronic check payment method users
            - Paperless billing Users
            - No online security/tech support
            - Monthly charges in `$70-$100` range
            - Senior citizens
            - No partner
            - No dependents
            """)
    
        with col2:
            st.markdown("""
            **🟢 Low Risk Indicators:**
            - One or two-year contracts
            - Longer tenure (>36 months)
            - Has partner
            - Has dependents
            - Has online security
            - Has tech support
            - Has device protection
            - Lower monthly charges (`~$20`)
            """)

# ============================================
# MAIN APP
# ============================================
def main():
    """Main application function"""
    # Load model
    model = load_model()
    
    # Render header
    render_header()
    
    # Render sidebar and get inputs
    input_data, predict_button, reset_button = render_sidebar()
    
    # Main content area
    if predict_button and model is not None:
        try:
            # Prepare input data
            df = prepare_input_data(input_data)
            
            # Make prediction
            prediction = model.predict(df)[0]
            probability = model.predict_proba(df)[0]
            
            # Store in session state
            st.session_state['prediction'] = prediction
            st.session_state['probability'] = probability
            st.session_state['input_data'] = input_data
            
        except Exception as e: 
            st.error(f"Error making prediction: {str(e)}")
            st.exception(e)
            return
    
    if reset_button:
        st.session_state.clear()
        st.rerun()

    # Display results if prediction exists
    if 'prediction' in st.session_state:
        prediction = st.session_state['prediction']
        probability = st. session_state['probability']
        input_data = st.session_state['input_data']
        
        # Render prediction result
        render_prediction_result(prediction, probability)
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # Customer summary
        render_customer_summary(input_data)
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # Two column layout for gauge and factors
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📊 Churn Probability")
            fig = render_probability_gauge(probability)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            render_risk_factors(input_data)
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # Recommendations
        render_recommendations(prediction, input_data)
        
    else:
        render_welcome_state()
    
    # Footer
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #a0aec0; padding: 1rem;">
        <p>Built with ❤️ using Streamlit by <a href="https://www.linkedin.com/in/samuel-o-momoh">SAMUEL MOMOH<a/> | Customer Churn Analysis Case Study</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()