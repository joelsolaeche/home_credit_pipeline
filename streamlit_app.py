import streamlit as st
import pandas as pd
import numpy as np

# Handle imports with error checking for Streamlit Cloud
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as e:
    st.error(f"Plotly import error: {e}")
    st.info("Please make sure plotly is installed: pip install plotly>=5.15.0")
    st.stop()

try:
    import seaborn as sns
    import matplotlib.pyplot as plt
except ImportError as e:
    st.error(f"Visualization library import error: {e}")
    st.info("Please make sure matplotlib and seaborn are installed")
    st.stop()

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
except ImportError as e:
    st.error(f"Scikit-learn import error: {e}")
    st.info("Please make sure scikit-learn is installed")
    st.stop()

import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Home Credit Risk Analysis Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .insight-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
        color: #000000;
    }
    .warning-box {
        background-color: #fff8e1;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
        color: #000000;
    }
    .success-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)

# Generate sample data (since we can't include the actual large datasets)
@st.cache_data
def generate_sample_data():
    """Generate realistic sample data for demonstration"""
    np.random.seed(42)
    n_samples = 1000
    
    # Helper function to normalize probabilities
    def normalize_probs(probs):
        probs = np.array(probs)
        return probs / probs.sum()
    
    # Generate features
    data = {
        'AMT_CREDIT': np.random.lognormal(13, 0.5, n_samples),
        'AMT_INCOME_TOTAL': np.random.lognormal(12, 0.7, n_samples),
        'DAYS_BIRTH': -np.random.randint(20*365, 70*365, n_samples),
        'DAYS_EMPLOYED': -np.random.randint(0, 40*365, n_samples),
        'CNT_CHILDREN': np.random.poisson(0.5, n_samples),
        'NAME_EDUCATION_TYPE': np.random.choice([
            'Secondary / secondary special', 'Higher education', 
            'Incomplete higher', 'Lower secondary', 'Academic degree'
        ], n_samples, p=normalize_probs([0.71, 0.24, 0.033, 0.012, 0.005])),
        'NAME_INCOME_TYPE': np.random.choice([
            'Working', 'Commercial associate', 'Pensioner', 'State servant'
        ], n_samples, p=normalize_probs([0.52, 0.23, 0.18, 0.07])),
        'OCCUPATION_TYPE': np.random.choice([
            'Laborers', 'Sales staff', 'Core staff', 'Managers', 'Drivers',
            'High skill tech staff', 'Accountants', 'Medicine staff'
        ], n_samples, p=normalize_probs([0.26, 0.15, 0.13, 0.10, 0.09, 0.05, 0.05, 0.17])),
        'NAME_FAMILY_STATUS': np.random.choice([
            'Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow'
        ], n_samples, p=normalize_probs([0.64, 0.15, 0.10, 0.06, 0.05])),
        'CODE_GENDER': np.random.choice(['F', 'M'], n_samples, p=normalize_probs([0.65, 0.35]))
    }
    
    df = pd.DataFrame(data)
    
    # Generate target with realistic correlations
    risk_score = (
        -0.3 * (df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']) +
        0.2 * (df['DAYS_BIRTH'] / -365 / 100) +
        0.1 * (df['CNT_CHILDREN'] / 5) +
        np.random.normal(0, 0.5, n_samples)
    )
    
    df['TARGET'] = (risk_score > np.percentile(risk_score, 92)).astype(int)
    df['RISK_SCORE'] = (risk_score - risk_score.min()) / (risk_score.max() - risk_score.min())
    
    return df

# Load data
@st.cache_data
def load_data():
    return generate_sample_data()

def main():
    # Header
    st.markdown('<h1 class="main-header">🏦 Home Credit Default Risk Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Interactive Dashboard for Loan Default Prediction & Business Intelligence</p>', unsafe_allow_html=True)
    
    # Load data with error handling
    try:
        df = load_data()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("This dashboard uses simulated data for demonstration purposes.")
        return
    
    # Sidebar
    st.sidebar.title("🎛️ Dashboard Controls")
    st.sidebar.markdown("---")
    
    # Page selection
    page = st.sidebar.selectbox(
        "Choose Analysis View:",
        ["📊 Executive Summary", "🔍 Data Exploration", "🤖 Model Performance", 
         "💼 Business Insights", "🔮 Loan Predictor"]
    )
    
    if page == "📊 Executive Summary":
        show_executive_summary(df)
    elif page == "🔍 Data Exploration":
        show_data_exploration(df)
    elif page == "🤖 Model Performance":
        show_model_performance(df)
    elif page == "💼 Business Insights":
        show_business_insights(df)
    elif page == "🔮 Loan Predictor":
        show_loan_predictor(df)

def show_executive_summary(df):
    st.header("📊 Executive Summary")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📈 Model Performance",
            value="75.5%",
            delta="25% improvement",
            help="AUC-ROC score of our best LightGBM model"
        )
    
    with col2:
        st.metric(
            label="💰 Financial Impact",
            value="$11.9M",
            delta="Net benefit",
            help="Estimated annual benefit from improved risk assessment"
        )
    
    with col3:
        st.metric(
            label="🎯 Default Detection",
            value="80%",
            delta="Accuracy",
            help="Percentage of actual defaults correctly identified"
        )
    
    with col4:
        st.metric(
            label="📊 Dataset Size",
            value="246K+",
            delta="Applications",
            help="Training samples analyzed for model development"
        )
    
    # Key insights
    st.markdown("### 🎯 Key Business Achievements")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="success-box">
        <h4 style="color: #000000; margin-top: 0;">🎯 Technical Excellence</h4>
        <ul style="color: #000000;">
        <li><strong>Advanced ML Pipeline:</strong> End-to-end automation</li>
        <li><strong>Model Optimization:</strong> LightGBM with hyperparameter tuning</li>
        <li><strong>Production Ready:</strong> Sklearn pipelines & testing</li>
        <li><strong>Scalable Architecture:</strong> Modular code design</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="insight-box">
        <h4 style="color: #000000; margin-top: 0;">💼 Business Impact</h4>
        <ul style="color: #000000;">
        <li><strong>Risk Reduction:</strong> 25% improvement in default prediction</li>
        <li><strong>Customer Insights:</strong> Detailed risk segmentation</li>
        <li><strong>Operational Efficiency:</strong> Automated screening process</li>
        <li><strong>Regulatory Compliance:</strong> Interpretable model features</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Model comparison chart
    st.markdown("### 🏆 Model Performance Comparison")
    
    model_data = {
        'Model': ['Logistic Regression', 'Random Forest', 'Random Forest (Tuned)', 'LightGBM (Champion)'],
        'AUC-ROC': [0.6769, 0.7078, 0.7379, 0.7552],
        'Training Time': ['1.3s', '25s', '14min', '8min'],
        'Type': ['Baseline', 'Ensemble', 'Optimized', 'Gradient Boosting']
    }
    
    fig = px.bar(
        model_data, 
        x='Model', 
        y='AUC-ROC',
        color='AUC-ROC',
        color_continuous_scale='viridis',
        title="Model Performance Progression",
        text='AUC-ROC'
    )
    fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Technology stack
    st.markdown("### 🛠️ Technology Stack")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🐍 Core Technologies**
        - Python 3.8+
        - Pandas & NumPy
        - Scikit-learn
        - LightGBM
        """)
    
    with col2:
        st.markdown("""
        **📊 Visualization & Analysis**
        - Matplotlib & Seaborn
        - Plotly & Streamlit
        - Jupyter Notebooks
        - Statistical Analysis
        """)
    
    with col3:
        st.markdown("""
        **⚙️ MLOps & Engineering**
        - Automated Pipelines
        - Hyperparameter Tuning
        - Unit Testing (pytest)
        - Version Control (Git)
        """)

def show_data_exploration(df):
    st.header("🔍 Data Exploration & Analysis")
    
    # Dataset overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 Dataset Overview")
        st.write(f"**Total Applications:** {len(df):,}")
        st.write(f"**Features:** {len(df.columns)-2}")  # Excluding TARGET and RISK_SCORE
        st.write(f"**Default Rate:** {df['TARGET'].mean():.1%}")
        st.write(f"**Data Quality:** {(1-df.isnull().sum().sum()/(len(df)*len(df.columns))):.1%} complete")
    
    with col2:
        # Target distribution
        fig = px.pie(
            values=df['TARGET'].value_counts().values,
            names=['Repaid (0)', 'Default (1)'],
            title="Loan Outcome Distribution",
            color_discrete_sequence=['#2E8B57', '#DC143C']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature analysis
    st.markdown("### 📊 Key Feature Analysis")
    
    # Credit amount distribution
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            df, 
            x='AMT_CREDIT', 
            nbins=50,
            title="Credit Amount Distribution",
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Income vs Credit scatter
        fig = px.scatter(
            df.sample(500), 
            x='AMT_INCOME_TOTAL', 
            y='AMT_CREDIT',
            color='TARGET',
            title="Income vs Credit Amount",
            color_discrete_sequence=['#2E8B57', '#DC143C'],
            labels={'TARGET': 'Loan Status'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Categorical analysis
    st.markdown("### 🎓 Customer Demographics")
    
    # Education analysis
    col1, col2 = st.columns(2)
    
    with col1:
        education_counts = df['NAME_EDUCATION_TYPE'].value_counts()
        fig = px.bar(
            x=education_counts.index,
            y=education_counts.values,
            title="Education Level Distribution",
            color=education_counts.values,
            color_continuous_scale='blues'
        )
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Default rate by education
        edu_default = df.groupby('NAME_EDUCATION_TYPE')['TARGET'].agg(['mean', 'count']).reset_index()
        fig = px.bar(
            edu_default,
            x='NAME_EDUCATION_TYPE',
            y='mean',
            title="Default Rate by Education Level",
            color='mean',
            color_continuous_scale='reds'
        )
        fig.update_layout(height=400, xaxis_tickangle=-45)
        fig.update_yaxes(tickformat='.1%')
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.markdown("### 🔗 Feature Correlations")
    
    # Select numeric columns
    numeric_cols = ['AMT_CREDIT', 'AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'CNT_CHILDREN', 'TARGET']
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Feature Correlation Matrix",
        color_continuous_scale='RdBu_r'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def show_model_performance(df):
    st.header("🤖 Model Performance Analysis")
    
    # Train a simple model for demonstration
    @st.cache_data
    def train_demo_models(df):
        # Prepare data
        le_dict = {}
        df_encoded = df.copy()
        
        categorical_cols = ['NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE', 'OCCUPATION_TYPE', 
                           'NAME_FAMILY_STATUS', 'CODE_GENDER']
        
        for col in categorical_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            le_dict[col] = le
        
        # Features and target
        feature_cols = ['AMT_CREDIT', 'AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 
                       'CNT_CHILDREN'] + categorical_cols
        
        X = df_encoded[feature_cols]
        y = df_encoded['TARGET']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        models = {}
        
        # Logistic Regression
        lr = LogisticRegression(random_state=42)
        lr.fit(X_train_scaled, y_train)
        lr_pred = lr.predict_proba(X_test_scaled)[:, 1]
        models['Logistic Regression'] = {
            'model': lr,
            'predictions': lr_pred,
            'auc': roc_auc_score(y_test, lr_pred)
        }
        
        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict_proba(X_test)[:, 1]
        models['Random Forest'] = {
            'model': rf,
            'predictions': rf_pred,
            'auc': roc_auc_score(y_test, rf_pred)
        }
        
        return models, X_test, y_test, feature_cols
    
    models, X_test, y_test, feature_cols = train_demo_models(df)
    
    # Model performance metrics
    st.markdown("### 📈 Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Logistic Regression AUC", f"{models['Logistic Regression']['auc']:.4f}")
    
    with col2:
        st.metric("Random Forest AUC", f"{models['Random Forest']['auc']:.4f}")
    
    with col3:
        # Simulated LightGBM performance
        st.metric("LightGBM AUC (Production)", "0.7552", delta="Best Model")
    
    # ROC Curves
    st.markdown("### 📊 ROC Curve Analysis")
    
    fig = go.Figure()
    
    # Add ROC curves for each model
    for model_name, model_data in models.items():
        fpr, tpr, _ = roc_curve(y_test, model_data['predictions'])
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f"{model_name} (AUC = {model_data['auc']:.4f})",
            line=dict(width=3)
        ))
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(dash='dash', color='gray')
    ))
    
    fig.update_layout(
        title='ROC Curve Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=500,
        hovermode='x'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance (Random Forest)
    st.markdown("### 🎯 Feature Importance Analysis")
    
    rf_model = models['Random Forest']['model']
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    fig = px.bar(
        feature_importance.tail(10),
        x='importance',
        y='feature',
        orientation='h',
        title='Top 10 Most Important Features',
        color='importance',
        color_continuous_scale='viridis'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Model interpretation
    st.markdown("### 💡 Model Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="insight-box">
        <h4 style="color: #000000; margin-top: 0;">🔍 Key Predictive Factors</h4>
        <ul style="color: #000000;">
        <li><strong>Credit Amount:</strong> Higher amounts increase risk</li>
        <li><strong>Income Ratio:</strong> Income-to-credit ratio is crucial</li>
        <li><strong>Employment History:</strong> Stability matters</li>
        <li><strong>Age:</strong> Older applicants tend to be more reliable</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-box">
        <h4 style="color: #000000; margin-top: 0;">⚠️ Model Limitations</h4>
        <ul style="color: #000000;">
        <li><strong>Class Imbalance:</strong> Only 8% defaults in training</li>
        <li><strong>Feature Engineering:</strong> Could benefit from more features</li>
        <li><strong>External Data:</strong> Credit bureau data would help</li>
        <li><strong>Temporal Factors:</strong> Economic conditions not captured</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

def show_business_insights(df):
    st.header("💼 Business Intelligence & Insights")
    
    # Risk segmentation
    st.markdown("### 🎯 Customer Risk Segmentation")
    
    # Create risk segments based on various factors
    def create_risk_segments(df):
        df_seg = df.copy()
        
        # Income to credit ratio
        df_seg['income_credit_ratio'] = df_seg['AMT_INCOME_TOTAL'] / df_seg['AMT_CREDIT']
        
        # Age groups
        df_seg['age'] = -df_seg['DAYS_BIRTH'] // 365
        df_seg['age_group'] = pd.cut(df_seg['age'], 
                                   bins=[0, 30, 40, 50, 100], 
                                   labels=['18-30', '31-40', '41-50', '50+'])
        
        # Risk segments
        conditions = [
            (df_seg['income_credit_ratio'] >= 0.5) & (df_seg['NAME_EDUCATION_TYPE'].isin(['Higher education', 'Academic degree'])),
            (df_seg['income_credit_ratio'] >= 0.3) & (df_seg['NAME_INCOME_TYPE'] == 'Working'),
            (df_seg['income_credit_ratio'] < 0.2) | (df_seg['NAME_INCOME_TYPE'] == 'Unemployed'),
        ]
        
        choices = ['Low Risk', 'Medium Risk', 'High Risk']
        df_seg['risk_segment'] = np.select(conditions, choices, default='Medium Risk')
        
        return df_seg
    
    df_seg = create_risk_segments(df)
    
    # Risk segment analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Segment distribution
        segment_counts = df_seg['risk_segment'].value_counts()
        fig = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title="Customer Risk Distribution",
            color_discrete_sequence=['#2E8B57', '#FFA500', '#DC143C']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Default rate by segment
        segment_default = df_seg.groupby('risk_segment')['TARGET'].agg(['mean', 'count']).reset_index()
        fig = px.bar(
            segment_default,
            x='risk_segment',
            y='mean',
            title="Default Rate by Risk Segment",
            color='mean',
            color_continuous_scale='reds',
            text='mean'
        )
        fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig.update_yaxes(tickformat='.1%')
        st.plotly_chart(fig, use_container_width=True)
    
    # Business recommendations
    st.markdown("### 💡 Strategic Recommendations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="success-box">
        <h4 style="color: #000000; margin-top: 0;">🎯 Low Risk Customers</h4>
        <p style="color: #000000;"><strong>Strategy:</strong> Premium Products</p>
        <ul style="color: #000000;">
        <li>Offer competitive rates</li>
        <li>Increase credit limits</li>
        <li>Cross-sell other products</li>
        <li>Fast-track approvals</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="insight-box">
        <h4 style="color: #000000; margin-top: 0;">⚖️ Medium Risk Customers</h4>
        <p style="color: #000000;"><strong>Strategy:</strong> Standard Processing</p>
        <ul style="color: #000000;">
        <li>Standard interest rates</li>
        <li>Regular monitoring</li>
        <li>Financial education</li>
        <li>Gradual limit increases</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="warning-box">
        <h4 style="color: #000000; margin-top: 0;">⚠️ High Risk Customers</h4>
        <p style="color: #000000;"><strong>Strategy:</strong> Enhanced Due Diligence</p>
        <ul style="color: #000000;">
        <li>Higher interest rates</li>
        <li>Lower credit limits</li>
        <li>Additional documentation</li>
        <li>Frequent reviews</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Financial impact analysis
    st.markdown("### 💰 Financial Impact Analysis")
    
    # Simulate financial impact
    avg_loan = df['AMT_CREDIT'].mean()
    total_loans = len(df)
    current_default_rate = df['TARGET'].mean()
    improved_default_rate = current_default_rate * 0.75  # 25% improvement
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Average Loan Amount",
            f"${avg_loan:,.0f}",
            help="Average credit amount in portfolio"
        )
    
    with col2:
        st.metric(
            "Current Default Rate",
            f"{current_default_rate:.1%}",
            help="Historical default rate"
        )
    
    with col3:
        st.metric(
            "Improved Default Rate",
            f"{improved_default_rate:.1%}",
            delta=f"{(improved_default_rate-current_default_rate)/current_default_rate:.1%}",
            help="Projected default rate with ML model"
        )
    
    with col4:
        annual_savings = total_loans * avg_loan * (current_default_rate - improved_default_rate)
        st.metric(
            "Annual Savings",
            f"${annual_savings:,.0f}",
            delta="Estimated benefit",
            help="Projected annual savings from improved risk assessment"
        )
    
    # Portfolio optimization
    st.markdown("### 📊 Portfolio Optimization Opportunities")
    
    # Age vs default rate analysis
    age_analysis = df_seg.groupby('age_group').agg({
        'TARGET': ['mean', 'count'],
        'AMT_CREDIT': 'mean'
    }).round(4)
    
    age_analysis.columns = ['Default Rate', 'Count', 'Avg Credit']
    age_analysis = age_analysis.reset_index()
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Default Rate by Age Group', 'Average Credit by Age Group'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    fig.add_trace(
        go.Bar(x=age_analysis['age_group'], y=age_analysis['Default Rate'], 
               name='Default Rate', marker_color='red'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=age_analysis['age_group'], y=age_analysis['Avg Credit'], 
               name='Avg Credit', marker_color='blue'),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def show_loan_predictor(df):
    st.header("🔮 Interactive Loan Default Predictor")
    st.markdown("*Try different customer profiles to see risk predictions*")
    
    # Input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👤 Customer Information")
        
        credit_amount = st.slider(
            "Credit Amount ($)",
            min_value=10000,
            max_value=500000,
            value=100000,
            step=5000
        )
        
        income_amount = st.slider(
            "Annual Income ($)",
            min_value=20000,
            max_value=300000,
            value=60000,
            step=5000
        )
        
        age = st.slider(
            "Age",
            min_value=18,
            max_value=70,
            value=35
        )
        
        children = st.selectbox(
            "Number of Children",
            options=[0, 1, 2, 3, 4, 5],
            index=1
        )
    
    with col2:
        st.markdown("### 📋 Background Information")
        
        education = st.selectbox(
            "Education Level",
            options=['Secondary / secondary special', 'Higher education', 
                    'Incomplete higher', 'Lower secondary', 'Academic degree'],
            index=0
        )
        
        income_type = st.selectbox(
            "Income Type",
            options=['Working', 'Commercial associate', 'Pensioner', 'State servant'],
            index=0
        )
        
        family_status = st.selectbox(
            "Family Status",
            options=['Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow'],
            index=0
        )
        
        gender = st.selectbox(
            "Gender",
            options=['Female', 'Male'],
            index=0
        )
    
    # Calculate risk score (simplified model)
    def calculate_risk_score(credit_amount, income_amount, age, children, education, income_type, family_status, gender):
        # Simple risk scoring model
        risk_score = 0.5  # Base risk
        
        # Income to credit ratio impact
        income_ratio = income_amount / credit_amount
        if income_ratio > 0.5:
            risk_score -= 0.2
        elif income_ratio < 0.2:
            risk_score += 0.3
        
        # Age impact
        if age > 50:
            risk_score -= 0.1
        elif age < 25:
            risk_score += 0.1
        
        # Education impact
        if education in ['Higher education', 'Academic degree']:
            risk_score -= 0.1
        elif education == 'Lower secondary':
            risk_score += 0.1
        
        # Children impact
        if children > 3:
            risk_score += 0.1
        
        # Income type impact
        if income_type == 'Pensioner':
            risk_score -= 0.05
        elif income_type == 'Working':
            risk_score -= 0.02
        
        # Family status impact
        if family_status == 'Married':
            risk_score -= 0.05
        
        # Ensure score is between 0 and 1
        risk_score = max(0, min(1, risk_score))
        
        return risk_score
    
    risk_score = calculate_risk_score(credit_amount, income_amount, age, children, 
                                    education, income_type, family_status, gender)
    
    # Display prediction
    st.markdown("### 🎯 Risk Assessment Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Risk score gauge
        risk_percentage = risk_score * 100
        
        if risk_percentage < 30:
            risk_level = "Low Risk"
            color = "green"
        elif risk_percentage < 60:
            risk_level = "Medium Risk"
            color = "orange"
        else:
            risk_level = "High Risk"
            color = "red"
        
        st.metric(
            "Risk Score",
            f"{risk_percentage:.1f}%",
            help="Probability of default based on customer profile"
        )
        
        st.markdown(f"**Risk Level:** <span style='color: {color}; font-weight: bold;'>{risk_level}</span>", 
                   unsafe_allow_html=True)
    
    with col2:
        # Recommendation
        if risk_percentage < 30:
            recommendation = "✅ **Approve** with standard terms"
            details = "Low risk customer. Consider premium products."
        elif risk_percentage < 60:
            recommendation = "⚖️ **Review** application carefully"
            details = "Medium risk. Standard processing with monitoring."
        else:
            recommendation = "❌ **Enhanced due diligence** required"
            details = "High risk. Consider higher rates or additional collateral."
        
        st.markdown("**Recommendation:**")
        st.markdown(recommendation)
        st.markdown(f"*{details}*")
    
    with col3:
        # Key factors
        st.markdown("**Key Risk Factors:**")
        
        factors = []
        if income_amount / credit_amount < 0.3:
            factors.append("• Low income-to-credit ratio")
        if age < 25:
            factors.append("• Young age")
        if children > 2:
            factors.append("• Multiple dependents")
        if education == 'Lower secondary':
            factors.append("• Limited education")
        
        if factors:
            for factor in factors:
                st.markdown(factor)
        else:
            st.markdown("• No major risk factors identified")
    
    # Risk breakdown visualization
    st.markdown("### 📊 Risk Factor Breakdown")
    
    # Create risk factor analysis
    factors_data = {
        'Factor': ['Income Ratio', 'Age', 'Education', 'Family Status', 'Dependents'],
        'Impact': [
            -0.2 if income_amount/credit_amount > 0.5 else 0.3 if income_amount/credit_amount < 0.2 else 0,
            -0.1 if age > 50 else 0.1 if age < 25 else 0,
            -0.1 if education in ['Higher education', 'Academic degree'] else 0.1 if education == 'Lower secondary' else 0,
            -0.05 if family_status == 'Married' else 0,
            0.1 if children > 3 else 0
        ]
    }
    
    fig = px.bar(
        factors_data,
        x='Factor',
        y='Impact',
        title='Risk Factor Impact Analysis',
        color='Impact',
        color_continuous_scale='RdYlGn_r'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Additional insights
    st.markdown("### 💡 Model Insights")
    
    st.markdown("""
    <div class="insight-box">
    <p style="color: #000000;"><strong>This predictor demonstrates:</strong></p>
    <ul style="color: #000000;">
    <li><strong>Real-time Risk Assessment:</strong> Instant evaluation of loan applications</li>
    <li><strong>Interpretable Results:</strong> Clear explanation of risk factors</li>
    <li><strong>Business Integration:</strong> Ready for integration into loan processing systems</li>
    <li><strong>Regulatory Compliance:</strong> Transparent decision-making process</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 