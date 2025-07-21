# 🏦 Home Credit Default Risk - ML Pipeline Project

> **Complete machine learning pipeline for predicting loan default risk with 75.5% AUC-ROC accuracy**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-3.0+-green.svg)](https://lightgbm.readthedocs.io)

## 📋 Quick Overview

This project implements an **end-to-end machine learning pipeline** for predicting loan default risk, featuring:

- 🎯 **75.5% AUC-ROC** with optimized LightGBM model
- 🚀 **Automated sklearn pipelines** for production-ready deployment
- 📊 **Comprehensive EDA** with business insights
- 🔧 **Modular architecture** with proper testing and documentation
- ⚡ **Multiple model comparison** (Logistic Regression → Random Forest → LightGBM)

## 🚀 Quick Start

### Installation
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Project
```bash
# Launch the main analysis notebook
jupyter notebook "AnyoneAI - Sprint Project 02.ipynb"

# Or run the interactive dashboard (great for demos!)
python run_dashboard.py

# Or run tests to verify setup
pytest tests/
```

## 📁 Project Structure

```
📁 fintech_pipeline_ml/
├── 📓 AnyoneAI - Sprint Project 02.ipynb  # Main analysis & implementation
├── 🌐 streamlit_app.py                   # 🎯 INTERACTIVE DASHBOARD
├── 🚀 run_dashboard.py                   # Dashboard launcher
├── 📄 README_Portfolio.md               # 📖 DETAILED PROJECT SHOWCASE
├── 📁 src/                              # Modular code architecture
│   ├── 🐍 data_utils.py                # Data loading & splitting functions
│   ├── 🐍 preprocessing.py             # Feature engineering pipeline
│   └── 🐍 config.py                    # Configuration management
├── 📁 dataset/                         # Data storage (auto-downloaded)
├── 📁 tests/                           # Unit tests
├── 📄 requirements.txt                 # Project dependencies
└── 📄 streamlit_requirements.txt       # Dashboard dependencies
```

## 🎯 Key Results

| Model | Validation AUC | Training Time | Key Features |
|-------|---------------|---------------|--------------|
| Logistic Regression | 0.6769 | ~1.3s | Baseline, interpretable |
| Random Forest (Tuned) | 0.7379 | ~14min | Hyperparameter optimization |
| **🏆 LightGBM (Tuned)** | **0.7552** | ~8min | **Best performance** |

## 💼 Business Impact

- **📈 Model Performance**: 75.5% AUC-ROC (25% improvement over baseline)
- **💰 Risk Reduction**: Identifies 8 out of 10 potential defaults
- **🎯 Financial Impact**: ~$11.9M net benefit improvement
- **📊 Data Insights**: 246K+ samples analyzed with actionable recommendations

## 🛠️ Technical Implementation

### Core Features
- **Automated Preprocessing Pipeline** with sklearn
- **Hyperparameter Optimization** using RandomizedSearchCV
- **Cross-validation** for robust model evaluation
- **Feature Engineering** for improved performance
- **Production-ready Code** with modular architecture

### Technologies Used
- **Python**: Core programming language
- **Pandas/NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms and pipelines
- **LightGBM**: Advanced gradient boosting
- **Matplotlib/Seaborn**: Data visualization
- **Streamlit/Plotly**: Interactive dashboard and modern visualizations
- **Jupyter**: Interactive development environment

## 🎯 Interactive Dashboard - NEW! 🚀

**👉 Launch the Interactive Dashboard: `python run_dashboard.py`**

Experience the complete analysis through a **professional web interface** featuring:

### **🎪 Dashboard Features**
- **📊 Executive Summary**: Key metrics, model comparison, and technology stack overview
- **🔍 Data Exploration**: Interactive visualizations, correlation heatmaps, and demographic analysis
- **🤖 Model Performance**: Live model training, ROC curves, and feature importance analysis
- **💼 Business Intelligence**: Risk segmentation, financial impact analysis, and strategic recommendations
- **🔮 Live Loan Predictor**: Interactive risk assessment tool - try different customer profiles!

### **💡 Perfect for:**
- 👔 **Job Interviews**: Live demonstration of your ML skills
- 📈 **Portfolio Presentations**: Professional showcase for recruiters
- 🎓 **Learning**: Interactive exploration of ML concepts
- 💼 **Business Stakeholders**: Non-technical friendly interface

### **🎨 Professional Design**
- Clean, corporate-style interface with readable fonts
- Interactive Plotly visualizations
- Real-time model predictions
- Mobile-responsive layout

## 🔍 For Detailed Analysis

**👉 See [README_Portfolio.md](README_Portfolio.md) for comprehensive project showcase including:**

- 📊 **Detailed Business Impact Analysis**
- 🔍 **Complete EDA Findings & Visualizations**
- ⚙️ **Advanced Technical Implementation Details**
- 📈 **Model Performance Deep-dive**
- 💡 **Business Intelligence & Recommendations**
- 🎓 **Key Learnings & Industry Best Practices**
- 🔮 **Future Enhancement Roadmap**

## 🧪 Testing & Quality Assurance

```bash
# Run unit tests
pytest tests/

# Code formatting
isort --profile=black . && black --line-length 88 .

# Check code quality
flake8 src/
```

## 📊 Development Workflow

1. **Data Exploration** → Comprehensive EDA with business insights
2. **Feature Engineering** → Automated preprocessing pipeline
3. **Model Development** → Progressive model complexity (LR → RF → LightGBM)
4. **Hyperparameter Tuning** → Optimized performance with cross-validation
5. **Pipeline Creation** → Production-ready sklearn pipelines
6. **Validation & Testing** → Robust evaluation and unit tests

## 🎯 Learning Outcomes

This project demonstrates:
- **End-to-end ML pipeline** development
- **Interactive dashboard** creation with Streamlit
- **Financial risk modeling** expertise
- **Production-ready code** practices
- **Business impact** quantification
- **Advanced model optimization** techniques
- **Professional presentation** skills for stakeholders

---

**💡 This project showcases advanced ML engineering skills suitable for fintech, banking, and risk management applications.**

*For the complete project showcase with detailed analysis, business insights, and technical deep-dive, please see [README_Portfolio.md](README_Portfolio.md)*
