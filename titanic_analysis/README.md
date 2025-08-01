# Titanic Dataset Analysis Pipeline

A comprehensive ETL and analysis pipeline for the classic Titanic dataset, featuring data extraction, transformation, exploratory data analysis, and machine learning-based survival prediction.

## 🚢 Overview

This project provides a complete end-to-end analysis of the Titanic dataset, including:

- **ETL Pipeline**: Automated data extraction, cleaning, and feature engineering
- **Exploratory Data Analysis**: Statistical analysis and visualizations
- **Machine Learning**: Survival prediction using multiple algorithms
- **Comprehensive Reporting**: Automated report generation with insights

## 📁 Project Structure

```
titanic_analysis/
├── data/
│   ├── raw/                    # Raw dataset storage
│   └── processed/              # Cleaned and processed datasets
├── src/
│   ├── etl/
│   │   ├── extract.py         # Data extraction and loading
│   │   ├── transform.py       # Data cleaning and feature engineering
│   │   ├── load.py           # Data saving and quality reporting
│   │   └── pipeline.py       # ETL orchestration
│   ├── analysis/
│   │   ├── exploratory_analysis.py  # EDA and visualizations
│   │   └── survival_prediction.py   # ML models and evaluation
│   └── visualization/
├── notebooks/                  # Jupyter notebooks (optional)
├── output/
│   ├── figures/               # Generated plots and charts
│   ├── models/                # Trained ML models
│   └── reports/               # Analysis reports
├── main.py                    # Main execution script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🛠️ Installation

1. **Clone or download the project**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Quick Start

Run the complete analysis pipeline:

```bash
python main.py
```

This will execute all three phases:
1. **ETL Pipeline** - Download, clean, and process the data
2. **Exploratory Data Analysis** - Generate insights and visualizations
3. **Survival Prediction** - Train and evaluate ML models

## 📊 Features

### ETL Pipeline
- **Data Extraction**: Automatic download from seaborn dataset
- **Data Cleaning**: Handle missing values, duplicates, and data quality issues
- **Feature Engineering**: Create new features like family size, age groups, fare per person
- **Data Validation**: Comprehensive data quality checks and reporting

### Exploratory Data Analysis
- **Survival Analysis**: Overall and group-specific survival rates
- **Statistical Analysis**: Descriptive statistics and correlation analysis
- **Visualizations**: 
  - Survival rate charts by passenger class, sex, age
  - Distribution plots for numerical variables
  - Correlation heatmaps
  - Interactive plots with professional styling

### Machine Learning
- **Multiple Algorithms**:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - Support Vector Machine
- **Model Evaluation**:
  - Cross-validation
  - Performance metrics (accuracy, precision, recall, F1-score, ROC-AUC)
  - Confusion matrices
  - ROC curves
  - Feature importance analysis

### Reporting
- **Automated Reports**: JSON and Markdown formats
- **Executive Summary**: Key findings and recommendations
- **Comprehensive Documentation**: All analysis steps and results

## 📈 Key Insights

The analysis typically reveals:

- **Overall survival rate**: ~38% of passengers survived
- **Gender impact**: Women had significantly higher survival rates
- **Class effect**: First-class passengers had better survival chances
- **Age factor**: Children had higher survival rates
- **Family size**: Moderate family sizes showed better survival rates

## 🔧 Customization

### Running Individual Components

**ETL Pipeline only**:
```python
from src.etl.pipeline import TitanicETLPipeline
pipeline = TitanicETLPipeline()
results = pipeline.run_pipeline()
```

**EDA only**:
```python
from src.analysis.exploratory_analysis import TitanicEDA
import pandas as pd

df = pd.read_csv('data/processed/titanic_basic_features.csv')
eda = TitanicEDA()
results = eda.run_complete_eda(df)
```

**ML Prediction only**:
```python
from src.analysis.survival_prediction import TitanicSurvivalPredictor
import pandas as pd

df = pd.read_csv('data/processed/titanic_ml_ready.csv')
predictor = TitanicSurvivalPredictor()
results = predictor.run_complete_prediction_analysis(df)
```

### Configuration Options

- **Feature Scaling**: Enable/disable feature scaling in the pipeline
- **Model Selection**: Choose specific ML algorithms to train
- **Output Directories**: Customize where results are saved
- **Visualization Styles**: Modify plot themes and colors

## 📋 Requirements

- Python 3.7+
- pandas >= 1.5.0
- numpy >= 1.21.0
- scikit-learn >= 1.1.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- rich >= 12.0.0 (for colored console output)
- xgboost >= 1.6.0
- scipy >= 1.9.0

## 📊 Output Files

### Data Files
- `titanic.csv` - Raw dataset
- `titanic_basic_features.csv` - Basic cleaned features
- `titanic_engineered_features.csv` - With engineered features
- `titanic_ml_ready.csv` - ML-ready dataset with encoding and scaling
- `data_dictionary.json` - Comprehensive data documentation
- `quality_report.json` - Data quality assessment

### Visualizations
- `survival_overall.png` - Overall survival pie chart
- `survival_by_class.png` - Survival rates by passenger class
- `survival_by_sex.png` - Survival rates by gender
- `distributions.png` - Distribution plots for numerical variables
- `correlation_heatmap.png` - Feature correlation matrix
- `model_comparison.png` - ML model performance comparison
- `confusion_matrices.png` - Confusion matrices for all models
- `roc_curves.png` - ROC curves comparison
- `feature_importance.png` - Feature importance ranking

### Models and Reports
- `*_model.joblib` - Trained ML models
- `titanic_analysis_report_*.json` - Comprehensive analysis results
- `titanic_analysis_summary_*.md` - Executive summary report

## 🎯 Use Cases

- **Data Science Education**: Learn ETL, EDA, and ML concepts
- **Portfolio Projects**: Demonstrate end-to-end data science skills
- **Research**: Historical analysis of Titanic disaster factors
- **Template**: Base for similar classification projects
- **Benchmarking**: Compare different ML approaches

## 🤝 Contributing

Feel free to contribute by:
- Adding new feature engineering techniques
- Implementing additional ML algorithms
- Improving visualizations
- Enhancing documentation
- Adding statistical tests

## 📝 License

This project is open source and available under the MIT License.

## 🔍 Technical Details

### Data Processing Pipeline
1. **Extraction**: Download from seaborn's built-in dataset
2. **Validation**: Check data structure and quality
3. **Cleaning**: Handle missing values using domain-specific strategies
4. **Feature Engineering**: Create meaningful derived features
5. **Encoding**: Convert categorical variables for ML
6. **Scaling**: Normalize numerical features
7. **Saving**: Export multiple dataset versions

### Machine Learning Pipeline
1. **Feature Preparation**: Select and prepare ML-ready features
2. **Data Splitting**: Stratified train-test split
3. **Model Training**: Train multiple algorithms with cross-validation
4. **Evaluation**: Comprehensive performance assessment
5. **Comparison**: Side-by-side model comparison
6. **Selection**: Identify best performing model
7. **Interpretation**: Feature importance and model insights

### Quality Assurance
- Comprehensive error handling
- Data validation at each step
- Automated testing of pipeline components
- Rich console output with progress tracking
- Detailed logging and reporting

## 📞 Support

For questions or issues:
1. Check the generated reports for detailed analysis results
2. Review the console output for any error messages
3. Ensure all dependencies are properly installed
4. Verify data files are accessible and not corrupted

---

*This pipeline was designed to be educational, comprehensive, and production-ready. It demonstrates best practices in data science project structure, code organization, and automated analysis workflows.*
