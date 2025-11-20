# CMT122 Coursework 1: Machine Learning for NLP
**Academic Year:** 2025/2026  
**Student:** Samidur Rahman (21053329)  
**Module:** CMT122 - Machine Learning for Natural Language Processing  

---

## 📋 Overview

This repository contains the complete submission for CMT122 Coursework 1, implementing machine learning pipelines for two text classification tasks:

- **Part 1**: SemEval2017 Task 4 - Twitter Sentiment Analysis (Classification & Regression)
- **Part 2**: 20_Newsgroups Multi-class Text Classification

Both parts demonstrate systematic ML methodology including data preprocessing, feature engineering, model selection, and comprehensive evaluation.

---

## 📂 Repository Structure

```
CMT122-Coursework1/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── NPart1_SentimentAnalysis_SemEval2017.ipynb    # Main notebook
├── sentiment_analysis_instructions.pdf            # Task specification
|── Classification_Performance_Experiment.pdf      # Question 1 calculations
│
├── Part2_20Newsgroups_Classification_Complete.ipynb  # Main notebook
├── Part2_Report.pdf                                   #  Report
│

```

---

##  Key Results

### Part 1: Sentiment Analysis
- **Classification**: 65.13% accuracy (SVC Linear)
- **Regression**: 0.5752 RMSE (Ridge α=1.0)
- **Dataset**: 19,699 tweets (3-class: positive/negative/neutral)

### Part 2: Text Classification  
- **Accuracy**: 88.74% (exceeds 65% requirement by +23.74pp)
- **Macro F1**: 0.8906
- **Model**: LinearSVC with chi-squared feature selection
- **Dataset**: 3,416 newsgroup articles (6 categories)

---

##  Quick Start

### Prerequisites

```bash
Python 3.8+
Jupyter Notebook / JupyterLab
```

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/CMT122-Coursework1.git
   cd CMT122-Coursework1
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data** (first run only)
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('punkt_tab')
   ```

### Running the Notebooks

**Part 1: Sentiment Analysis**
```bash
cd Part1_SentimentAnalysis
jupyter notebook NPart1_SentimentAnalysis_SemEval2017.ipynb
```

**Part 2: Text Classification**
```bash
cd Part2_TextClassification
jupyter notebook Part2_20Newsgroups_Classification_Complete.ipynb
```

 **Important**: Update file paths in notebooks to match your local data location.

---

## Part 1: Twitter Sentiment Analysis

### Task Description
Binary and regression approaches to sentiment analysis on SemEval2017 Task 4 dataset.

### Methodology
- **Preprocessing**: Lowercase conversion, URL/mention removal, hashtag extraction
- **Features**: TF-IDF (8,000 features, trigrams, optimized parameters)
- **Classification Model**: SVC (Linear kernel, C=1.0)
- **Regression Model**: Ridge Regression (α=1.0)

### Key Features
- Comprehensive model comparison (4 classifiers, 3 regressors)
- Optimized TF-IDF parameters (trigrams, sublinear_tf)
- Detailed performance analysis with confusion matrices

### Files
| File | Description |
|------|-------------|
| `NPart1_SentimentAnalysis_SemEval2017.ipynb` | Complete implementation |
| `sentiment_analysis_instructions.pdf` | Original task specification |
| `Classification_Performance_Experiment.pdf` | Manual metric calculations (Question 1) |

### Performance Summary
```
Classification:
  Model: SVC (Linear, C=1.0)
  Training instruction: svc_linear.fit(X_train_features, y_train_class)
  Prediction instruction: y_pred_lin = svc_linear.predict(X_test_features)
  Accuracy: 65.13%
  Macro F1: 0.60

Regression:
  Model: Ridge (α=1.0)
  Training instruction: ridge.fit(X_train_features, y_train_reg)
  Prediction instruction: y_pred_ridge = ridge.predict(X_test_features)
  RMSE: 0.5752
```

---

## Part 2: Multi-class Text Classification

### Task Description
Classify newsgroup articles into 6 categories using TF-IDF + statistical features with chi-squared selection.

### Methodology

**1. Preprocessing**
- Lowercase conversion, email/header removal
- Non-alphabetic character filtering
- Minimum length filtering (50 chars)

**2. Feature Engineering** (4 features required)
- **Feature 1**: TF-IDF (20,000 features, bigrams)
- **Feature 2**: Word count
- **Feature 3**: Average word length
- **Feature 4**: Lexical diversity (type-token ratio)

**3. Feature Selection**
- Chi-squared testing
- Development set experiments (k ∈ {1000, 5000, 10000, 15000, 19229})
- Optimal k=15,000 (90.34% dev accuracy)

**4. Model Selection**
- LinearSVC (C=1.0) - Best (90.34%)
- Logistic Regression (87.99%)
- Multinomial NB (88.58%)

**5. Evaluation Protocol**
- 60/20/20 train/dev/test split
- Stratified sampling
- Final model trained on train+dev

### Key Features
-  Systematic development set experiments
-  Four distinct feature types (TF-IDF + 3 statistical)
-  Chi-squared feature selection with justification
-  Comprehensive evaluation metrics
-  1200-word technical report

### Files
| File | Description |
|------|-------------|
| `Part2_20Newsgroups_Classification_Complete.ipynb` | Complete implementation with extensive documentation |
| `Part2_Report.pdf` | Final 1200-word technical report |
| `Part2_Report_20Newsgroups.tex` | LaTeX source for report |

### Performance Summary
```
Test Set Results:
  Accuracy: 88.74% (+23.74pp above requirement)
  Macro Precision: 0.8909
  Macro Recall: 0.8906
  Macro F1: 0.8906

Per-Class F1 Scores:
  class-1: 0.99 (near-perfect)
  class-2: 0.85
  class-3: 0.86
  class-4: 0.82
  class-5: 0.92
  class-6: 0.90
```

---

##  Technical Details

### Dependencies
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
nltk>=3.6
scipy>=1.7.0
jupyter>=1.0.0
```

### Hardware Requirements
- **RAM**: 8GB minimum
- **CPU**: Multi-core recommended for faster training

### Computational Complexity
- **Part 1**: ~2-3 minutes on standard laptop
- **Part 2**: ~5-7 minutes (feature selection experiments)

---

##  Methodology Highlights

### Part 1: Sentiment Analysis

**Feature Engineering:**
```python
TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 3),      # Trigrams
    stop_words='english',
    min_df=2,
    max_df=0.7,
    sublinear_tf=True
)
```

**Best Models:**
- Classification: `SVC(kernel='linear', C=1.0)`
- Regression: `Ridge(alpha=1.0)`

### Part 2: Text Classification

**Feature Pipeline:**
1. TF-IDF → 19,226 features
2. Statistical features (scaled) → 3 features
3. Concatenation → 19,229 total
4. Chi-squared selection → 15,000 optimal

**Training Protocol:**
```python
# Development experiments
for k in [1000, 5000, 10000, 15000, 19229]:
    selector = SelectKBest(chi2, k=k)
    X_selected = selector.fit_transform(X_train, y_train)
    model.fit(X_selected, y_train)
    evaluate_on_dev_set()

# Final model
X_traindev = vstack([X_train, X_dev])
final_model.fit(X_traindev, y_traindev)
```

---

## 📝 Reports

### Part 1: Manual Calculations (Question 1)
`Classification_Performance_Experiment.pdf` contains:
- Confusion matrix construction from given predictions
- Per-class precision/recall/F1 calculations (with formulas)
- Macro-averaged metrics computation
- Accuracy calculation
- **Final Results**: Accuracy=0.550, Macro-Precision=0.542, Macro-Recall=0.532, Macro-F1=0.534

### Part 2: Technical Report
`Part2_Report.pdf` (1200 words) includes:
1. **Methodology**: Complete pipeline description
2. **Justification**: Development set experiments with tables
3. **Results**: Comprehensive metrics, confusion matrix
4. **Critical Reflection**: Improvements, biases, ethics

---

##  Learning Outcomes Demonstrated

### Technical Skills
-  Text preprocessing (regex, normalization)
-  Feature engineering (TF-IDF, statistical features)
-  Feature selection (chi-squared)
-  Model selection (development set validation)
-  Evaluation (accuracy, precision, recall, F1, RMSE)
-  Proper ML methodology (train/dev/test splits, no data leakage)

### Documentation
-  Comprehensive Jupyter notebooks with markdown
-  Academic report writing (LaTeX)
-  Code documentation and comments
-  Reproducible research practices

---

##  Key Design Decisions

### Part 1
1. **Why Linear SVC?** High-dimensional sparse text → linear separability
2. **Why Ridge?** Closed-form solution, L2 regularization prevents overfitting
3. **Why Trigrams?** Captures negation ("not good") and longer phrases

### Part 2
1. **Why 60/20/20 split?** Balances training data with hyperparameter tuning
2. **Why chi-squared?** Fast, interpretable, works with sparse data
3. **Why k=15,000?** Development experiments showed peak at 90.34%
4. **Why statistical features?** Complement TF-IDF with document-level properties

---

##  Important Notes

### Data Files
Data files are **NOT included** in this repository due to size/licensing. Download from:
- **Part 1**: SemEval2017 Task 4 Dataset (provided by instructor)
- **Part 2**: Modified 20_Newsgroups (provided by instructor)

Update paths in notebooks:
```python
# Part 1
data_path = '/path/to/SemEval2017_Task4_Sentiment_Analysis.csv'

# Part 2
data_path = '/path/to/20_Newsgroups.csv'
```

### Reproducibility
All models use `random_state=42` for reproducibility. Results should be identical given same:
- Python version (3.8+)
- Package versions (see requirements.txt)
- Data splits (stratified with random_state=42)

---

## Author

**Student**: Samidur Rahman  
**Student ID**: 21053329  
**Module**: CMT122 - Machine Learning for NLP  
**Institution**: Cardiff University  

---

## License

This coursework is submitted for academic evaluation at Cardiff University. All rights reserved.

---

## Acknowledgments

- Course instructors: Nedjma Ousidhoum & Jose Camacho-Collados
- Datasets: SemEval2017 Task 4, 20_Newsgroups (modified)
- Libraries: scikit-learn, NLTK, pandas, numpy

---

## References

**Part 1:**
- SemEval-2017 Task 4: Sentiment Analysis in Twitter  
  Rosenthal et al., 2017

**Part 2:**
- The 20 Newsgroups Dataset  
  Ken Lang, 1995 (modified version)

**Methodology:**
- Scikit-learn Documentation: https://scikit-learn.org/
- NLTK Documentation: https://www.nltk.org/

---

**Last Updated**: November 2025  
**Version**: 1.0 (Final Submission)
