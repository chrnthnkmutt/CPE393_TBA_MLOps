# US Adult Census Income Prediction with Three-Models Classification Within Machine Learning Operation Process Repository

Created By TBA_MLOps Group for CPE393 Special Topics III: Machine Learning Operation

## About

 This repository is aimed at developing and deploying a robust MLOps pipeline for predicting the income level of adults in the United States. The core problem addresses the need to classify whether an individual's income exceeds $50K per year based on various demographic factors.

## Prerequisites

Before setting up the project, ensure you have the following software installed:

- **Python 3.10+**: The core language for backend, scripts, and ML workflows.
- **Conda (Anaconda/Miniconda)**: For managing environments and dependencies.
- **Docker**: For containerizing and running the application in isolated environments.
- **Git**: For cloning and managing the repository.
- **Jupyter Notebook/Lab**: For running and editing notebooks (optional, but recommended).

> Ensure your system PATH includes Python, Conda, Docker, and Git executables.

## Setting Up the Environment, Installing, and Running

1. Clone the repository:

```bash
git clone https://github.com/chrnthnkmutt/CPE393_TBA_MLOps.git
cd CPE393_TBA_MLOps
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. use conda to create the environment and activate it.

```
conda env create -f environment.yml
conda activate mlops-project
```
4. To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

5. To run the training script, use the following command:

```bash
python scripts/train.py
```

6. To run the web deployment, run the following command:

```
docker compose up --build --remove-orphans
```
and click at localhost:3000.

## Model Card

### 🔖 **1. Model Details**

- **Model Name**: XGBoost
- **Version**: Version 1
- **Developers**: TBA_MLOPs Group
- **Framework**: Scikit-learn
- **License**: Open-source

---

### 🧠 **2. Intended Use**

- **Primary use cases**: To predict whether an individual's income exceeds $50,000 per year based on demographic and employment-related attributes and to provide insights into the factors influencing income levels.
- **Users**: Socioeconomic analysts, Financial planners, Policymakers, Researchers 
- **Out-of-scope use**: The model should not be used to make discriminatory decisions based on protected attributes (e.g., race, sex). Also, the model should not be used for high-stakes decisions without careful consideration of other factors and potential biases and the model's predictions should not be considered definitive, as they are based on statistical probabilities and may not be accurate for every individual.

---

### 📊 **3. Model Architecture**

- **Type**: XGBoost
- **Description**: 
	- An ensemble learning method based on gradient boosting of decision trees.
	- Utilizes multiple weak learners (decision trees) combined sequentially, where each new tree corrects errors made by previous ones.
	- Known for its high performance, scalability, and ability to handle both numerical and categorical features efficiently.
	- In this case, the classes are income levels: ">50K" or "<=50K".
- **Input/Output**:
	- Input format: Numerical and categorical features (after preprocessing, including one-hot encoding for categorical variables). These features represent demographic and employment-related attributes like age, education, occupation, etc.
	- Output format: Probabilities representing the likelihood of an individual's income being ">50K" or "<=50K", and the predicted class label (">50K" or "<=50K") based on a probability threshold (typically 0.5).

---

### 🧪 **4. Training Data**

- **Dataset(s) used**:
	- Name: Adult Dataset
	- Source: UCI Machine Learning Repository
	- Size: 32,561 entries
	- License: This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license. This allows for the sharing and adaptation of the datasets for any purpose, provided that the appropriate credit is given.
- **Preprocessing**: 
	- Missing values in workclass, occupation, and native.country were imputed with the most frequent value.
	- Outliers in numerical features (age, fnlwgt, education.num, capital.gain, capital.loss, hours.per.week) were removed using the IQR method.
	- Categorical features were consolidated to reduce complexity (e.g., grouping education levels).
	- Numerical features were normalized using MinMaxScaler.
	- Categorical features were one-hot encoded.
- **Data balance**: Class distribution, potential biases.
	- The document includes some visualizations (pie charts) showing the distribution of gender within income groups (>$50K and <=$50K). However, it doesn't provide a detailed class distribution of the target variable (`income`). A full data balance assessment would typically involve showing the proportion of individuals with income > $50K versus <= $50K.
	- Potential biases are discussed in the document, noting that the dataset is based on 1994 census data, which may limit its representativeness of current income trends.

---

### 📈 **5. Evaluation Metrics**

- **Metrics**: 

```
	XGBoost:
--------------------------------------------------
  Train Accuracy:     0.8955
  Test Accuracy:      0.8690
  Overfitting:        0.0265
  Precision:          0.8640
  Recall:             0.8690
  F1-Score:           0.8639
  Confusion Matrix:
    True Neg (<=50K):  989
    False Pos:         579
    False Neg:         274
    True Pos (>50K):   4671
  Classification Report:
              precision    recall  f1-score   support

       <=50K     0.7831    0.6307    0.6987      1568
        >50K     0.8897    0.9446    0.9163      4945

    accuracy                         0.8690      6513
   macro avg     0.8364    0.7877    0.8075      6513
weighted avg     0.8640    0.8690    0.8639      6513
```

- **Test Dataset**: 
	- Splitting the dataset into training, validation, and testing sets to ensure robust evaluation.
	- The evaluation process involves using Stratified K-Fold cross-validation to maintain class distribution across the folds.
- **Benchmark results**: Performance compared to other models or baselines.
	- Gradient Boosting (GBM)
		-  Accuracy: 0.8390
		- Precision: 0.8677
		- Recall: 0.9387
		- F1 score: 0.9018
		- AUC: 0.7044
	- Random Forest (For Trial Deployment - as its data robustness - bias)
		- Accuracy: 0.7961
		- Precision: 0.8577
		- Recall: 0.8884
		- F1 score: 0.8728
		- AUC: 0.6713

---

### ⚖️ **6. Ethical Considerations**

- **Bias and fairness**: the dataset is based on 1994 census data, which may introduce bias and limit the model's ability to generalize to current income trends. Also, the potential for bias exists due to the inherent biases in the training data.
- **Risk factors**:
	- Potential misuses include using the model to make discriminatory decisions related to employment or lending, which can perpetuate societal inequalities.
	- Over-reliance on the model's predictions without considering other factors could lead to inaccurate or unfair assessments of individuals' financial situations.
- **Mitigations**: What was done to reduce risks or bias.
    - Emphasizes the importance of monitoring dataset drift to detect changes in the data distribution over time, which can help in identifying and mitigating potential biases.
    - The need for continuous improvement of the workflow to adapt to changes in dataset characteristics. 

---

### 🧪 **7. Limitations**

- **Performance caveats**:
    - While Logistic Regression performed well in this project, it is fundamentally a linear model. Therefore, it might not capture complex non-linear relationships present in the data as effectively as ensemble methods like Random Forest or Gradient Boosting.
    - The model's performance is also tied to the quality and representativeness of the training data. If the training data doesn't fully reflect the real-world population or contains biases, the model's predictions may be unreliable.
- **Generalization**:
    - The model's ability to generalize to other time periods or populations may be limited due to the use of 1994 census data. Income distributions and influencing factors can change significantly over time.
    - The model might not perform well in different geographical regions or countries with distinct socioeconomic characteristics.
- **Assumptions**:
    - Logistic Regression assumes a linear relationship between the features and the log-odds of the outcome. This assumption may not hold true for all features in the dataset.
    - The model assumes that the features are independent of each other, which might not be the case in reality. Multicollinearity among features can affect the model's coefficients and interpretability.
    - It's assumed that the preprocessing steps, including handling missing values and outlier removal, have adequately addressed data quality issues. However, if these steps are imperfect, they could impact the model's performance.

---

### 🔧 **8. Maintenance**

- **Contact info**:
    - Charunthon Limseelo: boat.charunthon@gmail.com
- **Planned updates**:
	- Automated model retraining based on detected dataset drift. This suggests a plan for ongoing model updates to maintain accuracy.
	- Continuous monitoring of model performance and data quality. 

## Directory Structure

The repository is structured as follows:

```
CPE393_TBA_MLOps/
├── back-app/                     # Backend application
│   ├── app.py                    # Main backend application file
│   ├── Dockerfile                # Dockerfile for backend
│   ├── environment.yml           # Conda environment file
│   ├── features.json             # Feature configuration
│   ├── REAME.md                  # Backend-specific README
│   └── scripts/                  # Backend scripts
│       └── inspect_model.py      # Script to inspect the model
├── data/                         # Dataset directory
│   └── adult.csv                 # Dataset file
├── front-app/                    # Frontend application
│   ├── .gitignore                # Git ignore file for frontend
│   ├── components/               # Frontend components
│   │   ├── home/                 # Home page components
│   │   │   ├── complete-analysis.tsx
│   │   │   └── main-content.tsx
│   │   └── ui/                   # UI components
│   │       ├── sidebar.tsx
│   │       └── tabs.tsx
│   ├── lib/                      # Frontend libraries
│   │   └── features.ts           # Feature definitions
│   ├── app/                      # Frontend application files
│   │   ├── layout.tsx
│   │   └── globals.css
│   ├── Dockerfile.dev            # Development Dockerfile
│   ├── Dockerfile.prod           # Production Dockerfile
│   ├── eslint.config.mjs         # ESLint configuration
│   └── tailwind.config.ts        # Tailwind CSS configuration
├── mlruns/                       # MLFlow experiment tracking
│   ├── 847288952177037969/       # Experiment ID
│   │   ├── meta.yaml             # Metadata for the experiment
│   │   ├── 376b4600a2d04d5f93abce79119105a1/
│   │   │   └── artifacts/        # Model artifacts
│   │   │       ├── gbm_model/
│   │   │       │   └── requirements.txt
│   │   │       └── rf_model/
│   │   │           └── requirements.txt
├── models/                       # Trained models
│   ├── GBM_metrics.json          # Metrics for GBM model
│   ├── GBM_model_20250514_131412.pkl
│   └── RF_model_20250515_112838.pkl
├── notebooks/                    # Jupyter notebooks
│   ├── 01_Analyze_and_Visualization.ipynb
│   ├── 02_compare-all-the-classification-models.ipynb
│   └── 03_ModelMornitoring.ipynb
├── scripts/                      # Scripts for training and pipelines
│   ├── DAG_TBA.py                # Airflow DAG script
│   ├── README.md                 # Scripts documentation
│   ├── tempCodeRunnerFile.py     # Temporary script
│   ├── train.py                  # Training script
│   ├── train_dag.py              # DAG training script
│   ├── train_updated.py          # Updated training script
│   └── trainforMLFlow.py         # MLFlow training script
├── requirements.txt              # Python dependencies
├── environment.yml               # Conda environment file
└── README.md                     # Main project documentation
```

### Using the Notebooks

The Jupyter notebooks are located in the `notebooks/` directory. You can open and run them using Jupyter Notebook or JupyterLab.

```bash
jupyter notebook notebooks/01_Analyze_and_Visualization.ipynb
jupyter notebook notebooks/02_compare-all-the-classification-models.ipynb
jupyter notebook notebooks/03_ModelMornitoring.ipynb
```

## Additional Information

This repository is structured to follow best practices for Machine Learning Operations (MLOps). The directory structure separates different functionalities such as data processing, model training, and evaluation. This helps in maintaining a clean and organized codebase, making it easier to manage and scale machine learning projects. Along with, each folder will provide you the information for running and installing inside within the folder. 