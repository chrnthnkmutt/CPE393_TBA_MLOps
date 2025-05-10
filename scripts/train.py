import numpy as np
import pandas as pd
import warnings
import pickle
import os
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,BaggingClassifier,ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score
from sklearn import metrics
from datetime import datetime
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from imblearn.under_sampling import RandomUnderSampler

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in matmul")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in matmul")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in matmul")

def feature_engineer():

    # Load the dataset
    data = pd.read_csv("data/adult.csv")

    # Display the first few rows of the dataset
    print("First few rows of the dataset:")
    print(data.head())

    # Display the info of the dataset
    print("\n--------------------------------------------")
    print("\nInfo of the dataset:")
    print(data.info())

    # Display column names in the dataset
    print("\n--------------------------------------------")
    print("\nColumn names in the dataset:")
    print(data.columns)

    # Display the shape of the dataset
    print("\n--------------------------------------------")
    print("\nShape of the dataset:")
    print(data.shape)

    # Display the number of missing values in each column
    print("\n--------------------------------------------")
    print("\nThe number of missing values in each column:")
    print(data.isnull().sum())

    # We found that there are no missing values in the dataset, but it has some question marks (?) in our dataset.
    print("\n--------------------------------------------")
    for column in data.columns:
        print(f"{column} = {data[data[column] == '?'].shape[0]}")

    # We replace the question marks (?) with mode value of each column that has question marks (?) values.
    # Fix: Use loc instead of chained assignment
    data.loc[data["workclass"] == "?", "workclass"] = data["workclass"].mode()[0]
    data.loc[data["occupation"] == "?", "occupation"] = data["occupation"].mode()[0]
    data.loc[data["native.country"] == "?", "native.country"] = data["native.country"].mode()[0]

    # Check for any remaining question marks (?) data
    print("\n--------------------------------------------")
    print("\nThe number of question marks (?) in each column after replacing:")
    for column in data.columns:
        print(f"{column} = {data[data[column] == '?'].shape[0]}")

    # Check for outliers in the dataset
    print("\n--------------------------------------------")
    # Display numeric columns in the dataset
    print("\nNumeric columns in the dataset:")
    numeric_columns = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
    
    # Identify outliers and remove them
    for column in numeric_columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        Lower = Q1 - 1.5 * IQR
        Upper = Q3 + 1.5 * IQR
        data = data[(data[column] >= Lower) & (data[column] <= Upper)]
    
     # Check for any remaining question marks (?) data
    print("\n--------------------------------------------")
    print("\nThe number of question marks (?) in each column after replacing:")
    for column in data.columns:
        print(f"{column} = {data[data[column] == '?'].shape[0]}")
    
    # Education
    #     9th, 10th, 11th, 12th comes under HighSchool Grad but it has mentioned separately
    #     Create Elementary object for 1st-4th, 5th-6th, 7th-8th
    hs_grad = ['HS-grad','11th','10th','9th','12th']
    elementary = ['1st-4th','5th-6th','7th-8th']

    # Fix: Copy data to avoid warning and use more modern replacement approach
    data = data.copy()
    data['education'] = data['education'].replace(hs_grad, 'HS-grad')
    data['education'] = data['education'].replace(elementary, 'elementary_school')
    print("\n--------------------------------------------")
    print(data['education'].value_counts())

    # Marital Status
    #     Married-civ-spouse,Married-spouse-absent,Married-AF-spouse comes under category Married
    #     Divorced, separated again comes under category separated.
    married= ['Married-spouse-absent','Married-civ-spouse','Married-AF-spouse']
    separated = ['Separated','Divorced']

    # Fix: Use direct replacement without inplace
    data['marital.status'] = data['marital.status'].replace(married, 'Married')
    data['marital.status'] = data['marital.status'].replace(separated, 'Separated')
    print("\n--------------------------------------------")
    print(data['marital.status'].value_counts())

    # Workclass
    #     Self-emp-not-inc, Self-emp-inc comes under category self employed
    #     Local-gov,State-gov,Federal-gov comes under category goverment emloyees
    self_employed = ['Self-emp-not-inc','Self-emp-inc']
    govt_employees = ['Local-gov','State-gov','Federal-gov']

    # Fix: Use direct replacement without inplace
    data['workclass'] = data['workclass'].replace(self_employed, 'Self_employed')
    data['workclass'] = data['workclass'].replace(govt_employees, 'Govt_employees')
    print("\n--------------------------------------------")
    print(data['workclass'].value_counts())

    # Delete the unuseful column
    del_cols = ['education.num']
    data = data.drop(columns=del_cols)  # Fix: Use drop with columns parameter

    # Separate data types
    # Numeric columns
    num_col_new = ['age','capital.gain', 'capital.loss', 'hours.per.week','fnlwgt']
    # Categorical columns
    cat_col_new = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'income']

    scaler = MinMaxScaler()
    print("\n--------------------------------------------")
    print(pd.DataFrame(scaler.fit_transform(data[num_col_new]),columns = num_col_new).head(5))

    class DataFrameSelector(TransformerMixin):
        def __init__(self,attribute_names):
            self.attribute_names = attribute_names
                
        def fit(self,X,y = None):
            return self
    
        def transform(self,X):
            return X[self.attribute_names]
    
    class num_trans(TransformerMixin):
        def __init__(self):
            pass
        
        def fit(self,X,y=None):
            return self
        
        def transform(self,X):
            df = pd.DataFrame(X)
            df.columns = num_col_new 
            return df
            
    pipeline = Pipeline([('selector',DataFrameSelector(num_col_new)),  
                        ('scaler',MinMaxScaler()),
                        ('transform',num_trans())])
    
    num_df = pipeline.fit_transform(data)
    num_df.shape

    # columns which we don't need after creating dummy variables dataframe
    cols = ['workclass_Govt_employess','education_Some-college',
        'marital-status_Never-married','occupation_Other-service',
        'race_Black','sex_Male','income_>50K']
    
    class dummies(TransformerMixin):
        def __init__(self,cols):
            self.cols = cols
        
        def fit(self,X,y = None):
            return self
        
        def transform(self,X):
            df = pd.get_dummies(X)
            df_new = df[df.columns.difference(cols)] 
    #difference returns the original columns, with the columns passed as argument removed.
            return df_new

    pipeline_cat=Pipeline([('selector',DataFrameSelector(cat_col_new)),
                        ('dummies',dummies(cols))])
    cat_df = pipeline_cat.fit_transform(data)
    print("\n--------------------------------------------")
    print(cat_df.shape)

    cat_df['id'] = pd.Series(range(cat_df.shape[0]))
    num_df['id'] = pd.Series(range(num_df.shape[0]))

    final_df = pd.merge(cat_df,num_df,how = 'inner', on = 'id')
    print("\n--------------------------------------------")
    print(f"Number of observations in final dataset: {final_df.shape}")

    # split the dataset
    y = final_df['income_<=50K']
    X = final_df.drop(columns=['id','income_<=50K','fnlwgt'])  # Fix: Use drop with columns parameter

    return X, y
    
def Model_Training():
    X, y = feature_engineer()
    # Split the dataset into training and testing sets
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.15, random_state=42)
    
    # Fitting the model
    # Fix: Use more stable solver and add regularization to prevent numerical issues
    lr = LogisticRegression(solver='liblinear', C=1.0)
    lr.fit(X_train1, y_train1)
    
    # Predict 
    y_pred4 = lr.predict(X_test1)
    print("Accuracy:", metrics.accuracy_score(y_test1, y_pred4))
    print("Precision:", metrics.precision_score(y_test1, y_pred4))
    print("Recall:", metrics.recall_score(y_test1, y_pred4))
    print("F1 score:", metrics.f1_score(y_test1, y_pred4))
    print("AUC :", metrics.roc_auc_score(y_test1, y_pred4))

    # Apply Random Under Sampling
    rus = RandomUnderSampler(random_state=42)
    X_rus, y_rus = rus.fit_resample(X, y)

    # Convert to DataFrame and use the original column names
    X_rus = pd.DataFrame(X_rus, columns=X.columns)
    y_rus = pd.Series(y_rus, name="income")  # Changed to Series instead of DataFrame

    # Split the undersampled data
    X_train, X_test, y_train, y_test = train_test_split(X_rus, y_rus, test_size=0.15, random_state=42)
    
    return X_train, y_train, X_test, y_test

def GetBasedModel():
    basedModels = []
    # Fix: Use more stable solver and add regularization to prevent numerical issues
    basedModels.append(('LR', LogisticRegression(solver='liblinear', C=1.0)))
    basedModels.append(('GBM', GradientBoostingClassifier()))
    basedModels.append(('RF', RandomForestClassifier(n_estimators=100, random_state=42))) # Fix: Add random_forest 
    return basedModels

def BasedLine2(X_train, y_train, models):
    # Test options and evaluation metric
    num_folds = 3
    scoring = 'accuracy'

    results = []
    names = []
    for name, model in models:
        kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=10)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
    return names, results

def ScoreDataFrame(names,results):
    def floatingDecimals(f_val, dec=3):
        prc = "{:."+str(dec)+"f}" 
    
        return float(prc.format(f_val))

    scores = []
    for r in results:
        scores.append(floatingDecimals(r.mean(),4))

    scoreDataFrame = pd.DataFrame({'Model':names, 'Score': scores})
    return scoreDataFrame

def save_model(model, model_type=None, filename=None):
    """
    Save the trained model to a pickle file

    Args:
        model: The trained model to save
        model_type: Type of model (e.g., 'LR', 'GBM', 'RF')
        filename: Custom filename for the saved model (optional)

    Returns:
        str: Path to the saved model file
    """
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')

    # Generate filename with timestamp if not provided
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_type = model_type if model_type else "MODEL"
        filename = f"{model_type}_model_{timestamp}.pkl"

    # Full path to save the model
    model_path = os.path.join('models', filename)

    # Save the model to a pickle file
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")

    return model_path

def load_model(filename, custom_path=None):
    """
    Load a model from a pickle file

    Args:
        filename: The filename of the saved model
        custom_path: Optional full path to the model file

    Returns:
        The loaded model
    """
    if custom_path:
        file_path = custom_path
    else:
        file_path = os.path.join('models', filename)

    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    return model

if __name__ == "__main__":
    # Get the training data
    X_train, y_train, X_test, y_test = Model_Training()

    # Get the models
    models = GetBasedModel()

    # Run the baseline evaluation
    names, results = BasedLine2(X_train, y_train, models)
    print("\n--------------------------------------------")
    print("\nModel Names:", names)
    print("Results:", results)

    basedLineScore = ScoreDataFrame(names, results)
    scores_df = basedLineScore.sort_values(by='Score', ascending=False)
    print(scores_df)

    # Train only GBM and RF models
    model_dict = {}
    for name, model in models:
        # Initialize and train models - only GBM and RF
        if name in ['GBM', 'RF']:
            print(f"\nTraining {name} model...")
            model.fit(X_train, y_train)
            model_dict[name] = model

            # Evaluate on test set
            y_pred = model.predict(X_test)
            print(f"\n{name} Model Test Set Evaluation:")
            print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred):.4f}")
            print(f"Precision: {metrics.precision_score(y_test, y_pred):.4f}")
            print(f"Recall: {metrics.recall_score(y_test, y_pred):.4f}")
            print(f"F1 score: {metrics.f1_score(y_test, y_pred):.4f}")
            print(f"AUC: {metrics.roc_auc_score(y_test, y_pred):.4f}")

            # Save the trained model
            save_model(model, name)

    print("\nAll models have been trained and saved successfully.")