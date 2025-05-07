import numpy as np
import pandas as pd
import warnings
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

def feature_engineer():

    # Load the dataset
    data = pd.read_csv("adult.csv")

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
    data["workclass"][data["workclass"] == "?"] = data["workclass"].mode()[0]
    data["occupation"][data["occupation"] == "?"] = data["occupation"].mode()[0]
    data["native.country"][data["native.country"] == "?"] = data["native.country"].mode()[0]

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

    # replace elements in list.
    data['education'].replace(to_replace = hs_grad,value = 'HS-grad',inplace = True)
    data['education'].replace(to_replace = elementary,value = 'elementary_school',inplace = True)
    print("\n--------------------------------------------")
    print(data['education'].value_counts())

    # Marital Status
    
    #     Married-civ-spouse,Married-spouse-absent,Married-AF-spouse comes under category Married
    #     Divorced, separated again comes under category separated.

    married= ['Married-spouse-absent','Married-civ-spouse','Married-AF-spouse']
    separated = ['Separated','Divorced']

    #replace elements in list.
    data['marital.status'].replace(to_replace = married ,value = 'Married',inplace = True)
    data['marital.status'].replace(to_replace = separated,value = 'Separated',inplace = True)
    print("\n--------------------------------------------")
    print(data['marital.status'].value_counts())

    # Workclass

    #     Self-emp-not-inc, Self-emp-inc comes under category self employed
    #     Local-gov,State-gov,Federal-gov comes under category goverment emloyees

    self_employed = ['Self-emp-not-inc','Self-emp-inc']
    govt_employees = ['Local-gov','State-gov','Federal-gov']

    #replace elements in list.
    data['workclass'].replace(to_replace = self_employed ,value = 'Self_employed',inplace = True)
    data['workclass'].replace(to_replace = govt_employees,value = 'Govt_employees',inplace = True)
    print("\n--------------------------------------------")
    print(data['workclass'].value_counts())

    # Delete the unuseful column
    del_cols = ['education.num']
    data.drop(labels = del_cols,axis = 1,inplace = True)

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
    final_df.drop(labels = ['id','income_<=50K','fnlwgt'],axis = 1,inplace = True)
    X = final_df

    return X, y
    
feature_engineer()   

def Model_Training():
    X, y = feature_engineer()
    # Split the dataset into training and testing sets
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.15, random_state=42)
    
    # Fitting the model
    lr = LogisticRegression()
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
    basedModels.append(('LR', LogisticRegression()))
    basedModels.append(('GBM', GradientBoostingClassifier()))
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
    print(basedLineScore.sort_values(by='Score', ascending=False))