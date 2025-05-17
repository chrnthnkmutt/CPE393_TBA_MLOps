import numpy as np
import pandas as pd
import warnings
import pickle
import os
import json
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

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in matmul")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in matmul")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in matmul")

def feature_engineer():

    # Load the dataset
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    csv_path = os.path.join(BASE_DIR, "adult.csv")

    data = pd.read_csv(csv_path)

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
    X = final_df.drop(columns=['id','income_<=50K','fnlwgt'])
    X_path = os.path.join(BASE_DIR, "X.csv")
    y_path = os.path.join(BASE_DIR, "y.csv")
    X.to_csv(X_path, index=False)
    y.to_csv(y_path, index=False)
    return {"X_path": X_path, "y_path": y_path}

def Model_Training():
    paths = feature_engineer()
    X = pd.read_csv(paths["X_path"])
    y = pd.read_csv(paths["y_path"])

    if len(y.columns) == 1:
        y = y.iloc[:, 0]
    # Split the dataset into training and testing sets
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.15, random_state=42)

    # Fitting the model
    lr = LogisticRegression(solver='liblinear', C=1.0)
    lr.fit(X_train1, y_train1)

    # Predict
    y_pred4 = lr.predict(X_test1)
    print("Accuracy:", metrics.accuracy_score(y_test1, y_pred4))
    print("Precision:", metrics.precision_score(y_test1, y_pred4))
    print("Recall:", metrics.recall_score(y_test1, y_pred4))
    print("F1 score:", metrics.f1_score(y_test1, y_pred4))
    print("AUC :", metrics.roc_auc_score(y_test1, y_pred4))

    # Split the data directly without undersampling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    X_train_path = os.path.join(BASE_DIR, "X_train.csv")
    y_train_path = os.path.join(BASE_DIR, "y_train.csv")
    X_test_path = os.path.join(BASE_DIR, "X_test.csv")
    y_test_path = os.path.join(BASE_DIR, "y_test.csv")
    X_train.to_csv(X_train_path, index=False)
    y_train.to_csv(y_train_path, index=False)
    X_test.to_csv(X_test_path, index=False)
    y_test.to_csv(y_test_path, index=False)

    # For direct Python usage
    if '__name__' in globals() and __name__ == "__main__":
        return X_train, y_train, X_test, y_test
    # For Airflow
    else:
        return {
            "X_train_path": X_train_path,
            "y_train_path": y_train_path,
            "X_test_path": X_test_path,
            "y_test_path": y_test_path
        }

def GetBasedModel():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    basedModels = [
        ('LR', LogisticRegression(solver='liblinear', C=1.0)),
        ('GBM', GradientBoostingClassifier()),
        ('RF', RandomForestClassifier(n_estimators=100, random_state=42))
    ]
    model_paths = []
    for name, model in basedModels:
        path = os.path.join(BASE_DIR, f"{name}_model.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)
        model_paths.append({'name': name, 'path': path})
    return model_paths

def BasedLine2(X_train, y_train, models):
    num_folds = 3
    scoring = 'accuracy'

    results = []
    names = []
    
    print("Initial X_train:", repr(X_train))
    print("Initial y_train:", repr(y_train))
    print("Initial models input:", repr(models))
    
    # Validate inputs
    if X_train is None or y_train is None:
        raise ValueError("X_train and y_train cannot be None")
    
    # Handle X_train and y_train inputs
    if isinstance(X_train, str):
        print(f"Loading X_train from {X_train}")
        if not os.path.exists(X_train):
            raise FileNotFoundError(f"X_train file not found: {X_train}")
        X_train = pd.read_csv(X_train)
    
    if isinstance(y_train, str):
        print(f"Loading y_train from {y_train}")
        if not os.path.exists(y_train):
            raise FileNotFoundError(f"y_train file not found: {y_train}")
        y_train = pd.read_csv(y_train)
        if len(y_train.columns) == 1:
            y_train = y_train.iloc[:, 0]
    
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError(f"X_train must be a DataFrame or path to CSV, got {type(X_train)}")
    if not isinstance(y_train, (pd.Series, pd.DataFrame)):
        raise TypeError(f"y_train must be a Series/DataFrame or path to CSV, got {type(y_train)}")
    
    # Add type checking and conversion if needed
    if isinstance(models, str):
        # Clean up the string
        models_str = models.strip()
        # Replace single quotes with double quotes
        models_str = models_str.replace("'", '"')
        
        try:
            # Try to parse the entire string as JSON
            models = json.loads(models_str)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON string: {e}")
            print(f"Attempting alternative parsing method...")
            
            # If that fails, try to extract the list content
            if models_str.startswith("[") and models_str.endswith("]"):
                # Remove the outer brackets and split by "}, {"
                items = models_str[1:-1].split("}, {")
                models_list = []
                
                for i, item in enumerate(items):
                    # Add back the curly braces except for first and last items
                    if i > 0:
                        item = "{" + item
                    if i < len(items) - 1:
                        item = item + "}"
                    
                    try:
                        # Parse each item individually
                        model_dict = json.loads(item)
                        models_list.append(model_dict)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing item {i}: {item}")
                        print(f"JSON error: {e}")
                        continue
                
                if not models_list:
                    raise ValueError("Failed to parse any valid model configurations")
                
                models = models_list
            else:
                raise ValueError("Models string must be a list enclosed in square brackets")
    
    if not isinstance(models, list):
        raise ValueError("models parameter must be a list")
    
    print("Processed models:", models)
    
    model_results = []
    
    for model_info in models:
        try:
            if isinstance(model_info, dict):
                name = model_info.get('name')
                path = model_info.get('path')
                
                if not name or not path:
                    print(f"Skipping invalid model info: {model_info}")
                    continue
                    
                print(f"Loading model {name} from {path}")
                if not os.path.exists(path):
                    print(f"Model file not found: {path}")
                    continue
                    
                try:
                    with open(path, "rb") as f:
                        model = pickle.load(f)
                except Exception as e:
                    print(f"Error loading model from {path}: {str(e)}")
                    continue
                    
                try:
                    kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=10)
                    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
                    results.append(cv_results.tolist())  # Convert numpy array to list
                    names.append(name)
                    model_results.append({
                        'name': name,
                        'mean_score': float(cv_results.mean()),  # Convert numpy float to Python float
                        'std_score': float(cv_results.std()),    # Convert numpy float to Python float
                        'scores': cv_results.tolist()            # Convert numpy array to list
                    })
                    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
                    print(msg)
                except Exception as e:
                    print(f"Error during cross validation for {name}: {str(e)}")
                    continue
            else:
                print(f"Skipping invalid model info type: {type(model_info)}")
                continue
            
        except Exception as e:
            print(f"Error processing model {model_info}: {str(e)}")
            continue
    
    if not results:
        raise ValueError("No valid models were processed. Please check the model paths and ensure they exist.")
    
    # Find the best model
    best_model_idx = max(range(len(model_results)), key=lambda i: model_results[i]['mean_score'])
    best_model = model_results[best_model_idx]
    
    return {
        'names': names,
        'results': results,
        'model_results': model_results,
        'best_model': {
            'name': best_model['name'],
            'mean_score': best_model['mean_score'],
            'std_score': best_model['std_score']
        }
    }

def ScoreDataFrame(names, results):
    def floatingDecimals(f_val, dec=3):
        prc = "{:."+str(dec)+"f}" 
        return float(prc.format(f_val))

    # Convert string inputs to proper types if needed
    if isinstance(names, str):
        try:
            names = eval(names)  # Convert string representation of list to actual list
        except:
            names = [names]  # If it's a single string, make it a list
            
    if isinstance(results, str):
        try:
            results = eval(results)  # Convert string representation of list to actual list
        except:
            print(f"Error parsing results string: {results}")
            results = []

    scores = []
    for result in results:
        # Handle string representation of list
        if isinstance(result, str):
            try:
                result = eval(result)
            except:
                print(f"Error parsing result string: {result}")
                continue
                
        # Convert result to numpy array if it's a list
        if isinstance(result, list):
            result = np.array(result)
            
        try:
            mean_val = result.mean()
            scores.append(floatingDecimals(mean_val, 4))
        except Exception as e:
            print(f"Error calculating mean: {str(e)}")
            print(f"Result type: {type(result)}")
            print(f"Result value: {result}")
            continue

    scoreDataFrame = pd.DataFrame({'Model': names, 'Score': scores})
    return scoreDataFrame

def save_model(model_type=None, filename=None):
    """
    Save the trained model to a pickle file

    Args:
        model_type: Type of model (e.g., 'LR', 'GBM', 'RF')
        filename: Custom filename for the saved model (optional)

    Returns:
        dict: Dictionary containing the saved model information
    """
    # Create models directory if it doesn't exist
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(BASE_DIR, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # Generate filename with timestamp if not provided
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_type = model_type if model_type else "MODEL"
        filename = f"{model_type}_model_{timestamp}.pkl"

    # Full path to save the model
    model_path = os.path.join(models_dir, filename)

    # Train a new model based on model_type
    if model_type == 'LR':
        model = LogisticRegression(solver='liblinear', C=1.0)
    elif model_type == 'GBM':
        model = GradientBoostingClassifier()
    elif model_type == 'RF':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load training data
    X_train = pd.read_csv(os.path.join(BASE_DIR, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(BASE_DIR, "y_train.csv"))
    if len(y_train.columns) == 1:
        y_train = y_train.iloc[:, 0]

    # Train the model
    model.fit(X_train, y_train)

    # Save the model to a pickle file
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")

    return {
        'filename': filename,
        'model_path': model_path,
        'model_type': model_type
    }

def load_model(filename):
    """
    Load a model from a pickle file and evaluate it

    Args:
        filename: The filename of the saved model

    Returns:
        dict: Dictionary containing the model's evaluation metrics
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(BASE_DIR, 'models')
    file_path = os.path.join(models_dir, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found at {file_path}")

    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load test data and evaluate model
    X_test = pd.read_csv(os.path.join(BASE_DIR, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(BASE_DIR, "y_test.csv"))
    if len(y_test.columns) == 1:
        y_test = y_test.iloc[:, 0]

    # Test the loaded model
    y_pred = model.predict(X_test)
    print("\nLoaded Model Test Set Evaluation:")
    print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {metrics.precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {metrics.recall_score(y_test, y_pred):.4f}")
    print(f"F1 score: {metrics.f1_score(y_test, y_pred):.4f}")
    print(f"AUC: {metrics.roc_auc_score(y_test, y_pred):.4f}")
    
    return {
        'metrics': {
            'accuracy': float(metrics.accuracy_score(y_test, y_pred)),
            'precision': float(metrics.precision_score(y_test, y_pred)),
            'recall': float(metrics.recall_score(y_test, y_pred)),
            'f1': float(metrics.f1_score(y_test, y_pred)),
            'auc': float(metrics.roc_auc_score(y_test, y_pred))
        },
        'model_file': file_path
    }

if __name__ == "__main__":
    # Get the training data
    X_train, y_train, X_test, y_test = Model_Training()

    # Get the models
    models = GetBasedModel()

    # Run the baseline evaluation
    baseline_results = BasedLine2(X_train, y_train, models)
    print("\n--------------------------------------------")
    print("\nModel Names:", baseline_results['names'])
    print("Results:", baseline_results['results'])

    # Create and display the scores DataFrame
    basedLineScore = ScoreDataFrame(baseline_results['names'], baseline_results['results'])
    scores_df = basedLineScore.sort_values(by='Score', ascending=False)
    print("\nModel Scores:")
    print(scores_df)

    # Get the best model name and score
    best_model_name = scores_df.iloc[0]['Model']
    best_model_score = scores_df.iloc[0]['Score']
    print(f"\nBest Model: {best_model_name} with score: {best_model_score}")

    # Train and save the best model
    if best_model_name == 'LR':
        best_model = LogisticRegression(solver='liblinear', C=1.0)
    elif best_model_name == 'GBM':
        best_model = GradientBoostingClassifier()
    elif best_model_name == 'RF':
        best_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the best model
    best_model.fit(X_train, y_train)
    
    # Save the best model
    model_info = save_model(best_model, best_model_name)
    print(f"\nBest model saved to: {model_info['model_path']}")

    # Test the best model
    y_pred = best_model.predict(X_test)
    print("\nBest Model Test Set Evaluation:")
    print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {metrics.precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {metrics.recall_score(y_test, y_pred):.4f}")
    print(f"F1 score: {metrics.f1_score(y_test, y_pred):.4f}")
    print(f"AUC: {metrics.roc_auc_score(y_test, y_pred):.4f}")

    print("\nAll models have been trained and saved successfully.")