import pandas as pd
import matplotlib.pyplot as plt

def EDA():

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
    
EDA()   

    