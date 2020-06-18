"""
This module implements the functions for parsing three data sets:
1. Adult
2. Law school
3. Communities and Crime


Racial encoding for Lawschool Dataset
1.0 : American Indian
2.0 : Asian
3.0 : Black, size 1201
4.0 : Mexican American
5.0 : Puerto Rican
6.0 : Other Hispanic
7.0 : White, size 17493
8.0 : Others

Documentation: For each data set 'name.csv' we create a function
clean_name clean name takes parameter num_sens, which is the number of
sensitive attributes to include clean_name returns pandas data frames
(x, a, y)
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
DATA_SPLIT_SEED = 4

def one_hot_code(df1):
    cols = df1.columns
    for c in cols:
        if isinstance(df1[c][1], str):
            column = df1[c]
            df1 = df1.drop(c, 1)
            unique_values = list(set(column))
            n = len(unique_values)
            if n > 2:
                for i in range(n):
                    col_name = '{}.{}'.format(c, i)
                    col_i = [1 if el == unique_values[i] else 0 for el in column]
                    df1[col_name] = col_i
            else:
                col_name = c
                col = [1 if el == unique_values[0] else 0 for el in column]
                df1[col_name] = col
    return df1

def log_numeric_features(df):
    cols = df.columns
    for c in cols:
        column =df[c]
        unique_values = list(set(column))
        n = len(unique_values)
        if n > 2:
            df[c] = np.log(1 + df[c])

# num_sens in 1:19
def clean_communities(num_sens):
    # Data Cleaning and Import
    df = pd.read_csv('./data/communities.csv')
    df = df.fillna(0)
    # sensitive variables are just racial distributions in the
    # population and police force as well as foreign status median
    # income and pct of illegal immigrants / related variables are not
    # labeled sensitive
    sens_features = [2, 3, 4, 5, 6, 22, 23, 24, 25, 26, 27, 61, 62, 92,
                     105, 106, 107, 108, 109]
    df_sens = df.iloc[:, sens_features[0:num_sens]]
    Y = df['ViolentCrimesPerPop']
    X = df.iloc[:, 0:122]
    X_prime = df_sens
    return X, X_prime, Y


# num_sens in 1:19
def clean_communities_short(num_sens, short):
    """
    Return a small number of the communities
    """
    df = pd.read_csv('./data/communities.csv')
    df = df.fillna(0)
    # sensitive variables are just racial distributions in the
    # population and police force as well as foreign status median
    # income and pct of illegal immigrants / related variables are not
    # labeled sensitive
    sens_features = [2, 3, 4, 5, 6, 22, 23, 24, 25, 26, 27, 61, 62, 92,
                     105, 106, 107, 108, 109]
    df_sens = df.iloc[:, sens_features[0:num_sens]]
    Y = df['ViolentCrimesPerPop']
    X = df.iloc[:, 0:122]
    X_prime = df_sens
    x = X.iloc[range(short)]
    y = Y[:short]
    a = X_prime[:short]
    return x, a, y


def clean_adult_full():
    """
    Parse the entire dataset of adult
    """
    df = pd.read_csv("./data/adult_full.csv", )
    df = df.dropna()
    df = df.replace({'?':np.nan}).dropna()
    df["income"] = df["income"].map({'<=50K': 0, '>50K': 1})
    y = df["income"]
    df = df.drop("income", 1)

    # hot code categorical variables
    df = one_hot_code(df)
    log_numeric_features(df)
    a = df['sex']
    return df, a, y

def majority_pop(a):
    """
    Identify the main ethnicity group of each community
    """
    B = "racepctblack"
    W = "racePctWhite"
    A = "racePctAsian"
    H = "racePctHisp"
    races = [B, W, A, H]
    maj = a.apply(pd.Series.idxmax, axis=1)
    return maj


def clean_communities_full():
    """
    Extract black and white dominant communities; 
    sub_size : number of communities for each group
    """
    df = pd.read_csv('./data/communities.csv')
    df = df.fillna(0)
    B = "racepctblack"
    W = "racePctWhite"
    A = "racePctAsian"
    H = "racePctHisp"
    sens_features = [2, 3, 4, 5]
    df_sens = df.iloc[:, sens_features]

    # creating labels using crime rate
    Y = df['ViolentCrimesPerPop']
    df = df.drop('ViolentCrimesPerPop', 1)

    maj = majority_pop(df_sens)

    # remap the values of maj
    a = maj.map({B : 0, W : 1, A : 0, H : 0})
   
    df['race'] = a
    df = df.drop(H, 1)
    df = df.drop(B, 1)
    df = df.drop(W, 1)
    df = df.drop(A, 1)
    return df, a, Y

def clean_lawschool_full():
    """

    Use race as the protected feature.
    sub_size : the number of observations to include for each group
    """
    df = pd.read_csv('./data/lawschool.csv')
    df = df.dropna()
    # remove y from df
    y = df['ugpa']
    y = y / 4
    df = df.drop('ugpa', 1)
    # convert gender variables to 0,1
    df['gender'] = df['gender'].map({'male': 1, 'female': 0})
    # add bar1 back to the feature set
    df_bar = df['bar1']
    df = df.drop('bar1', 1)
    df['bar1'] = [int(grade == 'P') for grade in df_bar]
    df['race'] = [int(race == 7.0) for race in df['race']]
    a = df['race']
    return df, a, y

def racial_gpa():
    """
    extract the racial percentage from the population
    """
    df = pd.read_csv('./data/lawschool.csv')
    df = df.dropna()
    races = pd.Series.unique(df['race'])
    for race in races:
        x_sub = df[df['race'] == race]
        print("Racial group " + str(race) + " GPA: " +
              str(pd.Series.mean(x_sub['ugpa'])) + " has size " +
              str(len(x_sub)))
        print(len(x_sub))


def subsample(x, a, y, size, random_seed=4):
    """
    Randomly subsample a smaller dataset of certain size
    """
    toss = 1 - size / (len(x))
    x1, _, a1, _, y1 ,_ = train_test_split(x, a, y, test_size=toss, random_state=random_seed)
    x1.index = range(len(x1))
    y1.index = range(len(x1))
    a1.index = range(len(x1))
    return x1, a1, y1




