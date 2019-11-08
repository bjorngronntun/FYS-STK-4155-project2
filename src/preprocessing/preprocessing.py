import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler

def get_clean_data_frame(path = '../../data/raw/'):
    df = pd.read_excel(path + 'default.xls')
    df_names = list(df.iloc[0])
    df_names[-1] = 'Y'
    df.columns = df_names
    df = df[1:]
    return df

def get_design_matrix(path = '../../data/raw/'):
    df = get_clean_data_frame(path)
    numerical_columns = [
        'LIMIT_BAL',
        'AGE',
        'BILL_AMT1',
        'BILL_AMT2',
        'BILL_AMT3',
        'BILL_AMT4',
        'BILL_AMT5',
        'BILL_AMT6',
        'PAY_AMT1',
        'PAY_AMT2',
        'PAY_AMT3',
        'PAY_AMT4',
        'PAY_AMT5',
        'PAY_AMT6'
    ]
    categorical_columns = [
        'SEX',
        'EDUCATION',
        'MARRIAGE',
        'PAY_0',
        'PAY_2',
        'PAY_3',
        'PAY_4',
        'PAY_5',
        'PAY_6'
    ]
    df[numerical_columns] = df[numerical_columns].astype(float)
    lb = LabelBinarizer()
    X = np.ones(len(df))
    X = np.c_[X, np.array(df[numerical_columns])]

    for cc in categorical_columns:
        X = np.c_[X, lb.fit_transform(np.array(df[cc]).astype('int'))]
    ss = StandardScaler()
    X = ss.fit_transform(X)
    return X

def get_target_values(path = '../../data/raw/'):
    df = get_clean_data_frame(path)
    y = np.array(df[['Y']].astype('int')).ravel()
    return y

if __name__ == '__main__':
    df = get_clean_data_frame()
    print(df[['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']].head())
