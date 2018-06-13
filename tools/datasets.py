import pandas as pd

PRUDENTIAL = './data/prudential/data.csv'
PRUDENTIAL_TRAIN = './data/prudential/train.csv'
PRUDENTIAL_TEST = './data/prudential/test.csv'

def get_prudential(test=False):
    if test:
        data = pd.read_csv(PRUDENTIAL_TEST)
    else:
        data = pd.read_csv(PRUDENTIAL_TRAIN)

    data.set_index('Id')
    data = data.drop('Id', axis=1)
    continuous = ['Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI', 'Employment_Info_1', 'Employment_Info_4',
                  'Employment_Info_6', 'Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4',
                  'Family_Hist_5']
    discrete = ['Medical_History_1', 'Medical_History_10', 'Medical_History_15', 'Medical_History_24', 'Medical_History_32']
    dummy = [col for col in data.columns if col.startswith('Medical_Keyword')]
    categorical = list(set(data.columns) - set(continuous) - set(discrete) - set(dummy) - {'Response', 'Id'})
    labels = data['Response']
    target = 'Response'
    missing_cols = continuous + discrete
    return data, labels, continuous, discrete, dummy, categorical, target, missing_cols


BOSTON = './data/boston/data.csv'
BOSTON_TRAIN = './data/boston/train.csv'
BOSTON_TEST = './data/boston/test.csv'
BOSTON_MISSING = './data/boston/data_missing.csv'
BOSTON_TRAIN_MISSING = './data/boston/train_missing.csv'
BOSTON_TEST_MISSING = './data/boston/test_missing.csv'

def get_boston(test=False, missing=False):
    if missing:
        if test:
            data = pd.read_csv(BOSTON_TEST_MISSING)
        else:
            data = pd.read_csv(BOSTON_TRAIN_MISSING)
    else:
        if test:
            data = pd.read_csv(BOSTON_TEST)
        else:
            data = pd.read_csv(BOSTON_TRAIN)


    continuous = ['crim', 'zn', 'nox', 'indus', 'rm', 'age', 'tax', 'ptratio', 'b', 'lstat','dis']
    discrete = ['rad']
    dummy = ['chas']
    categorical = []
    labels = data['medv']
    target = 'medv'
    missing_cols = ['crim', 'zn', 'nox', 'indus', 'rm', 'age', 'tax', 'ptratio', 'b','dis']
    return data, labels, continuous, discrete, dummy, categorical, target, missing_cols


HEART = './data/heart_disease/data.csv'
HEART_TRAIN = './data/heart_disease/train.csv'
HEART_TEST = './data/heart_disease/test.csv'
HEART_MISSING = './data/heart_disease/data_missing.csv'
HEART_TRAIN_MISSING = './data/heart_disease/train_missing.csv'
HEART_TEST_MISSING = './data/heart_disease/test_missing.csv'

def get_heart(test=False, missing=False):
    if missing:
        if test:
            data = pd.read_csv(HEART_TEST_MISSING)
        else:
            data = pd.read_csv(HEART_TRAIN_MISSING)
    else:
        if test:
            data = pd.read_csv(HEART_TEST)
        else:
            data = pd.read_csv(HEART_TRAIN)

    continuous = ['chol', 'thalach', 'oldpeak', 'trestbps']
    discrete = ['age', 'slope', 'ca']
    dummy = ['sex', 'fbs', 'exang']
    categorical = ['thal', 'chest_pain', 'restecg']
    labels = data['num']
    target = 'num'
    # 4 missing cases for ca
    missing_cols = ['trestbps', 'chol', 'thalach', 'oldpeak']
    return data, labels, continuous, discrete, dummy, categorical, target, missing_cols


HOUSES = './data/houses/data.csv'
HOUSES_TRAIN = './data/houses/train.csv'
HOUSES_TEST = './data/houses/test.csv'

def get_houses(test=False):
    if test:
        data = pd.read_csv(HOUSES_TEST)
    else:
        data = pd.read_csv(HOUSES_TRAIN)
    data.set_index('Id')
    data = data.drop('Id', axis=1)

    continuous = ['LotFrontage', 'LotArea',
       'MasVnrArea', 'BsmtFinSF1',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
       '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'GarageArea',  'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']
    discrete = ['OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars',
       'MoSold', 'YrSold']
    dummy = [ ]
    categorical = list(set(data.columns) - set(continuous) - set(discrete) - set(dummy) - {'SalePrice', 'Id'})
    labels = data['SalePrice']
    target = 'SalePrice'
    missing_cols = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']
    return data, labels, continuous, discrete, dummy, categorical, target, missing_cols
