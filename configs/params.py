from sklearn.preprocessing import StandardScaler

from transformers import CustomBinaryBinner, BoxCoxTransformer, OutliersClipper, CustomBinner
import numpy as np

BASE_HEART = {
    'binner': CustomBinaryBinner(
        configuration={}),
    'boxcox': BoxCoxTransformer(lambdas_per_column={'chol': 0, 'thalach': 2, 'trestbps': 0}),
    'clipper': None,
    'scaler': None
}

BASE_BOSTON = {
    'binner': CustomBinaryBinner(configuration={}),
    'boxcox': None,
    'clipper': OutliersClipper(columns=['crim', 'zn', 'nox', 'indus', 'rm', 'age', 'tax', 'ptratio', 'b', 'lstat']),
    'scaler': StandardScaler(copy=True, with_mean=True, with_std=True)}

BASE_HOUSES = {
    'binner': CustomBinaryBinner(configuration={}),
    'boxcox': None,
    'clipper': OutliersClipper(columns=['crim', 'zn', 'nox', 'indus', 'rm', 'age', 'tax', 'ptratio', 'b', 'lstat']),
    'scaler': StandardScaler(copy=True, with_mean=True, with_std=True)}

BASE_PRUDENTIAL = {
    'binner': CustomBinaryBinner(
        configuration={}),
    'binner2': CustomBinner(configuration={}),
    'boxcox': BoxCoxTransformer(lambdas_per_column={'Wt': 0.5}),
    'clipper': OutliersClipper(columns=['Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI', 'Employment_Info_1',
                                        'Employment_Info_4', 'Employment_Info_6', 'Insurance_History_5',
                                        'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5']),
    'scaler': None}
