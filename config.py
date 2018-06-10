import numpy as np

DATA_FILE = './data/data.csv'

BINARY_BINNER_CONFIG_PRUD = {
    'Medical_History_1': {
        'values': [0],
        'bins': [1e-9,10,30,100,200,241]
    },
    'Medical_History_10': {
        'values': [0, 240],
        'bins': [1e-9, 240]
    },
    'Medical_History_15': {
        'values': [0, 240],
        'bins': [1e-9, 240]
    },
    'Medical_History_24': {
        'values': [0, 240],
        'bins': [1e-9, 240]
    },
    'Medical_History_32': {
        'values': [0, 240],
        'bins': [1e-9, 240]
    },
}

BINNER_CONFIG_PRUD = {
    'Medical_History_1': {
        'values': [0, 240],
        'bins': [0, 1,10,30,100,200,240]
    },
    'Medical_History_10': {
        'values': [0, 240],
        'bins': [0, 1, 2, 240]
    },
    'Medical_History_15': {
        'values': [0, 240],
        'bins': [0, 1, 2, 240]
    },
    'Medical_History_24': {
        'values': [0, 240],
        'bins': [0, 1, 2, 240]
    },
    'Medical_History_32': {
        'values': [0, 240],
        'bins': [0, 1, 2, 240]
    },
}

BOX_COX_HE = {
    'chol': 0,
    'thalach': 2,
    'trestbps': 0,
}

# BOX_COX_P = {
#     'Product_Info_4': 0.5,
#     'Ht': 1.2,
#     'Wt': 0.5,
#     'BMI': 0.6,
#     'Employment_Info_1': 0.5,
#     'Employment_Info_4': 0.5,
#     'Employment_Info_6': 0.5,
#     'Insurance_History_5': 0.5,
#     'Family_Hist_2': 0.7,
#     'Family_Hist_3': 2,
#     'Family_Hist_4': 0.7,
#     'Family_Hist_5': 2
# }
BOX_COX_B = {
    'age': 2,
    'tax': 0,
    'lstat': 0,
}
BOX_COX_HO = {
    'BsmtFinSF1': 0,
    'LowQualFinSF': 0,
    'GrLivArea': 0,
    'WoodDeckSF': 0,
}
BOX_COX_P = {
    'Wt': 0.5,
}

FEATURES_TO_DROP = []