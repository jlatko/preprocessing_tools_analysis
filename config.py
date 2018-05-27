import numpy as np

DATA_FILE = './data/data.csv'

BINNER_CONFIG = {
    'Medical_History_1': {
        'values': [0],
        'nan': True,
        'drop': False,
        'bins': [1e-9,10,30,100,200,241]
    },
    'Medical_History_10': {
        'values': [0, 240],
        'nan': True,
        'drop': False,
        'bins': [1e-9, 240]
    },
    'Medical_History_15': {
        'values': [0, 240],
        'nan': True,
        'drop': False,
        'bins': [1e-9, 240]
    },
    'Medical_History_24': {
        'values': [0, 240],
        'nan': True,
        'drop': False,
        'bins': [1e-9, 240]
    },
    'Medical_History_32': {
        'values': [0, 240],
        'nan': True,
        'drop': False,
        'bins': [1e-9, 240]
    },
}

FEATURES_TO_DROP = []