import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from diagnostics.evaluation import fit_and_eval
from transformers.box_cox import BoxCoxTransformer
from transformers.feature_dropper import FeatureDropper
from transformers.fill_missing_transformer import FillNaTransformer
from transformers.one_hot_encoder import CustomOneHotEncoder
from config import BINNER_CONFIG
from transformers.custom_binner import CustomBinner

DATA_FILE = './data/data.csv'

def get_data():
    data = pd.read_csv(DATA_FILE)
    continuous = ['Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI', 'Employment_Info_1', 'Employment_Info_4',
                  'Employment_Info_6', 'Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4',
                  'Family_Hist_5']
    discrete = ['Medical_History_1', 'Medical_History_10', 'Medical_History_15', 'Medical_History_24', 'Medical_History_32']
    dummy = [col for col in data.columns if col.startswith('Medical_Keyword')]
    categorical = list(set(data.columns) - set(continuous) - set(discrete) - set(dummy) - {'Response', 'Id'})
    labels = data['Response']
    return data, labels, continuous, discrete, dummy, categorical


if __name__ == "__main__":

    data, labels, continuous, discrete, dummy, categorical = get_data()
    # chosen by manually from correlations
    features_to_drop = ['Medical_Keyword_45', 'Medical_Keyword_42']
    # features_to_drop = []
    categorical = [x for x in categorical if x not in features_to_drop]

    # add binned discrete, without Medical_History_1, which has less missing values
    features_to_drop += ['Medical_History_10', 'Medical_History_15',
                         'Medical_History_24', 'Medical_History_32']

    left_numerical = continuous + ['Medical_History_1']

    fill_with_zero = ['Employment_Info_4', 'Insurance_History_5'] + ['Medical_History_1']
    fill_with_median = [x for x in continuous if x not in fill_with_zero]
    lambdas_per_column = {
        'Product_Info_4': 0.5,
        'Ht': 1.2,
        'Wt': 0.5,
        'BMI': 0.6,
        'Employment_Info_1': 0.5,
        'Employment_Info_4': 0.5,
        'Employment_Info_6': 0.5,
        'Insurance_History_5': 0.5,
        'Family_Hist_2': 0.7,
        'Family_Hist_3': 2,
        'Family_Hist_4': 0.7,
        'Family_Hist_5': 2
    }


    pipe = Pipeline([
        ('binner', CustomBinner(BINNER_CONFIG)),
        ('fillna', FillNaTransformer(
            median=fill_with_median,
            zero=fill_with_zero,
            nan_flag=left_numerical,
            # chosen experimentally
            from_dict={'Medical_History_1': 300}
        )),
        ('boxcox', BoxCoxTransformer(lambdas_per_column)),
        ('drop', FeatureDropper(features_to_drop)),
        ('onehot', CustomOneHotEncoder(columns=categorical)),
        ('scale', StandardScaler()),
        ('classifier', LogisticRegression())
    ])
    # score = cross_val_score(pipe, data.copy(), labels, cv=3, n_jobs=8).mean()

    print(fit_and_eval(pipe, data.head(2000), labels.head(2000)))


    # print("Score: {}".format(score))

    # 0.7932331865263981