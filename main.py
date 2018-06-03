import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

from diagnostics.evaluation import fit_and_eval, weighted_quad_kappa
from imputers.hot_deck_full_imputer import HotDeckFullImputer, HotDeckColImputer
from imputers.hot_deck_simple_imputer import HotDeckSimpleImputer
from imputers.model_based import ModelBasedImputer
from imputers.model_based_imputer import ModelBasedFullImputer
from transformers.box_cox import BoxCoxTransformer
from transformers.clip_labels import LabelsClipper
from transformers.feature_dropper import FeatureDropper
from imputers.fill_missing_transformer import FillNaTransformer
from imputers.knn_filler import dist_fn1, KnnFiller
from transformers.one_hot_encoder import CustomOneHotEncoder
from config import BINNER_CONFIG
from transformers.custom_binner import CustomBinner
from imputers.regression_filler import RegressionFiller
from imputers.zero_filler import ZeroFiller

DATA_FILE = './data/data.csv'
TRAIN = './data/train.csv'
TEST = './data/test.csv'

def get_data():
    data = pd.read_csv(TRAIN)

    data.set_index('Id')
    continuous = ['Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI', 'Employment_Info_1', 'Employment_Info_4',
                  'Employment_Info_6', 'Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4',
                  'Family_Hist_5']
    discrete = ['Medical_History_1', 'Medical_History_10', 'Medical_History_15', 'Medical_History_24', 'Medical_History_32']
    dummy = [col for col in data.columns if col.startswith('Medical_Keyword')]
    categorical = list(set(data.columns) - set(continuous) - set(discrete) - set(dummy) - {'Response', 'Id'})
    labels = data['Response']
    return data, labels, continuous, discrete, dummy, categorical


if __name__ == "__main__":

    # test = pd.read_csv(TEST)
    data, labels, continuous, discrete, dummy, categorical = get_data()
    train = data.drop('Response', axis=1)

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
        # ('fillna', FillNaTransformer(
        #     median=fill_with_median,
        #     zero=fill_with_zero,
        #     nan_flag=left_numerical,
        #     # chosen experimentally
        #     from_dict={'Medical_History_1': 300}
        # )),
        ('drop', FeatureDropper(features_to_drop)),
        # ('zero_filler', ZeroFiller()),
        ('onehot', CustomOneHotEncoder(columns=categorical)),
        # TODO: implement own scaler that ignores missing values
        # ('filler', RegressionFiller(columns=left_numerical)),
        # ('boxcox', BoxCoxTransformer(lambdas_per_column)),
        # ('filler', ModelBasedFullImputer(columns=left_numerical, model=KNeighborsRegressor(n_neighbors=19))),
        # ('classifier', LogisticRegression()),
        # ('filler', HotDeckColImputer('Medical_History_1')),
        # ('filler', HotDeckFullImputer([(col, 20) for col in left_numerical])),
        ('filler', HotDeckSimpleImputer(left_numerical)),
        ('scale', StandardScaler()),
        # ('selector', SelectFromModel(Lasso(alpha=0.0003, normalize=True), threshold='mean')),
        ('classifier', LabelsClipper(LinearRegression())),
    ])



    # score = cross_val_score(pipe, train.copy(), labels, cv=3, n_jobs=8, scoring=make_scorer(weighted_quad_kappa)).mean()
    # print(score)
    print(fit_and_eval(pipe, train.copy(), labels, k=3))




    # print("Score: {}".format(score))
    # ([0.5451542189584215, 0.5518141795450093, 0.5408448116365189, 0.5521392732399414, 0.5592437127187462]
    # [0.5288078065499804, 0.5477213869792066, 0.5460956371087629, 0.5521371408514735, 0.5555445892529358], hot deck
    # [0.5364021796419012, 0.5376723241727355, 0.5330756546885753, 0.5427125809280473, 0.5547762823140613], manual
    # ([0.53446639140263, 0.5332623446185626, 0.5240618215953008, 0.5381708679217712, 0.5499139625996656], nothing
    # [0.1082600982019315, 0.1877032661454564, 0.28692956360847666] class