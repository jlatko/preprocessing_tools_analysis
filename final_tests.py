import json
import pickle
import traceback
import warnings
from sklearn import clone
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
from tqdm import tqdm
from configs.final_boston import get_test_config_boston
from configs.final_heart import get_test_config_heart
from configs.final_houses import get_test_config_houses
from configs.final_prudential import get_test_config_prudential
from diagnostics.grid_search import eval_single
from diagnostics.learning_curves import plot_triple_curve
from visualization.plot_results import plot_and_save_results

"""
This script is used to perform evaluation of chosen models on separated test sets 
and to draw learning curves using cross validation (depending on which lines are commented out). 
"""

warnings.filterwarnings('ignore')


dir_path = "./results/"
test_name = "prudential"
missing = True

# load all configurations
data, test, test_labels, scorer, pipe, params, target = get_test_config_prudential()

results = {}

# cv = KFold(5, shuffle=True, random_state=0)
cv = StratifiedKFold(5)
# run tests for each config
for key, config in tqdm(params.items()):
    model = clone(pipe)

    model.set_params(**config['params'])
    if not missing:
        model.set_params(main_imputer=None)
    # if "predictor__n_jobs" in model.get_params():
    #     model.set_params(predictor__n_jobs=8)

    # score = eval_single(model, data.copy(), target, test.copy(), test_labels, scorer, None)
    # print(key, score)
    # results[key] = score
    save_path = "./figures/learning_curves/" + test_name + "/"+ key + ".png"
    plot = plot_triple_curve(model, key, data, target,  test.copy(), test_labels, cv, scorer, outliers=None)
    plot.draw()
    plot.savefig(save_path)
# with open(dir_path + test_name + '.json', 'w') as fp:
#     json.dump(results, fp, indent=4)
# print(results)

