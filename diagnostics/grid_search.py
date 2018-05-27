import matplotlib.pyplot as plt
import numpy as np

# up to 3 grid dimensions (choose max from the last one)
def plot_grid(grid, k_options, l_options, m_options_length, k_label="k options"):
    mean_scores = np.array(-grid.cv_results_['mean_test_score'])
    # scores are in the order of param_grid iteration, which is alphabetical
    mean_scores = mean_scores.reshape(m_options_length, -1, len(k_options))
    # select score for best C
    mean_scores = mean_scores.max(axis=0)
    bar_offsets = (np.arange(len(k_options)) *
                   (len(l_options) + 1) + .5)

    plt.figure()
    COLORS = 'bgrcmyk'
    for i, (label, l_scores) in enumerate(zip(l_options, mean_scores)):
        plt.bar(bar_offsets + i, l_scores, label=label, color=COLORS[i])

    plt.title("Search for best hyperparams")
    plt.xlabel(k_label)
    plt.xticks(bar_offsets + len(l_options) / 2, k_options)
    plt.ylabel('Mean score')
    plt.legend(loc='best')
    plt.show()