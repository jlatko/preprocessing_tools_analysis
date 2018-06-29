import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
import json
import pickle
from collections import defaultdict
import traceback


W = 0.4
W2 = 0.26
W3 = 0.26
font = {'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

def plot_results(data, title=None, ylabel='error', save_path=None, ymax=1, ymin=0, precision=4, leg='best'):
    fig = plt.figure(figsize=(11,8))
    COLORS = 'bgrcmyk'
    legend = {}
    i = 0
    for model, settings in data.items():
        for setting in settings.keys():
            if setting not in legend:
                legend[setting] = COLORS[i % len(COLORS)]
                i += 1
    ticks = []
    offsets = []
    offset = 0
    if ymin < 0:
        ymin = 0
    for model, settings in data.items():
        offset += W
        ticks.append(model)
        offsets.append(offset + (len(settings) + 1) * W3 / 2.0)
        for setting, results in settings.items():
            offset += W2
            plt.bar(offset, results['mean'],
                    # yerr=results['std'],
                    width=W3, color=legend[setting])

            plt.errorbar(offset + 0.3 * W3, results['mean'], yerr=results['std'], color='k')
            # if (results['mean'] - results['std'] - ymin) < 0.4 * (ymax - ymin):
            if results['mean']  < ymax - 0.3 * (ymax - ymin):
                plt.text(offset - 0.46*W3, results['mean'] , " {num:.{pr}f}".format(num=results['mean'], pr=precision), ha='left', va='bottom', rotation='vertical')
                # plt.text(offset, results['mean'] + results['std'] * 1.05, "{num:.{pr}f}".format(num=results['mean'], pr=precision), ha='center', va='bottom', rotation='vertical')
            else:
                plt.text(offset - 0.46*W3 , results['mean'], "{num:.{pr}f} ".format(num=results['mean'], pr=precision), ha='left', va='top', rotation='vertical')
                # plt.text(offset - 0.5*W3 , results['mean'] - results['std'], "{num:.{pr}f}".format(num=results['mean'], pr=precision), ha='left', va='top', rotation='vertical')

    if ymax != None:
        plt.ylim(ymax=ymax)
    if ymin != None:
        plt.ylim(ymin=ymin)

    if title:
        plt.title(title)
    plt.xticks(offsets, ticks, rotation=20)
    if ylabel:
        plt.ylabel(ylabel)
    plt.legend(handles=[mpatches.Patch(color=c, label=l) for l, c in legend.items()], loc=leg)

    plt.tight_layout()

    if save_path:
        plt.draw()
        fig.savefig(save_path)
    else:
        plt.show(block=False)


def plot_and_save_results(results, get_model_and_label, path, precision=4, ymax_alpha=1.02, ymin_alpha=0.98,
                          robust=True):
    max_std = max(r['std'] for search_results in results.values() for r in search_results)
    max_score = max(r['score'] for search_results in results.values() for r in search_results)
    min_score = min(r['score'] for search_results in results.values() for r in search_results)
    output = defaultdict(dict)
    try:
        for setting, search_results in results.items():
            for result in search_results:
                prefix = setting + "-" if len(results) > 1 and setting else ""
                model_name, label = get_model_and_label(result['params'])
                output[prefix + model_name][label] = {
                    'mean': result['score'],
                    'std': result['std'],
                }
    except Exception as e:
        if robust:
            with open(path + "_error.p", 'wb') as fp:
                pickle.dump(results, fp, protocol=pickle.HIGHEST_PROTOCOL)
        print('Error occured', e)
        traceback.print_exc()
        return

    plot_results(output, save_path=path + ".png", precision=precision, ymax=ymax_alpha * max_score + max_std,
                 ymin=ymin_alpha * min_score - max_std)
    with open(path + '.json', 'w') as fp:
        json.dump(output, fp, indent=4)
    return output