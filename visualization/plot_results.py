import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

W = 0.5
W2 = 0.2
W3 = 0.2

def plot_results(data, title=None, ylabel='error', save_path=None, ymax=None, ymin=None, precision=4):
    fig = plt.figure()
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
    for model, settings in data.items():
        offset += W
        ticks.append(model)
        offsets.append(offset + (len(settings) + 1) * W3 / 2.0)
        for setting, results in settings.items():
            offset += W2
            plt.bar(offset, results['mean'], yerr=results['std'], width=W3, color=legend[setting])
            plt.text(offset, results['mean'] - results['std'], "{num:.{pr}f}".format(num=results['mean'], pr=precision), ha='center', va='top', rotation='vertical')

    if ymax != None:
        plt.ylim(ymax=ymax)
    if ymin != None:
        plt.ylim(ymin=ymin)

    if title:
        plt.title(title)
    plt.xticks(offsets, ticks)
    if ylabel:
        plt.ylabel(ylabel)
    plt.legend(handles=[mpatches.Patch(color=c, label=l) for l, c in legend.items()], loc='best')
    plt.show()

    if save_path:
        fig.savefig(save_path)