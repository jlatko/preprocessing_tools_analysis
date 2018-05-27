import matplotlib.pyplot as plt
import seaborn as sns

def corr_heatmap(data):
    corrmat = data.corr()
    plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True);

