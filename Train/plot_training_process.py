import os
import pandas as pd
from matplotlib import pyplot as plt

def plot_progress(df, n_rows, title, xaxis, yaxis, xlabel, ylabel, fig_fp=None):
    assert type(yaxis) is list, "xaxis should be list type"

    df = df.loc[:n_rows]
    plt.figure(figsize=(7, 4))

    for y in yaxis:
        plt.plot(df.loc[:n_rows, xaxis], df.loc[:n_rows, y], marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)

    if fig_fp is None:
        os.path.join(os.path.dirname(__file__), "temp", "training_progress.png")
    plt.savefig(fig_fp)

if __name__ == "__main__":
    from pathlib import Path
    df = pd.read_csv("assets/training_process/clip_nodecay_infoNCE_8_rand_augmix_000030_epoch30.csv")
    save_dir = os.path.join(os.path.dirname(__file__), "temp", "training_progress")
    Path(save_dir).mkdir(exist_ok=True, parents=True)

    title = 'Accuracy as a Function of Epochs in AFRICAN Model Training'
    fig_fp = os.path.join(save_dir, "clip_nodecay_infoNCE_8_rand_augmix_000030_epoch30.png")
    plot_progress(df, n_rows=50, title=title, xaxis='epoch', yaxis=['Accuracy'], xlabel='Epoch', ylabel='Accuracy', fig_fp=fig_fp)


