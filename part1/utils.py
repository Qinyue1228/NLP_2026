import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import itertools 
from collections import Counter, OrderedDict
import re 
from tabulate import tabulate
import random 
from tqdm.auto import tqdm


def infer_num_classes(pmfs: dict) -> int:
    """
    Inspects the conditional pmfs, dicts mapping conditions to K-dimensional probability vectors, to infer K.
    """
    K = None
    for x, probs in pmfs.items():
        if K is None:
            K = len(probs)
        else:
            if K != len(probs):
                raise ValueError(f"Every pmf must be specified by a {K}-dimensional vector, got {len(probs)} for {x}")
    return K
    
def pmfs_to_str(pmfs: dict):
    """
    Return a table-formatted view of the given conditional pmfs in a single string ready to be displayed.
    """
    table = []
    K = None
    for x, probs in pmfs.items():
        table.append([x] + [p for p in probs])
        if K is None:
            K = len(probs)
        else:
            if K != len(probs):
                raise ValueError(f"Every pmf must be specified by a {K}-dimensional vector, got {len(probs)} for {x}")
        
    return tabulate(table, headers=['x'] + [f'p({k+1}|x)' for k in range(K)])

def encodings_to_str(encodings: dict, coords=None):
    """
    Return a table-formatted view of a dict of encodings in a single string ready to be displayed.
    coords: headers for the columns ('names' of each axis)
    """
    table = []
    num_dims = 0
    for text, h in encodings.items():
        table.append([text] + [hd for hd in h])
        num_dims = len(h)
    if coords is None:
        headers = ['x'] + [f"h_{d}(x)" for d in range(num_dims)]
    else:
        headers = ['x'] + [f"h_{d}(x): {name}" for d, name in enumerate(coords)]
    return tabulate(table, headers=headers)


def plot_pmf(outcomes, probs, hlabel="Outcome (y)", vlabel="Probability p(y)", title="Probability mass funciton", figsize=(3,2)):
    """
    Plot a pmf as a bar plot.
    outcomes: possible outcomes (xticks labels)
    probs: heights of the bars
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()
    _ = ax.bar(outcomes, probs)
    _ = ax.set_xlabel(hlabel, fontsize=10)
    _ = ax.set_ylabel(vlabel, fontsize=10)
    _ = ax.set_ylim(0, 1)
    _ = ax.set_title(title, fontsize=10)
    return fig, ax

def plot_pmfs_barh(pmfs, category_names, legendtitle="category", plottitle=None, fmt="{:.2f}"):
    """
    Plot a collection of conditional pmfs using barh plots.
    
    pfms : dict
        A mapping from conditions to a list of K probability masses.
    category_names : list of category names
    fmt: format for probability values
    """
    labels = list(pmfs.keys())
    data = np.array(list(pmfs.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.colormaps['Blues'](
        np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(6, 2))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())
    if plottitle is not None:
        ax.set_title(plottitle, fontsize=10)

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color)
        
        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        ax.bar_label(rects, label_type='center', color=text_color, fmt=fmt)
    ax.legend(
        title=legendtitle,
        ncols=len(category_names), 
        bbox_to_anchor=(1, -0.15),
        loc="upper right",
        fontsize="small",
        title_fontsize='small',
    )
    ax.figure.subplots_adjust(bottom=0.25)


    return fig, ax

class ThreeWaySentimentExample:
    """
    Our running example in the earlier sections of the tutorial.
    """

    category_names = ['negative', 'neutral', 'positive']

    pmfs = OrderedDict(
        [
            ("what a great day", [0.2, 0.1, 0.7]),
            ("a lovely day", [0.05, 0.1, 0.85]),
            ("a day that never ends", [1., 0., 0.])
        ]
    )

    @staticmethod
    def plot(stacked_bars=False):
        """Plot the example pfms"""
        pmfs = ThreeWaySentimentExample.pmfs
        category_names = ThreeWaySentimentExample.category_names
        
        if stacked_bars:
            return plot_pmfs_barh(pmfs, category_names)
        else:        
            fig, axs = plt.subplots(1, len(pmfs), figsize=(3*len(pmfs), 2.5))
            
            for i, (x, probs) in enumerate(pmfs.items()):
                _ = axs[i].bar(category_names, probs)
                _ = axs[i].set_xlabel("Sentiment (y)", fontsize=10)
                _ = axs[i].set_ylabel("p(y|x)", fontsize=10)
                _ = axs[i].set_ylim(0, 1)
                _ = axs[i].set_title(f"Text (x): {x}", fontsize=10)
            
            fig.tight_layout(w_pad=1, h_pad=1)
        return fig, axs

