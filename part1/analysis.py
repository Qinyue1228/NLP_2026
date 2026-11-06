import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import baycomp
import scipy as sp
import scipy.stats as st

def make_data_frame(classifiers_data):
    """
    classifiers_data: a dict, where
        each key is an alias for a system
        which then maps to a list of runs
        
        each run is a dict with keys
        - encoder: this is the encoder used; there's one used for all folds;
        - models: the models trained; this is a list with one per training fold
        - predictions: the predictions of those models; this is a list with one array of predictions per classifier trained
        - accuracies: the validation accuracies of the trained classifiers; this is a list with one measurement per classifier trained
        
    """
    table = []
    for model_family, model_data in classifiers_data.items():
        for run, run_data in enumerate(model_data):
            for fold, acc in run_data['accuracies'].items():
                table.append([model_family, acc, run, fold])
    return pd.DataFrame(table, columns=['model', 'accuracy', 'run', 'fold'])

def plot_box_and_points(df, metric='accuracy', tickfmt="{x:.4f}"):
    """
    Plot the fold accuracies for various classifiers using one boxplot and a stripplot per classifier.

    df: a pandas DataFrame with two columns (model and metric)
    """     
    fig, ax = plt.subplots(figsize=(8, 4))
    order = df["model"].unique()
    
    sns.boxplot(
        data=df,
        x="model",
        y=metric,
        order=order,
        color="lightgray",
        showfliers=False,  # All measurements are shown by the strip plot
        ax=ax,
    )
    
    sns.stripplot(
        data=df,
        x="model",
        y=metric,
        hue='run',
        order=order,
        jitter=0.15,
        size=6,
        alpha=0.7,
        ax=ax,
    )
    
    ax.set(xlabel="model", ylabel=metric)
    ax.tick_params(axis="x", rotation=30)
    ax.yaxis.set_major_formatter(StrMethodFormatter(tickfmt))
    fig.tight_layout()
    plt.show()

class BenavoliCorrelatedT:
    """
    A convenient wrapper around baycomp.

    Estimates the posterior mean difference (B-A) in performance between system A and system B.
    """

    def __init__(self, sys_a, sys_b, runs=1, names=["sys_a", "sys_b"]):
        # the ROPE isn't actually needed at this point,
        # we can compute probs in relation to any given ROPE
        # after we get the posterior samples from baycomp (see the methods plot and probs below)
        self._ct = baycomp.CorrelatedTTest(
            np.array(sys_a), 
            np.array(sys_b),
            rope=0.,             
            runs=runs
        )
        self._names = names

    @property
    def samples(self):
        """Sampled mean differences"""
        return self._ct.sample

    @property
    def mean_diff(self):
        """Posterior mean difference"""
        return self._ct.mean

    @property
    def std(self):
        """Posterior standard deviation"""
        return np.sqrt(self._ct.var)

    @property
    def df(self):
        """Posterior degrees of freedom"""
        return self._ct.df

    @property
    def effect_size(self):
        """
        Return mean difference divided by posterior std.
        This can be interpreted as improvement in "units of stds".
        There is no universal way to read this, 
         but some people like to think that < 0.2 indicates a minor effect
         more than 0.8 indicates a large effect
         and in between indicates moderate effect
        (Don't take these 'rules' too seriously)
        """
        return self.mean_diff / self.std

    def plot(self, rope_param, hdi=0.95, wider_hdi=0.999, num_points=1000, rope_factor=2.5):
        """
        Plot the posterior for the mean difference (B-A), the ROPE and the region probs (ie, A wins, Tie, B wins).
        Return fig and ax.
        
        rope_param: the ROPE is treated as the interval (-rope_param, rope_param)
        hdi: draw a high posterior interval of size 100*hdi% 
        wider_hdi: this is used to control the size of the plot
        num_points: how many evaluates of the posterior density
        rope_factor: this is also used to control the size of the plot, the larger this is
         the more we zoom out
        """
        assert wider_hdi > hdi, "wider_hdi should be wider than hdi"
        
        t = st.t(self.df, loc=self.mean_diff, scale=self.std)
        
        fig, ax = plt.subplots()    

        # where do we evaluate the posterior
        ci = t.interval(wider_hdi)
        x = np.linspace(min(ci[0], -rope_param*rope_factor), max(ci[1], rope_param*rope_factor), num_points)
        _ = ax.set_xlabel(r"Mean difference: $\mu$")
        # plot it
        _ = ax.plot(x, t.pdf(x), label=r"$p(\mu|\mathcal{D})$")
        # draw the ROPE
        _ = ax.axvspan(-rope_param, rope_param, color="teal", alpha=0.25, label=f"ROPE ({rope_param})")

        # annotate the regions to the left of, inside of, and to the right of the ROPE
        transform = ax.get_xaxis_transform()    
        _ = ax.text(-rope_param, 0.2, f"{t.cdf(-rope_param)*100:.1f}% \nA>B ",
                ha="right", va="center", color="gray", transform=transform, fontsize=9)
        _ = ax.text(0, 0.5, f"{(t.cdf(rope_param)-t.cdf(-rope_param))*100:.1f}%\nTie",
                ha="center", va="center", color="gray", transform=transform, fontsize=9)
        _ = ax.text(rope_param, 0.7, f" {(1-t.cdf(rope_param))*100:.1f}%\n B>A",
                ha="left", va="center", color="gray", transform=transform, fontsize=9)

        # draw the chosen hdi and its median
        ci = t.interval(hdi)
        _ = ax.hlines(y=0.02, xmin=ci[0], xmax=ci[1], color="black", linewidth=2, label=f"{hdi*100}% hdi")
        _ = ax.plot(t.median(), 0.02, "d", color="black", label="median")        

        # plot title and legend
        _ = ax.set_title(f"A={self._names[0]} vs. B={self._names[1]}")
        _ = ax.legend()
        return fig, ax
    
    def sampled_probs(self, rope_size: float):
        """
        rope_size: defines a ROPE around 0 (ie, -rope_size, +rope_size)
        
        Return 
            - prob that mean diff is to the left of the ROPE (ie, A wins)
            - prob that mean diff is inside the ROPE (ie, tie)
            - prob that mean diff is to the right of the ROPE (ie, B wins)
        """
        p_A = np.mean(self.samples < - rope_size)
        p_tie = np.mean(np.abs(self.samples) < rope_size)
        p_B = np.mean(self.samples > rope_size)

        return p_A, p_tie, p_B

    def probs(self, rope_param: float):
        """
        rope_param: defines a ROPE around 0 (ie, -rope_param, +rope_param)
        
        Return 
            - prob that mean diff is to the left of the ROPE (ie, A wins)
            - prob that mean diff is inside the ROPE (ie, tie)
            - prob that mean diff is to the right of the ROPE (ie, B wins)
        """
        p_A = st.t.cdf(-rope_param, self.df, loc=self.mean_diff, scale=self.std)
        p_tie = st.t.cdf(rope_param, self.df, loc=self.mean_diff, scale=self.std) - st.t.cdf(-rope_param, self.df, loc=self.mean_diff, scale=self.std)
        p_B = 1 - st.t.cdf(rope_param, self.df, loc=self.mean_diff, scale=self.std)
        return p_A, p_tie, p_B
        