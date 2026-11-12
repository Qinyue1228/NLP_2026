import numpy as np
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax
import jax.nn
import arviz as az


def one_hot(y, num_classes):
    """
    Return a matrix [N, K] where each row is a K-dimensional one-hot encoding of the classes in the N-dimensional vector of classes.
    
    y: [N] a vector of N classes (each from 0 to K-1)
    num_classes: (int) number K of classes
    """
    return np.eye(num_classes, dtype=int)[y]

def numpyro_model(num_systems, sys_id, features, correctness=None, s_std=1., w_std=1.):
    """
    Defines a Bayesian regressor for correctness.

    The model looks roughly as follows.

    Observables:
        * c[n] is the correctness label for the nth observation
        * s[n] is the system that produced the nth observation
        * f[n] is the vector of I features that describe the nth input
    Latent parameters:
        * global_mean: a scalar
        * sys_shift: a vector of scalars, one per system
        * w: a collection of I-dimensional weights, one I-dimensional vector per system

    Likelihood function:

        c[n] | s[n], f[n] ~ Bernoulli(sigmoid(logit[n]))
            where logit[n] = global_mean + sys_shift[s[n]] + dot(w[s[n]], f[n])

    Priors:
        * Laplace(0,1) priors on all parameters

        
    num_systems: (int) a number S of systems
    sys_id: [N] a vector of size N, each position indicates which system (from 0 to S-1) produced the measurement
    features: [N, I] for each observation this is an I-dimensional vector of features 
    correctness: [N] a vector of size N, each position indicates whether the system made a correct prediction or not
    """
    n, d = features.shape

    # a prior for the global mean logit
    global_mean = numpyro.sample("global_mean", dist.Normal(0.0, 2.0))

    # per system, a prior for the linear shift and for the slope of each feature
    with numpyro.plate("systems", num_systems):
        sys_shift = numpyro.sample(
            "sys_shift", dist.Laplace(0.0, s_std)
        )        

        # One D-dimensional vector per system: shape (num_systems, d).
        w = numpyro.sample(
            "w",
            dist.Laplace(0.0, w_std).expand([d]).to_event(1),
        )
    
    # Define the logit as a linear combination of latent parameters and their linear interaction with features
    mu = (
        global_mean
        + sys_shift[sys_id]        
        + jnp.sum(w[sys_id] * features, axis=-1)
    )    
        
    mu = numpyro.deterministic('logit', mu)

    # Using logits directly is more numerically stable than sigmoid (ie, the Bernoulli distribution implementation uses sigmoid where necessary)
    numpyro.sample(
        "correctness",
        dist.Bernoulli(logits=mu),
        obs=correctness,
    )
    
    # Evaluate EVERY system on the SAME feature/item/class settings.
    # Shape: (n_observations, num_systems).
    comparison_logits = (
        global_mean
        + sys_shift[None, :]
        + features @ w.T
    )
        
    # Average correctness probability over this shared evaluation set.
    # Shape: (num_systems,).
    mean_probability = numpyro.deterministic(
        "accuracy",
        jnp.mean(jax.nn.sigmoid(comparison_logits), axis=0),
    )
    
    # Shape: (num_systems, num_systems).
    numpyro.deterministic(
        "accuracy_difference",
        mean_probability[:, None] - mean_probability[None, :],
    )

def fit_bda_model(
    num_systems,
    sys_id,    
    features,
    correctness,  
    s_std=1., 
    w_std=1.,
    seed=0,
    num_warmup=1000,
    num_samples=1000,
    num_chains=4,
):
    """
    Fit the BDA model above: draw posterior samples via MCMC.

    The parameters that are specific to this routine are
        - seed
        - num_warmup: we discard this many samples from the beginning of the MCMC chain, as a warmup step
        - num_samples: number of posterior samples to draw after warmpup
        - num_chains: number of chains to simulate (use 2 or more)

    The remaining parameters are documented by the BDA model's constructor above.

    This function takes care of standardising the input features (so they approximately have 0 mean and unit standard deviation).
    """
    sids = np.asarray(sys_id)
    x = np.asarray(features, dtype=float)
    x_mean = x.mean(axis=0)
    x_scale = jnp.sqrt(x.var(axis=0) + 1e-5)    
    x = (x - x_mean) / x_scale

    y = np.asarray(correctness)

    if not isinstance(num_systems, (int, np.integer)) or num_systems <= 0:
        raise ValueError("num_systems must be a positive integer.")
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("features must be a nonempty N-by-D array.")
    if sids.shape != (x.shape[0],):
        raise ValueError("sys_id must have length N.")
    if not np.issubdtype(sids.dtype, np.integer):
        raise ValueError("System IDs must be integers.")

    if np.any(sids < 0) or np.any(sids >= num_systems):
        raise ValueError("System IDs must be in [0, num_systems).")
    if not np.all(np.isfinite(x)):
        raise ValueError("Features must be finite.")
    if y.shape != (x.shape[0],) or not np.all((y == 0) | (y == 1)):
        raise ValueError("correctness must contain N binary values.")

    mcmc = MCMC(
        NUTS(numpyro_model, target_accept_prob=0.9),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="sequential",  # Works with a single device.
    )
    mcmc.run(
        random.PRNGKey(seed),
        num_systems=num_systems,
        sys_id=jnp.asarray(sids, dtype=jnp.int32),
        features=jnp.asarray(x),
        correctness=jnp.asarray(y, dtype=jnp.int32),        
        s_std=s_std, 
        w_std=w_std
    )
    return mcmc

def posterior_to_arviz(mcmc, system_names, feature_names):  
    """
    Postprocess the numpyro posterior object into a data frame that interacts well with various arviz plotting functions. 
    mcmc: the object returned by fit_model
    system_names: S names for the systems
    feature_names: I names for the features
    """
    posterior_data = az.from_numpyro(
        mcmc,
        dims={
            'global_mean': [],
            'sys_shift': ['system'],        
            'w': ['system', 'feature'],                
            "accuracy": ["system"],
            "accuracy_difference": ["system_a", "system_b"],
        },
        coords={
            "system": system_names, 
            "feature": feature_names,
            "system_a": system_names,
            "system_b": system_names,
        }   
    )
    return posterior_data