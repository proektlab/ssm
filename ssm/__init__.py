"""
This package has fast and flexible code for simulating, learning, and performing
inference in a variety of state space models.

Currently, it implements the following models:


Hidden Markov Models (HMM)
Auto-regressive HMMs (ARHMM)
Input-output HMMs (IOHMM)
Hidden Semi-Markov Models (HSMM)
Linear Dynamical Systems (LDS)
Switching Linear Dynamical Systems (SLDS)
Recurrent SLDS (rSLDS)
Hierarchical extensions of the above
Partial observations and missing data

For each model, you can choose from a variety of observation distributions:

Gaussian
Student's t
Bernoulli
Poisson
Categorical
Von Mises

HMM inference is done with either expectation maximization (EM) or stochastic
gradient descent (SGD). For SLDS, we use stochastic variational inference (SVI).

The model is trained on a dataset, which is represented as a list of dictionaries.
Each dict represents a single time series observation, and it must be of the
following form:

{
    # Required key-value pairs
    data : array_like (time_bins, observation_dim),

	# Optional data information
	covariance : array_like (time_bins, observation_dim, observation_dim),

    # Optional covariates
    input : array_like (time_bins, input_dim),
    mask : array_like (time_bins, observation_dim) (bool),
    tag : object,
}

Some functions, like those for sampling new data, only take in a dictionary
of covariates.  These are the optional keys from the data dictionary.

{
    # Optional key-value pairs
    input : array_like (time_bins, input_dim),
    mask : array_like (time_bins, observation_dim) (bool),
    tag : object
}

"""
