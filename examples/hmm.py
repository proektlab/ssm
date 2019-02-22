import autograd.numpy as np
import autograd.numpy.random as npr
npr.seed(0)

import matplotlib
import matplotlib.pyplot as plt

from ssm.models import HMM
from ssm.util import find_permutation

# Set the parameters of the HMM
T = 500                   # number of time bins
num_states = 5            # number of discrete states
observation_dim = 2       # number of observed dimensions

# Make an HMM with the true parameters
true_hmm = HMM(num_states, observation_dim, observations="diagonal_gaussian")
states, observations = true_hmm.sample(T)
data = dict(data=observations)
test_states, test_observations = true_hmm.sample(T)
test_data = dict(data=test_observations)
true_ll = true_hmm.log_probability(data)

# Fit models
N_sgd_iters = 1000
N_em_iters = 100

# A bunch of observation models that all include the
# diagonal Gaussian as a special case.
observation_models = [
    "diagonal_gaussian",
    "gaussian",
    "diagonal_t",
    "studentst",
    "diagonal_ar",
    "ar",
    "diagonal_robust_ar",
    "robust_ar"
]

# Fit with both SGD and EM
methods = [
    "sgd",
    "em"
]

results = {}
for obs_model in observation_models:
    for method in methods:
        print("Fitting {} HMM with {}".format(obs_model, method))
        model = HMM(num_states, observation_dim, observations=obs_model)
        train_lls = model.fit(data, method=method)
        test_ll = model.log_likelihood(test_data)
        smoothed_data = model.smooth(data)

        # Permute to match the true states
        model.permute(find_permutation(states, model.most_likely_states(data)))
        inferred_states = model.most_likely_states(data)
        results[(obs_model, method)] = (model, train_lls, test_ll, inferred_states, smoothed_data)

# Plot the inferred states
fig, axs = plt.subplots(len(observation_models) + 1, 1, figsize=(12, 8))

# Plot the true states
plt.sca(axs[0])
plt.imshow(states[None, :], aspect="auto", cmap="jet")
plt.title("true")
plt.xticks()

# Plot the inferred states
for i, obs in enumerate(observation_models):
    inferred_statess = []
    for method, ls in zip(methods, ['-', ':']):
        _, _, _, inferred_states, _ = results[(obs, method)]
        inferred_statess.append(inferred_states)

    plt.sca(axs[i+1])
    plt.imshow(np.row_stack(inferred_statess), aspect="auto", cmap="jet")
    plt.yticks([0, 1], methods)
    if i != len(observation_models) - 1:
        plt.xticks()
    else:
        plt.xlabel("time")
    plt.title(obs)

plt.tight_layout()

# Plot smoothed observations
fig, axs = plt.subplots(observation_dim, 1, figsize=(12, 8))

# Plot the true data
for d in range(observation_dim):
    plt.sca(axs[d])
    plt.plot(observations[:, d], '-k', lw=2, label="True")
    plt.xlabel("time")
    plt.ylabel("$y_{{}}$".format(d+1))

for obs in observation_models:
    line = None
    for method, ls in zip(methods, ['-', ':']):
        _, _, _, _, smoothed_data = results[(obs, method)]
        for d in range(observation_dim):
            plt.sca(axs[d])
            color = line.get_color() if line is not None else None
            line = plt.plot(smoothed_data[:, d], ls=ls, lw=1, color=color,
                            label="{}({})".format(obs, method))[0]

# Make a legend
plt.sca(axs[0])
plt.legend(loc="upper right")
plt.tight_layout()

# Plot log likelihoods
plt.figure(figsize=(12, 8))
for obs in observation_models:
    line = None
    for method, ls in zip(methods, ['-', ':']):
        _, lls, _, _, _ = results[(obs, method)]
        color = line.get_color() if line is not None else None
        line = plt.plot(lls, ls=ls, lw=1, color=color, label="{}({})".format(obs, method))[0]

xlim = plt.xlim()
plt.plot(xlim, true_ll * np.ones(2), '-k', label="true")
plt.xlim(xlim)

plt.legend(loc="lower right")
plt.tight_layout()

# Print the test log likelihoods
print("Test log likelihood")
print("True: ", true_hmm.log_likelihood(test_data))
for obs in observation_models:
    for method in methods:
        _, _, test_ll, _, _ = results[(obs, method)]
        print("{} ({}): {}".format(obs, method, test_ll))

plt.show()
