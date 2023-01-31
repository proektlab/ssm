import numpy as np
import numpy.random as npr
from scipy import stats
import warnings
from typing import Callable, Any, Tuple

import ssm_customized


# npr.seed(0)
rng = npr.default_rng()

INPUT_DIM = 6
INPUT_LABELS = ['Cue A', 'Cue B', 'Cue C', 'bias', 'prev correct',  'prev choice']
STATE_COLORS = ['#ff7f00', '#4daf4a', '#377eb8', '#FF69B4', '#377eb8']


def entropy(latent_states, trials, num_states):
    arr = np.ones(num_states) / trials   # initialize with 1 in each state to avoid -inf
    for t in range(trials):
        arr[latent_states[t]] += 1/trials
    return -sum(arr * np.log2(arr))


def cond_entropy(true_latent, likely_latent, trials, num_states):
    cond_ent = np.zeros(num_states)
    for i in range(num_states):
        in_state = true_latent == i
        num_in_state = sum(in_state)
        this_state_ent = entropy(likely_latent[in_state], num_in_state, num_states)
        prob = num_in_state / trials
        cond_ent[i] = prob * this_state_ent
    return sum(cond_ent)


def make_3state_learning_weights(n_sess, **weight_params_4state):
    """
    Make generative weights for an engaged & 2 biased states on the 3-cue task.
    A 4-D array with first dim of length n_sessions is returned, representing the weights as they change over
    sessions during learning.
    """
    weights_4state = make_4state_learning_weights(n_sess, **weight_params_4state)
    return weights_4state[:, 1:, ...]


def make_4state_learning_weights(n_sess, engaged_weight=6., disengaged_weight=2., bias=2.5):
    """Same as 3-state generative weights, with a cue-1-only state prepended."""   
    frac_to_end = np.linspace(0, 1, n_sess)

    return np.stack([
        np.array([
            # cue 1 engaged state
            [[1 / ((1 / engaged_weight) + np.exp(3 - 15 * f)), 0, 0,
               -0.2, 1 / (0.25 + np.exp(3 * f - 4)), 0.2]],
            # all-cue engaged state
            [[1 / ((1 / engaged_weight) + np.exp(3 - 15 * f)),
              1 / ((1 / engaged_weight) + np.exp(5 - 15 * f)),
              1 / ((1 / engaged_weight) + np.exp(7 - 15 * f)),
              -0.2, 1 / (0.25 + np.exp(3 * f - 4)), 0.2]],
            # biased right state
            [[1 / ((1 / disengaged_weight) + np.exp(3 - 15 * f)),
              1 / ((1 / disengaged_weight) + np.exp(5 - 15 * f)),
              1 / ((1 / disengaged_weight) + np.exp(7 - 15 * f)), bias, 0.2, 1]],
            # biased left state
            [[1 / ((1 / disengaged_weight) + np.exp(3 - 15 * f)),
             1 / ((1 / disengaged_weight) + np.exp(5 - 15 * f)),
             1 / ((1 / disengaged_weight) + np.exp(7 - 15 * f)), -bias, 0.2, 1]]])
        for f in frac_to_end])


def learning_to_final_weights(learning_weights_fn):
    """Converts a function producing learning weights to one producing final weights"""
    def final_weight_fn(n_sess, **learning_wts_params):
        final_state = learning_weights_fn(2, **learning_wts_params)[[1]]  # indexing by array prevents squeezing
        return np.tile(final_state, (n_sess, 1, 1, 1))
    return final_weight_fn

make_3state_final_weights = learning_to_final_weights(make_3state_learning_weights)
make_4state_final_weights = learning_to_final_weights(make_4state_learning_weights)


def get_mean_and_ci(series_set, alpha=0.05):
    """
    Given a set of N time series, compute and return the mean
    along with 95% confidence interval using a t-distribution.
    """
    n = series_set.shape[0]
    mean = np.mean(series_set, axis=0)
    stderr = np.std(series_set, axis=0) / np.sqrt(n)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        t_interval = stats.t.interval(1 - alpha, df=n - 1, loc=mean, scale=stderr)

    interval = np.where(stderr > 0, t_interval, np.broadcast_to(mean, (2, *mean.shape)))
    return mean, interval


def plot_glm_weights(weight_mat, ax, state_labels=None, input_names=None, **plot_overrides):
    input_dim = weight_mat.shape[-1]
    if input_names is None:
        if input_dim != len(INPUT_LABELS):
            raise RuntimeError('Must provide input names for non-3-cue-task')
        input_names = INPUT_LABELS

    num_states = len(weight_mat)
    if num_states > len(STATE_COLORS):
        raise RuntimeError(f'Not enough colors specified for {num_states} states.')

    if state_labels is None:
        state_labels = [f'State {k + 1}' for k in range(num_states)]

    plot_opts = {'marker': 'o', 'ls': '-', 'lw': 1.5, **plot_overrides}

    state_lines = []
    for weights, col, label in zip(weight_mat, STATE_COLORS, state_labels):
        h_line = ax.plot(range(input_dim), weights[0], color=col, label=label, **plot_opts)
        state_lines.append(h_line)

    ax.tick_params(axis='y', labelsize=10)
    ax.set_ylabel("GLM weight", fontsize=15)
    ax.set_xlabel("Covariate", fontsize=15)
    ax.set_xticks(range(input_dim), input_names, fontsize=12, rotation=45)
    ax.axhline(y=0, color="k", alpha=0.5, ls="-", lw=0.5)
    ax.legend()
    return state_lines


def plot_glm_weight_mean_and_ci(weight_mats, ax, state_labels=None, **plot_overrides):
    """"Plot a line for the mean and an area for the 95% CI of each state's weights."""
    mean, interval = get_mean_and_ci(weight_mats, alpha=0.05)
    mean_lines = plot_glm_weights(mean, ax, state_labels=state_labels, **plot_overrides)
    interval_lower, interval_upper = interval
    input_dim = mean.shape[-1]
    for il, iu, col in zip(interval_lower, interval_upper, STATE_COLORS):
        ax.fill_between(range(input_dim), il[0], iu[0], color=col, alpha=0.3)
    return mean_lines


def plot_trans_mat(trans_mat, ax):
    ax.imshow(trans_mat, vmin=-0.8, vmax=1, cmap='bone')
    num_states = len(trans_mat)
    for i in range(num_states):
        for j in range(num_states):
            ax.text(j, i, str(np.around(trans_mat[i, j], decimals=2)), ha="center", va="center",
                    color="k", fontsize=12)
    ax.set_xlim(-0.5, num_states - 0.5)
    ax.set_xticks(range(num_states), [str(i+1) for i in range(num_states)], fontsize=10)
    ax.set_yticks(range(num_states), [str(i+1) for i in range(num_states)], fontsize=10)
    ax.set_ylim(num_states - 0.5, -0.5)
    ax.set_ylabel("State t", fontsize=15)
    ax.set_xlabel("State t+1", fontsize=15)


def make_glmhmm(num_states=3, weights=None, trans_mat=None, use_prior=False,
                # only relevant for MAP:
                prior_mean=0, prior_sigma=2, prior_alpha=2, kappa=0):
    # Set the parameters of the GLM-HMM
    obs_dim = 1  # number of observed dimensions
    num_categories = 2  # number of categories for output
    # Make a GLM-HMM

    if use_prior:
        obs_kwargs = {'C': num_categories, 'prior_mean': prior_mean, 'prior_sigma': prior_sigma}
        trans_kwargs = {'alpha': prior_alpha, 'kappa': kappa}
        glmhmm = ssm_customized.HMM(num_states, obs_dim, INPUT_DIM, observations='input_driven_obs',
                         observation_kwargs=obs_kwargs, transitions='sticky',
                         transition_kwargs=trans_kwargs)
    else:
        glmhmm = ssm_customized.HMM(num_states, obs_dim, INPUT_DIM, observations="input_driven_obs",
                         observation_kwargs=dict(C=num_categories), transitions='standard')

    if weights is not None:
        glmhmm.observations.params = weights

    if trans_mat is not None:
        glmhmm.transitions.params = [np.log(trans_mat)]

    return glmhmm


def get_weights(glmhmm):
    return glmhmm.observations.params


def get_trans_mat(glmhmm):
    return np.exp(glmhmm.transitions.params[0])


def gen_cues_and_gt_outputs_notallequal(trials_shape, prob_incorrect=None):
    """
    Make 3-cue task inputs and ground-truth outputs where the cues are never in agreement
    trials_shape (tuple or scalar) is the shape of the outputs and up to last dimension of the cues.
    If given, prob_incorrect should be a 3-element arraylike given the relative probability that each cue
    is incorrect (when normalized to sum to 1).
    """
    if isinstance(trials_shape, int):
        trials_shape = (trials_shape,)
    total_trials = np.prod(trials_shape).item()

    if prob_incorrect is None:
        prob_incorrect = np.repeat(1/3, 3)
    prob_incorrect = np.array(prob_incorrect) / sum(prob_incorrect)

    gt_resp = rng.integers(2, size=total_trials)  # 0 = left, 1 = right
    which_cue_incorrect = rng.choice(3, size=total_trials, p=prob_incorrect)
    gt_resp_neg1to1 = 2 * gt_resp - 1
    cues = np.tile(gt_resp_neg1to1[:, np.newaxis], (1, 3))
    cues[range(total_trials), which_cue_incorrect] = -gt_resp_neg1to1

    return cues.reshape(trials_shape + (3,)), gt_resp.reshape(trials_shape)


def gen_cues_and_gt_outputs_legacy(trials_shape):
    """Original settings of 3-cue task"""
    return gen_cues_and_gt_outputs_notallequal(trials_shape, [1/6, 1/3, 1/2])


def gen_cues_and_gt_outputs_independent(trials_shape):
    """
    Make 3-cue task inputs and ground-truth outputs where each cue has an independent probability of 1/2 of being
    right vs. left. Thus, 1 of 4 trials on average have all cues on the same (correct) side.
    """
    if isinstance(trials_shape, int):
        trials_shape = (trials_shape,)
    total_trials = np.prod(trials_shape).item()
    
    cues = rng.choice([-1, 1], size=(total_trials, 3))
    gt_resp = (np.sum(cues, axis=2) > 0).astype(np.double)
    
    return cues.reshape(trials_shape + (3,)), gt_resp.reshape(trials_shape)


def get_all_input_generator(cue_gen_fn: Callable[[Any], Tuple[np.ndarray, np.ndarray]], include_output=True):
    def input_generator(trials_shape, use_prev_choice=True):
        # Assumption: last axis of trials_shape iterates over trials in a session.
        cues, gt_resp = cue_gen_fn(trials_shape)
        inpts = np.concatenate([cues, np.ones(trials_shape + (3 if use_prev_choice else 2,))], axis=-1)
        # column 3 is bias - leave as ones
        # column 4 is last correct
        inpts[..., 0, 4] = 0
        inpts[..., 1:, 4] = np.choose(gt_resp[..., :-1], [-1, 1])
        # column 5 is last choice and will be set during simulation
        
        if include_output:
            return inpts, gt_resp
        else:
            return inpts
    return input_generator


def simulate_glmhmm(num_sess=45, trials_per_sess=200, p_stay=0.98, cue_gen_fn=gen_cues_and_gt_outputs_legacy,
                    weight_fn=make_3state_learning_weights, weight_params=None):
    """
    Run a generative GLM-HMM with given cue generation function and weights.
    Returns a dict of:
    - 'model': the model
    - 'inputs': the inputs (includeing bias, prev choice, etc.)
    - 'latents': simulated latent states (array over sessions)
    - 'outputs': simulated outputs (array over sessions)
    - '
    """
    if weight_params is None:
        weight_params = {}
    
    all_gen_weights = weight_fn(num_sess, **weight_params)
    num_states = all_gen_weights.shape[1]

    p_trans = 1 - p_stay
    p_each_trans = p_trans / (num_states - 1)
    gen_trans_mat = p_each_trans + np.diag(np.tile(p_stay - p_each_trans, num_states))

    true_glmhmm = make_glmhmm(num_states, trans_mat=gen_trans_mat)
    
    input_generator = get_all_input_generator(cue_gen_fn, include_output=True)
    inpts = input_generator((num_sess, trials_per_sess))[0]
    inpts = list(inpts)  # convert inpts to correct format

    # Generate a sequence of latents and choices for each session
    true_latents = []
    true_choices = []

    for sess_inpts, sess_weights in zip(inpts, all_gen_weights):
        true_glmhmm.observations.params = sess_weights
        true_z, true_y = true_glmhmm.sample(trials_per_sess, [], input=sess_inpts, rnn=False,
                                            last_choice_input_ind=-1)
        true_latents.append(true_z)
        true_choices.append(true_y)
        
    return {
        'model': true_glmhmm,
        'inputs': inpts,
        'latents': true_latents,
        'outputs': true_choices
    }

