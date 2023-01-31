import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import ssm_customized
from ssm_customized.util import find_permutation
import sys
import itertools
import math
import time

npr.seed(0)


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


def closest_switch(true_latent, likely_latent, trials):
    dists = []
    seq = []
    curr = true_latent[0]
    i = 1
    while i in range(trials):
        if curr != true_latent[i]:
            start = true_latent[i-1]
            change = true_latent[i]
            j = i
            k = i - 2
            while j < trials or k >= 0:
                if j < trials and likely_latent[j] == change and likely_latent[j-1] == start:
                    if i-j > -70:
                        dists.append(i-j)
                    if i-j > -4:
                        seq.append(1)
                    else:
                        seq.append(0)
                    break
                elif k >= 0 and likely_latent[k] == change and likely_latent[k+1] == start:
                    if i-k < 70:
                        dists.append(i-k)
                    if i-k < 4:
                        seq.append(1)
                    else:
                        seq.append(0)
                    break
                else:
                    j += 1
                    k -= 1
        else:
            seq.append(-1)
        i += 1
        curr = true_latent[i-1]
    return dists, seq


def glmhmm(num_states=3, obs_dim=1, num_categories=2, input_dim=6):
    """
    Make a GLM-HMM.
    Parameters
    ----------
    num_states: # of discrete states (strategies)
    obs_dim: dimensionality of responses (1 for decision data)
    num_categories: # of different possible responses
    input_dim: # of factors that may affect the decision

    Returns
    -------

    """
    t = 60  # session/window
    np.set_printoptions(threshold=sys.maxsize)

    # Make a GLM-HMM
    true_glmhmm = ssm_customized.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs",
                          observation_kwargs=dict(C=num_categories), transitions="standard")

    gen_weights = np.array([[[1 / ((1 / 6) + np.exp(3 - (t / 2))), 1 / ((1 / 6) + np.exp(5 - (t / 2))),
                              1 / ((1 / 6) + np.exp(7 - (t / 2))), -0.2, 0.2, 1 / ((1 / 4) + np.exp((t / 10) - 4))]],
                            [[1 / ((1 / 2) + np.exp(3 - (t / 2))), 1 / ((1 / 2) + np.exp(5 - (t / 2))),
                              1 / ((1 / 2) + np.exp(7 - (t / 2))), 2.5, 1, 0.2]],
                            [[1 / ((1 / 2) + np.exp(3 - (t / 2))), 1 / ((1 / 2) + np.exp(5 - (t / 2))),
                              1 / ((1 / 2) + np.exp(7 - (t / 2))), -2.5, 1, 0.2]]])

    gen_log_trans_mat = np.log(np.array([[[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]]))

    true_glmhmm.observations.params = gen_weights
    true_glmhmm.transitions.params = gen_log_trans_mat

    # Plot generative parameters:
    fig = plt.figure(figsize=(8, 3), dpi=80, facecolor='w', edgecolor='k')
    plt.subplot(1, 2, 1)
    cols = ['#ff7f00', '#4daf4a', '#377eb8', '#FF69B4']  # , '#377eb8'
    for k in range(num_states):
        plt.plot(range(input_dim), gen_weights[k][0], marker='o',
                 color=cols[k], linestyle='-',
                 lw=1.5, label="state " + str(k + 1))
    plt.yticks(fontsize=10)
    plt.ylabel("GLM weight", fontsize=15)
    plt.xlabel("covariate", fontsize=15)
    plt.xticks([0, 1, 2, 3, 4, 5], ['Cue A', 'Cue B', 'Cue C', 'bias', 'prev choice', 'prev correct'], fontsize=12,
               rotation=45)
    plt.axhline(y=0, color="k", alpha=0.5, ls="--")
    plt.legend()
    plt.title("Generative weights", fontsize=15)

    plt.subplot(1, 2, 2)
    gen_trans_mat = np.exp(gen_log_trans_mat)[0]
    plt.imshow(gen_trans_mat, vmin=-0.8, vmax=1, cmap='bone')
    for i in range(gen_trans_mat.shape[0]):
        for j in range(gen_trans_mat.shape[1]):
            text = plt.text(j, i, str(np.around(gen_trans_mat[i, j], decimals=2)), ha="center", va="center",
                            color="k", fontsize=12)
    plt.xlim(-0.5, num_states - 0.5)
    plt.xticks(range(0, num_states), ('1', '2', '3'), fontsize=10)
    plt.yticks(range(0, num_states), ('1', '2', '3'), fontsize=10)
    plt.ylim(num_states - 0.5, -0.5)
    plt.ylabel("state t", fontsize=15)
    plt.xlabel("state t+1", fontsize=15)
    plt.title("Generative transition matrix", fontsize=15)

    num_sess = 42  # number of example sessions
    num_trials_per_sess = 204  # number of trials in a session

    inpts = np.ones((num_sess, num_trials_per_sess, input_dim))  # initialize inpts array

    numSequences = num_trials_per_sess
    wrongCueNums = np.repeat([0, 1, 2], [1, 2, 3])  # frequency each cue is wrong within a 6-trial block

    gtOutputs = [1, 2]
    combinations = np.array(
        list(itertools.product(wrongCueNums, gtOutputs)))  # incorrect cue and correct direction combinations (12)
    nCombos = len(combinations)
    nRepeats = numSequences // nCombos
    assert nCombos * nRepeats == numSequences, f'Number of sequences must be divisible by {nCombos}'
    combinations = np.tile(combinations, (nRepeats, 1))

    rng = npr.default_rng()
    combinations = rng.permutation(combinations)
    wrongCueNum, gtOutput = combinations.T

    for sess in range(num_sess):
        for i in range(num_trials_per_sess):
            if gtOutput[i] == 1:
                c = -1
                w = 1
            else:
                c = 1
                w = -1
            if wrongCueNum[i] == 0:
                inpts[sess, i, 0] = w
                inpts[sess, i, 1] = c
                inpts[sess, i, 2] = c
            elif wrongCueNum[i] == 1:
                inpts[sess, i, 0] = c
                inpts[sess, i, 1] = w
                inpts[sess, i, 2] = c
            else:
                inpts[sess, i, 0] = c
                inpts[sess, i, 1] = c
                inpts[sess, i, 2] = w

        for trial in range(num_trials_per_sess - 1):
            if inpts[sess, trial, 0] + inpts[sess, trial, 1] + inpts[sess, trial, 2] < 0:
                inpts[sess, trial + 1, 5] = -1
            elif inpts[sess, trial, 0] + inpts[sess, trial, 1] + inpts[sess, trial, 2] > 0:
                inpts[sess, trial + 1, 5] = 1
    inpts = list(inpts)  # convert inpts to correct format

    # Generate a sequence of latents and choices for each session
    logLikelihoods = []
    gen_weight_arr = []
    true_latents, true_choices = [], []
    ll_mean = 0

    for sess in range(num_sess):
        t = sess
        gen_weights = np.array([[[1 / ((1 / 6) + np.exp(3 - (t / 2))), 1 / ((1 / 6) + np.exp(5 - (t / 2))),
                                  1 / ((1 / 6) + np.exp(7 - (t / 2))),
                                  -0.2, 0.2, 1 / ((1 / 4) + np.exp((t / 10) - 4))]],
                                [[1 / ((1 / 2) + np.exp(3 - (t / 2))), 1 / ((1 / 2) + np.exp(5 - (t / 2))),
                                  1 / ((1 / 2) + np.exp(7 - (t / 2))), 2.5, 1, 0.2]],
                                [[1 / ((1 / 2) + np.exp(3 - (t / 2))), 1 / ((1 / 2) + np.exp(5 - (t / 2))),
                                  1 / ((1 / 2) + np.exp(7 - (t / 2))), -2.5, 1, 0.2]]])
        true_glmhmm.observations.params = gen_weights
        gen_weight_arr.append(gen_weights)

        true_z, true_y = true_glmhmm.sample(num_trials_per_sess, [], input=inpts[sess], rnn=False)
        true_latents.append(true_z)
        true_choices.append(true_y)

        # Calculate true loglikelihood
        true_ll = true_glmhmm.log_probability(true_choices, inputs=inpts)
        ll = true_glmhmm.log_likelihood(true_choices, inputs=inpts)
        logLikelihoods.append(ll)
        ll_mean += ll
        # print("true ll = " + str(true_ll))
    '''
    print(str(logLikelihoods))
    print(ll_mean/200)
    fig = plt.figure(figsize=(6,4), dpi=80, facecolor='w', edgecolor='k')
    plt.hist(logLikelihoods, 20)
    plt.ylabel("Amount")
    plt.xlabel("Values")
    '''

    N_iters = 200  # maximum number of EM iterations. Fitting with stop earlier if increase in LL is below tolerance specified by tolerance parameter

    new_glmhmm = ssm_customized.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs",
                         observation_kwargs=dict(C=num_categories), transitions="standard")

    fit_ll = new_glmhmm.fit(true_choices, inputs=inpts, method="em", num_iters=N_iters, tolerance=10 ** -4)

    # Plot the log probabilities of the true and fit models. Fit model final LL should be greater
    # than or equal to true LL.
    fig = plt.figure(figsize=(4, 3), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(fit_ll, label="EM")
    plt.plot([0, len(fit_ll)], true_ll * np.ones(2), ':k', label="True")
    plt.legend(loc="lower right")
    plt.xlabel("EM Iteration")
    plt.xlim(0, len(fit_ll))
    plt.ylabel("Log Probability")

    new_glmhmm.permute(
        find_permutation(true_latents[0], new_glmhmm.most_likely_states(true_choices[0], input=inpts[0]), 3, 3))

    fig = plt.figure(figsize=(4, 3), dpi=80, facecolor='w', edgecolor='k')
    cols = ['#ff7f00', '#4daf4a', '#377eb8', '#FF69B4']
    recovered_weights = new_glmhmm.observations.params

    gen_weights_tot = np.zeros(num_states)
    rec_weights_tot = np.zeros(num_states)
    for k in range(num_states):
        if k == 0:
            plt.plot(range(input_dim), gen_weights[k][0], marker='o',
                     color=cols[k], linestyle='-',
                     lw=1.5, label="generative")
            plt.plot(range(input_dim), recovered_weights[k][0], color=cols[k],
                     lw=1.5, label="recovered", linestyle='--')
        else:
            plt.plot(range(input_dim), gen_weights[k][0], marker='o',
                     color=cols[k], linestyle='-',
                     lw=1.5, label="")
            plt.plot(range(input_dim), recovered_weights[k][0], color=cols[k],
                     lw=1.5, label='', linestyle='--')
        for i in range(input_dim):
            gen_weights_tot[k] += gen_weights[k][0][i]
            rec_weights_tot[k] += recovered_weights[k][0][i]

    plt.yticks(fontsize=10)
    plt.ylabel("GLM weight", fontsize=15)
    plt.xlabel("covariate", fontsize=15)
    plt.xticks([0, 1, 2, 3, 4, 5], ['Cue A', 'Cue B', 'Cue C', 'bias', 'prev choice', 'prev correct'], fontsize=12,
               rotation=45)
    plt.axhline(y=0, color="k", alpha=0.5, ls="--")
    plt.legend()
    plt.title("Weight recovery", fontsize=15)

    # find correlation between true weights and recovered weights for each state
    gen_weights_mean = gen_weights_tot / input_dim
    rec_weights_mean = rec_weights_tot / input_dim
    numerator = np.zeros(num_states)
    gen_sd = np.zeros(num_states)
    rec_sd = np.zeros(num_states)
    corr = np.zeros(num_states)

    for k in range(num_states):
        for i in range(input_dim):
            numerator[k] += (gen_weights[k][0][i] - gen_weights_mean[k]) * (
                    recovered_weights[k][0][i] - rec_weights_mean[k])
            gen_sd[k] += (gen_weights[k][0][i] - gen_weights_mean[k]) ** 2
            rec_sd[k] += (recovered_weights[k][0][i] - rec_weights_mean[k]) ** 2
        corr[k] = numerator[k] / (math.sqrt(gen_sd[k] * rec_sd[k]))
        print(corr[k])

    fig = plt.figure(figsize=(5, 2.5), dpi=80, facecolor='w', edgecolor='k')
    plt.subplot(1, 2, 1)
    gen_trans_mat = np.exp(gen_log_trans_mat)[0]
    plt.imshow(gen_trans_mat, vmin=-0.8, vmax=1, cmap='bone')
    for i in range(gen_trans_mat.shape[0]):
        for j in range(gen_trans_mat.shape[1]):
            text = plt.text(j, i, str(np.around(gen_trans_mat[i, j], decimals=2)), ha="center", va="center",
                            color="k", fontsize=12)
    plt.xlim(-0.5, num_states - 0.5)
    plt.xticks(range(0, num_states), ('1', '2', '3'), fontsize=10)
    plt.yticks(range(0, num_states), ('1', '2', '3'), fontsize=10)
    plt.ylim(num_states - 0.5, -0.5)
    plt.ylabel("state t", fontsize=15)
    plt.xlabel("state t+1", fontsize=15)
    plt.title("generative", fontsize=15)

    plt.subplot(1, 2, 2)
    recovered_trans_mat = np.exp(new_glmhmm.transitions.log_Ps)
    plt.imshow(recovered_trans_mat, vmin=-0.8, vmax=1, cmap='bone')
    for i in range(recovered_trans_mat.shape[0]):
        for j in range(recovered_trans_mat.shape[1]):
            text = plt.text(j, i, str(np.around(recovered_trans_mat[i, j], decimals=2)), ha="center", va="center",
                            color="k", fontsize=12)
    plt.xlim(-0.5, num_states - 0.5)
    plt.xticks(range(0, num_states), ('1', '2', '3'), fontsize=10)
    plt.yticks(range(0, num_states), ('1', '2', '3'), fontsize=10)
    plt.ylim(num_states - 0.5, -0.5)
    plt.title("recovered", fontsize=15)
    plt.subplots_adjust(0, 0, 1, 1)

    # Get expected states:
    posterior_probs = [new_glmhmm.expected_states(data=data, input=inpt)[0]
                       for data, inpt
                       in zip(true_choices, inpts)]

    fig = plt.figure(figsize=(5, 2.5), dpi=80, facecolor='w', edgecolor='k')
    sess_id = 0  # session id; can choose any index between 0 and num_sess-1
    for k in range(num_states):
        plt.plot(posterior_probs[sess_id][:, k], label="State " + str(k + 1), lw=2,
                 color=cols[k])
    plt.ylim((-0.01, 1.01))
    plt.yticks([0, 0.5, 1], fontsize=10)
    plt.xlabel("trial #", fontsize=15)
    plt.ylabel("p(state)", fontsize=15)

    # concatenate posterior probabilities across sessions
    posterior_probs_concat = np.concatenate(posterior_probs)
    # get state with maximum posterior probability at particular trial:
    state_max_posterior = np.argmax(posterior_probs_concat, axis=1)
    # now obtain state fractional occupancies:
    _, state_occupancies = np.unique(state_max_posterior, return_counts=True)
    state_occupancies = state_occupancies / np.sum(state_occupancies)

    fig = plt.figure(figsize=(2, 2.5), dpi=80, facecolor='w', edgecolor='k')
    for z, occ in enumerate(state_occupancies):
        plt.bar(z, occ, width=0.8, color=cols[z])
    plt.ylim((0, 1))
    plt.xticks([0, 1, 2], ['1', '2', '3'], fontsize=10)
    plt.yticks([0, 0.5, 1], ['0', '0.5', '1'], fontsize=10)
    plt.xlabel('state', fontsize=15)
    plt.ylabel('frac. occupancy', fontsize=15)

    true_likelihood = true_glmhmm.log_likelihood(true_choices, inputs=inpts)
    mle_final_ll = new_glmhmm.log_likelihood(true_choices, inputs=inpts)

    fig = plt.figure(figsize=(6, 3), dpi=80, facecolor='w', edgecolor='k')
    cols = ['#ff7f00', '#4daf4a', '#377eb8', '#FF69B4']
    plt.subplot(1, 2, 1)
    recovered_weights = new_glmhmm.observations.params
    for k in range(num_states):
        if k == 0:  # show labels only for first state
            plt.plot(range(input_dim), gen_weights[k][0], marker='o',
                     color=cols[k],
                     lw=1.5, label="generative")
            plt.plot(range(input_dim), recovered_weights[k][0], color=cols[k],
                     lw=1.5, label='recovered', linestyle='--')
        else:
            plt.plot(range(input_dim), gen_weights[k][0], marker='o',
                     color=cols[k],
                     lw=1.5, label="")
            plt.plot(range(input_dim), recovered_weights[k][0], color=cols[k],
                     lw=1.5, label='', linestyle='--')
    plt.yticks(fontsize=10)
    plt.ylabel("GLM weight", fontsize=15)
    plt.xlabel("covariate", fontsize=15)
    plt.xticks([0, 1, 2, 3, 4, 5], ['Cue A', 'Cue B', 'Cue C', 'bias', 'prev choice', 'prev correct'], fontsize=12,
               rotation=45)
    plt.axhline(y=0, color="k", alpha=0.5, ls="--")
    plt.title("MLE", fontsize=15)
    plt.legend()

    all_inpts = np.zeros((num_sess * num_trials_per_sess, input_dim))
    all_choices = np.zeros((num_sess * num_trials_per_sess, 1), dtype=int)
    all_latent = np.zeros((num_sess * num_trials_per_sess))
    all_inpts[0:204, :] = inpts[0]
    all_choices[0:204, :] = true_choices[0]
    all_latent[0:204] = true_latents[0]
    for i in range(1, num_sess):
        sess_inpts = inpts[i]
        sess_choices = true_choices[i]
        sess_latent = true_latents[i]
        for j in range(num_trials_per_sess):
            all_inpts[i * num_trials_per_sess + j, :] = sess_inpts[j]
            all_choices[i * num_trials_per_sess + j, :] = sess_choices[j]
            all_latent[i * num_trials_per_sess + j] = sess_latent[j]

    end = num_trials_per_sess * num_sess
    map_recovered_weights = recovered_weights
    prior_rec_weights = recovered_weights
    window = 1020
    windows = int(((num_trials_per_sess * num_sess) - window) / num_trials_per_sess)

    map_state_change = np.empty((windows, num_states, 1, input_dim))
    prior_state_change = np.empty((windows, num_states, 1, input_dim))
    map_transitions = np.empty((windows, num_states, num_states))
    prior_transitions = np.empty((windows, num_states, num_states))
    epochs = np.empty((windows, 2))
    base = time.time()
    map_mut_info = []
    prior_mut_info = []
    map_dists = []
    prior_dists = []
    trans_dist = []

    for w in reversed(range(windows)):
        epochs[w, 0] = time.time() - base
        start = end - window

        # Instantiate GLM-HMM and set prior hyperparameters
        prior_sigma = 1
        prior_alpha = 2
        prior_mean = recovered_weights

        map_glmhmm = ssm_customized.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs",
                             observation_kwargs=dict(C=num_categories, prior_sigma=prior_sigma, prior_mean=0),
                             transitions="sticky", transition_kwargs=dict(alpha=prior_alpha, kappa=0))

        prior_glmhmm = ssm_customized.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs",
                               observation_kwargs=dict(C=num_categories, prior_sigma=prior_sigma,
                                                       prior_mean=prior_mean),
                               transitions="sticky", transition_kwargs=dict(alpha=prior_alpha, kappa=0))

        inpt_temp = np.array(all_inpts[start:end, :], dtype=int)
        choice_temp = np.array(all_choices[start:end, :], dtype=int)
        latent_temp = np.array(all_latent[start:end], dtype=int)

        # Fit GLM-HMM with MAP estimation:
        _ = map_glmhmm.fit(choice_temp, inputs=inpt_temp, method="em", num_iters=N_iters, tolerance=10 ** -4)
        _ = prior_glmhmm.fit(choice_temp, inputs=inpt_temp, method="em", num_iters=N_iters, tolerance=10 ** -4)

        # map_final_ll = map_glmhmm.log_likelihood(true_choices, inputs=inpts)

        try:
            map_glmhmm.permute(
                find_permutation(latent_temp, map_glmhmm.most_likely_states(choice_temp, input=inpt_temp), 3, 3))

            # map mutual information
            latent_entropy = entropy(latent_temp, window, num_states)
            map_entropy = entropy(map_glmhmm.most_likely_states(choice_temp, input=inpt_temp), window, num_states)
            map_norm = map_entropy + latent_entropy
            map_mutual_info = map_entropy - (
                cond_entropy(latent_temp, map_glmhmm.most_likely_states(choice_temp, input=inpt_temp), window,
                         num_states))
            map_mut_info.append(2 * map_mutual_info / map_norm)

            dists, seqs = closest_switch(latent_temp, map_glmhmm.most_likely_states(choice_temp, input=inpt_temp), window)
            map_dists.extend(dists)
        except:
            map_mut_info.append(0)
            print("Error in map session " + str(w) + "!")

        try:
            prior_glmhmm.permute(
                find_permutation(latent_temp, prior_glmhmm.most_likely_states(choice_temp, input=inpt_temp), 3, 3))

            # prior glmhmm mutual information
            prior_entropy = entropy(prior_glmhmm.most_likely_states(choice_temp, input=inpt_temp), window, num_states)
            prior_norm = latent_entropy + prior_entropy
            prior_mutual_info = prior_entropy - (
                cond_entropy(latent_temp, prior_glmhmm.most_likely_states(choice_temp, input=inpt_temp), window,
                             num_states))
            prior_mut_info.append(2 * prior_mutual_info / prior_norm)

            prior_dist, prior_seq = closest_switch(latent_temp, prior_glmhmm.most_likely_states(choice_temp, input=inpt_temp), window)
            prior_dists.extend(prior_dist)

            for p in range(len(prior_seq)-1):
                if prior_seq[p] == 0 and prior_seq[p+1] != 0:
                    start = p
                elif prior_seq[p] != 0 and prior_seq[p+1] == 0:
                    trans_dist.append(p-start)

        except:
            prior_mut_info.append(0)
            print("Error in prior session " + str(w) + "!")
        finally:
            end -= 204

        map_recovered_weights = map_glmhmm.observations.params
        prior_rec_weights = prior_glmhmm.observations.params
        map_state_change[w, :, :, :] = map_recovered_weights
        prior_state_change[w, :, :, :] = prior_rec_weights
        map_transitions[w, :, :] = np.exp(map_glmhmm.transitions.params)[0]
        prior_transitions[w, :, :] = np.exp(prior_glmhmm.transitions.params)[0]
        recovered_weights = prior_rec_weights

        epochs[w, 1] = time.time() - base

        gen_weights_tot = np.zeros(num_states)
        rec_weights_tot = np.zeros(num_states)

        for k in range(num_states):
            plt.plot(range(input_dim), gen_weight_arr[i][k][0], marker='o',
                     color=cols[k],
                     lw=1.5, label="", linestyle='-')
            plt.plot(range(input_dim), recovered_weights[k][0], color=cols[k],
                     lw=1.5, label='', linestyle='--')

            for i in range(input_dim):
                gen_weights_tot[k] += gen_weights[k][0][i]
                rec_weights_tot[k] += recovered_weights[k][0][i]

        plt.yticks(fontsize=10)
        plt.xticks([0, 1], ['', ''], fontsize=12, rotation=45)
        plt.axhline(y=0, color="k", alpha=0.5, ls="--")
        plt.title("MAP", fontsize=15)


    for i in range(input_dim):
        fig = plt.figure(figsize=(6, 3), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(range(windows), map_state_change[:, 0, 0, i], marker='o', color=cols[0], lw=1.5, label="prior = 0",
                 linestyle='--')
        plt.plot(range(windows), prior_state_change[:, 0, 0, i], marker='o', color=cols[1], lw=1.5,
                 label="prior = previous rec weights", linestyle='--')
        plt.plot(range(windows), np.array(gen_weight_arr)[0:windows, 0, 0, i], marker='o', color=cols[2], lw=1.5,
                 label="generative", linestyle='-')
        plt.xticks(np.arange(windows))
        plt.title("Smoothness", fontsize=15)
        plt.legend()

    gen_weight_arr = np.array(gen_weight_arr)

    for w in range(windows):
        mnorm = abs(map_state_change[w, 0, 0, 0]) + abs(map_state_change[w, 0, 0, 1]) + abs(map_state_change[w, 0, 0, 2])
        map_state_change[w, 0, 0, :] /= mnorm
        pnorm = abs(prior_state_change[w, 0, 0, 0]) + abs(prior_state_change[w, 0, 0, 1]) + abs(prior_state_change[w, 0, 0, 2])
        prior_state_change[w, 0, 0, :] /= pnorm
        gnorm = abs(gen_weight_arr[w, 0, 0, 0]) + abs(gen_weight_arr[w, 0, 0, 1]) + abs(gen_weight_arr[w, 0, 0, 2])
        gen_weight_arr[w, 0, 0, :] /= gnorm

    for i in range(3):
        fig = plt.figure(figsize=(6, 3), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(range(windows), map_state_change[:, 0, 0, i], marker='o', color=cols[0], lw=1.5, label="prior = 0",
                 linestyle='--')
        plt.plot(range(windows), prior_state_change[:, 0, 0, i], marker='o', color=cols[1], lw=1.5,
                 label="prior = previous rec weights", linestyle='--')
        plt.plot(range(windows), gen_weight_arr[0:windows, 0, 0, i], marker='o', color=cols[2], lw=1.5,
                 label="generative", linestyle='-')
        plt.xticks(np.arange(windows))
        plt.title("Smoothness", fontsize=15)
        plt.legend()

    print(map_dists)
    print(prior_dists)

    fig = plt.figure(figsize=(6, 3), dpi=80, facecolor='w', edgecolor='k')
    plt.hist(map_dists, label="prior = 0", bins=40)

    fig = plt.figure(figsize=(6, 3), dpi=80, facecolor='w', edgecolor='k')
    plt.hist(prior_dists, color=cols[0], label="prior = previous rec weights", bins=40)
    plt.title("Spike Triggered Average", fontsize=15)

    fig = plt.figure(figsize=(6, 3), dpi=80, facecolor='w', edgecolor='k')
    plt.hist(trans_dist, label="prior = previous rec weights", bins=40)
    plt.title("Distance Between Missed Transitions", fontsize=15)

    fig = plt.figure(figsize=(7, 3), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(range(windows), np.flip(np.array(map_mut_info), axis=0), marker='o', color=cols[0], lw=1.5, label="prior = 0")
    plt.plot(range(windows), np.flip(np.array(prior_mut_info), axis=0), marker='o', color=cols[1], lw=1.5,
             label="prior = previous rec weights")
    plt.xticks(np.arange(windows))
    plt.title("Mutual Information", fontsize=15)
    plt.legend()
    '''
     # Plot these values
    fig = plt.figure(figsize=(2, 2.5), dpi=80, facecolor='w', edgecolor='k')
    loglikelihood_vals = [true_likelihood, mle_final_ll, map_final_ll]
    colors = ['Red', 'Navy', 'Purple']
    for z, occ in enumerate(loglikelihood_vals):
        plt.bar(z, occ, width = 0.8, color = colors[z])
    plt.ylim((true_likelihood-5, true_likelihood+25))
    plt.xticks([0, 1, 2], ['true', 'mle', 'map'], fontsize = 10)
    plt.xlabel('model', fontsize = 15)
    plt.ylabel('loglikelihood', fontsize=15)

    plt.subplot(1,2,2)
    recovered_weights = map_glmhmm.observations.params
    gen_weights_tot = np.zeros(num_states)
    rec_weights_tot = np.zeros(num_states)
    for k in range(num_states):
        plt.plot(range(input_dim), gen_weights[k][0], marker='o',
                 color=cols[k],
                 lw=1.5, label="", linestyle = '-')
        plt.plot(range(input_dim), recovered_weights[k][0], color=cols[k],
                     lw=1.5,  label = '', linestyle='--')
        for i in range(input_dim):
            gen_weights_tot[k] += gen_weights[k][0][i]
            rec_weights_tot[k] += recovered_weights[k][0][i]
    plt.yticks(fontsize=10)
    plt.xticks([0, 1], ['', ''], fontsize=12, rotation=45)
    plt.axhline(y=0, color="k", alpha=0.5, ls="--")
    plt.title("MAP", fontsize = 15)

    gen_weights_mean = gen_weights_tot / input_dim
    rec_weights_mean = rec_weights_tot / input_dim
    numerator = np.zeros(num_states)
    gen_sd = np.zeros(num_states)
    rec_sd = np.zeros(num_states)
    corr = np.zeros(num_states)

    for k in range(num_states):
        for i in range(input_dim):
            numerator[k] += (gen_weights[k][0][i] - gen_weights_mean[k]) * (
                    recovered_weights[k][0][i] - rec_weights_mean[k])
            gen_sd[k] += (gen_weights[k][0][i] - gen_weights_mean[k]) ** 2
            rec_sd[k] += (recovered_weights[k][0][i] - rec_weights_mean[k]) ** 2
        corr[k] = numerator[k] / (math.sqrt(gen_sd[k] * rec_sd[k]))
        print("MAP: " + str(corr[k]))
    '''
    fig = plt.figure(figsize=(7, 2.5), dpi=80, facecolor='w', edgecolor='k')
    plt.subplot(1, 3, 1)
    gen_trans_mat = np.exp(gen_log_trans_mat)[0]
    plt.imshow(gen_trans_mat, vmin=-0.8, vmax=1, cmap='bone')
    for i in range(gen_trans_mat.shape[0]):
        for j in range(gen_trans_mat.shape[1]):
            text = plt.text(j, i, str(np.around(gen_trans_mat[i, j], decimals=2)), ha="center", va="center",
                            color="k", fontsize=12)
    plt.xlim(-0.5, num_states - 0.5)
    plt.xticks(range(0, num_states), ('1', '2', '3'), fontsize=10)
    plt.yticks(range(0, num_states), ('1', '2', '3'), fontsize=10)
    plt.ylim(num_states - 0.5, -0.5)
    plt.ylabel("state t", fontsize=15)
    plt.xlabel("state t+1", fontsize=15)
    plt.title("generative", fontsize=15)

    plt.subplot(1, 3, 2)
    recovered_trans_mat = np.exp(new_glmhmm.transitions.log_Ps)
    plt.imshow(recovered_trans_mat, vmin=-0.8, vmax=1, cmap='bone')
    for i in range(recovered_trans_mat.shape[0]):
        for j in range(recovered_trans_mat.shape[1]):
            text = plt.text(j, i, str(np.around(recovered_trans_mat[i, j], decimals=2)), ha="center", va="center",
                            color="k", fontsize=12)
    plt.xlim(-0.5, num_states - 0.5)
    plt.xticks(range(0, num_states), ('1', '2', '3'), fontsize=10)
    plt.yticks(range(0, num_states), ('1', '2', '3'), fontsize=10)
    plt.ylim(num_states - 0.5, -0.5)
    plt.title("recovered - MLE", fontsize=15)
    plt.subplots_adjust(0, 0, 1, 1)

    plt.subplot(1, 3, 3)
    recovered_trans_mat = np.exp(map_glmhmm.transitions.log_Ps)
    plt.imshow(recovered_trans_mat, vmin=-0.8, vmax=1, cmap='bone')
    for i in range(recovered_trans_mat.shape[0]):
        for j in range(recovered_trans_mat.shape[1]):
            text = plt.text(j, i, str(np.around(recovered_trans_mat[i, j], decimals=2)), ha="center", va="center",
                            color="k", fontsize=12)
    plt.xlim(-0.5, num_states - 0.5)
    plt.xticks(range(0, num_states), ('1', '2', '3'), fontsize=10)
    plt.yticks(range(0, num_states), ('1', '2', '3'), fontsize=10)
    plt.ylim(num_states - 0.5, -0.5)
    plt.title("recovered - MAP", fontsize=15)
    plt.subplots_adjust(0, 0, 1, 1)

    # Create additional input sequences to be used as held-out test data
    num_test_sess = 1
    test_inpts = np.ones((num_test_sess, num_trials_per_sess, input_dim))

    wrongCueNums = np.repeat([0, 1, 2], [1, 2, 3])  # frequency each cue is wrong within a 6-trial block

    gtOutputs = [1, 2]
    combinations = np.array(
        list(itertools.product(wrongCueNums, gtOutputs)))  # incorrect cue and correct direction combinations (12)
    nCombos = len(combinations)
    nRepeats = numSequences // nCombos
    assert nCombos * nRepeats == numSequences, f'Number of sequences must be divisible by {nCombos}'
    combinations = np.tile(combinations, (nRepeats, 1))

    rng = npr.default_rng()
    combinations = rng.permutation(combinations)
    wrongCueNum, gtOutput = combinations.T

    for i in range(num_trials_per_sess):
        if gtOutput[i] == 1:
            c = -1
            w = 1
        else:
            c = 1
            w = -1
        if wrongCueNum[i] == 0:
            test_inpts[0, i, 0] = w
            test_inpts[0, i, 1] = c
            test_inpts[0, i, 2] = c
        elif wrongCueNum[i] == 1:
            test_inpts[0, i, 0] = c
            test_inpts[0, i, 1] = w
            test_inpts[0, i, 2] = c
        else:
            test_inpts[0, i, 0] = c
            test_inpts[0, i, 1] = c
            test_inpts[0, i, 2] = w
    for sess in range(num_test_sess):
        for trial in range(num_trials_per_sess - 1):
            if test_inpts[sess, trial, 0] + test_inpts[sess, trial, 1] + test_inpts[sess, trial, 2] < 0:
                test_inpts[sess, trial + 1, 5] = -1
            elif test_inpts[sess, trial, 0] + test_inpts[sess, trial, 1] + test_inpts[sess, trial, 2] > 0:
                test_inpts[sess, trial + 1, 5] = 1

    test_inpts = list(test_inpts)  # convert inpts to correct format

    # Create set of test latents and choices to accompany input sequences:
    test_latents, test_choices = [], []
    for sess in range(num_test_sess):
        test_z, test_y = true_glmhmm.sample(num_trials_per_sess, [], input=test_inpts[sess], rnn=False)
        test_latents.append(test_z)
        test_choices.append(test_y)

    # Compare likelihood of test_choices for model fit with MLE and MAP:
    mle_test_ll = new_glmhmm.log_likelihood(test_choices, inputs=test_inpts)
    map_test_ll = map_glmhmm.log_likelihood(test_choices, inputs=test_inpts)

    fig = plt.figure(figsize=(2, 2.5), dpi=80, facecolor='w', edgecolor='k')
    loglikelihood_vals = [mle_test_ll, map_test_ll]
    colors = ['Navy', 'Purple']
    for z, occ in enumerate(loglikelihood_vals):
        plt.bar(z, occ, width=0.8, color=colors[z])
    plt.ylim((mle_test_ll - 2, mle_test_ll + 5))
    plt.xticks([0, 1], ['mle', 'map'], fontsize=10)
    plt.xlabel('model', fontsize=15)
    plt.ylabel('loglikelihood', fontsize=15)

    plt.show()


glmhmm()
