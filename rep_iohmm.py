import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import sys
import math
import time
import os

from ssm_customized.util import find_permutation
import ssm_customized.extensions.glm_hmm_helpers as ghh

# npr.seed(0)
rng = npr.default_rng()


def main(sessions_per_window=5, display=False, **sim_params):
    np.set_printoptions(threshold=sys.maxsize)
    
    # extract some parameters to reuse
    num_sess = sim_params['num_sess']
    trials_per_sess = sim_params['trials_per_sess']
    
    # simulate model to make ground truth
    sim_res = ghh.simulate_glmhmm(**sim_params)
    true_glmhmm = sim_res['model']
    inpts = sim_res['inputs']
    true_latents = sim_res['latents']
    true_choices = sim_res['outputs']
    
    last_gen_weights = ghh.get_weights(true_glmhmm)
    gen_trans_mat = ghh.get_trans_mat(true_glmhmm)
    num_states = true_glmhmm.K

    # Plot generative parameters after learning:
    if display:
        fig = plt.figure(figsize=(8, 3), dpi=80, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1, 2, 1)
        ghh.plot_glm_weights(last_gen_weights, ax)
        ax.set_title("Final generative weights", fontsize=15)

        ax = fig.add_subplot(1, 2, 2)
        ghh.plot_trans_mat(gen_trans_mat, ax)
        ax.set_title("Generative transition matrix", fontsize=15)

    N_iters = 200  # maximum number of EM iterations. Fitting with stop earlier if increase in LL is below tolerance specified by tolerance parameter

    new_glmhmm = ghh.make_glmhmm(num_states)
    # fit_ll = new_glmhmm.fit(true_choices, inputs=inpts, method="em", num_iters=N_iters, tolerance=10 ** -4)

    # # Plot the log probabilities of the true and fit models. Fit model final LL should be greater
    # # than or equal to true LL.
    # plt.figure(figsize=(4, 3), dpi=80, facecolor='w', edgecolor='k')
    # plt.plot(fit_ll, label="EM")
    # plt.plot([0, len(fit_ll)], true_ll * np.ones(2), ':k', label="True")
    # plt.legend(loc="lower right")
    # plt.xlabel("EM Iteration")
    # plt.xlim(0, len(fit_ll))
    # plt.ylabel("Log Probability")

    new_glmhmm.permute(
        find_permutation(true_latents[0], new_glmhmm.most_likely_states(true_choices[0], input=inpts[0]),
                         num_states, num_states)
    )

    if display:
        plt.figure(figsize=(4, 3), dpi=80, facecolor='w', edgecolor='k')
    recovered_weights = new_glmhmm.observations.params

    gen_weights_tot = np.zeros(num_states)
    rec_weights_tot = np.zeros(num_states)
    for k, col in zip(range(num_states), ghh.STATE_COLORS):
        if display:
            if k == 0:
                plt.plot(range(ghh.INPUT_DIM), last_gen_weights[k][0], marker='o',
                         color=col, linestyle='-',
                         lw=1.5, label="generative")
                plt.plot(range(ghh.INPUT_DIM), recovered_weights[k][0], color=col,
                         lw=1.5, label="recovered", linestyle='--')
            else:
                plt.plot(range(ghh.INPUT_DIM), last_gen_weights[k][0], marker='o',
                         color=col, linestyle='-',
                         lw=1.5, label="")
                plt.plot(range(ghh.INPUT_DIM), recovered_weights[k][0], color=col,
                         lw=1.5, label='', linestyle='--')
        for i in range(ghh.INPUT_DIM):
            gen_weights_tot[k] += last_gen_weights[k][0][i]
            rec_weights_tot[k] += recovered_weights[k][0][i]

    if display:
        plt.yticks(fontsize=10)
        plt.ylabel("GLM weight", fontsize=15)
        plt.xlabel("covariate", fontsize=15)
        plt.xticks(range(ghh.INPUT_DIM), ghh.INPUT_LABELS, fontsize=12,
                   rotation=45)
        plt.axhline(y=0, color="k", alpha=0.5, ls="--")
        plt.legend()
        plt.title("Weight recovery (parent model)", fontsize=15)

    # find correlation between true weights and recovered weights for each state
    gen_weights_mean = gen_weights_tot / ghh.INPUT_DIM
    rec_weights_mean = rec_weights_tot / ghh.INPUT_DIM
    numerator = np.zeros(num_states)
    gen_sd = np.zeros(num_states)
    rec_sd = np.zeros(num_states)
    corr = np.zeros(num_states)

    for k in range(num_states):
        for i in range(ghh.INPUT_DIM):
            numerator[k] += (last_gen_weights[k][0][i] - gen_weights_mean[k]) * (
                    recovered_weights[k][0][i] - rec_weights_mean[k])
            gen_sd[k] += (last_gen_weights[k][0][i] - gen_weights_mean[k]) ** 2
            rec_sd[k] += (recovered_weights[k][0][i] - rec_weights_mean[k]) ** 2
        corr[k] = numerator[k] / (math.sqrt(gen_sd[k] * rec_sd[k]))
        print(corr[k])

    if display:
        fig = plt.figure(figsize=(5, 2.5), dpi=80, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1, 2, 1)
        ghh.plot_trans_mat(gen_trans_mat, ax)
        ax.set_title('Generative', fontsize=15)

        ax = fig.add_subplot(1, 2, 2)
        recovered_trans_mat = np.exp(new_glmhmm.transitions.log_Ps)
        ghh.plot_trans_mat(recovered_trans_mat, ax)
        ax.set_title("Recovered", fontsize=15)
        fig.subplots_adjust(0, 0, 1, 1)

    # Get expected states:
    posterior_probs = [new_glmhmm.expected_states(data=data, input=inpt)[0]
                       for data, inpt
                       in zip(true_choices, inpts)]

    if display:
        plt.figure(figsize=(5, 2.5), dpi=80, facecolor='w', edgecolor='k')
        sess_id = 0  # session id; can choose any index between 0 and num_sess-1
        for k, col in zip(range(num_states), ghh.STATE_COLORS):
            plt.plot(posterior_probs[sess_id][:, k], label="State " + str(k + 1), lw=2, color=col)
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

    if display:
        plt.figure(figsize=(2, 2.5), dpi=80, facecolor='w', edgecolor='k')
        for z, (occ, col) in enumerate(zip(state_occupancies, ghh.STATE_COLORS)):
            plt.bar(z, occ, width=0.8, color=col)
        plt.ylim((0, 1))
        plt.xticks([0, 1, 2], ['1', '2', '3'], fontsize=10)
        plt.yticks([0, 0.5, 1], ['0', '0.5', '1'], fontsize=10)
        plt.xlabel('state', fontsize=15)
        plt.ylabel('frac. occupancy', fontsize=15)

    all_inpts = np.concatenate(inpts)
    all_choices = np.concatenate(true_choices)
    all_latent = np.concatenate(true_latents)

    end = trials_per_sess * num_sess
    trials_per_window = trials_per_sess * sessions_per_window
    windows = int(((trials_per_sess * num_sess) - trials_per_window) / trials_per_sess)

    map_state_change = np.empty((windows, num_states, 1, ghh.INPUT_DIM))
    prior_state_change = np.empty((windows, num_states, 1, ghh.INPUT_DIM))
    map_transitions = np.empty((windows, num_states, num_states))
    prior_transitions = np.empty((windows, num_states, num_states))
    epochs = np.empty((windows, 2))
    base = time.time()
    map_mut_info = []
    prior_mut_info = []

    for w in reversed(range(windows)):
        epochs[w, 0] = time.time() - base
        start = end - trials_per_window

        # Instantiate GLM-HMM and set prior hyperparameters
        prior_sigma = 1 if w < windows-1 else 2  # use wider prior for first one
        prior_alpha = 2
        prior_mean = recovered_weights

        map_glmhmm = ghh.make_glmhmm(num_states, use_prior=True, prior_sigma=prior_sigma, prior_alpha=prior_alpha)
        prior_glmhmm = ghh.make_glmhmm(num_states, use_prior=True, prior_mean=prior_mean, prior_sigma=prior_sigma,
                                       prior_alpha=prior_alpha)

        inpt_temp = np.array(all_inpts[start:end, :], dtype=int)
        choice_temp = np.array(all_choices[start:end, :], dtype=int)
        latent_temp = np.array(all_latent[start:end], dtype=int)

        # Fit GLM-HMM with MAP estimation:
        _ = map_glmhmm.fit(choice_temp, inputs=inpt_temp, method="em", num_iters=N_iters, tolerance=10 ** -4)
        _ = prior_glmhmm.fit(choice_temp, inputs=inpt_temp, method="em", num_iters=N_iters, tolerance=10 ** -4)

        # map_final_ll = map_glmhmm.log_likelihood(true_choices, inputs=inpts)
        # try:
        map_glmhmm.permute(
            find_permutation(latent_temp, map_glmhmm.most_likely_states(choice_temp, input=inpt_temp),
                             num_states, num_states))

        # map mutual information
        latent_entropy = ghh.entropy(latent_temp, trials_per_window, num_states)
        map_entropy = ghh.entropy(map_glmhmm.most_likely_states(choice_temp, input=inpt_temp), trials_per_window, num_states)
        map_norm = map_entropy + latent_entropy
        map_mutual_info = map_entropy - (ghh.cond_entropy(latent_temp, map_glmhmm.most_likely_states(choice_temp, input=inpt_temp), trials_per_window, num_states))
        map_mut_info.append(2 * map_mutual_info / map_norm)
        #
        # except:
        #     map_mut_info.append(0)
        #     print("Error in map session " + str(w) + "!")

        # try:
        prior_glmhmm.permute(
            find_permutation(latent_temp, prior_glmhmm.most_likely_states(choice_temp, input=inpt_temp),
                             num_states, num_states))

        # prior glmhmm mutual information
        prior_entropy = ghh.entropy(prior_glmhmm.most_likely_states(choice_temp, input=inpt_temp), trials_per_window, num_states)
        prior_norm = latent_entropy + prior_entropy
        prior_mutual_info = prior_entropy - (ghh.cond_entropy(latent_temp, prior_glmhmm.most_likely_states(choice_temp, input=inpt_temp), trials_per_window, num_states))
        prior_mut_info.append(2 * prior_mutual_info / prior_norm)

        # except:
        #     prior_mut_info.append(0)
        #     print("Error in prior session " + str(w) + "!")

        end -= trials_per_sess

        map_state_change[w, :, :, :] = map_glmhmm.observations.params
        prior_state_change[w, :, :, :] = prior_glmhmm.observations.params
        map_transitions[w, :, :] = np.exp(map_glmhmm.transitions.params)[0]
        prior_transitions[w, :, :] = np.exp(prior_glmhmm.transitions.params)[0]
        recovered_weights = prior_glmhmm.observations.params

        epochs[w, 1] = time.time() - base

        '''
        gen_weights_tot = np.zeros(num_states)
        rec_weights_tot = np.zeros(num_states)
    
        for k, col in zip(range(num_states), STATE_COLORS):
            plt.plot(range(INPUT_DIM), gen_weight_arr[i][k][0], marker='o',
                     color=col,
                     lw=1.5, label="", linestyle='-')
            plt.plot(range(INPUT_DIM), recovered_weights[k][0], color=col,
                     lw=1.5, label='', linestyle='--')
    
            for i in range(INPUT_DIM):
                gen_weights_tot[k] += gen_weights[k][0][i]
                rec_weights_tot[k] += recovered_weights[k][0][i]
            
        plt.yticks(fontsize=10)
        plt.xticks([0, 1], ['', ''], fontsize=12, rotation=45)
        plt.axhline(y=0, color="k", alpha=0.5, ls="--")
        plt.title("MAP", fontsize=15)
        

    for i in range(INPUT_DIM):
        fig = plt.figure(figsize=(6, 3), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(range(windows), map_state_change[:, 0, 0, i], marker='o', color=STATE_COLORS[0], lw=1.5, label="prior = 0",
                 linestyle='--')
        plt.plot(range(windows), prior_state_change[:, 0, 0, i], marker='o', color=STATE_COLORS[1], lw=1.5,
                 label="prior = previous rec weights", linestyle='--')
        plt.plot(range(windows), np.array(gen_weight_arr)[0:windows, 0, 0, i], marker='o', color=STATE_COLORS[2], lw=1.5,
                 label="generative", linestyle='-')
        plt.xticks(np.arange(windows))
        plt.title("Smoothness", fontsize=15)
        plt.legend()

    fig = plt.figure(figsize=(7, 3), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(range(windows), np.flip(np.array(map_mut_info), axis=0), marker='o', color=STATE_COLORS[0], lw=1.5, label="prior = 0")
    plt.plot(range(windows), np.flip(np.array(prior_mut_info), axis=0), marker='o', color=STATE_COLORS[1], lw=1.5,
             label="prior = previous rec weights")
    plt.xticks(np.arange(windows))
    plt.title("Mutual Information", fontsize=15)
    plt.legend()

    '''
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
    for k, col in zip(range(num_states), STATE_COLORS):
        plt.plot(range(INPUT_DIM), gen_weights[k][0], marker='o', color=col,
                 lw=1.5, label="", linestyle = '-')
        plt.plot(range(INPUT_DIM), recovered_weights[k][0], color=col,
                     lw=1.5,  label = '', linestyle='--')
        for i in range(INPUT_DIM):
            gen_weights_tot[k] += gen_weights[k][0][i]
            rec_weights_tot[k] += recovered_weights[k][0][i]
    plt.yticks(fontsize=10)
    plt.xticks([0, 1], ['', ''], fontsize=12, rotation=45)
    plt.axhline(y=0, color="k", alpha=0.5, ls="--")
    plt.title("MAP", fontsize = 15)
    
    gen_weights_mean = gen_weights_tot / INPUT_DIM
    rec_weights_mean = rec_weights_tot / INPUT_DIM
    numerator = np.zeros(num_states)
    gen_sd = np.zeros(num_states)
    rec_sd = np.zeros(num_states)
    corr = np.zeros(num_states)
    
    for k in range(num_states):
        for i in range(INPUT_DIM):
            numerator[k] += (gen_weights[k][0][i] - gen_weights_mean[k]) * (
                    recovered_weights[k][0][i] - rec_weights_mean[k])
            gen_sd[k] += (gen_weights[k][0][i] - gen_weights_mean[k]) ** 2
            rec_sd[k] += (recovered_weights[k][0][i] - rec_weights_mean[k]) ** 2
        corr[k] = numerator[k] / (math.sqrt(gen_sd[k] * rec_sd[k]))
        print("MAP: " + str(corr[k]))
    
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
    '''
    # Create additional input sequences to be used as held-out test data
    # num_test_sess = 1
    # test_inpts = np.ones((num_test_sess, trials_per_sess, INPUT_DIM))
    #
    # wrongCueNums = np.repeat([0, 1, 2], [1, 2, 3])  # frequency each cue is wrong within a 6-trial block
    #
    # gtOutputs = [1, 2]
    # combinations = np.array(
    #     list(itertools.product(wrongCueNums, gtOutputs)))  # incorrect cue and correct direction combinations (12)
    # nCombos = len(combinations)
    # nRepeats = numSequences // nCombos
    # assert nCombos * nRepeats == numSequences, f'Number of sequences must be divisible by {nCombos}'
    # combinations = np.tile(combinations, (nRepeats, 1))
    #
    # rng = npr.default_rng()
    # combinations = rng.permutation(combinations)
    # wrongCueNum, gtOutput = combinations.T
    #
    # for i in range(trials_per_sess):
    #     if gtOutput[i] == 1:
    #         c = -1
    #         w = 1
    #     else:
    #         c = 1
    #         w = -1
    #     if wrongCueNum[i] == 0:
    #         test_inpts[0, i, 0] = w
    #         test_inpts[0, i, 1] = c
    #         test_inpts[0, i, 2] = c
    #     elif wrongCueNum[i] == 1:
    #         test_inpts[0, i, 0] = c
    #         test_inpts[0, i, 1] = w
    #         test_inpts[0, i, 2] = c
    #     else:
    #         test_inpts[0, i, 0] = c
    #         test_inpts[0, i, 1] = c
    #         test_inpts[0, i, 2] = w
    # for sess in range(num_test_sess):
    #     for trial in range(trials_per_sess - 1):
    #         if test_inpts[sess, trial, 0] + test_inpts[sess, trial, 1] + test_inpts[sess, trial, 2] < 0:
    #             test_inpts[sess, trial + 1, 5] = -1
    #         elif test_inpts[sess, trial, 0] + test_inpts[sess, trial, 1] + test_inpts[sess, trial, 2] > 0:
    #             test_inpts[sess, trial + 1, 5] = 1
    #
    # test_inpts = list(test_inpts)  # convert inpts to correct format
    #
    # # Create set of test latents and choices to accompany input sequences:
    # test_latents, test_choices = [], []
    # for sess in range(num_test_sess):
    #     test_z, test_y = true_glmhmm.sample(trials_per_sess, [], input=test_inpts[sess], rnn=False)
    #     test_latents.append(test_z)
    #     test_choices.append(test_y)
    #
    # # Compare likelihood of test_choices for model fit with MLE and MAP:
    # mle_test_ll = new_glmhmm.log_likelihood(test_choices, inputs=test_inpts)
    # map_test_ll = map_glmhmm.log_likelihood(test_choices, inputs=test_inpts)
    #
    # fig = plt.figure(figsize=(2, 2.5), dpi=80, facecolor='w', edgecolor='k')
    # loglikelihood_vals = [mle_test_ll, map_test_ll]
    # colors = ['Navy', 'Purple']
    # for z, occ in enumerate(loglikelihood_vals):
    #     plt.bar(z, occ, width=0.8, color=colors[z])
    # plt.ylim((mle_test_ll - 2, mle_test_ll + 5))
    # plt.xticks([0, 1], ['mle', 'map'], fontsize=10)
    # plt.xlabel('model', fontsize=15)
    # plt.ylabel('loglikelihood', fontsize=15)
    #
    # plt.close('all')
    # # plt.show()

    return (map_state_change, prior_state_change, map_transitions,
            prior_transitions, np.flip(np.array(map_mut_info)), np.flip(np.array(prior_mut_info)), epochs)


def multi_runs(runs=25, savedir='Z:/kimliu/ssm', **kwargs):
    map_weights = []
    prior_weights = []
    map_trans = []
    prior_trans = []
    map_mut_info = []
    prior_mut_info = []
    all_epochs = []

    for i in range(runs):
        map_recovered_weights, prior_rec_weights, map_transitions, prior_transitions, map_mi, prior_mi, epochs = main(**kwargs)
        all_epochs.append(epochs)
        map_weights.append(map_recovered_weights)
        prior_weights.append(prior_rec_weights)
        map_trans.append(map_transitions)
        map_mut_info.append(map_mi)
        prior_mut_info.append(prior_mi)
        prior_trans.append(prior_transitions)

    np.savez(os.path.join(savedir, 'no_prior.npz'), glm_weights=map_weights, transition_probs=map_trans, mut_info=map_mut_info)
    np.savez(os.path.join(savedir, 'weights_prior.npz'), glm_weights=prior_weights, transition_probs=prior_trans, mut_info=prior_mut_info)
    np.savez(os.path.join(savedir, 'epochs.npz'), epochs=all_epochs)


if __name__ == '__main__':
    multi_runs()
