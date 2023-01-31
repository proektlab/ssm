import numpy as np
import numpy.random as npr
import sys
import matplotlib.pyplot as plt
import ssm_customized
import json
from ssm_customized.util import find_permutation

npr.seed(0)

# Set the parameters of the GLM-HMM
num_states = 2        # number of discrete states
obs_dim = 1           # number of observed dimensions
num_categories = 2    # number of categories for output
input_dim = 6         # input dimensions
t = 60                # window
fileNum = 1

inputs = np.zeros((fileNum, 1, 3, 64, 1200))
output = np.zeros((fileNum, 1, 1200), dtype=int)
for i in range(fileNum):
    infile = open("C:\\Users\\kimliu\\Documents\\glm-hmm-main\\glm-hmm-main\\ssm\\rnn_data\\gru_0.2_inputs"+str(3)+".json", 'r')
    inputs[i] = np.array(json.load(infile))

    outfile = open("C:\\Users\\kimliu\\Documents\\glm-hmm-main\\glm-hmm-main\\ssm\\rnn_data\\gru_0.2_outputs"+str(3)+".json", 'r')
    output[i] = np.array(json.load(outfile))[:, :, 63]

output = np.expand_dims(output, axis=3)
np.set_printoptions(threshold=sys.maxsize)

# Make a GLM-HMM
true_glmhmm = ssm_customized.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs",
                   observation_kwargs=dict(C=num_categories), transitions="standard")


gen_weights = np.array([[[1, 1, 1, 1, 1, 1]], [[1, 1, 1, 2, 1, 0.1]]])
    # np.array([[[1/((2/13)+np.exp(2-(t/8))), 1/((2/13)+np.exp(4-(t/6))), 1/((2/13)+np.exp(6-(t/5))), -0.2, 0.2, 1/((1/4)+np.exp((t/10)-4))]],
      #                  [[1/((1/2)+np.exp(2-(t/8))), 1/((1/2)+np.exp(4-(t/6))), 1/((1/2)+np.exp(6-(t/5))), 2.5, 1, 0.2]],
       #                 [[1/((1/2)+np.exp(2-(t/8))), 1/((1/2)+np.exp(4-(t/6))), 1/((1/2)+np.exp(6-(t/5))), -2.5, 1, 0.2]]])
gen_log_trans_mat = np.log(np.array([[[0.98, 0.02], [0.08, 0.92]]]))

true_glmhmm.observations.params = gen_weights
true_glmhmm.transitions.params = gen_log_trans_mat

# Plot generative parameters:
fig = plt.figure(figsize=(8, 3), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(1, 2, 1)
cols = ['#ff7f00', '#4daf4a', '#377eb8', '#FF69B4'] # , '#377eb8'
for k in range(num_states):
    plt.plot(range(input_dim), gen_weights[k][0], marker='o',
             color=cols[k], linestyle='-',
             lw=1.5, label="state " + str(k+1))
plt.yticks(fontsize=10)
plt.ylabel("GLM weight", fontsize=15)
plt.xlabel("covariate", fontsize=15)
plt.xticks([0, 1, 2, 3, 4, 5], ['Cue A', 'Cue B', 'Cue C', 'bias', 'prev choice', 'prev correct'], fontsize=12, rotation=45)
plt.axhline(y=0, color="k", alpha=0.5, ls="--")
plt.legend()
plt.title("Generative weights", fontsize = 15)

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
plt.ylabel("state t", fontsize = 15)
plt.xlabel("state t+1", fontsize = 15)
plt.title("Generative transition matrix", fontsize = 15)

num_sess = 1 # number of example sessions
num_trials_per_sess = 1200 # number of trials in a session

inpts = np.ones((num_sess, num_trials_per_sess, input_dim)) # initialize inpts array
stim_vals = [-1, 1]
prev_correct_vals = [-1, 1]
for sess in range(num_sess):
    for trial in range(num_trials_per_sess-1):
        cueA = 0
        cueB = 0
        cueC = 0
        for time in range(64):
            cueA += inputs[sess, 0, 0, time, trial]
            cueB += inputs[sess, 0, 1, time, trial]
            cueC += inputs[sess, 0, 2, time, trial]

        if cueA < 0:
            inpts[sess, trial, 0] = -1
        else:
            inpts[sess, trial, 0] = 1

        if cueB < 0:
            inpts[sess, trial, 1] = -1
        else:
            inpts[sess, trial, 1] = 1

        if cueC < 0:
            inpts[sess, trial, 2] = -1
        else:
            inpts[sess, trial, 2] = 1

        if cueA + cueB + cueC < 0:
            inpts[sess, trial+1, 5] = -1
        else:
            inpts[sess, trial+1, 5] = 1

cueA = 0
cueB = 0
cueC = 0

for time in range(64):
    cueA += inputs[fileNum-1, 0, 0, time, trial]
    cueB += inputs[fileNum-1, 0, 1, time, trial]
    cueC += inputs[fileNum-1, 0, 2, time, trial]

if cueA < 0:
    inpts[sess, trial, 0] = -1
else:
    inpts[sess, trial, 0] = 1

if cueB < 0:
    inpts[sess, trial, 1] = -1
else:
    inpts[sess, trial, 1] = 1

if cueC < 0:
    inpts[sess, trial, 2] = -1
else:
    inpts[sess, trial, 2] = 1

if cueA + cueB + cueC < 0:
    inpts[sess, trial + 1, 5] = -1
else:
    inpts[sess, trial + 1, 5] = 1

inpts = list(inpts) #convert inpts to correct format

# Generate a sequence of latents and choices for each session
true_latents, true_choices = [], []
for sess in range(num_sess):
    true_z, true_y = true_glmhmm.sample(num_trials_per_sess, output[sess, 0, :, :], input=inpts[sess], rnn=True)
    true_latents.append(true_z)
    true_choices.append(true_y)
print(str(inpts))
# print(true_choices)

# Calculate true loglikelihood
true_ll = true_glmhmm.log_probability(true_choices, inputs=inpts)
print("true ll = " + str(true_ll))

new_glmhmm = ssm_customized.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs",
                   observation_kwargs=dict(C=num_categories), transitions="standard")

N_iters = 200 # maximum number of EM iterations. Fitting with stop earlier if increase in LL is below tolerance specified by tolerance parameter
fit_ll = new_glmhmm.fit(true_choices, inputs=inpts, method="em", num_iters=N_iters, tolerance=10**-4)

# Plot the log probabilities of the true and fit models. Fit model final LL should be greater
# than or equal to true LL.
fig = plt.figure(figsize=(4, 3), dpi=80, facecolor='w', edgecolor='k')
plt.plot(fit_ll, label="EM")
plt.plot([0, len(fit_ll)], true_ll * np.ones(2), ':k', label="True")
plt.legend(loc="lower right")
plt.xlabel("EM Iteration")
plt.xlim(0, len(fit_ll))
plt.ylabel("Log Probability")
# plt.show()

# print(str(new_glmhmm.most_likely_states(true_choices[0], input=inpts[0])))
new_glmhmm.permute(find_permutation(true_latents[0], new_glmhmm.most_likely_states(true_choices[0], input=inpts[0])))

fig = plt.figure(figsize=(4, 4), dpi=80, facecolor='w', edgecolor='k')
cols = ['#ff7f00', '#4daf4a', '#377eb8', '#FF69B4']
recovered_weights = new_glmhmm.observations.params
# print(str(recovered_weights))
for k in range(num_states):
    if k == 0:
        '''
        plt.plot(range(input_dim), gen_weights[k][0], marker='o',
                 color=cols[k], linestyle='-',
                 lw=1.5, label="generative")
                 '''
        plt.plot(range(input_dim), recovered_weights[k][0], color=cols[k],
                     lw=1.5,  label = "recovered", linestyle = '--')
    else:
        '''
        plt.plot(range(input_dim), gen_weights[k][0], marker='o',
                 color=cols[k], linestyle='-',
                 lw=1.5, label="")
                 '''
        plt.plot(range(input_dim), recovered_weights[k][0], color=cols[k],
                     lw=1.5,  label = '', linestyle = '--')
plt.yticks(fontsize=10)
plt.ylabel("GLM weight", fontsize=15)
plt.xlabel("covariate", fontsize=15)
plt.xticks([0, 1, 2, 3, 4, 5], ['Cue A', 'Cue B', 'Cue C', 'bias', 'prev choice', 'prev correct'], fontsize=12, rotation=45)
plt.axhline(y=0, color="k", alpha=0.5, ls="--")
plt.legend()
plt.title("Weight recovery", fontsize=15)

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
plt.ylabel("state t", fontsize = 15)
plt.xlabel("state t+1", fontsize = 15)
plt.title("generative", fontsize = 15)


plt.subplot(1, 2, 2)
recovered_trans_mat = np.exp(new_glmhmm.transitions.log_Ps)
# print(str(recovered_trans_mat))
plt.imshow(recovered_trans_mat, vmin=-0.8, vmax=1, cmap='bone')
for i in range(recovered_trans_mat.shape[0]):
    for j in range(recovered_trans_mat.shape[1]):
        text = plt.text(j, i, str(np.around(recovered_trans_mat[i, j], decimals=2)), ha="center", va="center",
                        color="k", fontsize=12)
plt.xlim(-0.5, num_states - 0.5)
plt.xticks(range(0, num_states), ('1', '2', '3'), fontsize=10)
plt.yticks(range(0, num_states), ('1', '2', '3'), fontsize=10)
plt.ylim(num_states - 0.5, -0.5)
plt.title("recovered", fontsize = 15)
plt.subplots_adjust(0, 0, 1, 1)

# Get expected states:
posterior_probs = [new_glmhmm.expected_states(data=data, input=inpt)[0]
                for data, inpt
                in zip(true_choices, inpts)]


fig = plt.figure(figsize=(5, 2.5), dpi=80, facecolor='w', edgecolor='k')
sess_id = 0 #session id; can choose any index between 0 and num_sess-1
for k in range(num_states):
    plt.plot(posterior_probs[sess_id][:, k], label="State " + str(k + 1), lw=2,
             color=cols[k])
plt.ylim((-0.01, 1.01))
plt.yticks([0, 0.5, 1], fontsize = 10)
plt.xlabel("trial #", fontsize = 15)
plt.ylabel("p(state)", fontsize = 15)

# concatenate posterior probabilities across sessions
posterior_probs_concat = np.concatenate(posterior_probs)
# get state with maximum posterior probability at particular trial:
state_max_posterior = np.argmax(posterior_probs_concat, axis = 1)
# now obtain state fractional occupancies:
_, state_occupancies = np.unique(state_max_posterior, return_counts=True)
state_occupancies = state_occupancies/np.sum(state_occupancies)

fig = plt.figure(figsize=(2, 2.5), dpi=80, facecolor='w', edgecolor='k')
for z, occ in enumerate(state_occupancies):
    plt.bar(z, occ, width = 0.8, color = cols[z])
plt.ylim((0, 1))
plt.xticks([0, 1, 2], ['1', '2', '3'], fontsize = 10)
plt.yticks([0, 0.5, 1], ['0', '0.5', '1'], fontsize=10)
plt.xlabel('state', fontsize = 15)
plt.ylabel('frac. occupancy', fontsize=15)

# Instantiate GLM-HMM and set prior hyperparameters
prior_sigma = 2
prior_alpha = 2
map_glmhmm = ssm_customized.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs",
             observation_kwargs=dict(C=num_categories,prior_sigma=prior_sigma),
             transitions="sticky", transition_kwargs=dict(alpha=prior_alpha,kappa=0))

# Fit GLM-HMM with MAP estimation:
_ = map_glmhmm.fit(true_choices, inputs=inpts, method="em", num_iters=N_iters, tolerance=10**-4)

true_likelihood = true_glmhmm.log_likelihood(true_choices, inputs=inpts)
mle_final_ll = new_glmhmm.log_likelihood(true_choices, inputs=inpts)
map_final_ll = map_glmhmm.log_likelihood(true_choices, inputs=inpts)

# Plot these values
fig = plt.figure(figsize=(2, 2.5), dpi=80, facecolor='w', edgecolor='k')
loglikelihood_vals = [true_likelihood, mle_final_ll, map_final_ll]
print(str(mle_final_ll))
print(str(map_final_ll))
colors = ['Red', 'Navy', 'Purple']
for z, occ in enumerate(loglikelihood_vals):
    plt.bar(z, occ, width = 0.8, color = colors[z])
plt.ylim((true_likelihood-5, true_likelihood+25))
plt.xticks([0, 1, 2], ['true', 'mle', 'map'], fontsize = 10)
plt.xlabel('model', fontsize = 15)
plt.ylabel('loglikelihood', fontsize=15)

map_glmhmm.permute(find_permutation(true_latents[0], map_glmhmm.most_likely_states(true_choices[0], input=inpts[0])))

fig = plt.figure(figsize=(6, 4), dpi=80, facecolor='w', edgecolor='k')
cols = ['#ff7f00', '#4daf4a', '#377eb8', '#FF69B4']
plt.subplot(1,2,1)
recovered_weights = new_glmhmm.observations.params
for k in range(num_states):
    if k ==0: # show labels only for first state
        '''
        plt.plot(range(input_dim), gen_weights[k][0], marker='o',
                 color=cols[k],
                 lw=1.5, label="generative")
                 '''
        plt.plot(range(input_dim), recovered_weights[k][0], color=cols[k],
                     lw=1.5,  label = 'recovered', linestyle='--')
    else:
        '''
        plt.plot(range(input_dim), gen_weights[k][0], marker='o',
                 color=cols[k],
                 lw=1.5, label="")
                 '''
        plt.plot(range(input_dim), recovered_weights[k][0], color=cols[k],
                     lw=1.5,  label = '', linestyle='--')
plt.yticks(fontsize=10)
plt.ylabel("GLM weight", fontsize=15)
plt.xlabel("covariate", fontsize=15)
plt.xticks([0, 1, 2, 3, 4, 5], ['Cue A', 'Cue B', 'Cue C', 'bias', 'prev choice', 'prev correct'], fontsize=12, rotation=45)
plt.axhline(y=0, color="k", alpha=0.5, ls="--")
plt.title("MLE", fontsize = 15)
plt.legend()

plt.subplot(1,2,2)
recovered_weights = map_glmhmm.observations.params
for k in range(num_states):
    '''
    plt.plot(range(input_dim), gen_weights[k][0], marker='o',
             color=cols[k],
             lw=1.5, label="", linestyle = '-')
             '''
    plt.plot(range(input_dim), recovered_weights[k][0], color=cols[k],
                 lw=1.5,  label = '', linestyle='--')
plt.yticks(fontsize=10)
plt.xticks([0, 1, 2, 3, 4, 5], ['Cue A', 'Cue B', 'Cue C', 'bias', 'prev choice', 'prev correct'], fontsize=12, rotation=45)
plt.axhline(y=0, color="k", alpha=0.5, ls="--")
plt.title("MAP", fontsize = 15)

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
plt.ylabel("state t", fontsize = 15)
plt.xlabel("state t+1", fontsize = 15)
plt.title("generative", fontsize = 15)


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
plt.title("recovered - MLE", fontsize = 15)
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
plt.title("recovered - MAP", fontsize = 15)
plt.subplots_adjust(0, 0, 1, 1)

# Create additional input sequences to be used as held-out test data
num_test_sess = 1
test_inputs = np.zeros((1, 1, 3, 64, 1200))
test_output = np.zeros((1, 1, 1200), dtype=int)
for i in range(0, 1):
    infile = open("C:\\Users\\kimliu\\Documents\\glm-hmm-main\\glm-hmm-main\\ssm\\rnn_data\\gru_0.2_inputs"+str(i)+".json", 'r')
    test_inputs[i] = np.array(json.load(infile))

    outfile = open("C:\\Users\\kimliu\\Documents\\glm-hmm-main\\glm-hmm-main\\ssm\\rnn_data\\gru_0.2_outputs"+str(i)+".json", 'r')
    test_output[i] = np.array(json.load(outfile))[:, :, 63]

test_output = np.expand_dims(test_output, axis=3)
test_inpts = np.ones((num_test_sess, num_trials_per_sess, input_dim)) # initialize inpts array
stim_vals = [-1, 1]
prev_correct_vals = [-1, 1]
for sess in range(num_test_sess):
    for trial in range(num_trials_per_sess-1):
        cueA = 0
        cueB = 0
        cueC = 0
        for time in range(64):
            cueA += test_inputs[sess, 0, 0, time, trial]
            cueB += test_inputs[sess, 0, 1, time, trial]
            cueC += test_inputs[sess, 0, 2, time, trial]

        if cueA < 0:
            test_inpts[sess, trial, 0] = -1
        else:
            test_inpts[sess, trial, 0] = 1

        if cueB < 0:
            test_inpts[sess, trial, 1] = -1
        else:
            test_inpts[sess, trial, 1] = 1

        if cueC < 0:
            test_inpts[sess, trial, 2] = -1
        else:
            test_inpts[sess, trial, 2] = 1

        if cueA + cueB + cueC < 0:
            test_inpts[sess, trial+1, 5] = -1
        else:
            test_inpts[sess, trial+1, 5] = 1

cueA = 0
cueB = 0
cueC = 0

for time in range(64):
    cueA += test_inputs[0, 0, 0, time, trial]
    cueB += test_inputs[0, 0, 1, time, trial]
    cueC += test_inputs[0, 0, 2, time, trial]

if cueA < 0:
    test_inpts[sess, trial, 0] = -1
else:
    test_inpts[sess, trial, 0] = 1

if cueB < 0:
    test_inpts[sess, trial, 1] = -1
else:
    test_inpts[sess, trial, 1] = 1

if cueC < 0:
    test_inpts[sess, trial, 2] = -1
else:
    test_inpts[sess, trial, 2] = 1

if cueA + cueB + cueC < 0:
    test_inpts[sess, trial + 1, 5] = -1
else:
    test_inpts[sess, trial + 1, 5] = 1

test_inpts = list(test_inpts) #convert inpts to correct format

# Create set of test latents and choices to accompany input sequences:
test_latents, test_choices = [], []
for sess in range(num_test_sess):
    test_z, test_y = true_glmhmm.sample(num_trials_per_sess, test_output[sess, 0, :, :], input=test_inpts[sess], rnn=True)
    test_latents.append(test_z)
    test_choices.append(test_y)

# Compare likelihood of test_choices for model fit with MLE and MAP:
mle_test_ll = new_glmhmm.log_likelihood(test_choices, inputs=test_inpts)
map_test_ll = map_glmhmm.log_likelihood(test_choices, inputs=test_inpts)

fig = plt.figure(figsize=(2, 2.5), dpi=80, facecolor='w', edgecolor='k')
loglikelihood_vals = [mle_test_ll, map_test_ll]
colors = ['Navy', 'Purple']
for z, occ in enumerate(loglikelihood_vals):
    plt.bar(z, occ, width = 0.8, color = colors[z])
plt.ylim((mle_test_ll-2, mle_test_ll+5))
plt.xticks([0, 1], ['mle', 'map'], fontsize = 10)
plt.xlabel('model', fontsize = 15)
plt.ylabel('loglikelihood', fontsize=15)

plt.show()