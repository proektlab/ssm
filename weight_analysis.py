import numpy as np
import matplotlib.pyplot as plt
import sys

np.set_printoptions(threshold=sys.maxsize)

no_prior = np.load('no_prior.npz')
weights_prior = np.load('weights_prior.npz')
epochs = np.load('epochs.npz')

np_weights = no_prior['glm_weights']
np_transmat = no_prior['transition_probs']
np_mutinfo = no_prior['mut_info']

wp_weights = weights_prior.files[0]
wp_transmat = weights_prior['transition_probs']
wp_mutinfo = weights_prior['mut_info']

np_mt_mean = np.zeros(40)
wp_mt_mean = np.zeros(40)
np_mat_mean = np.zeros((40, 9))
wp_mat_mean = np.zeros((40, 9))

fig = plt.figure(figsize=(7, 3), dpi=80, facecolor='w', edgecolor='k')
start = 0
for i in range(25):
    end = start + 40
    plt.plot(range(40), np_mutinfo[start:end], marker='o', color='#ff7f00', lw=1.5)
    plt.plot(range(40), wp_mutinfo[start:end], marker='o', color='#4daf4a', lw=1.5)
    plt.xticks(np.arange(40))
    plt.title("Mutual Information", fontsize=15)
    np_mt_mean += np_mutinfo[start:end]
    wp_mt_mean += wp_mutinfo[start:end]
    start += 40
    for k in range(40):
        for j in range(9):
            np_mat_mean[k, j] += np_transmat[(i * 40 * 9) + (k * 9) + j]
            wp_mat_mean[k, j] += wp_transmat[(i * 40 * 9) + (k * 9) + j]

np_mt_mean /= 25
wp_mt_mean /= 25
np_mat_mean /= 25
wp_mat_mean /= 25

fig = plt.figure(figsize=(7, 3), dpi=80, facecolor='w', edgecolor='k')
plt.plot(range(40), np_mt_mean, marker='o', color='#ff7f00', lw=1.5, label="prior = 0")
plt.plot(range(40), wp_mt_mean, marker='o', color='#4daf4a', lw=1.5, label="prior = previous rec weights")
plt.xticks(np.arange(40))

np_mt_sd = np.zeros(40)
wp_mt_sd = np.zeros(40)

start = 0
for i in range(25):
    end = start + 40
    np_mt_sd += (np_mutinfo[start:end] - np_mt_mean[i])**2
    wp_mt_sd += (wp_mutinfo[start:end] - wp_mt_mean[i])**2
    start += 40

np_mt_sd = (np_mt_sd/40)**0.5
wp_mt_sd = (wp_mt_sd/40)**0.5

np_confi = 1.96 * (np_mt_sd/(40**0.5))
wp_confi = 1.96 * (wp_mt_sd/(40**0.5))

plt.fill_between(range(40), (np_mt_mean-np_confi), (np_mt_mean+np_confi), color='#ff7f00', alpha=0.25)
plt.fill_between(range(40), (wp_mt_mean-wp_confi), (wp_mt_mean+wp_confi), color='#4daf4a', alpha=0.25)
plt.title("Mutual Information Confidence Interval", fontsize=15)
plt.legend()

fig = plt.figure(figsize=(7, 3), dpi=80, facecolor='w', edgecolor='k')
plt.plot(range(40), np_mat_mean[:, 0], marker='o', color='#ff7f00', lw=1.5, label="prior = 0")
plt.plot(range(40), np_mat_mean[:, 4], marker='o', color='#ff7f00', lw=1.5, linestyle='--', label="(state 2)")
plt.plot(range(40), np_mat_mean[:, 8], marker='o', color='#ff7f00', lw=1.5, linestyle=':', label="(state 3)")
plt.plot(range(40), wp_mat_mean[:, 0], marker='o', color='#4daf4a', lw=1.5, label="prior = previous rec weights (state 1)")
plt.plot(range(40), wp_mat_mean[:, 4], marker='o', color='#4daf4a', lw=1.5, linestyle='--', label="(state 2)")
plt.plot(range(40), wp_mat_mean[:, 8], marker='o', color='#4daf4a', lw=1.5, linestyle=':', label="(state 3)")
plt.xticks(np.arange(40))
plt.title("Same State Transition Prob", fontsize=15)
plt.legend()

plt.show()
