import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio
import os
import pdb

n_1 = 5
n_2 = 10

np.random.seed(1)
torch.manual_seed(1)
eps = 1E-9
scaling = 0.5
log_p_C = torch.tensor(
    np.log(np.random.dirichlet(np.ones(n_1) * scaling) + eps))
log_p_C -= log_p_C.logsumexp(-1, keepdim=True)
log_p_i = torch.tensor(np.log(np.random.dirichlet(
    np.ones(n_2) * scaling, size=n_1) + eps))
log_p_i -= log_p_i.logsumexp(-1, keepdim=True)

dir_name = f'results/{n_1}_{n_2}_05'
try:
    os.makedirs(dir_name)
except FileExistsError:
    pass

fig_1 = plt.figure(1, figsize=(10, 10))
fig_2 = plt.figure(2, figsize=(10, 10))
kl_lists = []
var_lists = []

K_list = [10, 10, 10, 10, 10, 10]
# K_list = [1, 1, 1, 1, 1, 1]

base_baseline = 0.0005
# S_list = [1, 1, 1, 1, 1, 1]
# Z_list = [10, 10, 10, 10, 10, 10]
# lrs = [base_baseline, base_baseline/2, base_baseline/4,
#        base_baseline/8, base_baseline/16, base_baseline/32]

base_mix = 0.1
# S_list = [10, 10, 10, 10, 10, 10]
# Z_list = [1, 1, 1, 1, 1, 1]
# lrs = [base_mix, base_mix/2, base_mix/4, base_mix/8, base_mix/16, base_mix/32]


lrs = [base_baseline, base_baseline/2,
       base_baseline/4, base_mix, base_mix/2, base_mix/4]
S_list = [1, 1, 1, 10, 10, 10]
Z_list = [10, 10, 10, 1, 1, 1]

use_lr_schedule = [True, True, True, True, True, True]
lr_schduling_after = 10000

np.save(dir_name + '/learning_rates', lrs)


def get_matrix_index(Ci):
    c = Ci // n_2
    i = Ci % n_2
    return c, i


def sample_from_qi(log_q_i, s):
    Ci = torch.multinomial(
        log_q_i[s].flatten().exp(), 1, replacement=True)
    Ci, i = get_matrix_index(Ci)
    return Ci, i


def sample_iw_variance(log_q_C, log_q_i, Z, S, K):

    log_fs = torch.zeros(Z, S, K)

    for z in range(Z):
        C = torch.multinomial(log_q_C.exp(), K, replacement=True)
        i = torch.multinomial(log_q_i.exp(), K, replacement=True)
        for s in range(S):
            log_q = torch.logsumexp(
                log_q_C[:, C[s]] + log_q_i[:, i[s]], dim=0) - np.log(S)

            log_p = log_p_C[C[s]] + log_p_i[C[s], i[s]]

            log_f = log_p - log_q
            log_fs[z, s] = log_f

    # log_fs_sum = torch.logsumexp(log_fs, dim=(0, 1), keepdim=True)
    # log_fs_norm = (log_fs - log_fs_sum)
    # out = log_fs_norm.exp().var(dim=(0,1)).mean()
    # return out.detach().numpy()

    return log_fs.detach().exp().var().numpy()


print(list(zip(S_list, Z_list, K_list, lrs, use_lr_schedule)))
for S, Z, K, lr, lr_scheduling in zip(S_list, Z_list, K_list, lrs, use_lr_schedule):
    np.random.seed(0)
    torch.manual_seed(0)

    phi_C = torch.tensor(np.random.normal(
        0, 0.1, (S, n_1)), requires_grad=True)
    phi_i = torch.tensor(np.random.normal(
        0, 0.1, (S, n_2)), requires_grad=True)
    opt = torch.optim.SGD(params=[phi_C] + [phi_i], lr=lr)
    lr_schduler = torch.optim.lr_scheduler.StepLR(
        opt, step_size=10000, gamma=0.9)
    # K = K_tot # // S
    K_list.append(K)
    print(f"S = {S}, K = {K}, Z = {Z}, lr = {lr}")

    kl_list = []
    iw_var_list = []
    fig_names = []
    for iter_ in range(60001):

        loss = 0
        for z in range(Z):
            log_q_C = (phi_C - torch.logsumexp(phi_C, dim=-1, keepdim=True))
            log_q_i = (phi_i - torch.logsumexp(phi_i, dim=-1, keepdim=True))

            C = torch.multinomial(log_q_C.exp(), K, replacement=True)
            i = torch.multinomial(log_q_i.exp(), K, replacement=True)
            L_hat = 0
            for s in range(S):

                log_q_s = log_q_C[s, C[s]] + log_q_i[s, i[s]]
                log_q = torch.logsumexp(
                    log_q_C[:, C[s]] + log_q_i[:, i[s]], dim=0) - np.log(S)

                log_p = log_p_C[C[s]] + log_p_i[C[s], i[s]]

                log_f = log_p - log_q
                L_hat_s = torch.logsumexp(log_p - log_q, dim=-1) - np.log(K)
                L_hat += L_hat_s / S
                if K > 1:
                    mean_exclude_signal = (torch.sum(log_f) - log_f) / (K-1.)
                    control_variates = torch.logsumexp(
                        log_f.view(-1, 1).repeat(1, K) - log_f.diag() + mean_exclude_signal.diag() - np.log(K), dim=0)
                    loss += - \
                        torch.sum((L_hat_s - control_variates).detach()
                                  * log_q_s - log_q, dim=0) / S
                else:
                    # VIMCO is not applicable
                    loss += -torch.sum(L_hat_s.detach() *
                                       log_q_s - log_q, dim=0) / S
        loss = loss / Z

        # plot before parameter update
        if (iter_ % 2000) == 0:
            if iter_ > 0:
                print("Iter: ", iter_, "KL = ",
                      kl_list[-1].item(), "IW-var=", iw_var_list[-1], "lr=", lr_schduler.get_last_lr())

            # plt.figure(2)
            # max_ = np.max(log_p_C.exp().numpy()) + 0.2
            # plt.subplot(311)
            # plt.ylim(0, max_)
            # plt.bar(np.arange(n_1), log_p_C.exp().detach().numpy(), color='r')
            # plt.subplot(312)
            # plt.ylim(0, max_)
            # for s in range(S):
            #     plt.bar(np.arange(n_1), log_q_C[s].exp(
            #     ).detach().numpy() / S, alpha=0.5)
            # plt.subplot(313)
            # plt.ylim(0, max_)
            # plt.bar(np.arange(n_1), np.sum(log_q_C.exp().detach().numpy(), axis=0) / S, color='Black', align='edge',
            #         width=0.4)
            # plt.bar(np.arange(n_1), log_p_C.exp().detach().numpy(),
            #         color='r', align='edge', width=-0.4)
            # plt.title(f"{iter_}")
            # for k in range(1, 5):
            #     # five frames per batch
            #     filename = f"figs/{iter_ + k}_{S}_{lr}.png"
            #     fig_names.append(filename)
            #     plt.savefig(filename)
            # plt.close()

        loss.backward()
        # take step
        opt.step()
        # reset gradients
        opt.zero_grad()

        with torch.no_grad():

            kl = log_p_C.exp() @ torch.einsum('ci, ci -> c', log_p_i.exp(),
                                              (log_p_C.view((n_1, 1)) + log_p_i -
                                               torch.logsumexp(log_q_C.view((S, n_1, 1)) + log_q_i.view((S, 1, n_2)) - np.log(S), dim=0))
                                              )

            var = sample_iw_variance(log_q_C, log_q_i, Z, S, 1000)

        kl_list.append(kl)
        iw_var_list.append(var)

        if lr_scheduling:
            lr_schduler.step()

    np.save(dir_name + f"/upper_dist_{S}", phi_C.detach().numpy())
    np.save(dir_name + f"/lower_dist_{S}", phi_i.detach().numpy())
    kl_lists.append(kl_list)
    var_lists.append(iw_var_list)
    # with imageio.get_writer(f'S_{S}_lr_init={lr}.gif', mode='I') as writer:
    #     for filename in fig_names:
    #         image = imageio.v2.imread(filename)
    #         writer.append_data(image)
    #         os.remove(filename)

np.save(dir_name + "/var_lists", var_lists)
np.save(dir_name + "/kl_lists", kl_lists)
np.save(dir_name + "/K_list", K_list)

var_lists = np.load(dir_name + "/var_lists.npy")
kl_lists = np.load(dir_name + "/kl_lists.npy")
K_lists = np.load(dir_name + "/K_list.npy")


plt.figure(1, figsize=(10, 10))
plt.rcParams.update({'font.size': 20})
# plt.axis([0.0, 20000, 0.0, 0.05])

# test
for i, s in enumerate(S_list):
    plt.plot(var_lists[i], label=f"$S$={s} ($K$={K_list[i]}, $Z$={
             Z_list[i]}, lr_init={lrs[i]})", alpha=0.5)
plt.title(f'n_1={n_1}, n_2={n_2}')
plt.legend(loc='best')
plt.savefig(f'{n_1}_{n_2}_05_seed_1_iw_curves')
plt.show()
plt.close()
# test


for i, s in enumerate(S_list):
    plt.plot(kl_lists[i], label=f"$S$={s} ($K$={K_list[i]}, $Z$={
             Z_list[i]}, lr_init={lrs[i]})", alpha=0.5)
plt.title(f'n_1={n_1}, n_2={n_2}')
plt.legend(loc='best')
plt.savefig(f'{n_1}_{n_2}_05_seed_1_kl_curves')
plt.show()
plt.close()


# roll = 1000
# kl_lists_rolling_std = []
# for s in range(len(S_list)):
#     kl_ = torch.stack(kl_lists[s])
#     kl_lists_rolling_ = []
#     for i in range(20001-roll):
#         kl_lists_rolling_.append(
#             torch.std(kl_[i:i+roll]).item())
#     kl_lists_rolling_std.append(kl_lists_rolling_)
# plt.figure(1, figsize=(10, 10))
# # plt.axis([0.0, 20000, 0.0, 0.02])
# print("len", len(kl_lists_rolling_std[0]))
# for i, s in enumerate(S_list):
#     plt.plot(np.arange(1000, 20001, 1), kl_lists_rolling_std[i], label=f"S={
#              s} (K={K_list[i]}, Z={Z_list[i]}, lr_init={lrs[i]})", alpha=0.5)
# plt.title(f'Rolling({roll}) STD - n_1={n_1}, n_2={n_2}')
# plt.legend(loc='best')
# plt.savefig(f'{n_1}_{n_2}_05_seed_1_kl_curves_std')
# plt.show()
