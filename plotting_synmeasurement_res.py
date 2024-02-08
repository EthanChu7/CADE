import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('PDF')
# matplotlib.rcParams['text.usetex'] = True

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


import matplotlib.pyplot as plt

df_lin_lin = pd.read_excel('lin_lin.xlsx').values
df_lin_pgd_lin = pd.read_excel('lin_pdg_lin.xlsx').values
df_mlp_lin = pd.read_excel('mlp_lin.xlsx').values
df_mlp_pgd_lin = pd.read_excel('mlp_pgd_lin.xlsx').values

df_lin_mlp = pd.read_excel('lin_mlp.xlsx').values
df_lin_pgd_mlp = pd.read_excel('lin_pdg_mlp.xlsx').values
df_mlp_mlp = pd.read_excel('mlp_mlp.xlsx').values
df_mlp_pgd_mlp = pd.read_excel('mlp_pgd_mlp.xlsx').values

x = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
fig = plt.figure(figsize=(20, 3))

font_size = 19
tick_size = 15
legend_size = 16

labels = ['CP(i)', 'CP(p)', 'C1(i)', 'C1(p)', 'C1+C2(i)', 'C1+C2(p)']
markers = ['o', 'd', 's', 'p', 'h', 'v']
# print(df_lin_lin[1, 1:])

plt.subplot(181)
plt.tick_params(labelsize=tick_size)
plt.title('Linear(white-box)', fontdict={'fontsize': font_size})
plt.xlabel('$\epsilon$', fontdict={'fontsize': font_size})
plt.ylabel('RMSE', fontdict={'fontsize': font_size})
for i in range(6):
    plt.plot(x, df_lin_lin[i+1, 1:], label=labels[i], marker=markers[i])


plt.subplot(182)
plt.tick_params(labelsize=tick_size)
plt.title('Linear(D)', fontdict={'fontsize': font_size})
plt.xlabel('$\epsilon$', fontdict={'fontsize': font_size})
plt.ylabel('RMSE', fontdict={'fontsize': font_size})
for i in range(6):
    plt.plot(x, df_lin_pgd_lin[i+1, 1:], label=labels[i], marker=markers[i])


plt.subplot(183)
plt.tick_params(labelsize=tick_size)
plt.title('MLP', fontdict={'fontsize': font_size})
plt.xlabel('$\epsilon$', fontdict={'fontsize': font_size})
plt.ylabel('RMSE', fontdict={'fontsize': font_size})
for i in range(6):
    plt.plot(x, df_mlp_lin[i+1, 1:], label=labels[i], marker=markers[i])


plt.subplot(184)
plt.tick_params(labelsize=tick_size)
plt.title('MLP(D)', fontdict={'fontsize': font_size})
plt.xlabel('$\epsilon$', fontdict={'fontsize': font_size})
plt.ylabel('RMSE', fontdict={'fontsize': font_size})
for i in range(6):
    plt.plot(x, df_mlp_pgd_lin[i+1, 1:], label=labels[i], marker=markers[i])


plt.subplot(185)
plt.tick_params(labelsize=tick_size)
plt.title('Linear', fontdict={'fontsize': font_size})
plt.xlabel('$\epsilon$', fontdict={'fontsize': font_size})
plt.ylabel('RMSE', fontdict={'fontsize': font_size})
for i in range(6):
    plt.plot(x, df_lin_mlp[i+1, 1:], label=labels[i], marker=markers[i])


plt.subplot(186)
plt.tick_params(labelsize=tick_size)
plt.title('Linear(D)', fontdict={'fontsize': font_size})
plt.xlabel('$\epsilon$', fontdict={'fontsize': font_size})
plt.ylabel('RMSE', fontdict={'fontsize': font_size})
for i in range(6):
    plt.plot(x, df_lin_pgd_mlp[i+1, 1:], label=labels[i], marker=markers[i])


plt.subplot(187)
plt.tick_params(labelsize=tick_size)
plt.title('MLP(white-box)', fontdict={'fontsize': font_size})
plt.xlabel('$\epsilon$', fontdict={'fontsize': font_size})
plt.ylabel('RMSE', fontdict={'fontsize': font_size})
for i in range(6):
    plt.plot(x, df_mlp_mlp[i+1, 1:], label=labels[i], marker=markers[i])


plt.subplot(188)
plt.tick_params(labelsize=tick_size)
plt.title('MLP(D)', fontdict={'fontsize': font_size})
plt.xlabel('$\epsilon$', fontdict={'fontsize': font_size})
plt.ylabel('RMSE', fontdict={'fontsize': font_size})
for i in range(6):
    plt.plot(x, df_mlp_pgd_mlp[i+1, 1:], label=labels[i], marker=markers[i])


# fig.tight_layout()
fig.tight_layout(pad=3, h_pad=1, w_pad=1)
# fig.tight_layout(pad=1, h_pad=0., w_pad=0)

ax = fig.axes[0]
lines, labels_ax = ax.get_legend_handles_labels()
fig.legend(lines, labels_ax, loc='upper center', ncol=6)

plt.savefig('res_syn.pdf', dpi=300)
# plt.show()
