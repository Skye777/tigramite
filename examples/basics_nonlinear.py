import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import sklearn

import tigramite
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb

"""quadratic noise"""
np.random.seed(1)
data = np.random.randn(500, 3)
for t in range(1, 500):
    data[t, 0] += 0.4 * data[t - 1, 1] ** 2
    data[t, 2] += 0.3 * data[t - 2, 1] ** 2
var_names = [r'$X^0$', r'$X^1$', r'$X^2$', r'$X^3$']
dataframe = pp.DataFrame(data, var_names=var_names)
# tp.plot_timeseries(dataframe)
# plt.show()

gpdc = GPDC(significance='analytic', gp_params=None)
pcmci_gpdc = PCMCI(
    dataframe=dataframe,
    cond_ind_test=gpdc,
    verbosity=0)
results = pcmci_gpdc.run_pcmci(tau_max=2, pc_alpha=0.1)
pcmci_gpdc.print_significant_links(
    p_matrix=results['p_matrix'],
    val_matrix=results['val_matrix'],
    alpha_level=0.01)

array, dymmy, dummy = gpdc._get_array(X=[(0, -1)], Y=[(2, 0)], Z=[(1, -2)], tau_max=2)
x, meanx = gpdc._get_single_residuals(array, target_var=0, return_means=True)
y, meany = gpdc._get_single_residuals(array, target_var=1, return_means=True)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 3))
axes[0].scatter(array[2], array[0], color='grey')
axes[0].scatter(array[2], meanx, color='black')
axes[0].set_title("GP of %s on %s" % (var_names[0], var_names[1]))
axes[0].set_xlabel(var_names[1])
axes[0].set_ylabel(var_names[0])
axes[1].scatter(array[2], array[1], color='grey')
axes[1].scatter(array[2], meany, color='black')
axes[1].set_title("GP of %s on %s" % (var_names[2], var_names[1]))
axes[1].set_xlabel(var_names[1])
axes[1].set_ylabel(var_names[2])
axes[2].scatter(x, y, color='red')
axes[2].set_title("DC of residuals:" "\n val=%.3f / p-val=%.3f" % (gpdc.run_test(
    X=[(0, -1)], Y=[(2, 0)], Z=[(1, -2)], tau_max=2)))
axes[2].set_xlabel("resid. " + var_names[0])
axes[2].set_ylabel("resid. " + var_names[2])
plt.tight_layout()
plt.show()

"""multiplicative noise"""
np.random.seed(42)
data = np.random.randn(500, 3)
for t in range(1, 500):
    data[t, 0] *= 0.2 * data[t - 1, 1]
    data[t, 2] *= 0.3 * data[t - 2, 1]
dataframe = pp.DataFrame(data, var_names=var_names)
tp.plot_timeseries(dataframe)
plt.show()
#
# pcmci_gpdc = PCMCI(
#     dataframe=dataframe,
#     cond_ind_test=gpdc)
# results = pcmci_gpdc.run_pcmci(tau_max=2, pc_alpha=0.1)
# pcmci_gpdc.print_significant_links(
#     p_matrix=results['p_matrix'],
#     val_matrix=results['val_matrix'],
#     alpha_level=0.01)
#
# array, dymmy, dummy = gpdc._get_array(X=[(0, -1)], Y=[(2, 0)], Z=[(1, -2)], tau_max=2)
# x, meanx = gpdc._get_single_residuals(array, target_var=0, return_means=True)
# y, meany = gpdc._get_single_residuals(array, target_var=1, return_means=True)
#
# fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 3))
# axes[0].scatter(array[2], array[0], color='grey')
# axes[0].scatter(array[2], meanx, color='black')
# axes[0].set_title("GP of %s on %s" % (var_names[0], var_names[1]))
# axes[0].set_xlabel(var_names[1])
# axes[0].set_ylabel(var_names[0])
# axes[1].scatter(array[2], array[1], color='grey')
# axes[1].scatter(array[2], meany, color='black')
# axes[1].set_title("GP of %s on %s" % (var_names[2], var_names[1]))
# axes[1].set_xlabel(var_names[1])
# axes[1].set_ylabel(var_names[2])
# axes[2].scatter(x, y, color='red', alpha=0.3)
# axes[2].set_title("DC of residuals:" "\n val=%.3f / p-val=%.3f" % (gpdc.run_test(
#     X=[(0, -1)], Y=[(2, 0)], Z=[(1, -2)], tau_max=2)))
# axes[2].set_xlabel("resid. " + var_names[0])
# axes[2].set_ylabel("resid. " + var_names[2])
# plt.tight_layout()
# plt.show()

cmi_knn = CMIknn(significance='shuffle_test', knn=0.1, shuffle_neighbors=5, transform='ranks')
pcmci_cmi_knn = PCMCI(
    dataframe=dataframe,
    cond_ind_test=cmi_knn,
    verbosity=2)
results = pcmci_cmi_knn.run_pcmci(tau_max=2, pc_alpha=0.05)
pcmci_cmi_knn.print_significant_links(
    p_matrix=results['p_matrix'],
    val_matrix=results['val_matrix'],
    alpha_level=0.01)

link_matrix = pcmci_cmi_knn.return_significant_links(pq_matrix=results['p_matrix'],
                                                     val_matrix=results['val_matrix'], alpha_level=0.01)['link_matrix']
tp.plot_graph(
    val_matrix=results['val_matrix'],
    link_matrix=link_matrix,
    var_names=var_names,
    link_colorbar_label='cross-MCI',
    node_colorbar_label='auto-MCI',
    vmin_edges=0.,
    vmax_edges=0.3,
    edge_ticks=0.05,
    cmap_edges='OrRd',
    vmin_nodes=0,
    vmax_nodes=.5,
    node_ticks=.1,
    cmap_nodes='OrRd',
)
plt.show()
