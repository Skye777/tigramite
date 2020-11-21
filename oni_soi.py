import numpy as np
import sklearn
from matplotlib import pyplot as plt

from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb
from tigramite.models import Prediction


def linear_partial_correlation(dataframe, var_names):
    parcorr = ParCorr(significance='analytic')
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=1)
    # correlations = pcmci.get_lagged_dependencies(tau_max=40, val_only=True)['val_matrix']
    # lag_func_matrix = tp.plot_lagfuncs(val_matrix=correlations, setup_args={'var_names': var_names,
    #                                                                         'x_base': 5, 'y_base': .5})
    # plt.show()

    pcmci.verbosity = 1
    results = pcmci.run_pcmci(tau_max=36, pc_alpha=None)
    print("p-values")
    print(results['p_matrix'].round(3))
    print("MCI partial correlations")
    print(results['val_matrix'].round(2))

    q_matrix = pcmci.get_corrected_pvalues(p_matrix=results['p_matrix'], fdr_method='fdr_bh')
    pcmci.print_significant_links(
        p_matrix=results['p_matrix'],
        q_matrix=q_matrix,
        val_matrix=results['val_matrix'],
        alpha_level=0.01)

    link_matrix = pcmci.return_significant_links(pq_matrix=q_matrix,
                                                 val_matrix=results['val_matrix'], alpha_level=0.01)['link_matrix']
    tp.plot_graph(
        val_matrix=results['val_matrix'],
        link_matrix=link_matrix,
        var_names=var_names,
        link_colorbar_label='cross-MCI',
        node_colorbar_label='auto-MCI',
    )
    plt.show()

    # Plot time series graph
    tp.plot_time_series_graph(
        figsize=(24, 4),
        val_matrix=results['val_matrix'],
        link_matrix=link_matrix,
        var_names=var_names,
        link_colorbar_label='MCI',
    )
    plt.show()


def nonlinear_dependency_gdpc(dataframe, var_names):
    gpdc = GPDC(significance='analytic', gp_params=None)
    pcmci_gpdc = PCMCI(
        dataframe=dataframe,
        cond_ind_test=gpdc,
        verbosity=0)
    results = pcmci_gpdc.run_pcmci(tau_max=24, pc_alpha=0.1)
    pcmci_gpdc.print_significant_links(
        p_matrix=results['p_matrix'],
        val_matrix=results['val_matrix'],
        alpha_level=0.01)


def cmiknn(dataframe, var_names):
    cmi_knn = CMIknn(significance='shuffle_test', knn=0.1, shuffle_neighbors=5, transform='ranks')
    pcmci_cmi_knn = PCMCI(
        dataframe=dataframe,
        cond_ind_test=cmi_knn,
        verbosity=2)
    results = pcmci_cmi_knn.run_pcmci(tau_max=12, pc_alpha=0.05)
    pcmci_cmi_knn.print_significant_links(
        p_matrix=results['p_matrix'],
        val_matrix=results['val_matrix'],
        alpha_level=0.01)

    link_matrix = pcmci_cmi_knn.return_significant_links(pq_matrix=results['p_matrix'],
                                                         val_matrix=results['val_matrix'], alpha_level=0.01)[
        'link_matrix']
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


def prediction(dataframe, T, target, steps_ahead, tau_max):
    pred = Prediction(dataframe=dataframe,
                      # cond_ind_test=ParCorr(),  # CMIknn ParCorr
                      cond_ind_test=GPDC(),
                      # prediction_model=sklearn.linear_model.LinearRegression(),
                      prediction_model=sklearn.gaussian_process.GaussianProcessRegressor(alpha=0,
                                                                                         kernel=sklearn.gaussian_process.kernels.RBF() +
                                                                                                sklearn.gaussian_process.kernels.WhiteKernel()),
                      # prediction_model = sklearn.neighbors.KNeighborsRegressor(),
                      data_transform=sklearn.preprocessing.StandardScaler(),
                      train_indices=range(int(0.8 * T)),
                      test_indices=range(int(0.8 * T), T),
                      verbosity=1
                      )
    all_predictors = pred.get_predictors(
        selected_targets=[target],
        steps_ahead=steps_ahead,
        tau_max=tau_max,
        pc_alpha=None
    )
    # link_matrix = np.zeros((N, N, tau_max + 1), dtype='bool')
    # for j in [target]:
    #     for p in all_predictors[j]:
    #         link_matrix[p[0], j, abs(p[1])] = 1
    #
    # # Plot time series graph
    # tp.plot_time_series_graph(
    #     figsize=(18, 5),
    #     node_size=0.05,
    #     node_aspect=.3,
    #     val_matrix=np.ones(link_matrix.shape),
    #     link_matrix=link_matrix,
    #     var_names=None,
    #     link_colorbar_label='',
    #     label_fontsize=24
    # )
    # plt.show()

    # model fit
    pred.fit(target_predictors=all_predictors,
             selected_targets=[target],
             tau_max=tau_max)
    # predict the target variable at the test samples
    predicted = pred.predict(target)
    true_data = pred.get_test_array()[0]

    plt.scatter(true_data, predicted)
    plt.title(r"NRMSE = %.2f" % (np.abs(true_data - predicted).mean() / true_data.std()))
    plt.plot(true_data, true_data, 'k-')
    plt.xlabel('True test data')
    plt.ylabel('Predicted test data')
    plt.show()


# T = 837 (1951.1-2020.9)
oni = np.loadtxt('data/oni.ascii.txt', skiprows=1, usecols=3)
soi = np.loadtxt('data/soi.csv', skiprows=2, delimiter=",", usecols=1)
data = np.vstack((oni, soi)).transpose()
T, N = data.shape
var_names = ['ONI', 'SOI']
dataframe = pp.DataFrame(data, var_names=var_names)
# tp.plot_timeseries(dataframe)
# plt.show()
# linear_partial_correlation(dataframe, var_names)
# nonlinear_dependency_gdpc(dataframe, var_names)
# cmiknn(dataframe, var_names)
prediction(dataframe, T, target=0, steps_ahead=1, tau_max=12)
