import numpy as np
import matplotlib
from matplotlib import pyplot as plt

import sklearn

import tigramite
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb
from tigramite.models import LinearMediation, Prediction

np.random.seed(42)
T = 150
links_coeffs = {0: [((0, -1), 0.6)],
                1: [((1, -1), 0.6), ((0, -1), 0.8)],
                2: [((2, -1), 0.5), ((1, -1), 0.7)],  # ((0, -1), c)],
                }
N = len(links_coeffs)
data, true_parents = pp.var_process(links_coeffs, T=T)
dataframe = pp.DataFrame(data, var_names=[r'$X^0$', r'$X^1$', r'$X^2$', r'$X^3$'])

pred = Prediction(dataframe=dataframe,
                  cond_ind_test=ParCorr(),  # CMIknn ParCorr
                  prediction_model=sklearn.linear_model.LinearRegression(),
                  # prediction_model = sklearn.gaussian_process.GaussianProcessRegressor(),
                  # prediction_model = sklearn.neighbors.KNeighborsRegressor(),
                  data_transform=sklearn.preprocessing.StandardScaler(),
                  train_indices=range(int(0.8 * T)),
                  test_indices=range(int(0.8 * T), T),
                  verbosity=1
                  )

"""step ahead=1"""
# target = 2
# tau_max = 5
# predictors = pred.get_predictors(
#     selected_targets=[target],
#     steps_ahead=1,
#     tau_max=tau_max,
#     pc_alpha=None
# )
# link_matrix = np.zeros((N, N, tau_max + 1), dtype='bool')
# for j in [target]:
#     for p in predictors[j]:
#         link_matrix[p[0], j, abs(p[1])] = 1
#
# # Plot time series graph
# tp.plot_time_series_graph(
#     figsize=(6, 3),
#     #     node_aspect=2.,
#     val_matrix=np.ones(link_matrix.shape),
#     link_matrix=link_matrix,
#     var_names=None,
#     link_colorbar_label='',
# )
# plt.show()

"""step ahead=2"""
tau_max = 30
steps_ahead = 2
target = 2

all_predictors = pred.get_predictors(
    selected_targets=[target],
    steps_ahead=steps_ahead,
    tau_max=tau_max,
    pc_alpha=None
)
link_matrix = np.zeros((N, N, tau_max + 1), dtype='bool')
for j in [target]:
    for p in all_predictors[j]:
        link_matrix[p[0], j, abs(p[1])] = 1

# Plot time series graph
tp.plot_time_series_graph(
    figsize=(18, 5),
    node_size=0.05,
    node_aspect=.3,
    val_matrix=np.ones(link_matrix.shape),
    link_matrix=link_matrix,
    var_names=None,
    link_colorbar_label='',
    label_fontsize=24
)
plt.show()
# model fit
# pred.fit(target_predictors=all_predictors,
#          selected_targets=[target],
#          tau_max=tau_max)
# predict the target variable at the test samples
# predicted = pred.predict(target)
# true_data = pred.get_test_array()[0]
#
# plt.scatter(true_data, predicted)
# plt.title(r"NRMSE = %.2f" % (np.abs(true_data - predicted).mean()/true_data.std()))
# plt.plot(true_data, true_data, 'k-')
# plt.xlabel('True test data')
# plt.ylabel('Predicted test data')
# plt.show()

# predict other new data by supplying a new dataframe to new_data
new_data = pp.DataFrame(pp.var_process(links_coeffs, T=200)[0])
# predicted = pred.predict(target, new_data=new_data)
# true_data = pred.get_test_array()[0]
#
# plt.scatter(true_data, predicted)
# plt.title(r"NRMSE = %.2f" % (np.abs(true_data - predicted).mean() / true_data.std()))
# plt.plot(true_data, true_data, 'k-')
# plt.xlabel('True test data')
# plt.ylabel('Predicted test data')
# plt.show()

# This prediction is much better than using all past variables which leads to overfitting
whole_predictors = {2: [(i, -tau) for i in range(3) for tau in range(1, tau_max + 1)]}
pred.fit(target_predictors=whole_predictors,
         selected_targets=[target],
         tau_max=tau_max)

# new_data = pp.DataFrame(pp.var_process(links_coeffs, T=100)[0])
predicted = pred.predict(target, new_data=new_data)
# predicted = pred.predict(target)
true_data = pred.get_test_array()[0]

plt.scatter(true_data, predicted)
plt.plot(true_data, true_data, 'k-')
plt.title(r"NRMSE = %.2f" % (np.abs(true_data - predicted).mean() / true_data.std()))
plt.xlabel('True test data')
plt.ylabel('Predicted test data')
plt.show()

#  leave the data unscaled
pred = Prediction(dataframe=dataframe,
                  cond_ind_test=ParCorr(),
                  prediction_model=sklearn.linear_model.LinearRegression(),
                  #     data_transform=sklearn.preprocessing.StandardScaler(),
                  train_indices=range(int(0.8 * T)),
                  test_indices=range(int(0.8 * T), T),
                  verbosity=1
                  )
pred.fit(target_predictors=all_predictors,
         selected_targets=[target],
         tau_max=tau_max)
predicted = pred.predict(target)
# predicted = pred.predict(target)
true_data = pred.get_test_array()[0]

plt.scatter(true_data, predicted)
plt.plot(true_data, true_data, 'k-')
plt.title(r"NRMSE = %.2f" % (np.abs(true_data - predicted).mean() / true_data.std()))
plt.xlabel('True test data')
plt.ylabel('Predicted test data')
plt.show()
