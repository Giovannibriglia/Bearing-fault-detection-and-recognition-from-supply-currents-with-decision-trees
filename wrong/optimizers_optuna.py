import glob
import os
import sys
import warnings
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.visualization.matplotlib import plot_optimization_history, plot_slice, plot_param_importances
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import model_selection
import split_TEST
import split_TRAIN
import split_VALIDATION2
from scipy.ndimage import gaussian_filter1d
import random
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
seed_value = 42
random.seed(seed_value)
sampler = TPESampler(seed=seed_value)
label_encoder = LabelEncoder()
fontsize = 12

n_trials = 1000
path_res = 'results_curr'
path_input = '../dataframes_curr'
os.makedirs(path_res, exist_ok=True)


def fold_for_f(real_loads, dataTrainVal, features_names, n_loads_to_use=6, n_examples_for_load=20):
    groups_indices = []

    """n_loads_to_use = random.sample([2, 3, 4, 5, 6], 1)[0]
    n_examples_for_load = random.sample([10, 20, 40, 60], 1)[0]"""

    random.shuffle(real_loads)
    selected_real_loads = real_loads[:n_loads_to_use]

    for real_load in selected_real_loads:
        indexes = dataTrainVal[dataTrainVal['name_signal'].str.contains(real_load)].index.to_list()
        for ind in range(len(indexes)):
            if indexes[ind] > 0:
                indexes[ind] -= 1
        if n_examples_for_load <= len(indexes):
            indexes = random.sample(indexes, n_examples_for_load)
        groups_indices.append(indexes)

    x_opt = pd.DataFrame(columns=features_names)
    y_opt = []

    indexes_cols_features = [dataTrainVal.columns.get_loc(column_name) for column_name in features_names]

    for single_group_indices in groups_indices:
        x_values = dataTrainVal.iloc[single_group_indices, indexes_cols_features]
        x_opt = pd.concat([x_opt, x_values], ignore_index=True)
        y_values = dataTrainVal.iloc[single_group_indices, max(indexes_cols_features) + 1]
        y_opt.extend(y_values)

    y_opt = pd.Series(y_opt)

    n_groups = n_loads_to_use

    return x_opt, y_opt, n_groups, selected_real_loads


def dtree_objective(trial):
    max_depth = trial.suggest_int('max_depth', 2, 100)
    max_features = trial.suggest_float('max_features', 0, 1)
    min_samples_leaf = trial.suggest_float('min_samples_leaf', 0, 1)
    min_samples_split = trial.suggest_float('min_samples_split', 0, 1)
    min_weight_fraction_leaf = trial.suggest_float('min_weight_fraction_leaf', 0, 0.5)
    max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 2, 120)

    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_features=max_features,
                                 min_samples_split=min_samples_split, min_weight_fraction_leaf=min_weight_fraction_leaf,
                                 max_leaf_nodes=max_leaf_nodes, random_state=seed_value)
    """scores = model_selection.cross_val_score(model, X_val, y_val, cv=5, scoring='f1_weighted')
    return scores.mean()"""

    """acc = []
    for _ in range(20):
        n_loads_to_use = random.randint(2, 6)
        n_examples_for_load = random.randint(5, 60)
        x_opt, y_opt, n_groups, selected_real_loads = fold_for_f(real_loads, dataTrainVal, features_names,
                                                                 n_loads_to_use=n_loads_to_use, n_examples_for_load=n_examples_for_load)
        kf = model_selection.GroupKFold(n_splits=n_groups)

        groups = []
        count = 0
        for group_number in range(n_groups):
            count += 1
            for count in [count] * int(len(x_opt) / n_groups):
                groups.extend([count])

        x = x_opt
        y = y_opt
        val_score_fold = []
        for idx in kf.split(X=x, y=y, groups=groups):
            train_idx, test_idx = idx[0], idx[1]
            x_train_split = x.iloc[train_idx]

            y_train_split = y.iloc[train_idx]

            x_test_split = x.iloc[test_idx]
            y_test_split = y.iloc[test_idx]

            model.fit(x_train_split, y_train_split)
            preds_split = model.predict(x_test_split)
            # val_score_fold.append(np.mean(model_selection.cross_val_score(model, x, y, cv=n_groups, scoring='roc_auc')))
            val_score_fold.append(metrics.accuracy_score(y_test_split, preds_split))
        acc.append(np.mean(val_score_fold))

    return np.mean(acc)"""

    """kf = model_selection.StratifiedKFold(n_splits=5)
    accuracies = []
    x = X_val
    y = y_val

    for idx in kf.split(X=x, y=y):
        train_idx, test_idx = idx[0], idx[1]
        x_train_split = x.iloc[train_idx]
        y_train_split = y.iloc[train_idx]

        x_test_split = x.iloc[test_idx]
        y_test_split = y.iloc[test_idx]
        model.fit(x_train_split, y_train_split)
        preds = model.predict(x_test_split)
        accuracy = metrics.accuracy_score(y_test_split, preds)
    accuracies.append(accuracy)

    return np.mean(accuracies)"""

    model.fit(X_train, y_train)

    preds_val_split = model.predict(X_val)
    accuracy_val_split = metrics.accuracy_score(y_val, preds_val_split)
    vet_acc_val.append(accuracy_val_split)

    preds_test_split = model.predict(X_test)
    accuracy_test_split = metrics.accuracy_score(y_test, preds_test_split)
    vet_acc_test.append(accuracy_test_split)

    preds_train_split = model.predict(X_train)
    accuracy_train_split = metrics.accuracy_score(y_train, preds_train_split)
    vet_acc_train.append(accuracy_train_split)

    return np.mean(accuracy_test_split)


algorithms = ['DT', 'KNN', 'LR']

loads = ['R1', 'R2', 'R3',
         'T1', 'T2', 'T3',
         'R1_T1', 'R1_T2', 'R1_T3',
         'R2_T1', 'R2_T2', 'R2_T3',
         'R3_T1', 'R3_T2', 'R3_T3']

for alg in algorithms:

    path_alg = path_res + f'\\{alg}'
    os.makedirs(path_alg, exist_ok=True)

    for filename in glob.glob(f"{path_input}\*.pkl"):
        with open(os.path.join(os.getcwd(), filename), "r") as file:

            df_in = pd.read_pickle(filename)
            df_in.columns = df_in.columns.str.replace(' ', '')

            if alg == 'DT':
                table_out = pd.DataFrame([['', '', '', '', '']],
                                         columns=['load', 'acc_train', 'acc_test', 'hyperparameters',
                                                  'features_importance'])
            else:
                table_out = pd.DataFrame([['', '', '', '']],
                                         columns=['load', 'acc_train', 'acc_test', 'hyperparameters'])

            filename = filename.replace(f'{path_input}\\', '')
            filename = filename.replace('.pkl', '')
            path_alg_singleDf = path_alg + '\\' + filename
            os.makedirs(path_alg_singleDf, exist_ok=True)

            n_classes = len(df_in['D_class'].unique())
            if n_classes == 2:
                class_names = ['healthy', 'damaged']
            elif n_classes == 3:
                class_names = ['healthy', 'outer', 'brinn']

            mean_train_accuracy = 0
            mean_test_accuracy = 0

            for load in loads:
                real_loads = ['R1_T1', 'R1_T2', 'R1_T3',
                              'R2_T1', 'R2_T2', 'R2_T3',
                              'R3_T1', 'R3_T2', 'R3_T3']

                real_loads = [s for s in real_loads if load not in s]

                print(f'\n *** {alg} - {filename} - {load} ***')
                par1_test = load
                par2_test = load
                dataTrainVal = split_TRAIN.TRAIN(df_in, par1_test, par2_test)
                dataset_validation, dataset_train = split_VALIDATION.setVal(dataTrainVal, all_loads=loads,
                                                                            load_tested=par1_test,
                                                                            val_size=0.2)
                dataset_test = split_TEST.TEST(df_in, par1_test, par2_test)

                features_names = df_in.columns.to_list()[
                                 1:len(df_in.columns) - 1]  # first element is the name, the last is 'D'

                X_val = dataset_validation[features_names]
                y_val = dataset_validation['D_class']

                X_train = dataset_train[features_names]
                y_train = dataset_train['D_class']

                X_test = dataset_test[features_names]
                y_test = dataset_test['D_class']

                vet_acc_train = []
                vet_acc_val = []
                vet_acc_test = []

                dtree_study = optuna.create_study(direction='maximize', sampler=sampler)
                dtree_study.optimize(dtree_objective, show_progress_bar=True, n_trials=n_trials)
                best_params = dtree_study.best_params

                clf = DecisionTreeClassifier(**best_params, random_state=seed_value)

                clf.fit(X_train, y_train)
                p_train = clf.predict(X_train)
                p_val = clf.predict(X_val)
                p_test = clf.predict(X_test)

                acc_train = metrics.accuracy_score(y_train, p_train)
                mean_train_accuracy += acc_train
                acc_val = metrics.accuracy_score(y_val, p_val)
                acc_test = metrics.accuracy_score(y_test, p_test)
                mean_test_accuracy += acc_test

                print('Train accuracy: ', round(acc_train * 100, 2), ' %')
                print('Val accuracy: ', round(acc_val * 100, 2), ' %')
                print('Test accuracy: ', round(acc_test * 100, 2), ' %')

                """fig = plt.figure(dpi=500)
                plot_optimization_history(dtree_study)
                plt.show()"""

                fig = plt.figure(dpi=500)
                plt.plot(vet_acc_test, label='test')
                plt.plot(vet_acc_train, label='train')
                # plt.plot(vet_acc_val, label='val')
                plt.legend(loc='best')
                plt.show()


                """fig = plt.figure(dpi=500)
                plot_slice(dtree_study)
                plt.show()

                fig = plt.figure(dpi=500)
                plot_param_importances(dtree_study)
                plt.show()"""

                """path_alg_singleDf_OptGraph = path_alg_singleDf + '\\Optimization Graphs'
                os.makedirs(path_alg_singleDf_OptGraph, exist_ok=True)"""

                """fig = plt.figure(dpi=500)
                cm = confusion_matrix(y_test, p_test)
                cm_display = ConfusionMatrixDisplay(cm).plot()
                plt.title(f'{alg} - Confusion Matrix - Load {load} - {filename}')
                plt.ylabel('True Labels')
                plt.xlabel('Predicted Labels')
                path_alg_singleDf_confMatr = path_alg_singleDf + '\\Confusion Matrices'
                os.makedirs(path_alg_singleDf_confMatr, exist_ok=True)
                plt.savefig(f'{path_alg_singleDf_confMatr}\\ConfMat_{alg}_{load}_{filename}.jpg')

                if alg == 'DT':
                    fig = plt.figure(dpi=500)
                    plot_tree(clf, filled=True, feature_names=dataset_test.columns.to_list(), class_names=class_names)
                    plt.title(f"{alg} - Load {load} - {filename} - Accuracy: {round(acc_test * 100, 2)} %")
                    path_alg_singleDf_plotsTree = path_alg_singleDf + '\\PlotsTree'
                    os.makedirs(path_alg_singleDf_plotsTree, exist_ok=True)
                    plt.savefig(f'{path_alg_singleDf_plotsTree}\\PlotTree_{load}_{filename}.jpg')

                    vet_val_feat_imp, vet_names_feat_imp = [], []
                    path_alg_singleDf_FeatImp = path_alg_singleDf + '\\Features Importance'
                    os.makedirs(path_alg_singleDf_FeatImp, exist_ok=True)
                    for feat_num in range(len(clf.feature_importances_)):
                        if clf.feature_importances_[feat_num] > 0:
                            vet_val_feat_imp.append(round(clf.feature_importances_[feat_num], 3))
                            vet_names_feat_imp.append(feature_names[feat_num - 1])

                    dict_feat_imp = dict(zip(vet_names_feat_imp, vet_val_feat_imp))
                    dict_feat_imp = dict(sorted(dict_feat_imp.items(), key=lambda x: x[1], reverse=True))

                    fig = plt.figure(dpi=500)
                    plt.bar(vet_names_feat_imp, vet_val_feat_imp)
                    plt.title(f'{alg} - Load {load} - {filename} - Features Importance')
                    plt.xticks(rotation=90)
                    plt.subplots_adjust(bottom=0.3,
                                        top=0.9,
                                        wspace=0.5,
                                        hspace=0.35)
                    plt.savefig(f'{path_alg_singleDf_FeatImp}\\FeatImp_{load}_{filename}.jpg')
                    # plt.show()

                parameters = list(params_space[algorithms.index(alg)].keys())
                if len(parameters) > 3:
                    n_rows = int(len(parameters) / 2)
                    n_cols = len(parameters) - n_rows
                    row_plot, col_plot = 0, 0
                    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, dpi=500)
                else:
                    col_plot = 0
                    fig, axes = plt.subplots(nrows=1, ncols=len(parameters), dpi=500)

                fig.tight_layout(pad=4)
                fig.suptitle(f'{alg} - Load {load} - Tuning - {filename}', fontsize=15)
                cmap = plt.cm.jet
                for i, val in enumerate(parameters):
                    xs = np.array([t['misc']['vals'][val] for t in trials.trials]).ravel()
                    ys = [t['result']['loss'] for t in trials.trials]
                    xs, ys = zip(*sorted(zip(xs, ys)))
                    ys = np.array(ys)

                    if len(parameters) > 3:
                        axes[row_plot, col_plot].scatter(xs, ys, linewidth=0.01, alpha=0.5,
                                                         c=cmap(float(i) / len(parameters)))
                        axes[row_plot, col_plot].set_title(val, fontsize=10)

                        if col_plot + 1 == n_cols:
                            col_plot = 0
                            row_plot += 1
                        else:
                            col_plot += 1
                    else:
                        axes[col_plot].scatter(xs, ys, linewidth=0.01, alpha=0.5, c=cmap(float(i) / len(parameters)))
                        axes[col_plot].set_title(val, fontsize=10)

                        col_plot += 1

                fig.supxlabel('Parameter values')
                fig.supylabel('Loss')
                path_alg_singleDf_Hypers = path_alg_singleDf + '\\Hyper Parameters Tuning'
                os.makedirs(path_alg_singleDf_Hypers, exist_ok=True)
                plt.savefig(f'{path_alg_singleDf_Hypers}\\HyperPars_{alg}_{load}_{filename}.jpg')

                if alg == 'DT':
                    table_out.loc[loads.index(load)] = load, round(acc_train, 2), round(acc_test,
                                                                                        2), best, dict_feat_imp
                else:
                    table_out.loc[loads.index(load)] = load, round(acc_train, 2), round(acc_test, 2), best

                # plt.show()

            print('\nMean train accuracy: ', round(mean_train_accuracy * 100 / len(loads), 2), ' %')
            print('\nMean test accuracy: ', round(mean_test_accuracy * 100 / len(loads), 2), ' %')
            table_out.to_excel(f'{path_alg_singleDf}\\res_hyperopt.xlsx')
            table_out.to_pickle(f'{path_alg_singleDf}\\res_hyperopt.pkl')"""
