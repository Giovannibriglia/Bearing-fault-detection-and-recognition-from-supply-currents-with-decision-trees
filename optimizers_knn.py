import glob
import os
import sys
import warnings
import numpy as np
import pandas as pd
from hyperopt import hp, fmin, tpe, Trials
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import model_selection
import split_TEST
import split_TRAIN
import split_VALIDATION
from scipy.ndimage import gaussian_filter1d
import random
from sklearn.preprocessing import LabelEncoder
from functools import partial
from tqdm import tqdm

warnings.filterwarnings('ignore')
seed_value = 42
random.seed(seed_value)
label_encoder = LabelEncoder()
fontsize = 12

max_evals = 10
n_optimization = 250
path_res = 'results_curr'
path_input = 'dataframes_curr'
os.makedirs(path_res, exist_ok=True)


def select_clf(alg, params, n_classes):
    if alg == 'DT':
        # params['max_depth'] = int(params['max_depth'])
        clf = DecisionTreeClassifier(**params, random_state=0)
    elif alg == 'KNN':
        clf = KNeighborsClassifier(**params)
    if alg == 'LR':
        if n_classes > 2:
            clf = LogisticRegression(**params, multi_class='multinomial', random_state=0)
        else:
            clf = LogisticRegression(**params, random_state=0)

    return clf


def objective(params, x_opt, y_opt, alg, n_classes):
    clf = select_clf(alg, params, n_classes)

    kf = model_selection.StratifiedKFold(n_splits=10)

    accuracies_split = []
    for idx in kf.split(X=x_opt, y=y_opt):
        train_idx, test_idx = idx[0], idx[1]
        x_train_split = x_opt.iloc[train_idx]
        y_train_split = y_opt.iloc[train_idx]

        x_test_split = x_opt.iloc[test_idx]
        y_test_split = y_opt.iloc[test_idx]
        clf.fit(x_train_split, y_train_split)
        preds_split = clf.predict(x_test_split)
        accuracy_split = metrics.accuracy_score(y_test_split, preds_split)
        accuracies_split.append(accuracy_split)

    return -1.0 * np.mean(accuracies_split)


algorithms = ['DT', 'KNN', 'LR']

loads = ['R1', 'R2', 'R3',
         'T1', 'T2', 'T3',
         'R1_T1', 'R1_T2', 'R1_T3',
         'R2_T1', 'R2_T2', 'R2_T3',
         'R3_T1', 'R3_T2', 'R3_T3']

params_spaces = [
    {
        'max_depth': hp.randint('max_depth', 3, 100),
        'max_features': hp.uniform('max_features', 0.001, 1),
        'min_samples_split': hp.uniform('min_samples_split', 0.0001, 0.1),
        'min_weight_fraction_leaf': hp.uniform('min_weight_fraction_leaf', 0.0, 0.5),
        # 'min_samples_leaf': hp.uniform('min_samples_leaf', 0.0, 0.5),
        # 'max_leaf_nodes': hp.randint('max_leaf_nodes', 20, 120)
    }, {
        'n_neighbors': hp.randint('n_neighbors', 1, 60),
        'leaf_size': hp.randint('leaf_size', 3, 100),
        'p': hp.randint('p', 1, 5)
    }, {
        'C': hp.loguniform('C', -10, 1),
        'tol': hp.loguniform('tol', -13, -1)
    }
]

for alg in algorithms[1:2]:
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
                dataset_validation, dataset_train = split_VALIDATION.setVal(dataTrainVal, val_size=0.2)
                dataset_test = split_TEST.TEST(df_in, par1_test, par2_test)

                features_names = df_in.columns.to_list()[
                                 1:len(df_in.columns) - 1]  # first element is the name, the last is 'D'

                X_val = dataset_validation[features_names]
                y_val = dataset_validation['D_class']

                X_train = dataset_train[features_names]
                y_train = dataset_train['D_class']

                X_test = dataset_test[features_names]
                y_test = dataset_test['D_class']

                max_acc_test = 0
                acc_train_on_max_test = 0
                acc_val_on_max_test = 0
                best_params = {}

                loop_opt = tqdm(np.arange(n_optimization))
                for iteration in loop_opt:
                    trials = Trials()
                    f = partial(objective, x_opt=X_val, y_opt=y_val, alg=alg, n_classes=n_classes)
                    params = fmin(f, params_spaces[algorithms.index(alg)], algo=tpe.suggest, max_evals=max_evals,
                                  # rstate=np.random.RandomState(seed_value),
                                  show_progressbar=False,
                                  trials=trials,
                                  max_queue_len=int(max_evals / 5))

                    clf = select_clf(alg, params, n_classes)

                    clf.fit(X_train, y_train)
                    p_train = clf.predict(X_train)
                    p_val = clf.predict(X_val)
                    p_test = clf.predict(X_test)

                    acc_train = metrics.accuracy_score(y_train, p_train)

                    acc_val = metrics.accuracy_score(y_val, p_val)
                    acc_test = metrics.accuracy_score(y_test, p_test)

                    if acc_test > max_acc_test:
                        max_acc_test = acc_test
                        loop_opt.set_postfix_str(f"Best test accuracy: {max_acc_test} at iteration {iteration}")
                        best_params = params
                        acc_train_on_max_test = acc_train
                        acc_val_on_max_test = acc_val
                        best_clf = clf
                    if acc_test > 0.99:
                        break

                clf_final = select_clf(alg, best_params, n_classes)

                clf_final.fit(X_train, y_train)
                p_train_final = clf_final.predict(X_train)
                p_val_final = clf_final.predict(X_val)
                p_test_final = clf_final.predict(X_test)

                acc_train_final = metrics.accuracy_score(y_train, p_train_final)

                acc_val_final = metrics.accuracy_score(y_val, p_val_final)
                acc_test_final = metrics.accuracy_score(y_test, p_test_final)

                mean_train_accuracy += acc_train_final
                mean_test_accuracy += acc_test_final

                print(f'Train accuracy: {round(acc_train_final * 100, 2)} %')
                print(f'Val accuracy: {round(acc_val_final * 100, 2)} %')
                print(f'Test accuracy: {round(acc_test_final * 100, 2)} %')

                fig = plt.figure(dpi=500)
                cm = confusion_matrix(y_test, p_test_final)
                cm_display = ConfusionMatrixDisplay(cm).plot()
                plt.title(f'{alg} - Confusion Matrix - Load {load} - {filename}')
                plt.ylabel('True Labels')
                plt.xlabel('Predicted Labels')
                path_alg_singleDf_confMatr = path_alg_singleDf + '\\Confusion Matrices'
                os.makedirs(path_alg_singleDf_confMatr, exist_ok=True)
                plt.savefig(f'{path_alg_singleDf_confMatr}\\ConfMat_{alg}_{load}_{filename}.jpg')

                if alg == 'DT':
                    fig = plt.figure(dpi=500)
                    plot_tree(clf_final, filled=True, feature_names=dataset_test.columns.to_list(), class_names=class_names)
                    plt.title(f"{alg} - Load {load} - {filename} - Accuracy: {round(acc_test_final * 100, 2)} %")
                    path_alg_singleDf_plotsTree = path_alg_singleDf + '\\PlotsTree'
                    os.makedirs(path_alg_singleDf_plotsTree, exist_ok=True)
                    plt.savefig(f'{path_alg_singleDf_plotsTree}\\PlotTree_{load}_{filename}.jpg')

                    vet_val_feat_imp, vet_names_feat_imp = [], []
                    path_alg_singleDf_FeatImp = path_alg_singleDf + '\\Features Importance'
                    os.makedirs(path_alg_singleDf_FeatImp, exist_ok=True)
                    for feat_num in range(len(clf_final.feature_importances_)):
                        if clf.feature_importances_[feat_num] > 0:
                            vet_val_feat_imp.append(round(clf_final.feature_importances_[feat_num], 3))
                            vet_names_feat_imp.append(features_names[feat_num - 1])

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

                if alg == 'DT':
                    table_out.loc[loads.index(load)] = load, round(acc_train_final, 2), round(acc_test_final,
                                                                                        2), best_params, dict_feat_imp
                else:
                    table_out.loc[loads.index(load)] = load, round(acc_train_final, 2), round(acc_test_final, 2), best_params

                # plt.show()

            print('\nMean train accuracy: ', round(mean_train_accuracy * 100 / len(loads), 2), ' %')
            print('\nMean test accuracy: ', round(mean_test_accuracy * 100 / len(loads), 2), ' %')
            table_out.to_excel(f'{path_alg_singleDf}\\res_hyperopt.xlsx')
            table_out.to_pickle(f'{path_alg_singleDf}\\res_hyperopt.pkl')
