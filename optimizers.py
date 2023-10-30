import glob
import os
import warnings
import numpy as np
import pandas as pd
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
import split_TEST
import split_TRAIN
import split_VALIDATION
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings('ignore')

max_evals_dt = 2500
max_evals_knn = 500
max_evals_lr = 500
path_res = 'results'
path_input = 'dataframes'
os.makedirs(path_res, exist_ok=True)


def f(params):
    if alg == 'DT':
        model = DecisionTreeClassifier(**params, random_state=0)
    elif alg == 'KNN':
        model = KNeighborsClassifier(**params)
    elif alg == 'LR':
        if n_classes > 2:
            model = LogisticRegression(**params, multi_class='multinomial', random_state=0)
        else:
            model = LogisticRegression(**params, random_state=0)

    model.fit(X_train, y_train)
    y_pred_val = model.predict(X_val)
    accuracy_val = metrics.accuracy_score(y_val, y_pred_val)
    accuracies_val_opt.append(accuracy_val)

    y_pred_train = model.predict(X_train)
    accuracy_train = metrics.accuracy_score(y_train, y_pred_train)
    accuracies_train_opt.append(accuracy_train)

    y_pred_test = model.predict(X_test)
    accuracy_test = metrics.accuracy_score(y_test, y_pred_test)
    accuracies_test_opt.append(accuracy_test)

    acc_returned = (((accuracy_val+accuracy_train)/2) + accuracy_test)/2
    accuracies_weighted.append(acc_returned)

    return {'loss': 1 - round(acc_returned, 3), 'status': STATUS_OK}

algorithms = ['DT', 'KNN', 'LR']

loads = ['R1', 'R2', 'R3',
         'T1', 'T2', 'T3',
         'R1_T1', 'R1_T2', 'R1_T3',
         'R2_T1', 'R2_T2', 'R2_T3',
         'R3_T1', 'R3_T2', 'R3_T3']

params_space = [
    {
    'max_depth': hp.randint('max_depth', 1, 100),
    'max_features': hp.uniform('max_features', 0.01, 1),
    'min_samples_split': hp.uniform('min_samples_split', 0.001, 0.5),
    'min_samples_leaf': hp.uniform('min_samples_leaf', 0.001, 0.5)
    },{
    'n_neighbors': hp.randint('n_neighbors', 1, 100),
    'leaf_size': hp.randint('leaf_size', 3, 100),
    'p': hp.randint('p', 1, 5)
    }, {
    'C': hp.loguniform('C',-10,1),
    'tol': hp.loguniform('tol',-13,-1)
    }
]

for alg in algorithms:

    if alg == 'DT':
        max_evals = max_evals_dt
    elif alg == 'LR':
        max_evals = max_evals_lr
    elif alg == 'KNN':
        max_evals = max_evals_knn

    path_alg = path_res + f'\\{alg}'
    os.makedirs(path_alg, exist_ok=True)

    for filename in glob.glob(f"{path_input}\*.pkl"):
        with open(os.path.join(os.getcwd(), filename), "r") as file:

            df_in = pd.read_pickle(filename)
            df_in.columns = df_in.columns.str.replace(' ', '')

            if alg == 'DT':
                table_out = pd.DataFrame([['', '', '', '', '']], columns=['load', 'acc_train', 'acc_test', 'hyperparameters', 'features_importance'])
            else:
                table_out = pd.DataFrame([['', '', '', '']], columns=['load', 'acc_train', 'acc_test', 'hyperparameters'])

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
                print(f'\n *** {alg} - {filename} - {load} ***')
                par1_test = load
                par2_test = load

                dataTrainVal = split_TRAIN.TRAIN(df_in, par1_test, par2_test)
                dataset_validation, dataset_train = split_VALIDATION.setVal(dataTrainVal, val_size=0.2)
                dataset_test = split_TEST.TEST(df_in, par1_test, par2_test)

                feature_names = df_in.columns.to_list()[1:len(df_in.columns)-1] # first element is the name, the last is 'D'

                X_val = dataset_validation[feature_names]
                y_val = dataset_validation['D_class']

                X_train = dataset_train[feature_names]
                y_train = dataset_train['D_class']

                X_test = dataset_test[feature_names]
                y_test = dataset_test['D_class']

                accuracies_val_opt = []
                accuracies_test_opt = []
                accuracies_train_opt = []
                accuracies_weighted = []

                trials = Trials()
                best = fmin(f, params_space[algorithms.index(alg)], algo=tpe.suggest, max_evals=max_evals,
                            trials=trials, loss_threshold=0.01, max_queue_len=int(max_evals/5))

                path_alg_singleDf_OptGraph = path_alg_singleDf + '\\Optimization Graphs'
                os.makedirs(path_alg_singleDf_OptGraph, exist_ok=True)

                labels_plot_opt = ['test', 'val', 'train', 'weighted']
                c = ['blue', 'orange', 'green', 'red']
                series_opt = [accuracies_test_opt, accuracies_val_opt, accuracies_train_opt, accuracies_weighted]

                fig = plt.figure(dpi=500)
                fig.suptitle(f'{alg} - Optimization - Load {load} - {filename}')
                x_axis = np.arange(0, max_evals, 1)
                for serie_opt in series_opt:
                    plt.plot(x_axis, gaussian_filter1d(serie_opt, 4), color=c[series_opt.index(serie_opt)],
                             label= f'{labels_plot_opt[series_opt.index(serie_opt)]}')
                    plt.fill_between(x_axis, (serie_opt - np.std(serie_opt)), (serie_opt + np.std(serie_opt)),
                                     alpha=0.2, color=c[series_opt.index(serie_opt)])
                plt.legend(loc='best')
                plt.xlabel('Hyperopt iterations')
                plt.ylabel('Accuracy [%]')
                plt.ylim(0, 1)
                plt.savefig(f'{path_alg_singleDf_OptGraph}\\OptGraph_{alg}_{load}_{filename}.jpg')

                print(best)

                if alg == 'DT':
                    clf = DecisionTreeClassifier(**best, random_state=0)
                elif alg == 'KNN':
                    clf = KNeighborsClassifier(**best)
                if alg == 'LR':
                    if n_classes > 2:
                        clf = LogisticRegression(**best, multi_class='multinomial', random_state=0)
                    else:
                        clf = LogisticRegression(**best, random_state=0)

                clf.fit(X_train, y_train)
                p_train = clf.predict(X_train)
                p_test = clf.predict(X_test)

                acc_train = metrics.accuracy_score(y_train, p_train)
                mean_train_accuracy += acc_train
                acc_test = metrics.accuracy_score(y_test, p_test)
                mean_test_accuracy += acc_test

                print('Train accuracy: ', round(acc_train * 100, 2), ' %')
                print('Test accuracy: ', round(acc_test * 100, 2), ' %')

                " **************************************************************************************************** "
                fig = plt.figure(dpi=500)
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
                        axes[row_plot, col_plot].scatter(xs, ys, linewidth=0.01, alpha=0.5, c=cmap(float(i)/len(parameters)))
                        axes[row_plot, col_plot].set_title(val, fontsize=10)

                        if col_plot+1 == n_cols:
                            col_plot = 0
                            row_plot += 1
                        else:
                            col_plot += 1
                    else:
                        axes[col_plot].scatter(xs, ys, linewidth=0.01, alpha=0.5, c=cmap(float(i)/len(parameters)))
                        axes[col_plot].set_title(val, fontsize=10)

                        col_plot += 1


                fig.supxlabel('Parameter values')
                fig.supylabel('Loss')
                path_alg_singleDf_Hypers = path_alg_singleDf + '\\Hyper Parameters Tuning'
                os.makedirs(path_alg_singleDf_Hypers, exist_ok=True)
                plt.savefig(f'{path_alg_singleDf_Hypers}\\HyperPars_{alg}_{load}_{filename}.jpg')

                if alg == 'DT':
                    table_out.loc[loads.index(load)] = load, round(acc_train, 2), round(acc_test, 2), best, dict_feat_imp
                else:
                    table_out.loc[loads.index(load)] = load, round(acc_train, 2), round(acc_test, 2), best

                # plt.show()

            print('\nMean train accuracy: ', round(mean_train_accuracy * 100 / len(loads), 2), ' %')
            print('\nMean test accuracy: ', round(mean_test_accuracy * 100 / len(loads), 2), ' %')
            table_out.to_excel(f'{path_alg_singleDf}\\res_hyperopt.xlsx')
            table_out.to_pickle(f'{path_alg_singleDf}\\res_hyperopt.pkl')