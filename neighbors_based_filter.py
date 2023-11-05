import glob
import os
import random
import warnings
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import split_TEST
import split_TRAIN
import split_VALIDATION

warnings.filterwarnings('ignore')
seed_value = 42
random.seed(seed_value)
label_encoder = LabelEncoder()
labelsize = 12
fontsize = 12

start_k, max_k, step_k = 0, 20, 1
path_res = 'results_curr'
path_input = 'dataframes_curr'
os.makedirs(path_res, exist_ok=True)
n_most_imp_feat = 6


def get_best_parameters(df_par, par1, par2):
    if ((par1 == '') == True and (par2 == '') == False) or (par1 == par2):
        load = par2
    elif (par1 == '') == False and (par2 == '') == True:
        load = par1
    else:
        load = par1 + '_' + par2

    row_number = df_par.index[df_par['load'] == load]
    parameters = df_par.loc[row_number, 'hyperparameters'].to_list()[0]

    return parameters


def neighbors_based_filter(predictions, k):

    for i in range(len(predictions)):

        if i == 0:
            k_before = predictions[i]
        else:
            if i - k < 0:
                low_index = 0
            else:
                low_index = i - k

            if low_index == i:
                k_before = predictions[i]
            else:
                k_before = np.mean(predictions[low_index: i])

        if i == len(predictions)-1:
            k_after = predictions[len(predictions)-1]
        else:
            if i + k > len(predictions)-1:
                high_index = len(predictions)-1
            else:
                high_index = i + k

            if high_index == i:
                k_after = predictions[i]
            else:
                k_after = np.mean(predictions[i: high_index])

        if (k_before+k_after)/2 > 0.5:
            predictions[i] = 1
        elif (k_before+k_after)/2 < 0.5:
            predictions[i] = 0
        # if equal to 0.5: no changes

    return predictions


def select_clf(alg, params, n_classes):
    if alg == 'DT':
        params['max_depth'] = int(params['max_depth'])
        clf = DecisionTreeClassifier(**params, random_state=0)
    elif alg == 'KNN':
        clf = KNeighborsClassifier(**params)
    if alg == 'LR':
        if n_classes > 2:
            clf = LogisticRegression(**params, multi_class='multinomial', random_state=0)
        else:
            clf = LogisticRegression(**params, random_state=0)

    return clf


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

            df_input = pd.read_pickle(filename)
            df_input.columns = df_input.columns.str.replace(' ', '')

            filename = filename.replace(f'{path_input}\\', '')
            filename = filename.replace('.pkl', '')
            path_alg_singleDf = path_alg + '\\' + filename
            os.makedirs(path_alg_singleDf, exist_ok=True)

            path_alg_singleDf_post_process = path_alg_singleDf + '\\PostProcessing'
            os.makedirs(path_alg_singleDf_post_process, exist_ok=True)

            n_classes = len(df_input['D_class'].unique())
            if n_classes == 3:
                class_names = ['healthy', 'outer', 'brinn']
            elif n_classes == 2:
                class_names = ['healthy', 'damaged']

                mean_train_accuracy = 0
                mean_test_accuracy = 0

                vetAcc, vetPrec, vetF1, vetRec, vetAverages = [], [], [], [], []

                for k_bits in range(start_k, max_k + 1, step_k):
                    av_acc_test = 0
                    av_prec_test = 0
                    av_f1_test = 0
                    av_rec_test = 0
                    print('\n*************************************')
                    print(f'k = {k_bits} bit before and after - {alg} - {filename}')

                    for load in loads:
                        df_in = df_input.copy()

                        par1_test = load
                        par2_test = load

                        dataTrainVal = split_TRAIN.TRAIN(df_in, par1_test, par2_test)
                        dataset_validation, dataset_train = split_VALIDATION.setVal(dataTrainVal, val_size=0.2)
                        dataset_test = split_TEST.TEST(df_in, par1_test, par2_test)

                        features_names = df_in.columns.to_list()[1:len(df_in.columns) - 1]  # first element is the name, the last is 'D'

                        X_val = dataset_validation[features_names]
                        y_val = dataset_validation['D_class']

                        X_train = dataset_train[features_names]
                        y_train = dataset_train['D_class']

                        X_test = dataset_test[features_names]
                        y_test = dataset_test['D_class']

                        df_par = pd.read_pickle(path_alg_singleDf + '\\res_hyperopt.pkl')
                        best_params = get_best_parameters(df_par, par1_test, par2_test)

                        clf = select_clf(alg, best_params, n_classes)

                        clf.fit(X_train, y_train)

                        p_test = clf.predict(X_test)

                        prev_acc_test = metrics.accuracy_score(y_test, p_test)
                        prev_prec = metrics.precision_score(y_test, p_test)
                        prev_rec = metrics.recall_score(y_test, p_test)
                        prev_f1 = metrics.f1_score(y_test, p_test)

                        new_p_undamaged = neighbors_based_filter(p_test[0:int(len(p_test) / 3)], k_bits)
                        new_p_damaged = neighbors_based_filter(p_test[int(len(p_test) / 3):len(p_test)], k_bits)

                        p_test_new = []
                        p_test_new.extend(new_p_undamaged)
                        p_test_new.extend(new_p_damaged)

                        new_acc_test = metrics.accuracy_score(y_test, p_test_new)
                        new_rec = metrics.recall_score(y_test, p_test_new)
                        new_f1 = metrics.f1_score(y_test, p_test_new)
                        new_prec = metrics.precision_score(y_test, p_test_new)

                        """print(
                            f'{load}) New test accuracy: {round(prev_acc_test * 100)} % - Past test accuracy: {round(new_acc_test * 100)} %')
                        print(f'{load}) New test precision: {round(new_prec * 100)} % - Past test precision: {round(prev_prec * 100)} %')
                        print(f'{load}) New test recall: {round(new_rec * 100)} % - Past test recall: {round(prev_rec * 100)} %')
                        print(f'{load}) New test f1-score: {round(new_f1 * 100)} % - Past test f1-score: {round(prev_f1 * 100)} %')
                        print('\n')"""

                        av_acc_test += new_acc_test
                        av_f1_test += new_f1
                        av_prec_test += new_prec
                        av_rec_test += new_rec

                    average_averages = round(
                        (round(av_acc_test / len(loads) * 100, 2) + round(av_rec_test / len(loads) * 100, 2)
                         + round(av_f1_test / len(loads) * 100, 2) + round(av_prec_test / len(loads) * 100,
                                                                           2)) / 4, 2)

                    vetAcc.append(round(av_acc_test / len(loads) * 100, 2))
                    vetRec.append(round(av_rec_test / len(loads) * 100, 2))
                    vetPrec.append(round(av_prec_test / len(loads) * 100, 2))
                    vetF1.append(round(av_f1_test / len(loads) * 100, 2))
                    vetAverages.append(average_averages)

                    print(f'\nAverage test accuracy: {round(av_acc_test / len(loads) * 100, 2)} % ')
                    print(f'Average test recall: {round(av_rec_test / len(loads) * 100, 2)} % ')
                    print(f'Average test f1-score: {round(av_f1_test / len(loads) * 100, 2)} % ')
                    print(f'Average test precision: {round(av_prec_test / len(loads) * 100, 2)} % ')

                    print(f'\nAverage averages: {average_averages} % ')

                x_axis = np.arange(start_k, max_k + 1, step_k)
                fig = plt.figure(dpi=600)
                fig.suptitle('Neighbors-based filter', fontsize=fontsize + 5)
                plt.plot(x_axis, vetAcc, linewidth=3)
                plt.plot(x_axis, vetRec, linewidth=3)
                plt.plot(x_axis, vetF1, linewidth=3)
                plt.plot(x_axis, vetPrec, linewidth=3)
                plt.plot(x_axis, vetAverages, linewidth=5)
                plt.xlabel('K bits', fontsize=fontsize)
                plt.ylabel('Metrics [%]', fontsize=fontsize)
                plt.xticks(x_axis, fontsize=labelsize)
                plt.yticks(fontsize=labelsize)
                plt.legend(['Accuracy', 'Recall', 'F1-score', 'Precision', 'Averages'], fontsize=13)
                plt.grid()
                plt.tick_params(axis='x', labelsize=labelsize)
                plt.tick_params(axis='y', labelsize=labelsize)
                plt.savefig(f'{path_alg_singleDf_post_process}\\Neighbors_based_filter_{filename}.pdf')
                plt.show()

