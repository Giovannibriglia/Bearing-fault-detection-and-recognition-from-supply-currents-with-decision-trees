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


def evaluate_undamaged(predictions, k):

    predictions = list(predictions)
    aux = 0
    count_undamaged = 0
    for i in range(0, len(predictions)-1, 1):
        if predictions[i] == 1 and predictions[i+1] == 1:
            aux = aux + 1
        elif aux > count_undamaged:
            count_undamaged = aux
            aux = 0

    if count_undamaged > k:
        return 'no'
    else:
        return 'ok'


def evaluate_damaged(predictions, k):
    predictions = list(predictions)
    aux = 0
    count_damaged = 0
    for i in range(0, len(predictions)-1, 1):
        if predictions[i] == 1 and predictions[i+1] == 1:
            aux = aux + 1
        elif aux > count_damaged:
            count_damaged = aux
            aux = 0

    if count_damaged > k:
        return 'ok'
    else:
        return 'no'


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

                vet_damage, vet_undamage, vet_average = [], [], []

                for k_bits in range(start_k, max_k + 1, step_k):
                    list_with_damaged = []
                    list_without_damage = []
                    print('\n*************************************')
                    print(f'Threshold: k = {k_bits}, bits not consecutives - {alg} - {filename}')

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

                        acc_test = metrics.accuracy_score(y_test, p_test)

                        new_p_undamaged = evaluate_undamaged(p_test[0:int(len(p_test) / 3)], k_bits)
                        new_p_damaged = evaluate_damaged(p_test[int(len(p_test) / 3):len(p_test)], k_bits)

                        list_with_damaged.append(new_p_damaged)
                        list_without_damage.append(new_p_undamaged)

                    av_damage = list_with_damaged.count('ok') / len(list_with_damaged)
                    av_undamage = list_without_damage.count('ok') / len(list_without_damage)
                    print(f'Undamaged part: {round(av_undamage * 100, 2)} %')
                    print(f'Damaged part: {round(av_damage * 100, 2)} %')

                    total_av = av_damage * (2 / 3) + av_undamage * (1 / 3)
                    print('Average:', round(total_av * 100, 2), '%')

                    vet_damage.append(round(av_damage * 100, 2))
                    vet_undamage.append(round(av_undamage * 100, 2))
                    vet_average.append(round(total_av * 100, 2))

                x_axis = np.arange(start_k, max_k + 1, step_k)
                fig = plt.figure(dpi=600)
                fig.suptitle('Consecutive sliding window filter', fontsize=fontsize + 5)
                plt.plot(x_axis, vet_damage, linewidth=3)
                plt.plot(x_axis, vet_undamage, linewidth=3)
                plt.plot(x_axis, vet_average, linewidth=3)
                plt.xlabel('Thresholds', fontsize=fontsize)
                plt.ylabel('Correctness [%]', fontsize=fontsize)
                plt.xticks(x_axis, fontsize=labelsize)
                plt.yticks(fontsize=labelsize)
                plt.legend(['Damaged', 'Undamaged', 'Average'], fontsize=13)
                plt.grid()
                plt.tick_params(axis='x', labelsize=labelsize)
                plt.tick_params(axis='y', labelsize=labelsize)
                plt.savefig(f'{path_alg_singleDf_post_process}\\Consecutive_sliding_window_filter_{filename}.pdf')
                plt.show()
