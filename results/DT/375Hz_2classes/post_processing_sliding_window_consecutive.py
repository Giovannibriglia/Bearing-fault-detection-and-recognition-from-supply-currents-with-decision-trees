import split_TEST
import split_TRAIN
import split_VALIDATION
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import tree, metrics
import numpy as np
import matplotlib.pyplot as plt

label_size_plot = 12
fontsize = 12
start_k, max_k = 1, 20

path_parameters = 'res_hyperopt.pkl'
df_in = pd.read_pickle('375Hz_2classes.pkl')


def get_best_parameters(df_par, par1, par2):
    if ((par1 == '') == True and (par2 == '') == False) or (par1 == par2):
        load = par2
    elif (par1 == '') == False and (par2 == '') == True:
        load = par1
    else:
        load = par1 + '_' + par2

    row_number = df_par.index[df_par['name'] == load]
    parameters = df_par.loc[row_number, 'hyperparameters'].to_list()[0]

    return parameters


def evaluate_undamaged(predictions, k):

    predictions = list(predictions)
    predictions.append(0)
    aux = 0
    count_damaged = 0
    for i in range(0, len(predictions), 1):
        if predictions[i] == 1 and predictions[i+1] == 1:
            aux = aux + 1
        elif aux > count_damaged:
            count_damaged = aux
            aux = 0

    if count_damaged+1 > k:
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

    if count_damaged+1 > k:
        return 'ok'
    else:
        return 'no'


loads = ['R1', 'R2', 'R3',
         'T1', 'T2', 'T3',
         'R1_T1', 'R1_T2', 'R1_T3',
         'R2_T1', 'R2_T2', 'R2_T3',
         'R3_T1', 'R3_T2', 'R3_T3']
df_best_parameters = pd.read_pickle(path_parameters)

vet_damage = []
vet_undamage = []
vet_average = []

for k_bits in range(start_k, max_k + 1, 1):

    list_with_damaged = []
    list_without_damage = []
    print('\n*************************************')
    print('Threshold: k = ', k_bits, 'bits consecutives')

    for i in range(0, len(loads), 1):

        par1_test = loads[i]
        par2_test = loads[i]

        dataTrainVal = split_TRAIN.TRAIN(df_in, par1_test, par2_test)
        dataset_validation, dataset_train = split_VALIDATION.setVal(dataTrainVal, val_size=0.01)
        dataset_test = split_TEST.TEST(df_in, par1_test, par2_test)

        final_col = len(df_in.columns) - 1

        X_val = dataset_validation.values[:, 1:final_col - 1]
        y_val = dataset_validation['D_class']

        X_train = dataset_train.values[:, 1:final_col - 1]
        y_train = dataset_train['D_class']

        X_test = dataset_test.values[:, 1:final_col - 1]
        y_test = dataset_test['D_class']

        best_parameters = get_best_parameters(df_best_parameters, par1_test, par2_test)

        clf = tree.DecisionTreeClassifier(**best_parameters, random_state=0)

        clf.fit(X_train, y_train)

        p_train = clf.predict(X_train)
        p_test = clf.predict(X_test)

        acc_train = metrics.accuracy_score(y_train, p_train)
        acc_test = metrics.accuracy_score(y_test, p_test)

        new_p_undamaged = evaluate_undamaged(p_test[0:int(len(p_test)/3)], k_bits)
        new_p_damaged = evaluate_damaged(p_test[int(len(p_test)/3):len(p_test)], k_bits)

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


x_axis = np.arange(start_k, max_k+1, 1)
fig = plt.figure(dpi=500)
plt.plot(x_axis, vet_damage, linewidth = 3)
plt.plot(x_axis, vet_undamage, linewidth = 3)
plt.plot(x_axis, vet_average, linewidth = 3)
plt.xlabel('Thresholds', fontsize=fontsize)
plt.ylabel('Correctness [%]', fontsize=fontsize)
plt.xticks(np.arange(start_k, max_k+1, 1), fontsize=label_size_plot)
plt.yticks(fontsize=label_size_plot)
plt.legend(['Damaged', 'Undamaged', 'Average'], fontsize=13)
plt.grid()
plt.savefig('ThresholdsConsecutives.png')
plt.show()
