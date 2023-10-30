import glob
import os
import warnings
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import split_TEST
import split_TRAIN
import split_VALIDATION

warnings.filterwarnings('ignore')

path_res = 'results'
path_input = 'dataframes'
os.makedirs(path_res, exist_ok=True)

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
                dataset_validation, dataset_train = split_VALIDATION.setVal(dataTrainVal, val_size=0.01)
                dataset_test = split_TEST.TEST(df_in, par1_test, par2_test)

                feature_names = df_in.columns.to_list()[1:len(df_in.columns)-1] # first element is the name, the last is 'D'

                X_val = dataset_validation[feature_names]
                y_val = dataset_validation['D_class']

                X_train = dataset_train[feature_names]
                y_train = dataset_train['D_class']

                X_test = dataset_test[feature_names]
                y_test = dataset_test['D_class']

                if alg == 'DT':
                    clf = DecisionTreeClassifier(random_state=0)
                elif alg == 'KNN':
                    clf = KNeighborsClassifier()
                if alg == 'LR':
                    if n_classes > 2:
                        clf = LogisticRegression(multi_class='multinomial', random_state=0)
                    else:
                        clf = LogisticRegression(random_state=0)

                clf.fit(X_train, y_train)
                p_train = clf.predict(X_train)
                p_test = clf.predict(X_test)

                hyperparameters = clf.get_params()

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
                path_alg_singleDf_confMatr = path_alg_singleDf + '\\Confusion Matrices Basic Use'
                os.makedirs(path_alg_singleDf_confMatr, exist_ok=True)
                plt.savefig(f'{path_alg_singleDf_confMatr}\\ConfMat_{load}_{filename}.jpg')

                table_out.loc[loads.index(load)] = load, round(acc_train, 2), round(acc_test, 2), hyperparameters

                # plt.show()

            print('\nMean train accuracy: ', round(mean_train_accuracy * 100 / len(loads), 2), ' %')
            print('\nMean test accuracy: ', round(mean_test_accuracy * 100 / len(loads), 2), ' %')
            table_out.to_excel(f'{path_alg_singleDf}\\res_basic.xlsx')
            table_out.to_pickle(f'{path_alg_singleDf}\\res_basic.pkl')