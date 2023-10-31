import random
import pandas as pd
from sklearn import preprocessing
from sklearn import utils
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

seed_value = 42
random.seed(seed_value)


def setVal(dataset, all_loads, load_tested, val_size=0.2):
    valid_loads = [load for load in all_loads if (load.find(load_tested) == -1) and ('R' in load) and ('T' in load)]

    df_val = pd.DataFrame(columns=dataset.columns)
    df_train = pd.DataFrame(columns=dataset.columns)

    for load in valid_loads:
        for damage_type in [1, 2, 3]:
            combo = f'D{damage_type}_{load}'
            selected_rows = dataset[
                dataset['name_signal'].str.contains(combo)]

            n_samples_for_load = int(len(selected_rows) * val_size)

            indices_val = random.sample(selected_rows.index.tolist(), int(n_samples_for_load))

            indices_train = [s for s in selected_rows.index if s not in indices_val]

            rows_val = dataset.loc[indices_val]
            rows_train = dataset.loc[indices_train]

            # Append rows to the validation and training DataFrames
            df_val = pd.concat([df_val, rows_val], ignore_index=True)
            df_train = pd.concat([df_train, rows_train], ignore_index=True)

    df_train['D_class'] = label_encoder.fit_transform(df_train['D_class'])
    df_val['D_class'] = label_encoder.fit_transform(df_val['D_class'])

    return df_val, df_train
