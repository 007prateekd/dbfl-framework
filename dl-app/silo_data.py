import pandas as pd
from imblearn.over_sampling import SMOTE
from pathlib import Path


TRAIN_RATIO = 0.8
N_CLIENTS = 2


df = pd.read_csv("data.csv")
df = df.iloc[:20000]
smote = SMOTE(sampling_strategy=0.3, random_state=42)
df_train, df_test = df.iloc[:int(TRAIN_RATIO * len(df))], df.iloc[int(TRAIN_RATIO * len(df)):]
n_train, n_test = len(df_train), len(df_test)

for i in range(N_CLIENTS):
    fac = n_train // N_CLIENTS
    st = i * fac
    en = n_train if i == N_CLIENTS - 1 else (i + 1) * fac
    X, y = df_train.iloc[st:en, :-1], df_train.iloc[st:en, -1]
    X_smote, y_smote = smote.fit_resample(X.astype('float'), y)
    
    path_dir = Path(f"silo_{i}")
    path_dir.mkdir()
    X_smote.to_csv(f"silo_{i}/data.csv")
    y_smote.to_csv(f"silo_{i}/labels.csv")

for i in range(N_CLIENTS):
    fac = n_test // N_CLIENTS
    st = i * fac
    en = n_test if i == N_CLIENTS - 1 else (i + 1) * fac
    X, y = df_test.iloc[st:en, :-1], df_test.iloc[st:en, -1]
    path_dir = Path(f"test_{i}")
    path_dir.mkdir()
    X.to_csv(f"test_{i}/data.csv")
    y.to_csv(f"test_{i}/labels.csv")

