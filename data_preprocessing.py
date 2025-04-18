import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

def load_data(path):
    if path.endswith('.xlsx'):
        return pd.read_excel(path)
    return pd.read_csv(path)

def preprocess(X, fit=True, encoder_path=None, scaler_path=None):
    cat_cols = ['Gender', 'Education Level', 'Job Title']
    num_cols = ['Age', 'Years of Experience']

    if fit:
        # Usar sparse_output en lugar de sparse
        enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_cat = enc.fit_transform(X[cat_cols])
        joblib.dump(enc, encoder_path)

        scl = StandardScaler()
        X_num = scl.fit_transform(X[num_cols])
        joblib.dump(scl, scaler_path)
    else:
        enc = joblib.load(encoder_path)
        scl = joblib.load(scaler_path)
        X_cat = enc.transform(X[cat_cols])
        X_num = scl.transform(X[num_cols])

    df_num = pd.DataFrame(X_num, columns=num_cols, index=X.index)
    df_cat = pd.DataFrame(
        X_cat,
        columns=enc.get_feature_names_out(cat_cols),
        index=X.index
    )
    return pd.concat([df_num, df_cat], axis=1)

def split_data(df,
               target='Salary',
               test_size=0.2,
               random_state=42,
               encoder_path='results/encoder.joblib',
               scaler_path='results/scaler.joblib'):
    y = df[target]
    X = df.drop(columns=[target])
    X_proc = preprocess(
        X, fit=True,
        encoder_path=encoder_path,
        scaler_path=scaler_path
    )
    return train_test_split(
        X_proc, y,
        test_size=test_size,
        random_state=random_state
    )
