import pandas as pd
from data_preprocessing import load_data, preprocess
from keras.models import load_model

def main():
    df_new = load_data('DATASET-SALARY/prediccion.xlsx')
    desired = df_new['Desired salary']
    X_new = df_new.drop(columns=['Desired salary'])
    Xp = preprocess(
        X_new, fit=False,
        encoder_path='results/encoder.joblib',
        scaler_path='results/scaler.joblib'
    )
    best = pd.read_csv('results/tables/summary_results.csv') \
             .nsmallest(1, 'val_mae')['config'].iloc[0]
    model = load_model(f'results/{best}_model.h5')

    preds = model.predict(Xp).flatten()
    errs = (preds - desired) / desired * 100
    df_out = pd.DataFrame({
        'predicted_salary': preds,
        'desired_salary': desired,
        'error_%': errs
    })
    df_out.to_csv('results/tables/predictions_errors.csv', index=False)
    print(df_out)

if __name__ == '__main__':
    main()
