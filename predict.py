import argparse
import os
import pandas as pd
from data_preprocessing import load_data, preprocess
from keras.models import load_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predicción de salarios para nuevos empleados')
    parser.add_argument('--dataset',     default='DATASET-SALARY',
                        help='Carpeta que contiene prediccion.xlsx')
    parser.add_argument('--model',       required=True,
                        help='Ruta al archivo .h5 del modelo a usar')
    parser.add_argument('--results_dir', default='results',
                        help='Directorio raíz para resultados')
    args = parser.parse_args()

    # Nombre base del modelo (sin extensión)
    model_name = os.path.splitext(os.path.basename(args.model))[0]
    tables_dir = os.path.join(args.results_dir, 'tables', model_name)
    os.makedirs(tables_dir, exist_ok=True)

    # Cargar datos nuevos
    df_new = load_data(os.path.join(args.dataset, 'prediccion.xlsx'))
    desired = df_new['Desired salary']
    X_new = df_new.drop(columns=['Desired salary'])
    Xp    = preprocess(
        X_new,
        fit=False,
        encoder_path='results/encoder.joblib',
        scaler_path='results/scaler.joblib'
    )

    # Cargar modelo y predecir
    model = load_model(args.model)
    preds = model.predict(Xp).flatten()
    errors_pct = (preds - desired) / desired * 100

    # Guardar resultados
    df_out = pd.DataFrame({
        'predicted_salary': preds,
        'desired_salary':   desired,
        'error_%':          errors_pct
    })
    out_path = os.path.join(tables_dir, 'predictions_errors.csv')
    df_out.to_csv(out_path, index=False)

    print(df_out)
    print(f"\nPredicciones y errores guardados en: {out_path}")
