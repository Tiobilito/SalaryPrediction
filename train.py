import argparse
import os
import csv
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from data_preprocessing import load_data, split_data
from model import build_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, model_save_path):
    checkpoint = ModelCheckpoint(
        model_save_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )
    earlystop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=1,
        mode='min',
        restore_best_weights=True
    )
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, earlystop],
        verbose=1
    )
    return history

def plot_and_save(history, X_val, y_val, plots_dir, tables_dir, model_name):
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    # 1) Curvas de Loss y MAE
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Evolución de Pérdida')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f'{model_name}_loss.png'))

    plt.figure()
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title('Evolución de MAE')
    plt.xlabel('Época')
    plt.ylabel('MAE')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f'{model_name}_mae.png'))

    # 2) Predicción sobre set de validación
    preds = model.predict(X_val).flatten()
    # 3) Métricas RMSE y MAPE
    rmse = sqrt(mean_squared_error(y_val, preds))
    mape = mean_absolute_percentage_error(y_val, preds) * 100

    # Guardar métricas en CSV
    metrics_path = os.path.join(tables_dir, 'metrics.csv')
    with open(metrics_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['rmse', 'mape_%'])
        writer.writerow([rmse, mape])

    # 4) Gráfica real vs predicho
    plt.figure()
    plt.scatter(y_val, preds, alpha=0.6)
    mn, mx = min(min(y_val), min(preds)), max(max(y_val), max(preds))
    plt.plot([mn, mx], [mn, mx], '--', linewidth=1)
    plt.title('Salario Real vs Predicho')
    plt.xlabel('Real')
    plt.ylabel('Predicho')
    plt.savefig(os.path.join(plots_dir, f'{model_name}_real_vs_pred.png'))
    plt.close('all')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entrenar MLP para predicción de salarios')
    parser.add_argument('--dataset',      default='DATASET-SALARY',
                        help='Carpeta que contiene salarios.csv')
    parser.add_argument('--output',       default='models',
                        help='Directorio donde se guardará el .h5')
    parser.add_argument('--results_dir',  default='results',
                        help='Directorio raíz para resultados')
    parser.add_argument('--epochs',       type=int,   default=100,
                        help='Épocas de entrenamiento')
    parser.add_argument('--batch_size',   type=int,   default=32,
                        help='Tamaño de batch')
    parser.add_argument('--learning_rate',type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--hidden_layers',type=int,   default=2,
                        help='Capas ocultas')
    parser.add_argument('--neurons',      type=int,   default=64,
                        help='Neuronas por capa')
    parser.add_argument('--optimizer',
                        choices=['SGD','Adam','RMSprop','Adadelta'],
                        default='Adam',
                        help='Optimizador')
    args = parser.parse_args()

    # Nombre corto de modelo
    lr_tag = str(args.learning_rate).replace('0.', 'lr')
    model_name = f"{args.optimizer.lower()}_h{args.hidden_layers}_n{args.neurons}_{lr_tag}"

    # Crear carpetas
    os.makedirs(args.output, exist_ok=True)
    plots_dir  = os.path.join(args.results_dir, 'plots',  model_name)
    tables_dir = os.path.join(args.results_dir, 'tables', model_name)

    # Cargar y dividir datos
    df = load_data(os.path.join(args.dataset, 'salarios.csv'))
    X_train, X_val, y_train, y_val = split_data(df)

    # Construir modelo
    model = build_model(
        input_dim=X_train.shape[1],
        hidden_layers=args.hidden_layers,
        neurons=args.neurons,
        learning_rate=args.learning_rate,
        opt_name=args.optimizer
    )

    # Guardar ruta del modelo
    model_path = os.path.join(args.output, f'{model_name}.h5')
    if os.path.exists(model_path):
        os.remove(model_path)

    # Entrenar
    history = train_model(
        model,
        X_train, y_train,
        X_val,   y_val,
        args.epochs,
        args.batch_size,
        model_path
    )

    # Graficar y guardar métricas
    plot_and_save(history, X_val, y_val, plots_dir, tables_dir, model_name)

    # Guardar hiperparámetros + MAE final
    final_mae = history.history['val_mae'][-1]
    hyp_path  = os.path.join(tables_dir, 'hyperparameters.csv')
    with open(hyp_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'model_name','learning_rate','epochs','batch_size',
            'hidden_layers','neurons','optimizer','val_mae'
        ])
        writer.writerow([
            model_name,
            args.learning_rate,
            args.epochs,
            args.batch_size,
            args.hidden_layers,
            args.neurons,
            args.optimizer,
            final_mae
        ])

    print(f"\nModelo guardado en: {model_path}")
    print(f"Gráficas en:    {plots_dir}")
    print(f"Métricas en:    {tables_dir}")
