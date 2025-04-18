import itertools
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from data_preprocessing import load_data, split_data
from model import build_model
from utils import ensure_dirs, save_history, save_model

# Hiperparámetros a explorar
optimizers = ['SGD', 'Adam', 'RMSprop', 'Adadelta']
layers = [1, 2, 3]
neurons = [32, 64, 128]
lrs = [0.01, 0.001]
eps = [50, 100]
bs = [16, 32]

def main():
    ensure_dirs()
    df = load_data('DATASET-SALARY/salarios.csv')
    X_train, X_test, y_train, y_test = split_data(df)
    records = []

    for opt, hl, nr, lr, ep, bs in itertools.product(
        optimizers, layers, neurons, lrs, eps, bs
    ):
        name = f"opt-{opt}_hl{hl}_nr{nr}_lr{lr}_ep{ep}_bs{bs}"
        model = build_model(X_train.shape[1], hl, nr, lr, opt)
        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        hist = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=ep,
            batch_size=bs,
            callbacks=[es],
            verbose=0
        )
        save_history(hist, name)
        save_model(model, name)
        records.append({
            'config': name,
            'val_mae': hist.history['val_mae'][-1]
        })

    pd.DataFrame(records).to_csv('results/tables/summary_results.csv', index=False)

if __name__ == '__main__':
    main()
