import itertools
import pandas as pd
from keras.callbacks import EarlyStopping
from data_preprocessing import load_data, split_data
from model import build_model
from utils import ensure_dirs, save_history, save_model

# Hiperparámetros a explorar
optimizers = ['SGD', 'Adam', 'RMSprop', 'Adadelta']
hidden_layers_list = [1, 2, 3]
neuron_list = [32, 64, 128]
learning_rates = [0.01, 0.001]
epochs_list = [50, 100]
batch_sizes = [16, 32]

def main():
    ensure_dirs()
    df = load_data('DATASET-SALARY/salarios.csv')
    X_train, X_test, y_train, y_test = split_data(df)
    records = []

    for opt, hl, nr, lr, ep, bs in itertools.product(
        optimizers,
        hidden_layers_list,
        neuron_list,
        learning_rates,
        epochs_list,
        batch_sizes
    ):
        config_name = f"opt-{opt}_hl{hl}_nr{nr}_lr{lr}_ep{ep}_bs{bs}"
        model = build_model(X_train.shape[1], hl, nr, lr, opt)
        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=ep,
            batch_size=bs,
            callbacks=[es],
            verbose=0
        )
        save_history(history, config_name)
        save_model(model, config_name)
        records.append({
            'config': config_name,
            'val_mae': history.history['val_mae'][-1]
        })

    pd.DataFrame(records).to_csv('results/tables/summary_results.csv', index=False)

if __name__ == '__main__':
    main()
