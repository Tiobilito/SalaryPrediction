import os
import pandas as pd
import joblib

def ensure_dirs():
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/tables', exist_ok=True)

def save_history(history, config_name):
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(f'results/tables/history_{config_name}.csv', index=False)

def save_model(model, config_name):
    model.save(f'results/{config_name}_model.h5')
