import pandas as pd
import matplotlib.pyplot as plt
from utils import ensure_dirs

def main():
    ensure_dirs()
    summary = pd.read_csv('results/tables/summary_results.csv')
    top3 = summary.nsmallest(3, 'val_mae')['config']
    for cfg in top3:
        hist = pd.read_csv(f'results/tables/history_{cfg}.csv')
        plt.figure()
        plt.plot(hist['loss'], label='train_loss')
        plt.plot(hist['val_loss'], label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend()
        plt.savefig(f'results/plots/{cfg}_loss.png')

        plt.figure()
        plt.plot(hist['mae'], label='train_mae')
        plt.plot(hist['val_mae'], label='val_mae')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.savefig(f'results/plots/{cfg}_mae.png')

if __name__ == '__main__':
    main()
