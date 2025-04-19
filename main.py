import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(
        description='Orquestador: train y predict de salarios'
    )
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_train = sub.add_parser('train', help='Entrenar modelo')
    p_train.add_argument('--dataset',    default='DATASET-SALARY')
    p_train.add_argument('--output',     default='models')
    p_train.add_argument('--results_dir',default='results')
    p_train.add_argument('--epochs',     type=int,   default=100)
    p_train.add_argument('--batch_size', type=int,   default=32)
    p_train.add_argument('--learning_rate', type=float, default=0.001)
    p_train.add_argument('--hidden_layers', type=int, default=2)
    p_train.add_argument('--neurons',       type=int, default=64)
    p_train.add_argument('--optimizer',
                         choices=['SGD','Adam','RMSprop','Adadelta'],
                         default='Adam')

    p_pred = sub.add_parser('predict', help='Predecir salarios')
    p_pred.add_argument('--dataset',    default='DATASET-SALARY')
    p_pred.add_argument('--model',      required=True)
    p_pred.add_argument('--results_dir',default='results')

    args = parser.parse_args()

    if args.cmd == 'train':
        cmd = [
            'python', 'train.py',
            '--dataset', args.dataset,
            '--output', args.output,
            '--results_dir', args.results_dir,
            '--epochs', str(args.epochs),
            '--batch_size', str(args.batch_size),
            '--learning_rate', str(args.learning_rate),
            '--hidden_layers', str(args.hidden_layers),
            '--neurons', str(args.neurons),
            '--optimizer', args.optimizer
        ]
    else:  # predict
        cmd = [
            'python', 'predict.py',
            '--dataset', args.dataset,
            '--model', args.model,
            '--results_dir', args.results_dir
        ]

    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    main()
