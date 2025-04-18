import argparse
from train import main as train_main
from evaluate import main as evaluate_main
from predict import main as predict_main

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline: entrenamiento, evaluación y predicción de salarios"
    )
    parser.add_argument(
        "step",
        choices=["train", "evaluate", "predict", "all"],
        help="Paso a ejecutar"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.step in ("train", "all"):
        train_main()
    if args.step in ("evaluate", "all"):
        evaluate_main()
    if args.step in ("predict", "all"):
        predict_main()
