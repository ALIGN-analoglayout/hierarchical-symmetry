"""SimGNN runner."""
from os import path
from utils import tab_printer
from simgnn import SimGNNTrainer
from param_parser import parameter_parser

def main():
    """
    Parsing command line parameters, reading data.
    Fitting and scoring a SimGNN model.
    """
    args = parameter_parser()
    if path.isfile(args.saved_model):
        trainer = SimGNNTrainer(args)
        trainer.score()
    else:
        tab_printer(args)
        trainer = SimGNNTrainer(args)
        trainer.fit()
        trainer.score()

if __name__ == "__main__":
    main()
