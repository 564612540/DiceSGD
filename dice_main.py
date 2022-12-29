import torch
import copy
import numpy as np
from dicesgd.optimizer.optim import dicesgd
from dicesgd.data.loader import generate_dataset
from dicesgd.utils.logger import logger
from dicesgd.utils.utils import arg_parser
from dicesgd.model.constructor import generate_model

if __name__ == '__main__':
    args = arg_parser()
    log = logger()
    data_train, data_test, data_cfg = generate_dataset(args.data_cfg)
    optimizer = dicesgd(args)
    model = generate_model(args.model_cfg)
    