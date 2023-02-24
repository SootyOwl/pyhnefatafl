import torch
assert torch.cuda.is_available(), "CUDA is not available!"

from muzero_baseline.muzero import MuZero

from hnefatafl.muzero import HnefataflGame, MuZeroConfig
import os
config = MuZeroConfig()
config.max_num_gpus = 1
config.train_on_gpu = True
config.reanalyse_on_gpu = False
config.batch_size = 1
config.training_steps = 10000
config.num_workers = 1
config.support_size = 10 
mz = MuZero(HnefataflGame, config)



# load the model if we have a checkpoint to load (model.checkpoint in config.results_path)
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--load-path", default=None, type=str)
args = parser.parse_args()

run_path = os.path.join(args.load_path)
checkpoint_path = os.path.join(run_path, "model.checkpoint")

mz.load_model(checkpoint_path)

mz.test_direct(render=True, opponent="self", muzero_player=0)
