# %%
import torch
assert torch.cuda.is_available(), "CUDA is not available!"

# %%
from muzero_baseline.muzero import MuZero

from hnefatafl.muzero import HnefataflGame, MuZeroConfig
import os
config = MuZeroConfig()
config.network = "resnet"
config.max_num_gpus = 1
config.train_on_gpu = True
config.reanalyse_on_gpu = False
config.selfplay_on_gpu = False
config.batch_size = 1
config.training_steps = 20000
config.num_workers = 8
config.max_moves = 250
config.temperature_threshold = 30
config.support_size = 10
config.num_unroll_steps = 20
config.num_simulations = 50

mz = MuZero(HnefataflGame, config)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--load-path", type=str, default=None)
args = parser.parse_args()
# load the model if we have a checkpoint to load (model.checkpoint in config.results_path)
run_path = os.path.join(args.load_path)
checkpoint_path = os.path.join(run_path, "model.checkpoint")
replay_buffer_path = os.path.join(run_path, "replay_buffer.pkl")

if os.path.exists(checkpoint_path):
    if os.path.exists(replay_buffer_path):
        mz.load_model(checkpoint_path, replay_buffer_path)
    else:
        mz.load_model(checkpoint_path)
if os.path.exists(replay_buffer_path):
    mz.load_model(replay_buffer_path=replay_buffer_path)


mz.train(per_step_progress=True)
