import torch
assert torch.cuda.is_available(), "CUDA is not available!"

# %%
from muzero_baseline.muzero import MuZero
from hnefatafl import Board
from hnefatafl.muzero import HnefataflGame, MuZeroConfig, MuZeroResnet4096Policy, MuZeroResnetSmall, MuZeroResnetSmallLargerHead
import os
import json
import datetime

# %%
config = MuZeroResnet4096Policy()

# %%
config.num_workers = 8 # parallelize the self-play but allow some threads for the reanalysis
config.opponent = "expert"
config.max_moves = 150  # Most randomly generated games are shorter than 250, so this is a good upper bound. "expert" (i.e always capture) games are usually shorter than 100.

config.num_simulations = 50
config.training_steps = 1e6 # 1 million training steps
config.td_steps = config.max_moves + 1 # number of steps to unroll the game for the value function
# dump the config to a file so we can use it later if we want
# add the variables to the path so we can easily identify the results
# get class name
config_name = config.__class__.__name__.split(".")[-1].split("'")[0]
config.results_path = f"{config.results_path}_{config_name}__{config.opponent}_{config.num_simulations}_{config.max_moves}_{config.temperature_threshold}"
None if os.path.exists(config.results_path) else os.mkdir(config.results_path)
with open(os.path.join(config.results_path, "config.json"), "w") as f:
    # copy the config to a dict so we can serialize it
    config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith("_") and k != "results_path"}
    json.dump(config_dict, f)

# %%
mz = MuZero(HnefataflGame, config)

# %%
# load the model if we have a checkpoint to load (model.checkpoint in the previous config.results_path)
# use glob to find the latest checkpoint from a previous run
# run_path = max(glob.glob(os.path.join(config.results_path, "run_*")), key=os.path.getmtime)
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--run-path", default=None, type=str)

args = parser.parse_args()
run_path = args.run_path or config.results_path
checkpoint_path = os.path.join(run_path, "model.checkpoint")
replay_buffer_path = os.path.join(run_path, "replay_buffer.pkl")

if os.path.exists(checkpoint_path):
    if os.path.exists(replay_buffer_path):
        mz.load_model(checkpoint_path, replay_buffer_path)
    else:
        mz.load_model(checkpoint_path)
if os.path.exists(replay_buffer_path):
    mz.load_model(replay_buffer_path=replay_buffer_path)


# %%
mz.train(per_step_progress=True)