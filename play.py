import torch
assert torch.cuda.is_available(), "CUDA is not available!"

from muzero_baseline.muzero import MuZero

from hnefatafl.muzero import HnefEnv, HnefataflGame, MuZeroConfig, MuZeroResnetSmallLargerHead
import os


# load the model if we have a checkpoint to load (model.checkpoint in config.results_path)
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="model.checkpoint", type=str)
parser.add_argument("--board", type=str, default=None)
args = parser.parse_args()

# set the board if we have one
if args.board:
    # DEFAULT_BOARD_CODE = args.board doesn't work because it's a constant, so we have to do this
    HnefEnv.DEFAULT_BOARD_CODE = args.board

config = MuZeroResnetSmallLargerHead()
config.num_workers = 1
config.max_moves = 100
config.temperature_threshold = 30
mz = MuZero(HnefataflGame, config)

checkpoint_path = os.path.join(args.checkpoint)

mz.load_model(checkpoint_path)

results = mz.test_direct(render=True, opponent="human", muzero_player=0, num_tests=1)

print("Results: ", results)
