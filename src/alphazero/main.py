import gc
import logging
import sys

import coloredlogs

from alphazero.coach import Coach
from hnefatafl.game import HnefataflGame as Game
from hnefatafl.ai import NNetWrapper as nn
from alphazero.utils import dotdict

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')
# stop tensorflow from being noisy
logging.getLogger('tensorflow').setLevel(logging.ERROR)

args = dotdict({
    'numIters': 100,
    'numEps': 25,               # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        # Number of moves before dropping temperature to 0 (ie playing according to the max). Temp is 1 if the number of moves < tempThreshold.
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 100000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 30,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 20,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 3,                 # The amount of exploration vs exploitation to allow for MCTS. 
                                #   Larger values will increase exploration of the tree, smaller values will increase exploitation of known good moves. 
                                #   1-4 is a good range depending on the size of the action space.

    'move_limit': 25,           # Maximum number of moves in a game before it is considered a draw. This is to prevent games from going on forever, and to speed up training by exiting games early.

    'checkpoint': f'./checkpoint/{Game.__name__}/',
    'load_model': True,
    'load_folder_file': (f'./checkpoint/{Game.__name__}', 'checkpoint_1.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})

def main():
    log.info('Loading %s...', Game.__name__)
    g = Game(move_limit=args.move_limit)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    # Allow deep tree search by increasing the stack size and recursion limit of the main thread
    import threading
    threading.stack_size(1 << 27)
    sys.setrecursionlimit(1 << 20)
    thread = threading.Thread(target=main)
    
    thread.start()
    thread.join()
