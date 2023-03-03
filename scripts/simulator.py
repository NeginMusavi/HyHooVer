import sys
sys.path.append('..')
import zmq
import numpy as np
import msgpack
import argparse
from utils.general_utils import savepklz, temp_seed
import models
import signal

model_names = sorted(name for name in models.__dict__
    if not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='')

parser.add_argument('--model', metavar='MODEL',
                        default='Slplatoon',
                        help='models available: ' +
                            ' | '.join(model_names) +
                            ' (default: Slplatoon)')
parser.add_argument('--args', nargs='+', type=float, help='<Optional> This can be used to pass special arguments to the model.')
parser.add_argument('--initial_states', nargs='+', type=float, help='<Optional> specify the initial_states directly.')
parser.add_argument('--seed', type=int, default=1024, help='Random seed for reproducibility. (default: 1024)')
parser.add_argument('--port', type=int, help='port to listen on', required=True)
parser.add_argument('--save', type=str, help='pklz', required=True)

args = parser.parse_args()

np.random.seed(args.seed)

model = models.__dict__[args.model]()

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:"+str(args.port))

def random_initialization(seed, initial_states=None):
    if initial_states is not None:
        state = initial_states
    else:
        with temp_seed(np.abs(seed) % (2**32)):
            rnd_vector = np.random.rand(model.Theta[1].shape[0] + 1)
            state0 =  float(int(rnd_vector[0] * len(model.Theta[0])))
            state1 = rnd_vector[1:]\
              * (model.Theta[1][:,1] - model.Theta[1][:,0])\
              + model.Theta[1][:,0]

        state = [state0]
        state = state + state1.tolist()

    t = 0.
    is_unsafe = model.is_unsafe(state)
    return state + [t, is_unsafe]

result_table = {}
def save(sig, frame):
    savepklz(result_table, args.save)

signal.signal(signal.SIGUSR1, save)

prev_seed = None
prev_state = None
# print("1")
while True:
    #  Wait for next request from client
    message = socket.recv()
    _message = msgpack.unpackb(message, use_list=False, raw=False)
    # print('recv: ', _message)
    state = _message[:-1]
    seed = int(_message[-1])
    # from IPython import embed; embed()
    t = state[-2]
    if t < 0: # t == -1 for requesting initialization
        # print('----------------------------------')
        state = random_initialization(seed)
        if prev_seed is not None:
            result = (prev_state[-1] > 0 and prev_state[-2]<=model.k)
            if prev_seed in result_table:
                result_table[prev_seed].append(result)
            else:
                result_table[prev_seed] = [result,]
        prev_seed = seed
    else:
        t = t+1
        new_state = model.transition(list(state)[0:-2])
        is_unsafe = model.is_unsafe(new_state)
        state = new_state + [t, is_unsafe]

    prev_state = state
    socket.send(msgpack.packb(state, use_bin_type=True))
