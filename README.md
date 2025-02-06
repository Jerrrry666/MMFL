# New note in MMFL

## dataset

new **multimodal** processor in dataset_utils.py

* food101 (image, text)

### generate fold name temple: `food101-version`

use '-'to split the name, for load model module in `model/config.py`, `dataset_arg = args.dataset.split('-')[0]`

args.dataset = 'food101-version'

## model

* resnet for multimodal (image,text)

## alg

* [CVPR24] MLA (sync FL)

# ===========================

# A Simple Async FL Simulation Framework

A *simple and easy-to-extend*
asynchronous federated learning (AFL) simulation framework.

The asynchronous simulation is based on a prior queue.

+ Insert training-finished clients into the prior queue based on supposed finished time
+ Pop the first item in the prior queue
+ Update the simulated wall-clock time based on the item

We also provide simulation of synchronous federated learning like FedAvg.

## Getting Started

+ Download this project

```
git clone https://github.com/boyi-liu/Asynchronous-Federated-Learning-Simulation.git
```

+ Install required packages
+ Partition datasets

```
cd dataset
python generate_cifar10.py noniid balance dir
```

+ Config hyperparameters

There are two places to config, one in `/script/config.yaml`, another in `utils/options.py`.
The priority follows: `args>yaml`.
If you config hyperparameters in `utils/options.py`, it will overwrite that in `args`.

+ Run evaluation

```
cd script
bash run.sh {your_suffix}
```

## How to extend

### Create a new file

Create a new `{your_algorithm}.py` file inside `trainer.alg`

### Extend the Client and Server

If you are working on a **synchronous** FL algorithm, just extend the Client and Server class in `trainer.base`

```
from trainer.base import BaseServer, BaseClient

class Client(BaseClient):
    def __init__(self, id, args, dataset):
        super().__init__(id, args, dataset)

class Server(BaseServer):
    def __init__(self, id, args, dataset, clients):
        super().__init__(id, args, dataset, clients)
```

Otherwise, you may extend the Client and Server in `trainer.asyncbase`

```
from trainer.asyncbase import AsyncBaseServer, AsyncBaseClient

class Client(AsyncBaseClient):
    def __init__(self, id, args, dataset):
        super().__init__(id, args, dataset)

class Server(AsyncBaseServer):
    def __init__(self, id, args, dataset, clients):
        super().__init__(id, args, dataset, clients)
```

### Config your hyperparameters

For algorithm-specific hyperparameters,
it is recommended to add a `add_args()` function inside your file

```
def add_args(parser):
    parser.add_argument('--{your_param}', type=int, default=1)
    return parser.parse_args()
```

And all general args could be found in `utils/options.py`

### Implement your algorithms

We claim that each algorithm should overwrite the function `run()`,
because it stands for the main workflow of your algorithm.

You can overwrite or add any function as you want then.

## Acknowledgements

The data partitioning module is adopted from [PFLlib](https://github.com/TsingZ0/PFLlib).

