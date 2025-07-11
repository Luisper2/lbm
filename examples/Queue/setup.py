import json
from lbmpy.queue import Queue

with open('apiKeys.json', 'r') as f:
    api_keys = json.load(f)

simulations = [
    {
        'path': '../Re100/setup.py',
        'plots': True,
    },
    {
        'path': '../Re400/setup.py',
        'plots': True,
    },
    {
        'path': '../Re1000/setup.py',
        'plots': True,
    }
]

que = Queue(simulations_queue = simulations, apiKeys = api_keys)
que.launch()
