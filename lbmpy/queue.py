import os
import time
import requests
import subprocess
from pathlib import Path

class Queue:
    def __init__(self, simulations_queue: list = [], apiKeys: dict = {}):
        self.queue = simulations_queue
        self.apiKeys = apiKeys

    def sendMessage(self, message: str = '') -> bool:
        try:
            resp = requests.post(f'https://api.telegram.org/bot{self.apiKeys['telegramToken']}/sendMessage', data = {
                'chat_id':    self.apiKeys['chatID'],
                'text':       message,
                'parse_mode': 'Markdown'
            }, timeout = 10)

            resp.raise_for_status()
            
            return True
        except requests.exceptions.HTTPError as http_err:
            return False
        except requests.exceptions.RequestException as err:
            return False
        
    def sendPhoto(self, path: str = '', message: str = '') -> bool:
        try:
            with open(path, 'rb') as photo:
                resp = requests.post(f'https://api.telegram.org/bot{self.apiKeys['telegramToken']}/sendPhoto', data = {
                    'chat_id':    self.apiKeys['chatID'],
                    'text':       message,
                    'parse_mode': 'Markdown'
                }, files = { 'photo': photo }, timeout = 10)

                resp.raise_for_status()
            
            return True
        except requests.exceptions.HTTPError as http_err:
            return False
        except requests.exceptions.RequestException as err:
            return False
    
    def run(self, simulation: dict) -> str:
        start = time.perf_counter()
        
        try:
            subprocess.run(['python', simulation['path']], check=True, capture_output=True, text=True)

            return f'{simulation['path']} successfully ran ({(time.perf_counter() - start):.3f}s)'
        except subprocess.CalledProcessError as e:
            return f'{simulation['path']} failed ({e.stderr.strip()}) ({(time.perf_counter() - start):.3f}s)'
        except BaseException as e:
            return f'{simulation['path']} failed ({e}) ({(time.perf_counter() - start):.3f}s)'

    def launch(self):
        for simulation in self.queue:
            try:
                result = self.run(simulation)

                self.sendMessage(message = result)

                if 'plots' in simulation and simulation['plots']:
                    dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__import__('__main__').__file__)), simulation['path'], '..', 'results', 'plots'))

                    for img in Path(dir).rglob('Velocity.png'):
                        success = self.sendPhoto(path = str(img), message = f'`{img.relative_to(dir)}`')

                        time.sleep(1)
                    
                        if not success:
                            print(f'Fail submitting {img}')
            except Exception as e:
                print("Ocurrió un error:", e)
            