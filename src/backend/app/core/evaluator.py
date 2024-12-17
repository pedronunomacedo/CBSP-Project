from genericpath import isfile
from ntpath import join
from os import listdir
import requests

class Evaluator:
    # Get files from specific directory
    @staticmethod
    def get_files(directory):
        return [f for f in listdir(directory) if isfile(join(directory, f))]

    @staticmethod
    def evaluate():
        mp3_files = Evaluator.get_files('../data/')

        responses = []
        # Go through each music file
        for music in mp3_files:
            file_path = f'../data/{music}'
            print(f'Processing {music}...')

            # HTTP request to '/bpm_per_second' endpoint
            response = requests.post('http://localhost:8000/bpm_per_second', files={'file': open(file_path, 'rb')})
            print("response: ", response.json())
            responses.append(response.json())


    




