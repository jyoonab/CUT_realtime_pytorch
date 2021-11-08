from createModel import create_model
from utility import get_transform, tensor2im
from PIL import Image

import torch.utils.data
import numpy as np
import cv2
import argparse
import json

import time

class CUTGan():
    def __init__(self):
        '''Create Options'''
        parser = argparse.ArgumentParser()

        with open('option.json', 'r') as f:
            json_data = json.load(f)

        for key in json_data:
            parser.add_argument(key, nargs='?', default=json_data[key])

        self.opt = parser.parse_args()

        # hard-code some parameters for test
        self.opt.num_threads = 0   # test code only supports num_threads = 1
        self.opt.batch_size = 1    # test code only supports batch_size = 1
        self.opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
        self.opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
        self.opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

        '''Create Model'''
        self.model = create_model(self.opt)      # create a model given opt.model and other options


    '''Start Converting Image'''
    def start_converting(self, A_path, B_path):
        '''Preprocess Image'''
        A_image = Image.open(A_path).convert('RGB')
        B_image = Image.open(B_path).convert('RGB')

        transform = get_transform()
        A = transform(A_image)
        B = transform(B_image)

        preprocessed_data = torch.utils.data.DataLoader(
            [{'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}],
            batch_size = self.opt.batch_size,
            shuffle = not self.opt.serial_batches,
            num_workers = int(0),
            drop_last = False,
            )


        '''Start Converting Image'''
        for i, data in enumerate(preprocessed_data):
            self.model.data_dependent_initialize(data)

            self.model.setup(self.opt)               # regular setup: load and print networks; create schedulers
            self.model.parallelize()

            self.model.set_input(data)  # unpack data from data loader
            self.model.test()           # run inference

            visuals = self.model.get_current_visuals()  # get image results

            start = time.time()

            image_result = tensor2im(visuals['fake_B'])

            image_result = cv2.cvtColor(image_result, cv2.COLOR_RGB2BGR)

            print("time :", time.time() - start)

            h, w, c = image_result.shape
            print('width:  ', w)
            print('height: ', h)
            print('channel:', c)

            cv2.imshow('video', image_result)
            cv2.waitKey(0)

if __name__ == '__main__':
    A_path = './images\\3.png'
    B_path = './images\\4.png'

    cut = CUTGan()
    cut.start_converting(A_path, B_path)
