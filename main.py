from transformers import pipeline
import glob
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from collections import Counter
import warnings
from sklearn.metrics import classification_report
import torch
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import random
import pandas as pd
import glob
from tqdm import tqdm
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import json
warnings.filterwarnings("ignore")

## logger 設定
now = datetime.now()
log_filename = 'cath_project.log'
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
log_dir = './var/log/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
file_handler = TimedRotatingFileHandler('./var/log/'+log_filename, when="midnight", interval=1, encoding="utf-8", backupCount=9)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

## config 讀取
cfg_path = "./config/config.json"
with open(cfg_path, 'r', encoding='utf-8-sig') as f:
    cfg = json.load(f)

class MLpipeline:
    def __init__(self, model, loss, optimizer, metrics, logger):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.logger = logger

    def compile(self):
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

    def fit(self, x, y, batch_size, epochs, validation_data):
        self.model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_data=validation_data)

    def evaluate(self, x, y):
        self.model.evaluate(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model.load(path)