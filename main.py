from transformers import pipeline
import warnings
from sklearn.metrics import classification_report
import torch
import cv2 as cv
from torchvision import datasets,transforms
import numpy as np
import matplotlib.pyplot as plt
import traceback
import os
import pandas as pd
import glob
from tqdm import tqdm
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import json
from common.model import DeepLearningModels
from common.ETL import ETL
from manager.image_manager import image_manager
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
    def __init__(self, logger):
        self.logger = logger

    def img_processes(self,val_size, test_size, shuffle_dataset, random_seed, batch_size):
        IMG = image_manager(cfg["img"]['img_path'], cfg["img"]['df_path'],logger)
        # ETL = ETL()
        IMG.get_image_info()
        dataset = IMG.creat_dataset(height = 32, width=32,
                                    transform = transforms.Compose([
                                    transforms.Resize((32,32)),  # 將圖片從原先大小28x28改成LeNet可以接受的輸入大小32x32
                                    transforms.ToTensor(),  # 轉換成tensor並且將像素範圍(range)從[0, 255]改到[0,1]
                                    transforms.Normalize(mean = (0.1307,), std = (0.3081,))]))
        self.train_loader, self.val_loader, self.test_loader = IMG.train_test_split(val_size, test_size, shuffle_dataset, random_seed, batch_size)

    def model_training(self, num_classes ,learning_rate ,num_epochs ,device):
        DL = DeepLearningModels(train_loader=self.train_loader, 
                                val_loader=self.val_loader, 
                                test_loader=self.test_loader, 
                                model_save_path = cfg["setting"]["model_save_path"],
                                model_save_timeformat = cfg["setting"]["timeformat"],
                                logger=logger)
        DL.model_LeNet5(num_classes ,learning_rate ,num_epochs ,device)
        

    def evaluate(self, x, y):
        self.model.evaluate(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model.load(path)

def main():
    val_size, test_size = 0.1, 0.1  # train:val:test=0.8:0.1:0.1
    shuffle_dataset = True
    random_seed= 42
    batch_size = 64  # 每個batch有64張圖片
    num_classes = 2  # 圖片共分成10種類別
    learning_rate = 0.001  # 學習率
    num_epochs = 50  #　訓練總共要跑的回合數，一回合(epoch)即將所有訓練數據都掃過一遍
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = "LeNet5"

    logger.info("Starting ML pipeline")
    pipeline = MLpipeline(logger=logger)
    pipeline.img_processes(val_size, test_size, shuffle_dataset, random_seed, batch_size)
    pipeline.model_training(num_classes ,learning_rate ,num_epochs ,device)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(traceback.format_exc())
        logging.error("Error Message : {}".format(traceback.format_exc()))