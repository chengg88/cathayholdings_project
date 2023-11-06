import warnings
import traceback
from datetime import datetime
import logging
import torch
from torchvision import transforms
import yaml
from common.model import ImageClassifierTrainer
from manager.image_manager import image_manager
warnings.filterwarnings("ignore")

# config 讀取
cfg_path = './config/config.yaml'
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

# logger 設定
timestamp = datetime.now().strftime(cfg["setting"]["timeformat"])
log_filename = cfg["setting"]["log_path"].format(timestamp)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
file_handler = logging.FileHandler(log_filename + ".log", encoding="utf-8")
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


class MLpipeline:
    def __init__(self, logger):
        self.logger = logger

    def img_processes(self, val_size, test_size, shuffle_dataset, random_seed, batch_size):
        IMG = image_manager(cfg["img"]['img_path'],
                            cfg["img"]['info_path'], logger)
        # ETL = ETL()
        IMG.get_image_info()
        transform=transforms.Compose([transforms.Resize((32, 32)),# 將圖片從原先大小28x28改成LeNet可以接受的輸入大小32x32
                                        transforms.ToTensor(),# 轉換成tensor並且將像素範圍(range)從[0, 255]改到[0,1]
                                        transforms.Normalize(mean=(0.1307,), std=(0.3081,))])
        dataset = IMG.creat_dataset(height=32, width=32,transform=transform)
        self.train_loader, self.val_loader, self.test_loader = IMG.train_test_split(
            val_size, test_size, shuffle_dataset, random_seed, batch_size)

    def model_training(self,model_name, num_classes, learning_rate, num_epochs, device):
        DL = ImageClassifierTrainer(train_loader=self.train_loader,
                                val_loader=self.val_loader,
                                test_loader=self.test_loader,
                                model_save_path=cfg["setting"]["model_save_path"],
                                model_save_timeformat=cfg["setting"]["timeformat"],
                                logger=logger)
        DL.train_pytorch_model(model_name, num_classes, learning_rate, num_epochs, device)



def main():
    val_size, test_size = 0.1, 0.1  # train:val:test=0.8:0.1:0.1
    shuffle_dataset = True
    random_seed = 42
    batch_size = 64  # 每個batch有64張圖片
    num_classes = 2  # 圖片共分成2種類別
    learning_rate = 0.001  # 學習率
    num_epochs = 50  # 　訓練總共要跑的回合數，一回合(epoch)即將所有訓練數據都掃過一遍
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = "LeNet5"

    logger.info("Starting ML pipeline")
    pipeline = MLpipeline(logger=logger)
    pipeline.img_processes(val_size, test_size,
                           shuffle_dataset, random_seed, batch_size)
    pipeline.model_training(model_name, num_classes, learning_rate, num_epochs, device)


if __name__ == "__main__":
    try:
        main()
        file_handler.close()
    except Exception as e:
        print(traceback.format_exc())
        logger.error("Error Message : {}".format(traceback.format_exc()))
