from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
import warnings
import torch
import os
import datetime
from torch import nn
warnings.filterwarnings("ignore")


class DeepLearningModels:
    def __init__(self , train_loader, val_loader, test_loader, model_save_path,model_save_timeformat, logger):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model_save_path = model_save_path
        self.model_save_timeformat= model_save_timeformat
        self.logger = logger


    def model_huggingface(self):
        """
        Model from HuggingFace
        """
        model = AutoModelForImageClassification.from_pretrained("Devarshi/Brain_Tumor_Detector_swin")
        return model

    def model_keras(self):
        """
        Model from Keras
        """
        pass

    def model_tensorflow(self):
        """
        Model from Tensorflow
        """
        pass

    def model_LeNet5(self ,
                     num_classes ,
                     learning_rate ,
                     num_epochs ,
                     device):
        """
        Model from PyTorch
        """
        model = LeNet5(num_classes).to(device)
        cost = nn.CrossEntropyLoss()  # 交叉墒損失函數，適用多分類任務的損失函數
        #Setting the optimizer with the model parameters and learning rate
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adam優化器
        #this is defined to print how many steps are remaining when training
        total_step = len(self.train_loader)
        for epoch in range(num_epochs):  # 總共進行共num_epochs個回合的訓練
            for i, (images, labels) in enumerate(self.train_loader):  
                images = images.to(device)  # 將tensor移動到GPU或CPU上訓練
                labels = labels.to(device)
                
                # 前向傳播(Forward pass)
                outputs = model(images)
                loss = cost(outputs, labels)
                    
                # 反向傳播(Backward pass) and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                        
            if (epoch+1) % 10 == 0:
                print ('Epoch [{}/{}], Step [{}/{}] ====== Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        DeepLearningModels.evaluate(self,model,device)
        DeepLearningModels.model_save(self,model,model_name = "LeNet5",path = self.model_save_path)
        return model

    def evaluate(self, model,device):
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in self.test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item() 
        # 混沌矩陣report
        print("Recall of the model on the test images: {} %" .format(round(recall_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro')*100,4)))
        print('Accuracy of the model on the test images: {} %'.format(round(100 * correct / total,4)))
        print("F1-score of the model on the test images: {} %" .format(round(f1_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro')*100,4)))


    def model_save(self, model, model_name, path):
        model_save_timeformat = self.model_save_timeformat

		# 查看timestamp是否已經存在這個class中
        if hasattr(self, "timestamp"):
            timestamp = self.timestamp
        else:
            timestamp = datetime.datetime.now().strftime(model_save_timeformat)
            self.timestamp = timestamp

		# 創造一個資料夾儲存這次訓練的所有模型
        mkdir_path = path[:-2].format(timestamp)
        if not os.path.exists(mkdir_path):
            print(f"模型儲存位置: {mkdir_path}")
            os.mkdir(mkdir_path)

		# 得到最終的位置
        model_save_fullPath = path.format(timestamp, model_name)
        model_save_fullPath = model_save_fullPath + ".pkl"
        torch.save(model.state_dict(), model_save_fullPath)
        print(
		    "{0}_Path:{1} =========> Saved.".format(
		        model_name, model_save_fullPath
		        )
		    )
        self.logger.info(
		    "{0}_Path:{1} =========> Saved.".format(
		        model_name, model_save_fullPath
		        )
		    )

          
class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out