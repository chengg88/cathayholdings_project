import datetime
import warnings
import os
from transformers import AutoModelForImageClassification
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
import torch
import seaborn as sns
from matplotlib import pyplot as plt
from torch import nn
from common.DLModel.LeNet5 import LeNet5


warnings.filterwarnings("ignore")

class ImageClassifierTrainer:
	def __init__(self, train_loader, val_loader, test_loader, model_save_path, model_save_timeformat, logger):
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.test_loader = test_loader
		self.model_save_path = model_save_path
		self.model_save_timeformat = model_save_timeformat
		self.logger = logger

	def model_huggingface(self):
		# 模型來自 HuggingFace
		model = AutoModelForImageClassification.from_pretrained("Devarshi/Brain_Tumor_Detector_swin")
		return model

	def train_pytorch_model(self, model_name, num_classes, learning_rate, num_epochs, device):
		# 訓練 PyTorch 模型
		if model_name == "LeNet5":
			model = LeNet5(num_classes).to(device)
		self.logger.info("Model:" + model_name)
		criterion = nn.CrossEntropyLoss()  # 多分類任務的損失函數
		optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
		total_step = len(self.train_loader)

		for epoch in range(num_epochs):
			for i, (images, labels) in enumerate(self.train_loader):
				images = images.to(device)
				labels = labels.to(device)

				outputs = model(images)
				loss = criterion(outputs, labels)

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

			if (epoch + 1) % 10 == 0:
				self.logger.info('Epoch [{}/{}], Step [{}/{}] - Loss: {:.4f}'.format(
					epoch + 1, num_epochs, i + 1, total_step, loss.item()))

		self.evaluate(model, device)
		self.save_model(model, model_name=model_name, path=self.model_save_path)
		self.logger.info("Model training complete")
		return model

	def evaluate(self, model, device):
		# 評估模型
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

			cm = confusion_matrix(labels.cpu().numpy(), predicted.cpu().numpy())
			sns.heatmap(cm, annot=True, fmt="d")
			plt.xlabel("Predicted")
			plt.ylabel("True")
			plt.title("Confusion Matrix")
			plt.savefig("image/" + "confusion_matrix.png")

			recall = recall_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro')
			accuracy = accuracy_score(labels.cpu().numpy(), predicted.cpu().numpy())
			f1 = f1_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro')
			precision = precision_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro')

			self.logger.info("Recall on test images: {:.2f}%".format(recall * 100))
			self.logger.info('Accuracy on test images: {:.2f}%'.format(accuracy * 100))
			self.logger.info("F1-score on test images: {:.2f}%".format(f1 * 100))
			self.logger.info("Precision on test images: {:.2f}%".format(precision * 100))

	def save_model(self, model, model_name, path):
		timestamp = datetime.datetime.now().strftime(self.model_save_timeformat)
		if not hasattr(self, "timestamp"):
			self.timestamp = timestamp
		model_save_folder = path[:-2].format(self.timestamp)
		if not os.path.exists(model_save_folder):
			print(f"Model save location: {model_save_folder}")
			os.mkdir(model_save_folder)

		model_save_full_path = path.format(self.timestamp, model_name) + ".pkl"
		torch.save(model.state_dict(), model_save_full_path)
		self.logger.info("{0} Path: {1} => Saved.".format(model_name, model_save_full_path))