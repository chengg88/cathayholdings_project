from transformers import AutoFeatureExtractor, AutoModelForImageClassification

class Classification:
    def __init__(self) -> None:
        pass

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

    def model_pytorch(self):
        """
        Model from PyTorch
        """
        pass