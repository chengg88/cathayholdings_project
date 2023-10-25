import cv2
"""
影像讀取，篩選/前處理等在這
"""
class load_image:
    def __init__(self, path, logger):
        self.path = path
        self.logger = logger
        self.img = cv2.imread(self.path)

    def get_image(self):
        return self.img

    def get_path(self):
        return self.path

    def get_size(self):
        return self.img.shape

