from model import MTCNN, InceptionResnetV1
import numpy as np
from PIL import Image
import torch
import time
import requests
import json
from io import BytesIO


# only for downloading models.
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


class FaceNet:
    def __init__(self):
        self.mtcnn = MTCNN()
        self._model = InceptionResnetV1(pretrained="vggface2").eval()

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    def resize_image(self, img: Image, dimensions: tuple = (400, 400)) -> Image:
        """Resize an image to the required size. By default, VGG16 has been trained on (160, 160) images."""
        if isinstance(img, str):
            img: Image = self.load_image(img)
            img = img.resize(dimensions, Image.ANTIALIAS)  # preserve aspect ratio
        return img

    def load_image(self, img: str) -> Image:
        """Load an image from a file path."""
        if img.startswith("http"):
            img = BytesIO(requests.get(img).content)
        img = Image.open(img)
        return img

    def convert_img_to_rgb(self, img: Image) -> np.ndarray:
        """if an image is in grayscale, convert it to RGB"""
        if img.mode == "RGB":
            return img
        # use PIL to convert image to RGB
        img = Image.new("RGB", img.size)
        img = np.asarray(img)
        # img = cv2.cvtColor(rgb_img, cv2.COLOR_GRAY2RGB) # TODO: opencv version
        return img

    def get_embedding(self, img: Image) -> torch.Tensor:
        """Get the embeddings for an image."""
        img = self.resize_image(img)
        img = self.convert_img_to_rgb(img)  # convert to RGB, if grayscale

        img_cropped = self.mtcnn(img)  # get the cropped face -- feature extraction
        if img_cropped is None:  # no face found
            return None
        img_embedding = self._model(img_cropped.unsqueeze(0))  # get the embeddings

        return img_embedding

    def get_distance(
        self, embeddings_1: torch.Tensor, embeddings_2: torch.Tensor
    ) -> torch.Tensor:
        """Get L2 norm b/w two embeddings"""
        return torch.norm(embeddings_1 - embeddings_2).item()


def load_model():
    model = FaceNet()
    return model

def run(model:FaceNet):
    """ generates embeddings for a given image """
    fp = open("./data/data.json", "r")
    data = json.load(fp)
    frame = data["data"]

    embedding = model.get_embedding(frame).numpy()
    result = {"data": embedding.decode("utf-8")}
    return result
