import fiftyone as fo
from fiftyone.core.models import Model
import torch
from PIL import Image
import numpy as np
from transformers import AutoImageProcessor, HieraModel
from importlib.util import find_spec
from typing import List, Dict

HIERA_ARCHS = [
    "facebook/hiera-tiny-224-hf",
    "facebook/hiera-small-224-hf",
    "facebook/hiera-base-224-hf",
    "facebook/hiera-base-plus-224-hf",
    "facebook/hiera-large-224-hf",
    "facebook/hiera-huge-224-hf",
]


def hiera_activator():
    return find_spec("transformers") is not None

def get_device():
    """Helper function to determine the best available device."""
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Silicon (MPS) device")
    else:
        device = "cpu"
        print("Using CPU device")
    return device


class HieraEmbeddingModel(Model):
    """
    AIMv2EmbeddingModel is a flexible class for extracting embeddings using various vision models.

    Attributes:
        model_name (str): Name of the pretrained model to use
        embedding_types (List[str]): Types of embeddings to extract ('cls', 'mean', or both)
        processor (AutoImageProcessor): The processor for preparing inputs
        model (HieraModel): The pretrained vision model
        device (str): The device (CPU/GPU) where the model will run
    """

    def __init__(self, model_name, embedding_types):
        self.model_name = model_name
        self.embedding_types = embedding_types

        # Validate embedding types
        valid_types = ["cls", "mean"]
        if self.embedding_types not in valid_types:
            raise ValueError(f"Invalid embedding type: {embedding_types}. Must be one of {valid_types}")

        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = HieraModel.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        # Set up device
        self.device = get_device()

        # Move model to appropriate device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

    @property
    def media_type(self):
        return "image"

    def extract_embeddings(self, last_hidden_state: torch.Tensor) -> np.ndarray:
        """
        Extracts specified type of embedding from the model's output.

        Args:
            last_hidden_state (torch.Tensor): The model's last hidden state

        Returns:
            np.ndarray: The extracted embedding array (cls or mean based on initialization)
        """

        if self.embedding_types == "cls":
            cls_embedding = last_hidden_state[0, 0].cpu().numpy()
            return cls_embedding

        if self.embedding_types == "mean":
            mean_embedding = last_hidden_state[0].mean(dim=0).cpu().numpy()
            return mean_embedding


    def _predict(self, image: Image.Image) -> np.ndarray:
        """
        Performs embedding extraction on a single image.

        Args:
            image (PIL.Image.Image): The input image

        Returns:
            np.ndarray: Requested embeddings
        """
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state

        return self.extract_embeddings(last_hidden_state)

    def predict(self, args: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predicts embeddings for the given image.

        Args:
            args (np.ndarray): The input image as a numpy array

        Returns:
            np.ndarray: Requested embeddings
        """
        image = Image.fromarray(args)
        predictions = self._predict(image)
        return predictions

    def _predict_all(self, images: List[Image.Image]) -> List[np.ndarray]:
        """
        Performs prediction on a list of images.

        Args:
            images (List[PIL.Image.Image]): List of PIL images

        Returns:
            List[np.ndarray]: List of embedding arrays for each image
        """
        return [self._predict(image) for image in images]

def run_embeddings_model(
    dataset,
    model_name,
    emb_field,
    embedding_types
    ):

    model = HieraEmbeddingModel(model_name, embedding_types)

    dataset.apply_model(model, label_field=emb_field)