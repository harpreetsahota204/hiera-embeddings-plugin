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
    """Helper function to determine the best available device.
    
    Returns:
        str: Device identifier ('cuda', 'mps', or 'cpu')
    """
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
    """A model for extracting embeddings from images using Hiera vision models.

    This class supports extraction of CLS token embeddings or mean pooled embeddings
    from various sizes of Hiera models.

    Args:
        model_name (str): Name of the pretrained Hiera model to use
        embedding_types (str): Type of embedding to extract ('cls' or 'mean')

    Attributes:
        processor (AutoImageProcessor): The processor for preparing inputs
        model (HieraModel): The pretrained vision model
        device (str): The device (CPU/GPU) where the model will run

    Raises:
        ValueError: If embedding_types is not 'cls' or 'mean'
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
        """Extract embeddings from the model's output tensor.

        Args:
            last_hidden_state (torch.Tensor): Output tensor of shape (batch_size, sequence_length, hidden_size)

        Returns:
            np.ndarray: Embedding vector of shape (hidden_size,)
        """
        if self.embedding_types == "cls":
            cls_embedding = last_hidden_state[0, 0].cpu().numpy()
            return cls_embedding

        if self.embedding_types == "mean":
            mean_embedding = last_hidden_state[0].mean(dim=0).cpu().numpy()
            return mean_embedding

    def _predict(self, image: Image.Image) -> np.ndarray:
        """Extract embeddings from a single image.

        Args:
            image (PIL.Image.Image): Input image to process

        Returns:
            np.ndarray: Embedding vector of shape (hidden_size,)
        """
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state

        return self.extract_embeddings(last_hidden_state)

    def predict(self, args: np.ndarray) -> np.ndarray:
        """Extract embeddings from an image array.

        Args:
            args (np.ndarray): Input image as a numpy array of shape (height, width, channels)

        Returns:
            np.ndarray: Embedding vector of shape (hidden_size,)
        """
        image = Image.fromarray(args)
        predictions = self._predict(image)
        return predictions

    def predict_all(self, images: List[Image.Image]) -> List[np.ndarray]:
        """Extract embeddings from multiple images.

        Args:
            images (List[PIL.Image.Image]): List of input images to process

        Returns:
            List[np.ndarray]: List of embedding vectors, each of shape (hidden_size,)
        """
        return [self.predict(image) for image in images]

def run_embeddings_model(
    dataset: fo.Dataset,
    model_name: str,
    emb_field: str,
    embedding_types: str
) -> None:
    """Apply the Hiera embedding model to a FiftyOne dataset.

    Args:
        dataset (fo.Dataset): The FiftyOne dataset to process
        model_name (str): Name of the pretrained Hiera model to use
        emb_field (str): Name of the field to store embeddings in
        embedding_types (str): Type of embedding to extract ('cls' or 'mean')
    """

    model = HieraEmbeddingModel(model_name, embedding_types)

    dataset.apply_model(model, label_field=emb_field)