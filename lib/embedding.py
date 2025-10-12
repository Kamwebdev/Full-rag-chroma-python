import os
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Load environment variables from .env file
load_dotenv()


class Embedder:
    """
    Class responsible for generating embeddings — either locally (using SentenceTransformer) or through the OpenAI API.

    It also manages local model caching.
    """

    # Cache to store locally loaded models
    _local_models: Dict[str, SentenceTransformer] = {}

    def __init__(self, provider: str = "local", model_name: str = "all-mpnet-base-v2"):
        """
        Initializes the Embedder class.

        Args:
            provider (str): The provider to use for generating embeddings. Can be either 'openai' for OpenAI API or 'local' for local models.
            model_name (str): The model name to use for embedding generation. Defaults to 'all-mpnet-base-v2'.

        Raises:
            ValueError: If the provider is not supported or if the OpenAI API key is missing.
        """
        self.provider = provider
        self.model_name = model_name
        self.openai_client = None

        # Initialize OpenAI client if using OpenAI API
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "Missing OpenAI API key. Set the OPENAI_API_KEY environment variable in .env."
                )
            self.openai_client = OpenAI(api_key=api_key)

        # Initialize SentenceTransformer client if using local models
        elif provider == "local":
            if model_name not in self._local_models:
                self._local_models[model_name] = SentenceTransformer(model_name)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def initialize(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of input texts.

        Depending on the selected provider, this method will either call the OpenAI API or use a local model.

        Args:
            texts (List[str]): A list of strings for which embeddings should be generated.

        Returns:
            List[List[float]]: A list of embeddings, where each embedding is a list of floats.
        """
        if self.provider == "openai":
            return self._get_openai_embeddings(texts)
        else:
            return self._get_local_embeddings(texts)

    def _get_openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Fetches embeddings from OpenAI's API.

        Args:
            texts (List[str]): A list of strings for which embeddings should be generated.

        Returns:
            List[List[float]]: A list of embeddings returned by OpenAI API.
        """
        response = self.openai_client.embeddings.create(
            input=texts, model=self.model_name
        )
        return [r.embedding for r in response.data]

    def _get_local_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings using a locally loaded model.

        Args:
            texts (List[str]): A list of strings for which embeddings should be generated.

        Returns:
            List[List[float]]: A list of embeddings generated using the local SentenceTransformer model.
        """
        model = self._local_models[self.model_name]
        return model.encode(texts, show_progress_bar=False).tolist()
