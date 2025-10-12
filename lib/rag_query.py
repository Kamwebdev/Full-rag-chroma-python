import os
import requests
from typing import Dict, Any
from openai import OpenAI
from rich.console import Console
from rich.table import Table
from rich.panel import Panel


class LLMSearch:
    """Universal class for handling RAG queries with OpenAI or Ollama."""

    def __init__(self, provider: str, model: str, verbose: bool = False):
        """
        Initializes the LLMSearch instance with the selected provider and model.

        Args:
            provider (str): The provider to use for querying ("openai" or "local").
            model (str): The model to use for querying.
            verbose (bool): Whether to enable verbose logging.
        """
        self.provider = provider
        self.model = model
        self.verbose = verbose
        self.console = Console()

        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("Missing environment variable OPENAI_API_KEY")
            self.client = OpenAI(api_key=api_key)
        elif self.provider == "local":
            self.client = None
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def ask(self, prompt: str) -> str:
        """
        Sends a query to the selected model.

        Args:
            prompt (str): The prompt to send to the model.

        Returns:
            str: The response from the model.
        """
        if self.provider == "openai":
            return self._ask_openai(prompt)
        elif self.provider == "local":
            return self._ask_ollama(prompt)
        return None

    def search_with_context(self, user_query: str, results: Dict[str, Any]) -> str:
        """
        Builds a prompt based on RAG results and sends it to the LLM.

        Args:
            user_query (str): The user query to ask the model.
            results (Dict[str, Any]): The RAG results containing documents to be used as context.

        Returns:
            str: The response from the model.
        """
        if self.verbose:
            self._print_results_table(results)

        retrieved_docs = results.get("documents", [[]])
        if not retrieved_docs or not retrieved_docs[0]:
            return "[No results to process.]"

        context = "\n".join(retrieved_docs[0])
        prompt = f"""
        Answer the user's question based on the context.

        Context:
        {context}

        Question:
        {user_query}
        """

        if self.verbose:
            self.console.print(Panel(prompt.strip(), title="Sent Prompt"))

        try:
            return self.ask(prompt.strip())
        except Exception as e:
            return f"[LLM query error]: {e}"

    # -------------------------------
    # PROVIDERS
    # -------------------------------

    def _ask_openai(self, prompt: str) -> str:
        """
        Sends the prompt to the OpenAI API.

        Args:
            prompt (str): The prompt to send to the OpenAI model.

        Returns:
            str: The response from OpenAI.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    def _ask_ollama(self, prompt: str) -> str:
        """
        Sends the prompt to the local Ollama model.

        Args:
            prompt (str): The prompt to send to the Ollama model.

        Returns:
            str: The response from Ollama.
        """
        url = "http://localhost:11434/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }

        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            if "message" in data:
                return data["message"]["content"]
            elif "messages" in data:
                return data["messages"][-1]["content"]
            else:
                return "[Error]: Unexpected response format from Ollama."
        except requests.exceptions.RequestException as e:
            return f"[Connection error with Ollama]: {e}"

    def _print_results_table(self, results: Dict[str, Any]):
        """
        Prints the results of the RAG search in a formatted table.

        Args:
            results (Dict[str, Any]): The RAG search results containing documents and distances.
        """
        table = Table(title="RAG Results")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Document", style="magenta")
        table.add_column("Distance", style="green")

        for ids_row, docs_row, dist_row in zip(
                results.get("ids", []),
                results.get("documents", []),
                results.get("distances", []),
        ):
            for i, d, dist in zip(ids_row, docs_row, dist_row):
                doc_preview = (d[:1000] + "...") if len(d) > 1000 else d
                table.add_row(i, doc_preview, f"{dist:.2f}")

        self.console.print(table)

    def get(self):
        """
        Returns the configuration as an object.

        Returns:
            argparse.Namespace: The configuration object.
        """
        return self.args

    def as_dict(self):
        """
        Returns the configuration as a dictionary.

        Returns:
            dict: A dictionary representation of the configuration.
        """
        return vars(self.args)
