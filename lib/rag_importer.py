import json
from typing import Callable, List
from chromadb import PersistentClient
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


class DataImporter:
    """
    Class responsible for importing data into the Chroma collection.
    Handles chunking, embedding, and optional verbose logging.
    """

    def __init__(
        self,
        chroma_collection: PersistentClient,
        embed_fn: Callable[[List[str]], List[List[float]]] = None,
        chunk_size: int = 500,
        overlap: int = 50,
        verbose: bool = False,
    ):
        """
        Initializes the DataImporter instance.

        Args:
            chroma_collection (PersistentClient): The Chroma collection to which the data will be added.
            embed_fn (Callable[[List[str]], List[List[float]]]): A function that generates embeddings for the text chunks.
            chunk_size (int): The maximum size of each text chunk in characters.
            overlap (int): The number of characters that overlap between adjacent chunks.
            verbose (bool): Whether to enable verbose logging.
        """
        self.chroma_collection = chroma_collection
        self.embed_fn = embed_fn
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.verbose = verbose

    def __show_table(self, items: List[dict]):
        """
        Helper method to visualize input data in a table format.

        Args:
            items (List[dict]): A list of items to display in the table.
        """
        table = Table(title="Data from URLs")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Text (snippet)", style="green")
        table.add_column("Source", style="magenta", no_wrap=True)

        for item in items:
            doc_snippet = item["doc"][:1000] + (
                "..." if len(item["doc"]) > 1000 else ""
            )
            table.add_row(item["id"], doc_snippet, item["meta"]["source"])

        console.print(table)

    def load_data(self, json_data: json):
        """
        Adds documents to the Chroma collection.

        Args:
            json_data (str or List[dict]): JSON string or a list of dictionaries containing document data.
                Example: [{"id": ..., "doc": ..., "meta": {...}}]
        """
        items = json.loads(json_data) if isinstance(json_data, str) else json_data

        if self.verbose:
            self.__show_table(items)

        for item in items:
            doc_id = item["id"]
            doc_text = item["doc"]
            meta = item.get("meta", {})

            chunks = self.__chunk_text(doc_text)

            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk{i}"
                if not self.chroma_collection.get(ids=[chunk_id]).get("ids"):
                    self.chroma_collection.add(
                        ids=[chunk_id],
                        documents=[chunk],
                        embeddings=self.embed_fn([chunk])[0] if self.embed_fn else None,
                        metadatas=[meta],
                    )
                    if self.verbose:
                        panel_text = (
                            "[bold cyan]Document:[/bold cyan]\n"
                            f"id: [green]{chunk_id}[/green]\n"
                            f"chunk: [green]{chunk}[/green]\n"
                        )

                        console.print(Panel(panel_text, title="Added documents"))

        if self.verbose:
            console.print(
                Panel("[green]Import completed successfully![/green]", title="Success")
            )

    def __chunk_text(self, text: str) -> List[str]:
        """
        Splits the text into chunks of the specified size while maintaining overlap.

        Args:
            text (str): The text to be chunked.

        Returns:
            List[str]: A list of text chunks.
        """
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if self.overlap < 0 or self.overlap >= self.chunk_size:
            raise ValueError("overlap must be >= 0 and smaller than chunk_size")

        chunks: List[str] = []
        step = self.chunk_size - self.overlap
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            if end >= len(text):
                chunk = text[-self.chunk_size :]
                chunks.append(chunk)
                break
            chunks.append(text[start:end])
            start += step
        return chunks
