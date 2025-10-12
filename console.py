from chromadb import PersistentClient
from lib.embedding import Embedder
from lib.rag_importer import DataImporter
from lib.sample_data import DataLoader
from lib.rag_query import LLMSearch
from lib.config_parser import RAGConfig

from rich.console import Console
from rich.panel import Panel
import json


def main():
    console = Console()
    config = RAGConfig().get()
    embed_fn = Embedder(config.embedder_provider, config.embedder_model).initialize
    collection = PersistentClient(path=config.db_location).get_or_create_collection(
        "documents"
    )

    if config.do_import:
        # Load data to chromadb
        loader = DataLoader()
        urls = ["https://blog.kamdev.pl",]
        documents = loader.load_docs(urls)

        if config.verbose:
            console.print(
                Panel(
                    json.dumps(documents, indent=4, ensure_ascii=False), title="Raw data"
                )
            )

        # Import data to chromadb
        importer = DataImporter(
            collection,
            embed_fn=embed_fn,
            overlap=config.overlap,
            chunk_size=config.chunk_size,
            verbose=True,
        )
        importer.load_data(
            documents,
        )

    if config.query:

        # Search in chromadb
        results = collection.query(
            query_embeddings=embed_fn([config.query]), n_results=3
        )

        if (
            results.get("documents")
            and results["documents"]
            and results["documents"][0]
        ):
            rag = LLMSearch(
                provider=config.search_provider, model=config.search_model, verbose=True
            )
            response = rag.search_with_context(config.query, results)
            console.print(Panel(response, title="Results"))


if __name__ == "__main__":
    main()
