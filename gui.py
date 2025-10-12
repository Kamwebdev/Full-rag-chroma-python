import gradio as gr
from chromadb import PersistentClient
from lib.embedding import Embedder
from lib.rag_query import LLMSearch
from lib.config_parser import RAGConfig
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import json

from rich.console import Console

console = Console()
config = RAGConfig().get()
embed_fn = Embedder(config.embedder_provider, config.embedder_model).initialize
collection = PersistentClient(path=config.db_location).get_or_create_collection(
    "documents"
)


def chat_fn(message, history):
    results = collection.query(
        query_embeddings=embed_fn([message]),
        n_results=config.n_results if hasattr(config, "n_results") else 3,
    )

    rag = LLMSearch(
        provider=config.search_provider, model=config.search_model, verbose=True
    )
    answer = rag.search_with_context(message, results)

    history = history or []
    history.append((message, answer))

    if config.verbose:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Question", style="dim", width=40)
        table.add_column("Answer", width=80)

        for question, answer in history:
            table.add_row(message, answer)

        console.print(table)
        console.print(Panel(answer, title="Conversation"))

    return answer


with gr.ChatInterface(chat_fn, title="Chat RAG + LLM") as demo:
    print(config)
    demo.launch(share=False)
