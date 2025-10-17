import gradio as gr
from chromadb import PersistentClient

from lib.embeding import embeder_loader
from lib.parser import parse_args
from lib.search import search

from typing import List, Tuple, Optional, Literal


def chat_fn(
    message: str,
    history: Optional[List[Tuple[str, str]]],
    search_provider: Literal["openai", "local"],
    n_results: int
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Handles a single chat interaction.

    Args:
        message (str): The user's message.
        history (Optional[List[Tuple[str, str]]]): Conversation history as list of (user, assistant) message pairs.
        search_provider (Literal["openai", "local"]): The search backend to use.
        n_results (int): Number of top documents to retrieve from ChromaDB.

    Returns:
        Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
            - First element: messages to display in the chat UI (usually full or truncated history).
            - Second element: updated history to store in state.
    """
    history = history or []

    results = collection.query(
        query_embeddings=embed_fn([message]), n_results=int(n_results)
    )

    if search_provider == "openai":
        search_model = "gpt-4o-mini"
    else:
        search_model = "llama3.1:8b"

    if history:
        previous_messages = "\n".join(
            [f"    User: {q.strip()}\n    Assistant: {a.strip()}" for q, a in history[-2:]]
        )
        full_message = f"{message}\n\n    History:\n{previous_messages}"
    else:
        full_message = f"User: {message}"

    answer = search(full_message, results, search_provider, search_model, args.verbose)
    history.append((message, answer))

    return history, history


with gr.Blocks() as my_rag:
    gr.Markdown("## Chat RAG + LLM")
    chatbot = gr.Chatbot(label="KamDev.pl")
    msg = gr.Textbox(label="Enter message", placeholder="Ask question...")
    send_btn = gr.Button("Send")
    state = gr.State([])

    with gr.Row():
        search_provider = gr.Dropdown(
            choices=["openai", "local"], value="local", label="Search Provider"
        )
        n_results = gr.Slider(
            minimum=1, maximum=10, step=1, value=3, label="Set a limit on documents returned by ChromaDB."
        )

    send_btn.click(
        fn=chat_fn,
        inputs=[msg, state, search_provider, n_results],
        outputs=[chatbot, state],
    )

if __name__ == "__main__":
    args = parse_args()
    collection = PersistentClient(path=args.db_location).get_or_create_collection(
        "documents"
    )
    embed_fn = embeder_loader(args)
    my_rag.launch()
