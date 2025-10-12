import argparse
from rich.console import Console
from rich.panel import Panel


class RAGConfig:
    """
    Class responsible for parsing and storing the configuration for the RAG application.

    This class handles the parsing of command-line arguments, validates them, and stores the configuration.
    It also provides methods to retrieve the configuration and display a summary of the current settings.
    """

    def __init__(self):
        """
        Initializes the RAGConfig class, sets up the console for rich text output,
        and parses the command-line arguments.
        """
        self.console = Console()
        self.args = self._parse_args()

    def _parse_args(self):
        """
        Parses command-line arguments and returns them as a Namespace object.

        Sets default values for arguments if they are not provided.

        Returns:
            argparse.Namespace: The parsed arguments with default values applied.
        """
        parser = argparse.ArgumentParser(description="RAG by Kamil v0.1")

        parser.add_argument(
            "--embedder_provider",
            choices=["openai", "local"],
            default="local",
            help="Choice of embedding provider",
        )

        parser.add_argument(
            "--embedder_model",
            choices=[
                "sdadas/st-polish-paraphrase-from-distilroberta",  # 768
                "all-mpnet-base-v2",  # 768
                "text-embedding-3-small",  # 1536
                "text-embedding-3-large",  # 3072
            ],
            help="Model name to use for embeddings",
        )

        parser.add_argument(
            "--chunk_size",
            type=int,
            default=500,
            help="Size of each text chunk (in characters)",
        )

        parser.add_argument(
            "--overlap",
            type=int,
            default=50,
            help="Number of overlapping characters between chunks",
        )

        parser.add_argument(
            "--search_provider",
            choices=["openai", "local"],
            default="local",
            help="Choice of search provider",
        )

        parser.add_argument(
            "--search_model",
            choices=[
                "gpt-4o-mini",
                "llama3.1:8b",
            ],
            help="LLM model name",
        )

        parser.add_argument(
            "--import",
            dest="do_import",
            action="store_true",
            help="Import data to chroma",
        )

        parser.add_argument(
            "--query",
            help="User query",
        )

        parser.add_argument(
            "-v",
            "--verbose",
            dest="verbose",
            action="store_true",
            help="Enable debug mode",
        )

        args = parser.parse_args()

        # --- Default values ---
        if args.embedder_model is None:
            args.embedder_model = (
                "text-embedding-3-small"
                if args.embedder_provider == "openai"
                else "sdadas/st-polish-paraphrase-from-distilroberta"
            )

        if args.search_model is None:
            args.search_model = (
                "gpt-4o-mini" if args.search_provider == "openai" else "llama3.1:8b"
            )

        args.db_location = f"./chroma_db_{args.embedder_provider}-model-{args.embedder_model.replace('/', '-')}"

        if args.verbose:
            self._print_summary(args)

        return args

    def _print_summary(self, args):
        """
        Prints a summary of the current configuration in a formatted way.

        Args:
            args (argparse.Namespace): The parsed arguments to display in the summary.
        """
        panel_text = (
            "[bold cyan]Embedding:[/bold cyan]\n"
            f"Provider: [green]{args.embedder_provider}[/green]\n"
            f"Model: [green]{args.embedder_model}[/green]\n"
            f"Database Location: [green]{args.db_location}[/green]\n"
            f"Chunk size: [green]{args.chunk_size}[/green]\n"
            f"Overlap: [green]{args.overlap}[/green]\n\n"
            "[bold cyan]LLM Search:[/bold cyan]\n"
            f"Provider: [green]{args.search_provider}[/green]\n"
            f"Model: [green]{args.search_model}[/green]\n\n"
            "[bold cyan]Flags:[/bold cyan]\n"
            f"Import Data: [green]{'yes' if args.do_import else 'no'}[/green]\n"
            f"Query Search: [green]{'yes' if args.query else 'no'}[/green]"
        )
        self.console.print(Panel(panel_text, title="Application Settings"))

    def get(self):
        """
        Returns the parsed arguments as an argparse.Namespace object.

        Returns:
            argparse.Namespace: The parsed arguments.
        """
        return self.args

    def as_dict(self):
        """
        Returns the configuration as a dictionary.

        Returns:
            dict: A dictionary representation of the parsed arguments.
        """
        return vars(self.args)
