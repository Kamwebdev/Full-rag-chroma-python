import json
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional


class DataLoader:
    """
    Class responsible for fetching data from URLs or loading sample data.

    This class provides methods to load documents either by fetching content from a list of URLs or
    by returning predefined sample data. It supports use cases like extracting plain text from web pages
    or providing mock data for testing and development.

    Methods:
        - load_docs: Loads documents by either fetching content from provided URLs or returning sample data.
        - _fetch_url_text: Fetches and extracts plain text content from a given URL.
        - _load_data_from_url: Fetches content from a list of URLs and returns it in a structured format.
        - _load_sample_data: Returns predefined sample data in JSON format.

    Example usage:
        data_loader = DataLoader()
        sample_docs = data_loader.load_docs()  # Returns sample data
        fetched_docs = data_loader.load_docs(urls=["http://example.com", "http://example.org"])  # Fetches documents from URLs
    """

    def __init__(self):
        pass

    @staticmethod
    def _fetch_url_text(url: str) -> str:
        """
        Fetches the page content from the provided URL and extracts plain text.

        Args:
            url (str): The URL of the page to fetch.

        Returns:
            str: The plain text content of the page, or an empty string if the content could not be retrieved.
        """
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
        except Exception as e:
            print(f"Nie udało się pobrać {url}: {e}")
            return ""

        soup = BeautifulSoup(r.text, "html.parser")
        for s in soup(["script", "style", "noscript", "iframe"]):
            s.extract()
        text = soup.get_text(separator=" ")
        text = " ".join(text.split())
        return text

    @staticmethod
    def _load_data_from_url(urls: List[str]) -> List[Dict]:
        """
        Fetches the content of pages from the provided list of URLs and returns a list of dictionaries in the following format:
        [{"id": ..., "doc": ..., "meta": {"source": ...}}, ...]

        Args:
            urls (List[str]): A list of URLs to fetch content from.

        Returns:
            List[Dict]: A list of dictionaries where each dictionary contains the following keys:
                - "id": A unique identifier for the document (e.g., a combination of URL and index).
                - "doc": The plain text content fetched from the URL.
                - "meta": A dictionary with metadata, including the "source" key that stores the URL.

        Example:
            [
                {"id": "doc-0-example_com", "doc": "Page content", "meta": {"source": "http://example.com"}},
                {"id": "doc-1-example_org", "doc": "Another page content", "meta": {"source": "http://example.org"}}
            ]
        """

        data = []
        for idx, url in enumerate(urls):
            page_body = DataLoader._fetch_url_text(url)
            if page_body:
                data.append(
                    {
                        "id": f"doc-{idx}-{url.replace('https://', '').replace('/', '_')}",
                        "doc": page_body,
                        "meta": {"source": url},
                    }
                )
        return data

    @staticmethod
    def _load_sample_data() -> List[Dict]:
        """
        Returns sample data in JSON format.

        This method provides a predefined set of sample documents, each containing an identifier (`id`),
        text content (`doc`), and metadata (`meta`) with the source of the document.

        Returns:
            List[Dict]: A list of dictionaries, each representing a sample document with the following keys:
                - "id": A unique identifier for the document.
                - "doc": The content of the document as a string.
                - "meta": A dictionary with metadata about the document, including the "source" key.

        Example:
            [
                {"id": "doc1", "doc": "Chroma is an engine for vector databases.", "meta": {"source": "notes"}},
                {"id": "doc2", "doc": "OpenAI embeddings allow text to be converted into vectors.", "meta": {"source": "blog"}}
            ]
        """
        json_data = """
        [
            {"id":"doc1", "doc":"Chroma is an engine for vector databases.", "meta":{"source":"notes"}},
            {"id":"doc2", "doc":"OpenAI embeddings allow text to be converted into vectors.", "meta":{"source":"blog"}},
            {"id":"doc3", "doc":"LangChain makes it easier to create applications based on LLMs.", "meta":{"source":"documentation"}},
            {"id":"doc4", "doc":"Vector databases enable efficient searching through large text data sets.", "meta":{"source":"article"}},
            {"id":"doc5", "doc":"RAG combines text generation with real-time information retrieval.", "meta":{"source":"presentation"}}
        ]
        """
        return json.loads(json_data)

    @staticmethod
    def load_docs(urls: Optional[List[str]] = None) -> List[Dict]:
        """
        Loads documents. If URLs are provided, fetches their content; otherwise, returns sample data.

        If a list of URLs is provided, this method will attempt to fetch the content from each URL and return it in the
        form of a list of dictionaries. If no URLs are provided, it will return a predefined set of sample data.

        Args:
            urls (Optional[List[str]], optional): A list of URLs to fetch content from. Defaults to None.

        Returns:
            List[Dict]: A list of dictionaries, where each dictionary contains the following:
                - "id": A unique identifier for the document.
                - "doc": The content of the document (text from the URL or sample data).
                - "meta": A dictionary with metadata about the document, including the "source" key with the URL or source of sample data.

        Example:
            If URLs are provided:
            [
                {"id": "doc-0-example_com", "doc": "Page content", "meta": {"source": "http://example.com"}},
                {"id": "doc-1-example_org", "doc": "Another page content", "meta": {"source": "http://example.org"}}
            ]

            If no URLs are provided, returns sample data:
            [
                {"id": "doc1", "doc": "Chroma is an engine for vector databases.", "meta": {"source": "notes"}},
                {"id": "doc2", "doc": "OpenAI embeddings allow text to be converted into vectors.", "meta": {"source": "blog"}}
            ]
        """
        if urls:
            return DataLoader._load_data_from_url(urls)
        return DataLoader._load_sample_data()
