import pandas as pd

import json
import re
import uuid
from __future__ import annotations
import os
from typing import List, Dict, Any
from huggingface_hub import InferenceClient
import json
import logging

class CourtroomDB:
    def __init__(self, dataframe: pd.DataFrame):
        """
        Initializes the database wrapper.
        Expects a DataFrame with a 'text' column.
        """
        if "text" not in dataframe.columns:
            raise ValueError("DataFrame must contain a 'text' column.")
        self.df = dataframe

    def _search(self, query: str, n_results: int = 3):
        """
        Perform a basic keyword search in the 'text' column.
        """
        keyword = query.lower()
        matches = self.df[self.df["text"].str.lower().str.contains(keyword, na=False)]
        return matches["text"].head(n_results).tolist()

    def query_legal(self, query: str, n_results: int = 3):
        """
        Returns text snippets related to legal statutes or laws.
        """
        return self._search(query, n_results)

    def query_case_metadatas(self, query: str, n_results: int = 3):
        """
        Returns precedent cases matching the legal topic.
        """
        return self._search(query, n_results)

    def query_experience_metadatas(self, query: str, n_results: int = 3):
        """
        Returns summaries or examples of experience-based logic or legal patterns.
        """
        return self._search(query, n_results)

    def add_to_legal(self, id: str, content: str, metadata: Dict[str, Any]):
        self._append_to_db("Legal", id, content, metadata)

    def add_to_case(self, id: str, content: str, metadata: Dict[str, Any]):
        self._append_to_db("Case", id, content, metadata)

    def add_to_experience(self, id: str, content: str, metadata: Dict[str, Any]):
        self._append_to_db("Experience", id, content, metadata)

    def _append_to_db(self, category: str, id: str, content: str, metadata: Dict[str, Any]):
        entry = {
            "id": id,
            "text": content,
            "category": category,
            "metadata": metadata
        }
        self.df = pd.concat([self.df, pd.DataFrame([entry])], ignore_index=True)
