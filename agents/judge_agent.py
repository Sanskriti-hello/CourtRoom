from __future__ import annotations

import pandas as pd
import json
import re
import uuid

import os
from typing import List, Dict, Any
from huggingface_hub import InferenceClient
import json
import logging
from transformers import AutoTokenizer

class JudgeAgent:

    def __init__(self,
                 name: str,
                 system_prompt: str,
                 description: str = "",
                 model: str = "microsoft/Phi-3-mini-4k-instruct",
                 db=None):
        self.name = name
        self.role = name
        self.system_prompt = system_prompt.strip()
        self.description = description.strip()   
        self.client = InferenceClient(
            model,
            token="hf_EhWyVApfuVivnwnrfQDhaRbPagYcEFSDf"          # make sure this env‑var is set
        )
        self.db = db
        self.logger = logging.getLogger(name)
        self.log_think = True

     # ---- helper bitches --------------------------

    def prepare_history_context(self, history: List[Dict[str, str]]) -> str:
        return "\n\n".join(f"{entry['role']} ({entry['name']}):\n  {entry['content']}" for entry in history)

    def prepare_case_content(self, history_context: str) -> str:
        instruction = f"You are a judge. Summarize the following case in 3 sentences."
        return self._hf_generate(instruction, history_context)

    def _hf_generate(self, instruction: str, prompt: str) -> str:
        full_prompt = f"<|system|>\n{instruction}\n<|user|>\n{prompt}\n<|assistant|>\n"

        max_total_tokens = 4096
        max_new_tokens = 512

        input_ids = self.tokenizer(full_prompt, return_tensors="pt")["input_ids"][0]
        if len(input_ids) + max_new_tokens > max_total_tokens:
            max_prompt_tokens = max_total_tokens - max_new_tokens
            input_ids = input_ids[-max_prompt_tokens:]  # keep only last tokens
            full_prompt = self.tokenizer.decode(input_ids, skip_special_tokens=True)

        return self.client.text_generation(full_prompt, max_new_tokens=max_new_tokens, temperature=0.7).strip()

    def _parse_json(self, response: str) -> Dict[str, Any]:
        try:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                return json.loads(match.group())
        except json.JSONDecodeError:
            self.logger.warning("Failed to parse JSON from reflection output.")
        return {}

    # ---- reflect bitches --------------------------

    def reflect(self, history_list: List[Dict[str, str]]) -> Dict[str, Any]:
        history_context = self.prepare_history_context(history_list)
        case_content = self.prepare_case_content(history_context)

        legal_reflection = self._reflect_on_legal_knowledge(history_context)
        experience_reflection = self._reflect_on_experience(case_content, history_context)
        case_reflection = self._reflect_on_case(case_content, history_context)

        return {
            "legal_reflection": legal_reflection,
            "experience_reflection": experience_reflection,
            "case_reflection": case_reflection
        }

    def _reflect_on_legal_knowledge(self, history_context: str) -> Dict[str, Any]:
        if not self._need_legal_reference(history_context):
            return {"needed_reference": False}

        query = self._prepare_legal_query(history_context)
        laws = self.db.query_legal(query, n_results=3) if self.db else []
        processed = [
            self._process_law(law) for law in laws
        ]
        for law in processed:
            self.add_to_legal(str(uuid.uuid4()), law["content"], law["metadata"])
        return {"needed_reference": True, "query": query, "laws": processed}

    def _need_legal_reference(self, history_context: str) -> bool:
        instruction = f"You are a {self.role}. {self.description}"
        prompt = (
            "Review the court history. Would referencing specific laws improve reasoning?\n"
            "Respond with 'true' or 'false'.\n\n"
            f"{history_context}"
        )
        result = self._hf_generate(instruction, prompt).lower()
        return "true" in result

    def _reflect_on_experience(self, case_content: str, history_context: str) -> Dict[str, Any]:
        instruction = f"You are {self.role}. {self.description}"
        prompt = f""" Given the following case and history, summarize actionable courtroom experience.
                      Return as JSON with keys: context, content, focus_points, guidelines.

                      Case:\n{case_content}\n\nHistory:\n{history_context}
                  """
        response = self._hf_generate(instruction, prompt)
        print("LLM raw experience response:", response)  # 🧪 For debugging
        summary = self._parse_json(response)

        entry = {
            "id": str(uuid.uuid4()),
            "content": summary.get("context", "[missing context]"),
            "metadata": {
                "context": summary.get("content", "[missing content]"),
                "focusPoints": summary.get("focus_points", ""),
                "guidelines": summary.get("guidelines", "")
            }
        }
        self.add_to_experience(entry["id"], entry["content"], entry["metadata"])
        return entry


    def _reflect_on_case(self, case_content: str, history_context: str) -> Dict[str, Any]:
        summary = self._generate_case_summary(case_content, history_context)
        entry = {
            "id": str(uuid.uuid4()),
            "content": summary["content"],
            "metadata": {
                "caseType": summary["case_type"],
                "keywords": summary["keywords"],
                "quick_reaction_points": summary["quick_reaction_points"],
                "response_directions": summary["response_directions"],
            }
        }
        self.add_to_case(entry["id"], entry["content"], entry["metadata"])
        return entry

    # ---- verdict bitches -----------------------------

    def deliberate(self, reflections: Dict[str, Any], history_context: str) -> str:
        instruction = f"You are {self.role}. {self.description}"
        prompt = f""" You have reviewed the trial. Use the reflections and court history to make a final ruling.

                      Legal Reflection:\n{reflections['legal_reflection']}
                      Experience Reflection:\n{reflections['experience_reflection']}
                      Case Reflection:\n{reflections['case_reflection']}
                      Court History:\n{history_context}

                      Write a clear, fair, and reasoned verdict.
                  """
        return self._hf_generate(instruction, prompt)

    # ---- trim history because exceeding tokens bitches --------------------------

    '''def trim_history(self, history_list: List[Dict[str, str]], max_tokens: int = 3000) -> List[Dict[str, str]]:
        trimmed = []
        total_tokens = 0
        for entry in reversed(history_list):  # Start from the most recent
            tokens = len(entry["content"]) // 4  # Approx 1 token ≈ 4 characters
            if total_tokens + tokens > max_tokens:
                break
            trimmed.insert(0, entry)
            total_tokens += tokens
        return trimmed
'''
     # ---- database bitches --------------------------

    def add_to_legal(self, id: str, content: str, metadata: Dict[str, Any]):
        if self.db:
            self.db.add_to_legal(id, content, metadata)

    def add_to_case(self, id: str, content: str, metadata: Dict[str, Any]):
        if self.db:
            self.db.add_to_case(id, content, metadata)

    def add_to_experience(self, id: str, content: str, metadata: Dict[str, Any]):
        if self.db:
            self.db.add_to_experience(id, content, metadata)

    # ---- summarisations bitches --------------------------

    def _prepare_legal_query(self, history_context: str) -> str:
        instruction = f"You are a {self.role}. {self.description}"
        prompt = "Generate a keyword query to find applicable laws for the case.\n\n" + history_context
        result = self._hf_generate(instruction, prompt)
        return result.strip()

    def _generate_experience_summary(self, case_content: str, history_context: str) -> Dict[str, Any]:
        instruction = f"You are {self.role}. {self.description}"
        prompt = f""" Given the following case and history, summarize actionable courtroom experience.
                      Return as JSON with keys: context, content, focus_points, guidelines.

                      Case:\n{case_content}\n\nHistory:\n{history_context}
                  """
        return self._parse_json(self._hf_generate(instruction, prompt))

    def _generate_case_summary(self, case_content: str, history_context: str) -> Dict[str, Any]:
        instruction = f"You are {self.role}. {self.description}"
        prompt = f""" Summarize this case to support quick judicial decision-making.
                      Return as JSON with keys: content, case_type, keywords, quick_reaction_points, response_directions.

                      Case:\n{case_content}\n\nHistory:\n{history_context}
                  """
        return self._parse_json(self._hf_generate(instruction, prompt))

    def _process_law(self, law: dict) -> Dict[str, Any]:
        content = f"{law['lawsName']} {law['articleTag']} {law['articleContent']}"
        return {"content": content, "metadata": {"lawName": law["lawsName"], "articleTag": law["articleTag"]}}
