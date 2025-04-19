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
import torch

#the agents are gonna follow 2 approaches, the lawyer agents follow react and judge follows reflection
#firstly coming to the lawyer agent,
class LawyerAgent:


    def __init__(self,
                 name: str,
                 system_prompt: str,
                 model: str = "microsoft/Phi-3-mini-4k-instruct",
                 db=None):
        self.name = name
        self.role = name
        self.description = system_prompt

        self.logger = logging.getLogger(name)
        self.log_think = True
        self.system_prompt = system_prompt.strip()
        self.history: List[Dict[str, str]] = []      # list of {"role": ..., "content": ...}
        self.client = InferenceClient(
                        model="microsoft/Phi-3-mini-4k-instruct",
                        token="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"
                    ) # make sure this env‑var is set
        self.tokenizer = AutoTokenizer.from_pretrained(model) 
        self.db = db        
        


    # ---- helper for HF prompt formatting ----------
    def _format_prompt(self, user_msg: str) -> str:
        """
        Formats a full prompt that includes
        * system prompt
        * prior turns
        * new user message
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_msg})

        # HF text-generation endpoints expect a single string.

        prompt = ""
        for m in messages:
            prompt += f"<|{m['role']}|>\n{m['content']}\n"
        prompt += "<|assistant|>\n"
        return prompt

    # ---- produce a reply --------------------------
    def respond(self, user_msg: str, **gen_kwargs) -> str:
        prompt = self._format_prompt(user_msg)
        completion = self.client.text_generation(
            prompt,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            stream=False,
            **gen_kwargs
        )
        answer = completion.strip()
        # keep chat memory
        self.history.append({"role": "user", "content": user_msg})
        self.history.append({"role": "assistant", "content": answer})
        return answer

    # ---- think bitch ------------------------------
    def plan(self, history_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.log_think:
            self.logger.info(f"Agent ({self.role}) starting planning phase")
        history_context = self.prepare_history_context(history_list)
        plans = self._get_plan(history_context)
        if self.log_think:
            self.logger.info(f"Agent ({self.role}) generated plans: {plans}")
        queries = self._prepare_queries(plans, history_context)
        if self.log_think:
            self.logger.info(f"Agent ({self.role}) prepared queries: {queries}")
        return {"plans": plans, "queries": queries}

    def prepare_history_context(self, history: List[Dict[str, str]]) -> str:
        return "\n".join(f"<|{m['role']}|>\n{m['content']}\n" for m in history)

    def _get_plan(self, history_context: str) -> Dict[str, bool]:
        instruction = f"You are a {self.role}. {self.description}\n\n"
        prompt = (
            "Your Honor, I need to access the courtroom database to inform my next course of action."
            "Can you please direct me to the database where I can retrieve the necessary information to generate a plan for the case at hand?"
            "Upon accessing the database, I will provide a well-structured JSON file outlining the plans and queries required to move forward with the trial."
        )
        response = self._hf_generate(instruction, prompt + "\n\n" + history_context)
        return self._extract_plans(self.extract_response(response))

    def _prepare_queries(self, plans: Dict[str, bool], history_context: str) -> Dict[str, str]:
        queries = {}
        if plans.get("experience"):
            queries["experience"] = self._prepare_experience_query(history_context)
        if plans.get("case"):
            queries["case"] = self._prepare_case_query(history_context)
        if plans.get("legal"):
            queries["legal"] = self._prepare_legal_query(history_context)
        return queries

    def _prepare_experience_query(self, history_context: str) -> str:
        instruction = f"You are a {self.role}. {self.description}\n\n"
        prompt = (
            "Your Honor, I need to access the courtroom database to retrieve information about my experience."
            "Can you please direct me to the database where I can find details about my qualifications, training, and previous cases?"
            "Upon accessing the database, I will provide a well-structured JSON file outlining the queries required to gather the necessary information."
        )
        response = self._hf_generate(instruction, prompt + "\n\n" + history_context)
        return self.extract_response(response)
    def _prepare_case_query(self, history_context: str) -> str:
        instruction = f"You are a {self.role}. {self.description}\n\n"
        prompt = (
            "Your Honor, I need to access the courtroom database to retrieve information about the case at hand."
            "Can you please direct me to the database where I can find details about the facts, evidence, and legal issues involved in this case?"
            "Upon accessing the database, I will provide a well-structured JSON file outlining the queries required to gather the necessary information."
        )
        response = self._hf_generate(instruction, prompt + "\n\n" + history_context)
        return self.extract_response(response)

    def _prepare_legal_query(self, history_context: str) -> str:
        instruction = f"You are a {self.role}. {self.description}\n\n"
        prompt = (
            "Your Honor, I need to access the courtroom database to retrieve information about relevant legal precedents and statutes."
            "Can you please direct me to the database where I can find details about similar cases and applicable legal provisions?"
            "Upon accessing the database, I will provide a well-structured JSON file outlining the queries required to gather the necessary information."
        )
        response = self._hf_generate(instruction, prompt + "\n\n" + history_context)
        return self.extract_response(response)

    # ---- execute bitch ------------------------------
    def execute(self, queries: Dict[str, str]) -> Dict[str, Any]:
        results = {}
        for query_type, query in queries.items():
            results[query_type] = self._execute_query(query_type, query)
        return results

    def _prepare_context(self, plan: Dict[str, Any], history_list: List[Dict[str, str]]) -> str:
        context = ""
        queries = plan["queries"]

        if self.db:
            if "experience" in queries:
                exp = self.db.query_experience_metadatas(queries["experience"], n_results=3)
                context += f"\nRelevant Experience:\n{exp}\n"
            if "case" in queries:
                cs = self.db.query_case_metadatas(queries["case"], n_results=3)
                context += f"\nCase Precedents:\n{cs}\n"
            if "legal" in queries:
                law = self.db.query_legal(queries["legal"], n_results=3)
                context += f"\nLegal References:\n{law}\n"

        context += "\nConversation History:\n" + self.prepare_history_context(history_list)
        return context

    def speak(self, context: str, prompt: str) -> str:
        instruction = f"You are a {self.role}. {self.description}"
        full_prompt = f"{context}\n\n{prompt}"
        return self._hf_generate(instruction, full_prompt)

    # ---- this is for helpers --------------------------------------------

    def prepare_history_context(self, history: List[Dict[str, str]]) -> str:
        return "\n".join(f"<|{m['role']}|>\n{m['content']}" for m in history)

    def _hf_generate(self, instruction: str, prompt: str) -> str:
        full_prompt = f"<|system|>\n{instruction}\n<|user|>\n{prompt}\n<|assistant|>\n"
        max_total_tokens=4096
        max_new_tokens=512

        tokens = self.tokenizer(full_prompt, return_tensors="pt")["input_ids"][0]
        if len(tokens) + max_new_tokens > max_total_tokens:
            max_prompt_tokens = max_total_tokens - max_new_tokens
            tokens = tokens[-max_prompt_tokens:]  # keep only last tokens
            full_prompt = self.tokenizer.decode(tokens, skip_special_tokens=True)

        return self.client.text_generation(
            full_prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            stream=False
        ).strip()

    def extract_response(self, response: str) -> Any:
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            self.logger.warning(f"Failed to parse JSON: {response}")
            return {"experience": False, "case": False, "legal": False}

    def _extract_plans(self, response: dict) -> Dict[str, bool]:
        return {
            "experience": bool(response.get("experience")),
            "case": bool(response.get("case")),
            "legal": bool(response.get("legal"))
        }

    # ---- extra step for making it better framed ------------------------------

    def step(self, history_list: List[Dict[str, str]], prompt: str) -> str:
        plan_result = self.plan(history_list)
        response = self.execute(plan_result, history_list, prompt)
        self.history.append({"role": "user", "content": prompt})
        self.history.append({"role": "assistant", "content": response})
        return response