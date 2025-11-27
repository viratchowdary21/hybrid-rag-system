import os
from typing import List, Dict, Tuple
from openai import OpenAI
import google.generativeai as genai
from constants import OPENAI_API_KEY, GEMINI_API_KEY

class AnswerGenerator:
    def __init__(self):
        self.llm_provider = "openai"  # or "gemini"
        self.max_context_length = 4000

        if self.llm_provider == "openai":
            self.client = OpenAI(api_key=OPENAI_API_KEY)
        elif self.llm_provider == "gemini":
            genai.configure(api_key=GEMINI_API_KEY)
            self.model_name = "gemini-pro"   # set from env / config in production
            self.model = genai.GenerativeModel(self.model_name)

    def generate_answer(
        self, query: str, contexts: List[Dict]
    ) -> Tuple[str, List[str]]:
        """Generate answer using LLM with retrieved contexts"""
        if not contexts:
            return "I couldn't find relevant information to answer your question.", []

        # Prepare context string
        context_text = self._prepare_context(contexts)

        # Generate prompt
        prompt = self._build_prompt(query, context_text)

        try:
            if self.llm_provider == "openai":
                answer = self._call_openai(prompt)
            else:
                answer = self._call_gemini(prompt)

            # Extract source chunks
            sources = [ctx["chunk_id"] for ctx in contexts[:3]]  # Top 3 sources

            return answer, sources

        except Exception as e:
            return f"Error generating answer: {str(e)}", []

    def _prepare_context(self, contexts: List[Dict]) -> str:
        """Prepare context from retrieved chunks"""
        context_parts = []
        total_length = 0

        for ctx in contexts:
            chunk_text = f"Source {ctx['chunk_id']}: {ctx['content']}"
            if total_length + len(chunk_text) > self.max_context_length:
                break
            context_parts.append(chunk_text)
            total_length += len(chunk_text)

        return "\n\n".join(context_parts)

    def _build_prompt(self, query: str, context: str) -> str:
        """Build LLM prompt"""
        return f"""
    Based on the following context, please answer the question. If the context doesn't contain relevant information, say so.

    Context:
    {context}

    Question: {query}

    Please provide a comprehensive answer based only on the provided context. Cite relevant source IDs when applicable.
    """

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API"""
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on provided context.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0.1,
        )
        return response.choices[0].message.content

    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini API"""
        response = self.model.generate_content(prompt)
        return response.text
