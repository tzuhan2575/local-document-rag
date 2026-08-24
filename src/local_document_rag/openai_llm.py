"""OpenAI Responses API adapter for grounded RAG prompts."""

from typing import Any

from local_document_rag.prompting import GroundedPrompt


DEFAULT_OPENAI_MODEL = "gpt-5.6-luna"


class OpenAILLMClient:
    """Generate answers through the OpenAI Responses API."""

    def __init__(
        self,
        model: str = DEFAULT_OPENAI_MODEL,
        max_output_tokens: int = 500,
        client: Any | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")

        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be greater than 0")

        if client is None:
            from openai import OpenAI

            client = OpenAI()

        self.model = model
        self.max_output_tokens = max_output_tokens
        self._client = client

    def generate(self, prompt: GroundedPrompt) -> str:
        """Send one grounded prompt and return its output text."""

        response = self._client.responses.create(
            model=self.model,
            instructions=prompt.system,
            input=prompt.user,
            max_output_tokens=self.max_output_tokens,
        )

        output_text = response.output_text

        if not isinstance(output_text, str):
            raise RuntimeError("OpenAI response did not contain text output")

        return output_text
