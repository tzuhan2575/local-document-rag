from types import SimpleNamespace

import pytest

from local_document_rag.openai_llm import OpenAILLMClient
from local_document_rag.prompting import GroundedPrompt


class FakeResponses:
    def __init__(self, output_text="answer"):
        self.output_text = output_text
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(output_text=self.output_text)


class FakeOpenAI:
    def __init__(self, output_text="answer"):
        self.responses = FakeResponses(output_text)


def test_generate_maps_grounded_prompt_to_responses_api():
    client = FakeOpenAI("Grounded answer. [Source 1]")
    llm = OpenAILLMClient(
        model="test-model",
        max_output_tokens=123,
        client=client,
    )
    prompt = GroundedPrompt(
        system="system instructions",
        user="question and context",
    )

    answer = llm.generate(prompt)

    assert answer == "Grounded answer. [Source 1]"
    assert client.responses.requests == [
        {
            "model": "test-model",
            "instructions": "system instructions",
            "input": "question and context",
            "max_output_tokens": 123,
        }
    ]


@pytest.mark.parametrize(
    ("model", "max_output_tokens", "message"),
    [
        ("", 100, "model must not be empty"),
        ("   ", 100, "model must not be empty"),
        ("test-model", 0, "max_output_tokens"),
    ],
)
def test_invalid_configuration_is_rejected(
    model,
    max_output_tokens,
    message,
):
    with pytest.raises(ValueError, match=message):
        OpenAILLMClient(
            model=model,
            max_output_tokens=max_output_tokens,
            client=FakeOpenAI(),
        )


def test_non_text_response_is_rejected():
    llm = OpenAILLMClient(client=FakeOpenAI(output_text=None))

    with pytest.raises(RuntimeError, match="did not contain text"):
        llm.generate(GroundedPrompt(system="system", user="user"))
