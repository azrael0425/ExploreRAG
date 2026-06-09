from __future__ import annotations

import base64

from app.langchain_rag.chains.answer import AnswerInput, build_answer_prompt
from app.schemas.rag import ChatSourceChunk
from app.services.llm.types import LLMImagePart


def test_answer_prompt_keeps_history_and_untrusted_material_boundary() -> None:
    value = build_answer_prompt(AnswerInput(
        system_prompt="system",
        history=[{"role": "user", "content": "old question"}, {"role": "assistant", "content": "old answer"}],
        context="Source [KB-test]: evidence",
        question="What happened?",
        sources=[ChatSourceChunk(index="KB-test", chunk_id="chunk", content="evidence")],
    ))
    messages = value.to_messages()

    assert messages[0].content == "system"
    assert messages[1].content == "old question"
    assert messages[2].content == "old answer"
    assert "=== UNTRUSTED RETRIEVED MATERIAL ===" in messages[3].content
    assert "[KB-test]" in messages[3].content


def test_answer_prompt_encodes_images_as_langchain_multimodal_blocks() -> None:
    value = build_answer_prompt(AnswerInput(
        system_prompt="system",
        context="evidence",
        question="describe image",
        llm_images=[LLMImagePart(data=b"image", mime_type="image/png")],
    ))
    content = value.to_messages()[-1].content

    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[1]["image_url"]["url"] == (
        "data:image/png;base64," + base64.b64encode(b"image").decode("ascii")
    )


def test_answer_prompt_keeps_greetings_as_direct_user_messages() -> None:
    value = build_answer_prompt(AnswerInput(
        system_prompt="system",
        context="",
        question="Hello!",
        grounded=False,
    ))
    assert value.to_messages()[-1].content == "Hello!"
