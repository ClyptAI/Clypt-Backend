from __future__ import annotations

from backend.services.creator_knowledge import (
    CreatorChannel,
    CreatorKnowledgeService,
    CreatorVideoDocument,
)


class StubClient:
    def __init__(self) -> None:
        self.created_content: list[dict] = []
        self.prompt_calls: list[str] = []
        self.search_calls: list[dict] = []

    def get_kb_root(self):
        return type("Root", (), {"kb_node_id": "root_1", "name": "Root", "type": "folder"})()

    def create_raw_content(self, *, title, text, summary=None, kb_folder_node_id=None, tag_ids=None):
        self.created_content.append(
            {
                "title": title,
                "text": text,
                "summary": summary,
                "kb_folder_node_id": kb_folder_node_id,
                "tag_ids": tag_ids,
            }
        )
        return type("Content", (), {"id": f"content_{len(self.created_content)}"})()

    def get_prompt(self, prompt_id: str):
        self.prompt_calls.append(prompt_id)
        return type(
            "Prompt",
            (),
            {
                "prompt_id": prompt_id,
                "text": "What are the defining content patterns of this creator?",
                "type": "consideration",
            },
        )()

    def wait_for_search_results(self, **kwargs):
        self.search_calls.append(kwargs)
        return type(
            "SearchResponse",
            (),
            {
                "answer": "{\"creator_archetype\":\"Educator\"}",
                "results": [{"content_id": "content_1", "chunk_text": "test", "score": 0.8}],
                "processing_time_ms": 123,
            },
        )()

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return type("SearchResponse", (), {"answer": "", "results": []})()


def test_ensure_creator_workspace_uses_root_kb_node():
    client = StubClient()
    service = CreatorKnowledgeService(client)

    workspace = service.ensure_creator_workspace(
        CreatorChannel(channel_id="UC123", channel_name="Theo - t3.gg", description="Developer content")
    )

    assert workspace.kb_folder_node_id == "root_1"
    assert workspace.kb_folder_name == "Root"
    assert workspace.creator_label == "Theo - t3.gg (UC123)"


def test_ingest_video_documents_binds_content_to_workspace():
    client = StubClient()
    service = CreatorKnowledgeService(client)
    workspace = service.ensure_creator_workspace(
        CreatorChannel(channel_id="UC123", channel_name="Theo - t3.gg")
    )

    documents = [
        CreatorVideoDocument(title="React hot takes", summary="Short summary", text="Transcript one"),
        CreatorVideoDocument(title="TypeScript myths", text="Transcript two"),
    ]

    created = service.ingest_video_documents(workspace, documents)

    assert len(created) == 2
    assert client.created_content == [
        {
            "title": "Theo - t3.gg (UC123) :: React hot takes",
            "text": "Transcript one",
            "summary": "Short summary",
            "kb_folder_node_id": "root_1",
            "tag_ids": None,
        },
        {
            "title": "Theo - t3.gg (UC123) :: TypeScript myths",
            "text": "Transcript two",
            "summary": None,
            "kb_folder_node_id": "root_1",
            "tag_ids": None,
        },
    ]


def test_build_creator_profile_uses_prompt_and_scoped_content_ids():
    client = StubClient()
    service = CreatorKnowledgeService(client)
    workspace = service.ensure_creator_workspace(
        CreatorChannel(channel_id="UC123", channel_name="Theo - t3.gg")
    )

    response = service.build_creator_profile(
        workspace,
        prompt_id="prompt_123",
        content_ids=["content_1", "content_2"],
        template_id="template_123",
        max_results=6,
        save=True,
    )

    assert response.generated_text == "{\"creator_archetype\":\"Educator\"}"
    assert client.prompt_calls == ["prompt_123"]
    assert client.search_calls == [
        {
            "query": service._build_profile_query(
                "What are the defining content patterns of this creator?",
                "Theo - t3.gg (UC123)",
            ),
            "content_ids": ["content_1", "content_2"],
            "max_results": 6,
            "include_answer": True,
        }
    ]


def test_search_creator_context_scopes_search_to_content_ids():
    client = StubClient()
    service = CreatorKnowledgeService(client)

    service.search_creator_context(
        query="What is this creator's style?",
        content_ids=["content_1", "content_2"],
        max_results=4,
    )

    assert client.search_calls == [
        {
            "query": "What is this creator's style?",
            "max_results": 4,
            "content_ids": ["content_1", "content_2"],
            "require_scoped_ids": True,
            "include_answer": False,
        }
    ]
