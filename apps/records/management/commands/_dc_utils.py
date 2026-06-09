from django.conf import settings
from documentcloud import DocumentCloud


def get_dc_client():
    return DocumentCloud(settings.DOCUMENTCLOUD_USERNAME, settings.DOCUMENTCLOUD_PASSWORD)


def get_project_documents(client, project_id: str, doc_type: str | None = None):
    """Yield documents from a DC project filtered by Type metadata key."""
    project = client.projects.get(id=project_id)
    for doc in project.document_list:
        if doc_type is None or doc_type in doc.data.get("Type", []):
            yield doc


def fetch_pages(document) -> list[dict]:
    """
    Return page data from DC's get_json_text() as a list of dicts.
    Each dict has 'page' (0-indexed int) and 'contents' (str).
    Single API call — more efficient than fetching pages individually.
    """
    data = document.get_json_text()
    return data.get("pages", [])


def pages_to_llm_text(pages: list[dict]) -> str:
    """
    Concatenate pages into a single string with [Page N] markers (1-indexed).
    Use this only when a flat string is needed, e.g. for LLM prompts.
    """
    lines = []
    for p in pages:
        page_num = p["page"] + 1  # DC pages are 0-indexed
        text = p["contents"].strip()
        lines.append(f"[Page {page_num}]\n{text}")
    return "\n\n".join(lines)

