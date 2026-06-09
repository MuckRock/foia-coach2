"""
Import supporting document guides from a DocumentCloud project.

Usage:
    python manage.py import_guides_dc \
        --project <dc_project_id> \
        --jurisdiction <state> \
        [--type "Guide"] \
        [--replace]
"""

import time

import openai
from django.conf import settings
from django.core.management.base import BaseCommand

from apps.records.models import DocumentChunk, SupportingDocument

from ._dc_utils import fetch_pages, get_dc_client, get_project_documents

BATCH_SIZE = 100


def chunk_pages(pages: list[dict], max_tokens=800, overlap_tokens=100, min_tokens=50):
    """
    Split DC pages into overlapping chunks, yielding dicts with:
        chunk_index, page_number, text, token_count

    pages: list of {'page': int (0-indexed), 'contents': str} from DC get_json_text().
    """
    chunk_index = 0
    pending_overlap = ""

    for p in pages:
        page_num = p["page"] + 1  # DC pages are 0-indexed; store 1-indexed
        page_text = p["contents"].strip()

        if pending_overlap:
            text = pending_overlap + "\n" + page_text
        else:
            text = page_text

        words = text.split()
        if len(words) < min_tokens:
            continue

        if len(words) <= max_tokens:
            yield {
                "chunk_index": chunk_index,
                "page_number": page_num,
                "text": text,
                "token_count": len(words),
            }
            chunk_index += 1
            pending_overlap = (
                " ".join(words[-overlap_tokens:])
                if len(words) >= overlap_tokens
                else text
            )
        else:
            start = 0
            while start < len(words):
                end = start + max_tokens
                chunk_words = words[start : min(end, len(words))]
                yield {
                    "chunk_index": chunk_index,
                    "page_number": page_num,
                    "text": " ".join(chunk_words),
                    "token_count": len(chunk_words),
                }
                chunk_index += 1
                if end >= len(words):
                    break
                start = end - overlap_tokens
            pending_overlap = (
                " ".join(words[-overlap_tokens:])
                if len(words) >= overlap_tokens
                else text
            )


class Command(BaseCommand):
    help = "Import supporting document guides from a DocumentCloud project."

    def add_arguments(self, parser):
        parser.add_argument("--project", required=True, help="DocumentCloud project ID")
        parser.add_argument(
            "--jurisdiction", required=True, help="Jurisdiction (e.g. 'Colorado')"
        )
        parser.add_argument(
            "--type",
            dest="doc_type",
            default="Guide",
            help="Type metadata value to filter on",
        )
        parser.add_argument(
            "--replace",
            action="store_true",
            default=False,
            help="Delete existing chunks before re-importing",
        )

    def handle(self, *args, **options):
        project_id = options["project"]
        jurisdiction = options["jurisdiction"]
        doc_type = options["doc_type"]
        replace = options["replace"]

        client = get_dc_client()
        openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

        docs = list(get_project_documents(client, project_id, doc_type))
        self.stdout.write(
            f"Found {len(docs)} document(s) with Type='{doc_type}' in project {project_id}."
        )

        total_chunks = 0
        total_embedded = 0

        for document in docs:
            dc_id = str(document.id)
            dc_updated_at = document.updated_at

            existing = SupportingDocument.objects.filter(documentcloud_id=dc_id).first()
            if existing and not replace:
                if (
                    existing.documentcloud_updated_at
                    and existing.documentcloud_updated_at >= dc_updated_at
                ):
                    self.stdout.write(f"  Skipping '{document.title}' (unchanged).")
                    continue

            self.stdout.write(f"  Fetching '{document.title}'...")
            pages = fetch_pages(document)

            supporting_doc, _ = SupportingDocument.objects.update_or_create(
                documentcloud_id=dc_id,
                defaults={
                    "document_title": document.title,
                    "filename": document.slug,
                    "jurisdiction": jurisdiction,
                    "documentcloud_url": document.canonical_url,
                    "documentcloud_updated_at": dc_updated_at,
                },
            )

            if replace or existing:
                deleted_count = supporting_doc.chunks.all().delete()[0]
                if deleted_count:
                    self.stdout.write(f"    Deleted {deleted_count} existing chunks.")

            chunks_data = list(chunk_pages(pages))
            if not chunks_data:
                self.stdout.write(
                    self.style.WARNING(
                        f"    No chunks produced for '{document.title}'."
                    )
                )
                continue

            DocumentChunk.objects.bulk_create(
                [
                    DocumentChunk(
                        supporting_document=supporting_doc,
                        chunk_index=c["chunk_index"],
                        page_number=c["page_number"],
                        text=c["text"].replace("\x00", ""),
                        token_count=c["token_count"],
                    )
                    for c in chunks_data
                ]
            )
            supporting_doc.chunk_count = len(chunks_data)
            supporting_doc.save(update_fields=["chunk_count"])
            total_chunks += len(chunks_data)
            self.stdout.write(f"    Created {len(chunks_data)} chunks.")

            # Generate embeddings inline
            chunk_qs = list(supporting_doc.chunks.filter(embedding__isnull=True))
            for i in range(0, len(chunk_qs), BATCH_SIZE):
                batch = chunk_qs[i : i + BATCH_SIZE]
                texts = [c.to_chunk_text() for c in batch]
                response = openai_client.embeddings.create(
                    model=settings.EMBEDDING_MODEL,
                    input=texts,
                )
                for chunk, emb_obj in zip(batch, response.data):
                    chunk.embedding = emb_obj.embedding
                    chunk.save(update_fields=["embedding"])
                total_embedded += len(batch)
                if i + BATCH_SIZE < len(chunk_qs):
                    time.sleep(0.5)
            self.stdout.write(f"    Embedded {len(chunk_qs)} chunks.")

        self.stdout.write(
            self.style.SUCCESS(
                f"Done. {total_chunks} chunks created, {total_embedded} embedded."
            )
        )
