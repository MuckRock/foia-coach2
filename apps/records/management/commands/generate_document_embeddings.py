"""
Management command to generate OpenAI embeddings for DocumentChunks.

Usage:
    python manage.py generate_document_embeddings [--supporting-document <id>] [--force]
"""
import time

from django.conf import settings
from django.core.management.base import BaseCommand

import openai

from apps.records.models import DocumentChunk


BATCH_SIZE = 100


class Command(BaseCommand):
    help = "Generate embeddings for supporting document chunks."

    def add_arguments(self, parser):
        parser.add_argument(
            "--supporting-document",
            dest="supporting_document",
            type=int,
            default=None,
            help="Limit to chunks from this SupportingDocument ID",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            default=False,
            help="Re-embed chunks that already have embeddings",
        )

    def handle(self, *args, **options):
        force = options["force"]
        supporting_document_id = options["supporting_document"]

        qs = DocumentChunk.objects.all()
        if supporting_document_id is not None:
            qs = qs.filter(supporting_document_id=supporting_document_id)
        if not force:
            qs = qs.filter(embedding__isnull=True)

        chunks = list(qs)
        total = len(chunks)
        if total == 0:
            self.stdout.write("No chunks to embed.")
            return

        self.stdout.write(f"Embedding {total} chunks...")

        client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        processed = 0

        for i in range(0, total, BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            texts = [c.to_chunk_text() for c in batch]

            response = client.embeddings.create(
                model=settings.EMBEDDING_MODEL,
                input=texts,
            )

            for chunk, embedding_obj in zip(batch, response.data):
                chunk.embedding = embedding_obj.embedding
                chunk.save(update_fields=["embedding"])

            processed += len(batch)
            self.stdout.write(f"  {processed}/{total} embedded")

            if i + BATCH_SIZE < total:
                time.sleep(0.5)

        self.stdout.write(self.style.SUCCESS(f"Done. {processed} chunks embedded."))
