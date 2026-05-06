"""
Management command to generate OpenAI embeddings for RetentionRecords.

Usage:
    python manage.py generate_embeddings [--source-document <id>] [--force]
"""
import time

from django.conf import settings
from django.core.management.base import BaseCommand

import openai

from apps.records.models import RetentionRecord


BATCH_SIZE = 100


class Command(BaseCommand):
    help = "Generate embeddings for retention records."

    def add_arguments(self, parser):
        parser.add_argument(
            "--source-document",
            dest="source_document",
            type=int,
            default=None,
            help="Limit to records from this SourceDocument ID",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            default=False,
            help="Re-embed records that already have embeddings",
        )

    def handle(self, *args, **options):
        force = options["force"]
        source_document_id = options["source_document"]

        qs = RetentionRecord.objects.filter(is_cross_reference=False)
        if source_document_id is not None:
            qs = qs.filter(source_document_id=source_document_id)
        if not force:
            qs = qs.filter(embedding__isnull=True)

        records = list(qs)
        total = len(records)
        if total == 0:
            self.stdout.write("No records to embed.")
            return

        self.stdout.write(f"Embedding {total} records...")

        client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        processed = 0

        for i in range(0, total, BATCH_SIZE):
            batch = records[i : i + BATCH_SIZE]
            texts = [r.to_chunk_text() for r in batch]

            response = client.embeddings.create(
                model=settings.EMBEDDING_MODEL,
                input=texts,
            )

            for record, embedding_obj in zip(batch, response.data):
                record.embedding = embedding_obj.embedding
                record.save(update_fields=["embedding"])

            processed += len(batch)
            self.stdout.write(f"  {processed}/{total} embedded")

            if i + BATCH_SIZE < total:
                time.sleep(0.5)

        self.stdout.write(self.style.SUCCESS(f"Done. {processed} records embedded."))
