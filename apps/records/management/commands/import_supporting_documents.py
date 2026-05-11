"""
Management command to import a PDF as chunked DocumentChunks.

Usage:
    python manage.py import_supporting_documents <pdf_file> --title "..." [--document-type "..."] [--jurisdiction "..."] [--replace]
"""
import os

from django.core.management.base import BaseCommand, CommandError
from pypdf import PdfReader

from apps.records.models import DocumentChunk, SupportingDocument


def chunk_pdf(pdf_path, max_tokens=800, overlap_tokens=100, min_tokens=50):
    """
    Split a PDF into overlapping text chunks, yielding dicts with:
        chunk_index, page_number, text, token_count
    """
    reader = PdfReader(pdf_path)
    chunk_index = 0
    pending_overlap = ""

    for page_num, page in enumerate(reader.pages, 1):
        page_text = page.extract_text()
        if page_text is None:
            continue

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
            pending_overlap = " ".join(words[-overlap_tokens:]) if len(words) >= overlap_tokens else text
        else:
            start = 0
            while start < len(words):
                end = start + max_tokens
                chunk_words = words[start:min(end, len(words))]
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
            pending_overlap = " ".join(words[-overlap_tokens:]) if len(words) >= overlap_tokens else text


class Command(BaseCommand):
    help = "Import a PDF as chunked supporting document records."

    def add_arguments(self, parser):
        parser.add_argument("pdf_file", help="Path to the PDF file to import")
        parser.add_argument("--title", required=True, help="Document title")
        parser.add_argument("--document-type", dest="document_type", default="", help="Document type (e.g. 'FOIA Guide')")
        parser.add_argument("--jurisdiction", default="", help="Jurisdiction (e.g. 'Colorado')")
        parser.add_argument(
            "--replace",
            action="store_true",
            default=False,
            help="Delete existing chunks before re-importing",
        )

    def handle(self, *args, **options):
        pdf_path = options["pdf_file"]
        if not os.path.exists(pdf_path):
            raise CommandError(f"File not found: {pdf_path}")

        filename = os.path.basename(pdf_path)
        title = options["title"]
        document_type = options["document_type"]
        jurisdiction = options["jurisdiction"]
        replace = options["replace"]

        supporting_doc, created = SupportingDocument.objects.get_or_create(
            document_title=title,
            defaults={
                "filename": filename,
                "document_type": document_type,
                "jurisdiction": jurisdiction,
            },
        )

        if not created:
            existing_count = supporting_doc.chunks.count()
            if existing_count > 0:
                if replace:
                    supporting_doc.chunks.all().delete()
                    self.stdout.write(f"Deleted {existing_count} existing chunks.")
                else:
                    self.stdout.write(
                        self.style.WARNING(
                            f"Document '{title}' already has {existing_count} chunks. "
                            "Use --replace to re-import."
                        )
                    )
                    return

        self.stdout.write(f"Chunking {pdf_path}...")
        chunks_data = list(chunk_pdf(pdf_path))
        total = len(chunks_data)

        if total == 0:
            self.stdout.write(self.style.WARNING("No chunks produced — check the PDF has extractable text."))
            return

        self.stdout.write(f"Creating {total} chunks...")
        batch_size = 100
        for i in range(0, total, batch_size):
            batch = chunks_data[i : i + batch_size]
            DocumentChunk.objects.bulk_create([
                DocumentChunk(
                    supporting_document=supporting_doc,
                    chunk_index=c["chunk_index"],
                    page_number=c["page_number"],
                    text=c["text"],
                    token_count=c["token_count"],
                )
                for c in batch
            ])

        supporting_doc.chunk_count = total
        supporting_doc.save(update_fields=["chunk_count"])

        self.stdout.write(self.style.SUCCESS(f"Done. Imported {total} chunks for '{title}'."))
