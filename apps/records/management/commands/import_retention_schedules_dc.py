"""
Import retention schedules from a DocumentCloud project via LLM extraction.

Usage:
    python manage.py import_retention_schedules_dc \
        --project <dc_project_id> \
        --jurisdiction <state> \
        [--type "Retention Schedule"] \
        [--force-reparse]
"""
import json
import re
import time

from django.conf import settings
from django.core.management.base import BaseCommand

import openai

from apps.records.models import RetentionRecord, SourceDocument
from ._dc_utils import fetch_pages, get_dc_client, get_project_documents, pages_to_llm_text


BATCH_SIZE = 100
TITLE_RE = re.compile(r"SCHEDULE NO\.\s+(\S+).*?\(([^)]+)\)", re.IGNORECASE)

EXTRACTION_PROMPT = """\
You are extracting structured records from a government records retention schedule.
The text below was extracted by AWS Textract (tables are formatted as markdown).

Extract every retention record entry and return a JSON object with a single key "records"
containing an array. Each element must have exactly these fields:
  - record_number: str (schedule item/section number, or "" if absent)
  - record_title: str (name of the record type)
  - record_description: str (what records fall under this type)
  - record_custodian_preservation_destruction_requirement: str (custody/handling requirements, or "")
  - minimum_retention_period: str (e.g. "4 years", "Permanent", "Until superseded")
  - regulatory_citation_statutes_rules_notations: str (relevant statutes/rules, or "")
  - page_number: int or null (source page from [Page N] markers in the text)

Rules:
- Omit header rows, table-of-contents entries, and any row that is not a retention record entry.
- If a field is absent, use "" (or null for page_number).
- Do not infer or fabricate any values — extract only what is present in the text.

DOCUMENT TEXT:
{full_text}"""


def parse_schedule_info(document_title: str) -> tuple[str, str]:
    match = TITLE_RE.search(document_title)
    if match:
        return match.group(1), match.group(2)
    return "", document_title


def extract_records_with_llm(openai_client, full_text: str) -> list[dict]:
    prompt = EXTRACTION_PROMPT.format(full_text=full_text)
    response = openai_client.chat.completions.create(
        model=settings.LLM_MODEL,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    data = json.loads(response.choices[0].message.content)
    return data.get("records", [])


class Command(BaseCommand):
    help = "Import retention schedules from a DocumentCloud project via LLM extraction."

    def add_arguments(self, parser):
        parser.add_argument("--project", required=True, help="DocumentCloud project ID")
        parser.add_argument("--jurisdiction", required=True, help="Jurisdiction (e.g. 'Colorado')")
        parser.add_argument(
            "--type",
            dest="doc_type",
            default="Retention Schedule",
            help="Type metadata value to filter on",
        )
        parser.add_argument(
            "--force-reparse",
            dest="force_reparse",
            action="store_true",
            default=False,
            help="Re-run LLM extraction even if document hasn't changed",
        )

    def handle(self, *args, **options):
        project_id = options["project"]
        jurisdiction = options["jurisdiction"]
        doc_type = options["doc_type"]
        force_reparse = options["force_reparse"]

        dc_client = get_dc_client()
        openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

        docs = list(get_project_documents(dc_client, project_id, doc_type))
        self.stdout.write(f"Found {len(docs)} document(s) with Type='{doc_type}' in project {project_id}.")

        total_created = 0
        total_updated = 0

        for document in docs:
            dc_id = str(document.id)
            dc_updated_at = document.updated_at

            existing = SourceDocument.objects.filter(documentcloud_id=dc_id).first()
            if existing and not force_reparse:
                if existing.documentcloud_updated_at and existing.documentcloud_updated_at >= dc_updated_at:
                    self.stdout.write(f"  Skipping '{document.title}' (unchanged).")
                    continue

            self.stdout.write(f"  Fetching '{document.title}'...")
            pages = fetch_pages(document)
            full_text = pages_to_llm_text(pages)

            self.stdout.write(f"  Extracting records via LLM...")
            records_data = extract_records_with_llm(openai_client, full_text)
            if not records_data:
                self.stdout.write(self.style.WARNING(f"    No records extracted for '{document.title}'."))
                continue

            schedule_number, entity_type = parse_schedule_info(document.title)

            source_doc, _ = SourceDocument.objects.update_or_create(
                documentcloud_id=dc_id,
                defaults={
                    "document_title": document.title,
                    "filename": document.slug,
                    "jurisdiction": jurisdiction,
                    "entity_type": entity_type,
                    "schedule_number": schedule_number,
                    "documentcloud_url": document.canonical_url,
                    "documentcloud_updated_at": dc_updated_at,
                },
            )

            source_doc.records.all().delete()

            new_records = []
            for raw in records_data:
                record_title = raw.get("record_title", "").strip()
                if not record_title:
                    continue
                period = raw.get("minimum_retention_period", "").strip()
                new_records.append(RetentionRecord(
                    source_document=source_doc,
                    record_number=raw.get("record_number", ""),
                    record_title=record_title,
                    record_description=raw.get("record_description", ""),
                    custodian_requirement=raw.get(
                        "record_custodian_preservation_destruction_requirement", ""
                    ),
                    minimum_retention_period=period,
                    regulatory_citations=raw.get(
                        "regulatory_citation_statutes_rules_notations", ""
                    ),
                    page_number=raw.get("page_number"),
                    is_cross_reference=period.startswith("See "),
                    is_permanent=period.strip().lower() == "permanent",
                ))

            RetentionRecord.objects.bulk_create(new_records)
            source_doc.record_count = len(new_records)
            source_doc.save(update_fields=["record_count"])
            self.stdout.write(f"    {len(new_records)} records.")
            total_created += len(new_records)

            # Generate embeddings inline for newly created records
            record_qs = list(
                RetentionRecord.objects.filter(
                    source_document__documentcloud_id=dc_id,
                    is_cross_reference=False,
                    embedding__isnull=True,
                )
            )
            embedded = 0
            for i in range(0, len(record_qs), BATCH_SIZE):
                batch = record_qs[i : i + BATCH_SIZE]
                texts = [r.to_chunk_text() for r in batch]
                response = openai_client.embeddings.create(
                    model=settings.EMBEDDING_MODEL,
                    input=texts,
                )
                for record, emb_obj in zip(batch, response.data):
                    record.embedding = emb_obj.embedding
                    record.save(update_fields=["embedding"])
                embedded += len(batch)
                if i + BATCH_SIZE < len(record_qs):
                    time.sleep(0.5)
            self.stdout.write(f"    Embedded {embedded} records.")
            total_updated += embedded

        self.stdout.write(
            self.style.SUCCESS(
                f"Done. {total_created} records imported, {total_updated} embedded."
            )
        )
