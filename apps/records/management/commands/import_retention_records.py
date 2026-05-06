"""
Management command to import retention records from a parsed JSON file.

Usage:
    python manage.py import_retention_records <json_file> \
        --jurisdiction "Colorado" \
        --filename "Colorado_Schedules.pdf"
"""
import json
import re
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from apps.records.models import RetentionRecord, SourceDocument

TITLE_RE = re.compile(r"SCHEDULE NO\.\s+(\S+).*?\(([^)]+)\)", re.IGNORECASE)


def parse_schedule_info(document_title: str) -> tuple[str, str]:
    """Extract schedule_number and entity_type from document_title."""
    match = TITLE_RE.search(document_title)
    if match:
        return match.group(1), match.group(2)
    return "", document_title


class Command(BaseCommand):
    help = "Import retention records from a parsed JSON file."

    def add_arguments(self, parser):
        parser.add_argument("json_file", type=str, help="Path to JSON file")
        parser.add_argument("--jurisdiction", type=str, default="Colorado")
        parser.add_argument("--filename", type=str, default="")

    def handle(self, *args, **options):
        json_path = Path(options["json_file"])
        if not json_path.exists():
            raise CommandError(f"File not found: {json_path}")

        with open(json_path) as f:
            records_data = json.load(f)

        if isinstance(records_data, dict):
            records_data = records_data["retention_schedule_entries"]


        jurisdiction = options["jurisdiction"]
        filename = options["filename"] or json_path.name

        # Group records by document_title
        by_title: dict[str, list[dict]] = {}
        for record in records_data:
            title = record.get("document_title", "")
            by_title.setdefault(title, []).append(record)

        created_total = 0
        updated_total = 0

        for document_title, group in by_title.items():
            schedule_number, entity_type = parse_schedule_info(document_title)

            source_doc, _ = SourceDocument.objects.get_or_create(
                document_title=document_title,
                defaults={
                    "filename": filename,
                    "jurisdiction": jurisdiction,
                    "entity_type": entity_type,
                    "schedule_number": schedule_number,
                },
            )

            for raw in group:
                period = raw.get("minimum_retention_period", "").strip()

                defaults = {
                    "record_number": raw.get("record_number", ""),
                    "record_description": raw.get("record_description", ""),
                    "custodian_requirement": raw.get(
                        "record_custodian_preservation_destruction_requirement", ""
                    ),
                    "minimum_retention_period": period,
                    "regulatory_citations": raw.get(
                        "regulatory_citation_statutes_rules_notations", ""
                    ),
                    "page_number": raw.get("page_number"),
                    "is_cross_reference": period.startswith("See "),
                    "is_permanent": period.strip().lower() == "permanent",
                }

                record_title = raw.get("record_title")
                if not record_title:
                    raise CommandError(
                        f"Record missing 'record_title' in {document_title!r}: {raw}"
                    )

                _, _created = RetentionRecord.objects.update_or_create(
                    source_document=source_doc,
                    record_title=record_title,
                    defaults=defaults,
                )
                if _created:
                    created_total += 1
                else:
                    updated_total += 1

            source_doc.record_count = source_doc.records.count()
            source_doc.save(update_fields=["record_count"])

        self.stdout.write(
            self.style.SUCCESS(
                f"Import complete: {created_total} created, {updated_total} updated."
            )
        )
