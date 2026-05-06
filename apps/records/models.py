from django.db import models
from pgvector.django import VectorField


class SystemPrompt(models.Model):
    """
    Editable system prompt for the FOIA Coach assistant.
    Only one prompt is active at a time.
    """
    name = models.CharField(max_length=255)
    content = models.TextField()
    is_active = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self):
        return f"{self.name}{' (active)' if self.is_active else ''}"

    def save(self, *args, **kwargs):
        if self.is_active:
            SystemPrompt.objects.exclude(pk=self.pk).update(is_active=False)
        super().save(*args, **kwargs)

    @classmethod
    def get_active(cls) -> str:
        prompt = cls.objects.filter(is_active=True).first()
        if prompt is None:
            raise RuntimeError("No active system prompt configured.")
        return prompt.content


class SourceDocument(models.Model):
    """A source retention schedule PDF."""
    filename = models.CharField(max_length=512)
    document_title = models.CharField(max_length=512)
    jurisdiction = models.CharField(max_length=255)
    entity_type = models.CharField(max_length=255)
    schedule_number = models.CharField(max_length=50, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    record_count = models.IntegerField(default=0)

    class Meta:
        ordering = ["jurisdiction", "document_title"]

    def __str__(self):
        return self.document_title


class RetentionRecord(models.Model):
    """A single parsed record entry from a retention schedule."""

    source_document = models.ForeignKey(
        SourceDocument, on_delete=models.CASCADE, related_name="records"
    )

    # Core fields from parsed JSON
    record_number = models.CharField(max_length=50, blank=True)
    record_title = models.CharField(max_length=512)
    record_description = models.TextField()
    custodian_requirement = models.TextField(blank=True)
    minimum_retention_period = models.TextField()
    regulatory_citations = models.TextField(blank=True)
    page_number = models.IntegerField(null=True, blank=True)

    # Derived / normalized fields
    is_cross_reference = models.BooleanField(default=False)
    is_permanent = models.BooleanField(default=False)

    # Search fields
    embedding = VectorField(dimensions=1536, null=True)
    # search_vector is a generated tsvector column added via RunSQL migration
    # It is NOT declared as a Django model field to avoid ORM conflicts.

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["source_document", "record_number"]
        indexes = [
            models.Index(fields=["source_document"]),
            models.Index(fields=["is_cross_reference"]),
            models.Index(fields=["is_permanent"]),
        ]

    def __str__(self):
        return f"{self.record_number} — {self.record_title}"

    def to_chunk_text(self) -> str:
        """Render the record as a natural language string for embedding."""
        parts = [
            f"Title: {self.record_title}",
            f"Description: {self.record_description}",
            f"Retention period: {self.minimum_retention_period}",
        ]
        if self.custodian_requirement:
            parts.append(f"Disposition: {self.custodian_requirement}")
        if self.regulatory_citations:
            parts.append(f"Legal citations: {self.regulatory_citations}")
        parts.append(f"Source: {self.source_document.document_title}, page {self.page_number}")
        return "\n".join(parts)
