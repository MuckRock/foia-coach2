from django.contrib import admin
from django.core.management import call_command

from apps.records.models import (DocumentChunk, NFOICChapter, RetentionRecord,
                                 SourceDocument, SupportingDocument,
                                 SystemPrompt)


@admin.register(SystemPrompt)
class SystemPromptAdmin(admin.ModelAdmin):
    list_display = ["name", "is_active", "updated_at"]
    list_filter = ["is_active"]
    readonly_fields = ["created_at", "updated_at"]
    fields = ["name", "content", "is_active", "created_at", "updated_at"]


@admin.register(NFOICChapter)
class NFOICChapterAdmin(admin.ModelAdmin):
    list_display = ["name", "jurisdiction", "website", "email", "phone"]
    list_filter = ["jurisdiction"]
    search_fields = ["name", "jurisdiction", "description"]
    readonly_fields = ["created_at", "updated_at"]


@admin.register(SourceDocument)
class SourceDocumentAdmin(admin.ModelAdmin):
    list_display = [
        "filename",
        "document_title",
        "jurisdiction",
        "entity_type",
        "record_count",
        "uploaded_at",
    ]
    readonly_fields = ["record_count", "uploaded_at"]
    actions = ["regenerate_embeddings"]

    @admin.action(description="Regenerate embeddings for selected documents")
    def regenerate_embeddings(self, request, queryset):
        for doc in queryset:
            call_command("generate_embeddings", source_document=doc.pk, force=True)
        self.message_user(request, "Embeddings regenerated.")


@admin.register(RetentionRecord)
class RetentionRecordAdmin(admin.ModelAdmin):
    list_display = [
        "record_title",
        "record_number",
        "source_document",
        "minimum_retention_period",
        "is_permanent",
        "is_cross_reference",
        "has_embedding",
    ]
    list_filter = ["source_document", "is_permanent", "is_cross_reference"]
    search_fields = ["record_title", "record_description", "record_number"]
    readonly_fields = ["has_embedding", "created_at", "updated_at"]
    actions = ["regenerate_embeddings"]

    @admin.display(boolean=True, description="Embedded")
    def has_embedding(self, obj):
        return obj.embedding is not None

    @admin.action(description="Regenerate embeddings for selected records")
    def regenerate_embeddings(self, request, queryset):
        for record in queryset:
            call_command(
                "generate_embeddings",
                source_document=record.source_document_id,
                force=True,
            )
        self.message_user(request, "Embeddings regenerated.")


@admin.register(SupportingDocument)
class SupportingDocumentAdmin(admin.ModelAdmin):
    list_display = [
        "filename",
        "document_title",
        "document_type",
        "jurisdiction",
        "chunk_count",
        "uploaded_at",
    ]
    list_filter = ["document_type", "jurisdiction"]
    readonly_fields = ["chunk_count", "uploaded_at"]
    actions = ["regenerate_embeddings"]

    @admin.action(description="Regenerate embeddings for selected documents")
    def regenerate_embeddings(self, request, queryset):
        for doc in queryset:
            call_command(
                "generate_document_embeddings", supporting_document=doc.pk, force=True
            )
        self.message_user(request, "Embeddings regenerated.")


@admin.register(DocumentChunk)
class DocumentChunkAdmin(admin.ModelAdmin):
    list_display = [
        "chunk_index",
        "supporting_document",
        "page_number",
        "token_count",
        "has_embedding",
    ]
    list_filter = ["supporting_document"]
    search_fields = ["text"]
    readonly_fields = [
        "has_embedding",
        "token_count",
        "chunk_index",
        "page_number",
        "created_at",
        "updated_at",
    ]

    @admin.display(boolean=True, description="Embedded")
    def has_embedding(self, obj):
        return obj.embedding is not None
