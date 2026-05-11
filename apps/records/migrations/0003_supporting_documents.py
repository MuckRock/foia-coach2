from django.db import migrations, models
import django.db.models.deletion
import pgvector.django


class Migration(migrations.Migration):

    dependencies = [
        ("records", "0002_seed_system_prompt"),
    ]

    operations = [
        migrations.CreateModel(
            name="SupportingDocument",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("filename", models.CharField(max_length=512)),
                ("document_title", models.CharField(max_length=512)),
                ("document_type", models.CharField(blank=True, max_length=100)),
                ("jurisdiction", models.CharField(blank=True, max_length=255)),
                ("chunk_count", models.IntegerField(default=0)),
                ("uploaded_at", models.DateTimeField(auto_now_add=True)),
            ],
            options={
                "ordering": ["document_title"],
            },
        ),
        migrations.CreateModel(
            name="DocumentChunk",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("supporting_document", models.ForeignKey(
                    on_delete=django.db.models.deletion.CASCADE,
                    related_name="chunks",
                    to="records.supportingdocument",
                )),
                ("chunk_index", models.IntegerField()),
                ("page_number", models.IntegerField()),
                ("text", models.TextField()),
                ("token_count", models.IntegerField(default=0)),
                ("embedding", pgvector.django.VectorField(dimensions=1536, null=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={
                "ordering": ["supporting_document", "chunk_index"],
            },
        ),
        migrations.AddIndex(
            model_name="documentchunk",
            index=models.Index(fields=["supporting_document"], name="records_doc_support_idx"),
        ),
        migrations.RunSQL(
            sql="""
                CREATE INDEX document_chunk_embedding_idx
                ON records_documentchunk USING hnsw (embedding vector_cosine_ops);
            """,
            reverse_sql="DROP INDEX IF EXISTS document_chunk_embedding_idx;",
        ),
    ]
