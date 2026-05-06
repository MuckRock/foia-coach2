from django.db import migrations, models
import django.db.models.deletion
import pgvector.django


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        # Enable extensions first
        migrations.RunSQL(
            sql="CREATE EXTENSION IF NOT EXISTS vector;",
            reverse_sql="DROP EXTENSION IF EXISTS vector;",
        ),
        migrations.RunSQL(
            sql="CREATE EXTENSION IF NOT EXISTS pg_trgm;",
            reverse_sql="DROP EXTENSION IF EXISTS pg_trgm;",
        ),
        migrations.CreateModel(
            name="SystemPrompt",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("name", models.CharField(max_length=255)),
                ("content", models.TextField()),
                ("is_active", models.BooleanField(default=False)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={
                "ordering": ["-updated_at"],
            },
        ),
        migrations.CreateModel(
            name="SourceDocument",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("filename", models.CharField(max_length=512)),
                ("document_title", models.CharField(max_length=512)),
                ("jurisdiction", models.CharField(max_length=255)),
                ("entity_type", models.CharField(max_length=255)),
                ("schedule_number", models.CharField(blank=True, max_length=50)),
                ("uploaded_at", models.DateTimeField(auto_now_add=True)),
                ("record_count", models.IntegerField(default=0)),
            ],
            options={
                "ordering": ["jurisdiction", "document_title"],
            },
        ),
        migrations.CreateModel(
            name="RetentionRecord",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("source_document", models.ForeignKey(
                    on_delete=django.db.models.deletion.CASCADE,
                    related_name="records",
                    to="records.sourcedocument",
                )),
                ("record_number", models.CharField(blank=True, max_length=50)),
                ("record_title", models.CharField(max_length=512)),
                ("record_description", models.TextField()),
                ("custodian_requirement", models.TextField(blank=True)),
                ("minimum_retention_period", models.TextField()),
                ("regulatory_citations", models.TextField(blank=True)),
                ("page_number", models.IntegerField(blank=True, null=True)),
                ("is_cross_reference", models.BooleanField(default=False)),
                ("is_permanent", models.BooleanField(default=False)),
                ("embedding", pgvector.django.VectorField(dimensions=1536, null=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={
                "ordering": ["source_document", "record_number"],
            },
        ),
        migrations.AddIndex(
            model_name="retentionrecord",
            index=models.Index(fields=["source_document"], name="records_ret_source__idx"),
        ),
        migrations.AddIndex(
            model_name="retentionrecord",
            index=models.Index(fields=["is_cross_reference"], name="records_ret_is_cros_idx"),
        ),
        migrations.AddIndex(
            model_name="retentionrecord",
            index=models.Index(fields=["is_permanent"], name="records_ret_is_perm_idx"),
        ),
        # Add the generated tsvector column and its GIN index
        migrations.RunSQL(
            sql="""
                ALTER TABLE records_retentionrecord
                ADD COLUMN search_vector tsvector
                GENERATED ALWAYS AS (
                    to_tsvector('english',
                        coalesce(record_title, '') || ' ' ||
                        coalesce(record_description, '') || ' ' ||
                        coalesce(minimum_retention_period, '') || ' ' ||
                        coalesce(regulatory_citations, '')
                    )
                ) STORED;

                CREATE INDEX retention_record_search_vector_idx
                ON records_retentionrecord USING GIN(search_vector);
            """,
            reverse_sql="""
                DROP INDEX IF EXISTS retention_record_search_vector_idx;
                ALTER TABLE records_retentionrecord DROP COLUMN IF EXISTS search_vector;
            """,
        ),
        # Add pgvector HNSW index for embedding similarity search
        migrations.RunSQL(
            sql="""
                CREATE INDEX retention_record_embedding_idx
                ON records_retentionrecord
                USING hnsw (embedding vector_cosine_ops);
            """,
            reverse_sql="DROP INDEX IF EXISTS retention_record_embedding_idx;",
        ),
    ]
