from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("records", "0004_documentcloud_fields"),
    ]

    operations = [
        migrations.CreateModel(
            name="NFOICChapter",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("name", models.CharField(max_length=255)),
                ("jurisdiction", models.CharField(max_length=255)),
                ("website", models.URLField(blank=True)),
                ("email", models.EmailField(blank=True, max_length=254)),
                ("phone", models.CharField(blank=True, max_length=50)),
                ("description", models.TextField(blank=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={
                "ordering": ["jurisdiction", "name"],
            },
        ),
    ]
