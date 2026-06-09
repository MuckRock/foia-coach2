from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("records", "0003_supporting_documents"),
    ]

    operations = [
        migrations.AddField(
            model_name="sourcedocument",
            name="documentcloud_id",
            field=models.CharField(blank=True, max_length=100, null=True, unique=True),
        ),
        migrations.AddField(
            model_name="sourcedocument",
            name="documentcloud_url",
            field=models.URLField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="sourcedocument",
            name="documentcloud_updated_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="supportingdocument",
            name="documentcloud_id",
            field=models.CharField(blank=True, max_length=100, null=True, unique=True),
        ),
        migrations.AddField(
            model_name="supportingdocument",
            name="documentcloud_url",
            field=models.URLField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="supportingdocument",
            name="documentcloud_updated_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
    ]
