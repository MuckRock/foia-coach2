from django.db import migrations


def seed_prompt(apps, schema_editor):
    SystemPrompt = apps.get_model("records", "SystemPrompt")
    SystemPrompt.objects.create(
        name="Initial FOIA Coach Prompt",
        content=(
            "You are a FOIA Coach assistant specializing in Colorado public records retention "
            "schedules. Answer questions about how long specific types of records must be kept, "
            "citing the relevant retention schedule entries provided to you. Be precise about "
            "retention periods and note any special conditions or exceptions. If the retrieved "
            "context does not contain enough information to answer definitively, say so clearly."
        ),
        is_active=True,
    )


def unseed_prompt(apps, schema_editor):
    SystemPrompt = apps.get_model("records", "SystemPrompt")
    SystemPrompt.objects.filter(name="Initial FOIA Coach Prompt").delete()


class Migration(migrations.Migration):

    dependencies = [
        ("records", "0001_initial"),
    ]

    operations = [
        migrations.RunPython(seed_prompt, unseed_prompt),
    ]
