import factory

from apps.records.models import RetentionRecord, SourceDocument, SystemPrompt


class SystemPromptFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = SystemPrompt

    name = factory.Sequence(lambda n: f"Prompt {n}")
    content = factory.Faker("paragraph")
    is_active = False


class SourceDocumentFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = SourceDocument

    filename = factory.Faker("file_name", extension="pdf")
    document_title = factory.Sequence(lambda n: f"SCHEDULE NO. {n} - TEST RECORDS (Colorado Special Districts)")
    jurisdiction = "Colorado"
    entity_type = "Special Districts"
    schedule_number = factory.Sequence(lambda n: str(n))


class RetentionRecordFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = RetentionRecord

    source_document = factory.SubFactory(SourceDocumentFactory)
    record_number = factory.Sequence(lambda n: str(n))
    record_title = factory.Faker("sentence", nb_words=4)
    record_description = factory.Faker("paragraph")
    minimum_retention_period = "7 years"
    custodian_requirement = ""
    regulatory_citations = ""
