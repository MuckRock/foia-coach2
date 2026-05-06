from django.contrib import admin
from django.urls import path

from apps.records.views import ChatCompletionsView, ModelsView

urlpatterns = [
    path("admin/", admin.site.urls),
    path("v1/chat/completions", ChatCompletionsView.as_view()),
    path("v1/models", ModelsView.as_view()),
]
