"""
Base settings for FOIA Coach API.
"""
from pathlib import Path

import environ

ROOT_DIR = Path(__file__).resolve(strict=True).parent.parent.parent
APPS_DIR = ROOT_DIR / "apps"

env = environ.Env()

READ_DOT_ENV_FILE = env.bool("DJANGO_READ_DOT_ENV_FILE", default=False)
if READ_DOT_ENV_FILE:
    env.read_env(str(ROOT_DIR / ".env"))

# GENERAL
DEBUG = env.bool("DJANGO_DEBUG", False)
TIME_ZONE = "America/Denver"
LANGUAGE_CODE = "en-us"
USE_I18N = False
USE_TZ = True
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# APPS
DJANGO_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
]
LOCAL_APPS = [
    "apps.records",
]
INSTALLED_APPS = DJANGO_APPS + LOCAL_APPS

# MIDDLEWARE
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

# URLS
ROOT_URLCONF = "config.urls"

# TEMPLATES
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

# WSGI / ASGI
WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"

# DATABASES
DATABASES = {
    "default": env.db("DATABASE_URL", default="postgres://postgres:postgres@localhost:5432/foia_coach_api"),
}
DATABASES["default"]["CONN_MAX_AGE"] = env.int("CONN_MAX_AGE", default=60)

# AUTHENTICATION
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

# LOGGING
import os as _os
_log_file = env("DJANGO_LOG_FILE", default="")
if _log_file and not _os.path.isdir(_os.path.dirname(_log_file) or "."):
    _log_file = ""  # disable if parent directory doesn't exist

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {"format": "%(levelname)s %(name)s: %(message)s"},
        "timestamped": {"format": "%(asctime)s %(levelname)s %(name)s: %(message)s"},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
        },
        **(
            {
                "file": {
                    "class": "logging.FileHandler",
                    "filename": _log_file,
                    "formatter": "timestamped",
                    "delay": True,
                }
            }
            if _log_file else {}
        ),
    },
    "loggers": {
        "apps.records": {
            "handlers": ["console", "file"] if _log_file else ["console"],
            "level": "INFO",
            "propagate": False,
        },
        "django": {
            "handlers": ["console"],
            "level": "INFO",
        },
        "django.request": {
            "handlers": ["console"],
            "level": "DEBUG",
            "propagate": False,
        },
    },
}

# STATIC
STATIC_URL = "/static/"
STATIC_ROOT = str(ROOT_DIR / "staticfiles")

# SECRET KEY
SECRET_KEY = env("DJANGO_SECRET_KEY", default="!!!SET DJANGO_SECRET_KEY!!!")

# ALLOWED HOSTS
ALLOWED_HOSTS = env.list("DJANGO_ALLOWED_HOSTS", default=["*"])

# FOIA Coach settings
DOCUMENTCLOUD_USERNAME = env("DOCUMENTCLOUD_USERNAME", default="")
DOCUMENTCLOUD_PASSWORD = env("DOCUMENTCLOUD_PASSWORD", default="")
LLM_MODEL = env("LLM_MODEL", default="gpt-4o")
LLM_BASE_URL = env("LLM_BASE_URL", default="")  # leave empty for OpenAI; set to https://api.anthropic.com/v1 for Claude
_llm_api_key_override = env("LLM_API_KEY", default="")
OPENAI_API_KEY = env("OPENAI_API_KEY", default="")
LLM_API_KEY = _llm_api_key_override if _llm_api_key_override else OPENAI_API_KEY
QUERY_REWRITE_MODEL = env("QUERY_REWRITE_MODEL", default="gpt-4o-mini")
EXTRACTION_MODEL = env("EXTRACTION_MODEL", default="gpt-5.2")  # used for import commands (always OpenAI)
EMBEDDING_MODEL = env("EMBEDDING_MODEL", default="text-embedding-3-small")
LLM_TEMPERATURE = env.float("LLM_TEMPERATURE", default=0.3)
LLM_TEMPERATURE_ENABLED = env.bool("LLM_TEMPERATURE_ENABLED", default=True)  # set False for models that deprecated temperature (gpt-5.5, claude)
