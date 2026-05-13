from .base import *  # noqa
from .base import env

DEBUG = True
INSTALLED_APPS = INSTALLED_APPS + ["django_extensions"]  # noqa: F405
SECRET_KEY = env(
    "DJANGO_SECRET_KEY",
    default="local-dev-secret-key-change-in-production",
)
ALLOWED_HOSTS = ["*"]

# Read .env files in local dev
import os as _os
if _os.path.exists(_os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(__file__))), ".envs", ".local", ".django")):
    import environ as _environ
    _environ.Env.read_env(_os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(__file__))), ".envs", ".local", ".django"))
