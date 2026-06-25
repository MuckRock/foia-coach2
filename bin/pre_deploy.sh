#!/usr/bin/env bash
set -e
python manage.py migrate --no-input
python manage.py update_system_prompt
