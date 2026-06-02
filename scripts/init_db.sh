#!/usr/bin/env sh
set -eu

psql "$DATABASE_URL" -f db/postgres_init.sql
