"""Semantic package for web-facing integrations and schedulers.

Kept intentionally bare: the ``app.web.scheduler`` / ``app.web.email_digest``
submodules alias the underlying services, and all consumers import those
services from ``app.services.*`` directly — no re-export facade is needed here.
"""
