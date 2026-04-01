"""OAI-PMH ingest backend stub for future bulk backfill support."""

from __future__ import annotations

from datetime import date
from typing import Any
from xml.etree import ElementTree

import requests

from app.services.ingest.base import PaperCandidate

OAI_PMH_NAMESPACE = {"oai": "http://www.openarchives.org/OAI/2.0/"}


class OaiPmhBackend:
    def __init__(
        self,
        *,
        base_url: str = "https://export.arxiv.org/oai2",
        metadata_prefix: str = "arXiv",
    ):
        self.base_url = base_url
        self.metadata_prefix = metadata_prefix

    @property
    def name(self) -> str:
        return "oai_pmh"

    def build_list_records_params(
        self,
        *,
        start_dt: date,
        end_dt: date,
        set_spec: str | None = None,
        resumption_token: str | None = None,
    ) -> dict[str, str]:
        if resumption_token:
            return {
                "verb": "ListRecords",
                "resumptionToken": resumption_token,
            }

        params = {
            "verb": "ListRecords",
            "metadataPrefix": self.metadata_prefix,
            "from": start_dt.isoformat(),
            "until": end_dt.isoformat(),
        }
        if set_spec:
            params["set"] = set_spec
        return params

    @staticmethod
    def extract_resumption_token(payload: bytes | str) -> str | None:
        root = ElementTree.fromstring(payload)
        token = root.findtext(".//oai:resumptionToken", namespaces=OAI_PMH_NAMESPACE)
        if token is None:
            return None

        normalized = token.strip()
        return normalized or None

    def fetch(self, *, session: requests.Session | None = None, **kwargs: Any) -> list[PaperCandidate]:
        del session, kwargs
        raise NotImplementedError("OAI-PMH ingest is not implemented yet")
