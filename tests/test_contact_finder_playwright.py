from __future__ import annotations

import sys
import types

# Stub dns.resolver so the heavy dependency is not required during import
dns_module = types.ModuleType("dns")
dns_module.resolver = types.SimpleNamespace()
sys.modules.setdefault("dns", dns_module)
sys.modules.setdefault("dns.resolver", dns_module.resolver)

from vexto.enrichment import contact_finder as cf


class DummyHttpClient:
    def __init__(self, responses: list[tuple[int, str]] | None = None):
        self._responses = responses or [(200, "")]
        self.calls: list[str] = []

    def get(self, url: str) -> tuple[int, str]:
        self.calls.append(url)
        if len(self.calls) <= len(self._responses):
            return self._responses[len(self.calls) - 1]
        # Return the last response if we run out
        return self._responses[-1]


def test_fetch_text_smart_uses_playwright_for_blank_html(monkeypatch):
    dummy_client = DummyHttpClient([(200, "   "), (200, "   ")])
    finder = cf.ContactFinder(use_browser="auto", pw_budget=2, http_client=dummy_client)

    calls = {}

    def fake_fetch_with_playwright(url: str, pre_html: str | None, pre_status, http_client):  # type: ignore[override]
        calls["called"] = True
        calls["url"] = url
        calls["pre_html"] = pre_html
        return "<html><body>Rendered</body></html>"

    monkeypatch.setattr(cf, "_fetch_with_playwright_sync", fake_fetch_with_playwright)

    html = finder._fetch_text_smart("https://example.com/contact")

    assert calls.get("called") is True
    assert calls.get("url") == "https://example.com/contact"
    assert calls.get("pre_html") == "   "
    assert html == "<html><body>Rendered</body></html>"

