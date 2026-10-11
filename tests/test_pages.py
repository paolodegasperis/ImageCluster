"""Every page is reachable, and carries the controls its script expects.

A page is HTML and a script that assumes the HTML. When an element id is
renamed on one side and not the other the script throws on load and the view is
simply blank - no error the user can act on, no failing test either, because
nothing on the Python side noticed. These checks are the cheap half of that
problem: the route answers, and the ids the script reaches for by name are in
the markup.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.main import app

APP_DIR = Path(__file__).resolve().parents[1] / "app"

PAGES = {
    "/": "index.html",
    "/clip": "clip_projection.html",
    "/graph": "neighbor_graph.html",
    "/lens": "lens.html",
    "/compare": "compare.html",
    "/guide": "guide.html",
    "/guide/it": "guide.it.html",
    "/models": "models_tokens.html",
}


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.mark.parametrize("route", sorted(PAGES))
def test_every_page_answers(client, route):
    response = client.get(route)
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


@pytest.mark.parametrize("route", sorted(PAGES))
def test_every_page_stamps_its_assets_with_the_version(client, route):
    """Without the stamp a browser keeps last version's stylesheet against this
    version's markup, and the page looks broken for no visible reason."""
    body = client.get(route).text
    for url in re.findall(r'(?:href|src)="(/static/[^"]+)"', body):
        assert "?v=" in url, url


def test_the_compare_page_has_what_its_script_reaches_for():
    """compare.js calls getElementById on these the moment it loads: a missing
    one is a blank page, not an error message."""
    markup = (APP_DIR / "compare.html").read_text(encoding="utf-8")
    script = (APP_DIR / "js" / "compare.js").read_text(encoding="utf-8")
    wanted = sorted(set(re.findall(r"\$\('([A-Za-z0-9_]+)'\)", script)))
    assert wanted, "the script stopped using the $ helper; this check needs updating"
    missing = [name for name in wanted if f'id="{name}"' not in markup]
    assert not missing, f"compare.html is missing: {', '.join(missing)}"


def test_compare_is_linked_from_the_pages_that_lead_to_it():
    for name in ("index.html", "clip_projection.html", "neighbor_graph.html", "lens.html"):
        markup = (APP_DIR / name).read_text(encoding="utf-8")
        assert 'href="/compare"' in markup, name


def test_the_chain_lives_only_on_the_compare_page():
    """It was briefly a panel on the projection page as well. Two homes for one
    feature is two things to keep in step, and the projection page is already
    the busiest screen in the application."""
    projection = (APP_DIR / "clip_projection.html").read_text(encoding="utf-8")
    assert 'data-analysis-panel="chain"' not in projection
    assert 'data-analysis-tool="chain"' not in projection
    panels = (APP_DIR / "js" / "analysis_panels.js").read_text(encoding="utf-8")
    assert "runChain" not in panels
    assert "chainResults" not in panels
