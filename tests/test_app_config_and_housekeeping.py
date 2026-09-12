"""config.json is now load-bearing, and the output folder has ceilings."""

from __future__ import annotations

import json

import pytest

from backend import app_config, housekeeping
from backend.app_config import DEFAULTS, _deep_merge, get
from backend.schemas import ClipProjectionRequest
from backend.version import __app_name__, __version__


def test_defaults_are_returned_for_unknown_paths():
    assert get("nope.not.here") is None
    assert get("nope.not.here", "fallback") == "fallback"


def test_dotted_lookup_reaches_nested_values():
    assert isinstance(get("semantic_search.default_top_k"), int)
    assert isinstance(get("limits.max_jobs_in_memory"), int)


def test_deep_merge_keeps_unspecified_defaults():
    merged = _deep_merge(DEFAULTS, {"semantic_search": {"default_top_k": 7}})
    assert merged["semantic_search"]["default_top_k"] == 7
    assert merged["semantic_search"]["enabled"] is True
    assert merged["limits"]["max_jobs_in_memory"] == DEFAULTS["limits"]["max_jobs_in_memory"]


def test_malformed_config_falls_back_to_defaults(tmp_path, monkeypatch):
    broken = tmp_path / "config.json"
    broken.write_text("{ not json", encoding="utf-8")
    monkeypatch.setattr(app_config, "CONFIG_PATH", broken)
    restored = app_config.reload_config()
    try:
        assert restored["port"] == DEFAULTS["port"]
    finally:
        monkeypatch.undo()
        app_config.reload_config()


def test_shipped_config_only_contains_keys_the_app_reads():
    raw = json.loads(app_config.CONFIG_PATH.read_text(encoding="utf-8"))
    unknown = set(raw) - set(DEFAULTS)
    assert not unknown, f"config.json declares keys nothing reads: {sorted(unknown)}"


def test_request_defaults_come_from_config():
    request = ClipProjectionRequest()
    assert request.reducer == get("default_reducer")
    assert request.batch_size == get("batch_size")
    assert request.cluster_k == get("optional_clustering.fixed_k")
    assert request.cluster_min_k == get("optional_clustering.auto_min_k")
    assert request.cluster_max_k == get("optional_clustering.auto_max_k")
    assert request.cluster_enabled == get("optional_clustering.enabled_by_default")


def test_version_is_a_single_source_of_truth():
    assert __version__.count(".") == 2
    assert __app_name__ == "ImageCluster"


def test_log_pruning_keeps_the_newest_files(tmp_path, monkeypatch):
    monkeypatch.setattr(housekeeping, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(housekeeping, "config_get", lambda path, fallback=None: 3 if path == "limits.max_log_files" else fallback)
    for index in range(10):
        target = tmp_path / f"job_{index}.log"
        target.write_text("x", encoding="utf-8")
        import os
        os.utime(target, (1_600_000_000 + index, 1_600_000_000 + index))
    removed = housekeeping.prune_log_dir()
    assert removed == 7
    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert remaining == ["job_7.log", "job_8.log", "job_9.log"]


def test_log_pruning_is_a_no_op_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(housekeeping, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(housekeeping, "config_get", lambda path, fallback=None: 0)
    (tmp_path / "a.log").write_text("x", encoding="utf-8")
    assert housekeeping.prune_log_dir() == 0
    assert (tmp_path / "a.log").exists()


def test_thumb_cache_pruning_trims_to_the_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(housekeeping, "THUMB_CACHE_DIR", tmp_path)
    # 1 KB budget, ten 512-byte thumbnails: the two newest must survive.
    monkeypatch.setattr(
        housekeeping,
        "config_get",
        lambda path, fallback=None: (1 / 1024) if path == "limits.thumb_cache_max_mb" else fallback,
    )
    import os

    for index in range(10):
        target = tmp_path / f"{index:032x}.jpg"
        target.write_bytes(b"0" * 512)
        os.utime(target, (1_600_000_000 + index, 1_600_000_000 + index))
    housekeeping.prune_thumb_cache(force=True)
    survivors = sorted(p.name for p in tmp_path.glob("*.jpg"))
    assert len(survivors) == 2
    assert survivors == [f"{8:032x}.jpg", f"{9:032x}.jpg"]


def test_thumb_cache_pruning_is_throttled_between_sweeps(tmp_path, monkeypatch):
    monkeypatch.setattr(housekeeping, "THUMB_CACHE_DIR", tmp_path)
    monkeypatch.setattr(housekeeping, "_thumb_writes_since_sweep", 0)
    # Without force, the first call must not walk the directory at all.
    assert housekeeping.prune_thumb_cache() == 0


# --------------------------------------------------------------------------
# A hand-edited config.json must never be able to break the app.
#
# Before this was fixed, a section replaced by a value of the wrong type
# propagated into the running configuration, and dict(CONFIG["limits"]) raised —
# so /api/config answered 500 and the dashboard failed to load.


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    """Point app_config at a throwaway config.json and restore the real one after."""
    target = tmp_path / "config.json"
    monkeypatch.setattr(app_config, "CONFIG_PATH", target)
    yield target
    monkeypatch.undo()
    app_config.reload_config()


def _write(config_file, payload):
    config_file.write_text(json.dumps(payload), encoding="utf-8")
    app_config.reload_config()


ACCESSORS = [
    app_config.limits,
    app_config.clustering_defaults,
    app_config.search_defaults,
    app_config.view_defaults,
    app_config.debug_flags,
]


@pytest.mark.parametrize("payload", [
    {"limits": "nonsense"},
    {"optional_clustering": [1, 2, 3]},
    {"views": None},
    {"debug": 42},
    {"semantic_search": True},
])
def test_a_section_of_the_wrong_type_falls_back_to_defaults(config_file, payload):
    _write(config_file, payload)
    for accessor in ACCESSORS:
        assert isinstance(accessor(), dict)
    section = next(iter(payload))
    assert app_config.CONFIG[section] == DEFAULTS[section]


@pytest.mark.parametrize("payload", [
    {"port": {"oops": 1}},
    {"port": "8765"},
    {"block_start_if_dependencies_missing": "yes"},
    {"default_reducer": 7},
    {"batch_size": None},
])
def test_a_scalar_of_the_wrong_type_keeps_its_default(config_file, payload):
    _write(config_file, payload)
    key = next(iter(payload))
    assert app_config.CONFIG[key] == DEFAULTS[key]


def test_a_nested_value_of_the_wrong_type_keeps_its_default(config_file):
    _write(config_file, {"semantic_search": {"default_top_k": "many", "enabled": False}})
    # The bad key is discarded; the good one alongside it still applies.
    assert get("semantic_search.default_top_k") == DEFAULTS["semantic_search"]["default_top_k"]
    assert get("semantic_search.enabled") is False


def test_a_nullable_setting_accepts_null_and_a_number(config_file):
    _write(config_file, {"semantic_search": {"default_threshold": None}})
    assert get("semantic_search.default_threshold") is None
    _write(config_file, {"semantic_search": {"default_threshold": 0.25}})
    assert get("semantic_search.default_threshold") == 0.25


def test_a_bool_is_not_accepted_where_a_number_is_expected(config_file):
    _write(config_file, {"batch_size": True})
    assert get("batch_size") == DEFAULTS["batch_size"]


def test_a_valid_override_still_applies(config_file):
    _write(config_file, {"port": 9100, "optional_clustering": {"fixed_k": 9}})
    assert get("port") == 9100
    assert get("optional_clustering.fixed_k") == 9
    # Untouched siblings keep their defaults.
    assert get("optional_clustering.auto_max_k") == DEFAULTS["optional_clustering"]["auto_max_k"]


def test_unknown_keys_are_preserved_rather_than_trimmed(config_file):
    _write(config_file, {"my_own_note": "keep me"})
    assert get("my_own_note") == "keep me"


def test_request_schemas_survive_a_malformed_config(config_file):
    import importlib

    import backend.schemas as project_schemas
    import backend.search.schemas as search_schemas

    _write(config_file, {"batch_size": "lots", "semantic_search": {"default_top_k": "many"}})
    importlib.reload(project_schemas)
    importlib.reload(search_schemas)
    try:
        assert project_schemas.ClipProjectionRequest().batch_size == DEFAULTS["batch_size"]
        assert search_schemas.TextSearchRequest(query="q", model_key="m").top_k == DEFAULTS["semantic_search"]["default_top_k"]
    finally:
        app_config.reload_config()
        importlib.reload(project_schemas)
        importlib.reload(search_schemas)


def test_thumbnail_opacity_bounds_match_between_config_and_ui():
    """The floor lives in three places; they must not drift apart."""
    from pathlib import Path

    root = Path(app_config.CONFIG_PATH).parent
    js = (root / "app" / "js" / "clip_projection.js").read_text(encoding="utf-8")
    assert "const MIN_THUMB_OPACITY = 0.05;" in js

    for page, control in [
        ("clip_projection.html", "thumbOpacity"),
        ("neighbor_graph.html", "graphOpacity"),
    ]:
        html = (root / "app" / page).read_text(encoding="utf-8")
        marker = f'id="{control}"'
        start = html.index(marker)
        tag = html[html.rindex("<input", 0, start): html.index(">", start) + 1]
        assert 'min="5"' in tag, f"{page}: {control} floor is not 5%"
        assert 'max="100"' in tag, f"{page}: {control} ceiling is not 100%"


def test_every_graph_control_carries_a_tooltip():
    """A control the user cannot interpret is a control they will not use."""
    import re
    from pathlib import Path

    root = Path(app_config.CONFIG_PATH).parent
    html = (root / "app" / "neighbor_graph.html").read_text(encoding="utf-8")

    missing = []
    for match in re.finditer(r"<(input|select|button|a)\b[^>]*\bid=\"([^\"]+)\"[^>]*>", html):
        # The title may sit on the control or on the label/field wrapping it.
        context = html[max(0, match.start() - 420): match.end()]
        if "title=" not in context:
            missing.append(match.group(2))
    assert not missing, f"controls without a tooltip: {missing}"


def test_the_graph_page_ships_the_explanation_panel():
    from pathlib import Path

    root = Path(app_config.CONFIG_PATH).parent
    html = (root / "app" / "neighbor_graph.html").read_text(encoding="utf-8")
    assert 'id="graphHelpModal"' in html
    # Reachable from the topbar, from the sidebar sentence and from the empty state.
    assert html.count("data-open-help") >= 3
    for heading in ["What it answers", "How it is built", "Reading the picture"]:
        assert heading in html, heading


def test_pages_stamp_their_assets_with_the_app_version():
    """A cached stylesheet from a previous version must not survive an update."""
    import re

    from fastapi.testclient import TestClient

    import backend.main as main

    client = TestClient(main.app)
    for page in ["/", "/clip", "/graph", "/models"]:
        html = client.get(page).text
        assets = re.findall(r'(?:href|src)="(/static/[^"]+)"', html)
        assert assets, f"{page} references no static assets"
        unstamped = [url for url in assets if f"?v={__version__}" not in url]
        assert not unstamped, f"{page}: {unstamped}"


def test_asset_stamping_leaves_other_urls_alone():
    from backend.main import _ASSET_URL

    html = '<a href="/api/status"></a><img src="https://cdn/x.png"><link href="/static/css/a.css">'
    stamped = _ASSET_URL.sub("MATCH", html)
    assert "/api/status" in stamped
    assert "https://cdn/x.png" in stamped
    assert "MATCH" in stamped


def test_no_frontend_code_mistakes_a_missing_value_for_zero():
    """`Number(null)` is 0, and 0 is finite.

    Testing a possibly-absent field with `Number.isFinite(Number(x))` therefore
    reads "this image has no cluster" as "this image is in cluster 0". That is
    what veiled every thumbnail on an unclustered projection: the gallery invented
    a group, auto-selected it, and dimmed everything outside it to 30%. The guard
    helpers are the only place the bare expression may appear.
    """
    from pathlib import Path

    root = Path(app_config.CONFIG_PATH).parent
    offenders = []
    for script in sorted((root / "app" / "js").glob("*.js")):
        for number, line in enumerate(script.read_text(encoding="utf-8").splitlines(), start=1):
            if "Number.isFinite(Number(" not in line:
                continue
            # The helper itself is the guard; everything else must go through it.
            if line.strip() == "return Number.isFinite(Number(value));":
                continue
            offenders.append(f"{script.name}:{number}: {line.strip()}")
    assert not offenders, "use hasNumber() instead:\n" + "\n".join(offenders)


def test_the_guard_helper_is_defined_wherever_it_is_used():
    from pathlib import Path

    root = Path(app_config.CONFIG_PATH).parent
    scripts = {path.name: path.read_text(encoding="utf-8") for path in (root / "app" / "js").glob("*.js")}

    # clip_projection.js defines it for itself and for analysis_panels.js, which
    # already depends on its globals; neighbor_graph.js is a standalone page.
    for name in ["clip_projection.js", "neighbor_graph.js"]:
        assert "function hasNumber(value)" in scripts[name], name
        assert "value === null" in scripts[name], name


def _graph_script() -> str:
    from pathlib import Path

    root = Path(app_config.CONFIG_PATH).parent
    return (root / "app" / "js" / "neighbor_graph.js").read_text(encoding="utf-8")


def test_a_node_selection_and_a_community_focus_cannot_both_dim_the_graph():
    """Regression: choosing a community appeared to do nothing.

    The graph dims by node selection and by community focus, and both used to
    stay active at once. Because a selected node is almost never inside the
    community you just picked, the two filters intersected to nearly nothing -
    measured on a real graph, 1471 vivid pixels fell to 332 where a 27-image
    community should have lit up thousands. The view looked frozen on the
    previous selection.

    The fix is that the most recent choice wins, so each handler clears the
    other. Asserted on the source because there is no JS test runner here.
    """
    js = _graph_script()

    select = js.split("function selectNode(", 1)[1].split("\n}", 1)[0]
    assert "state.highlightCommunity = null" in select, (
        "selectNode must drop the community focus"
    )

    # The community handler is the one that reads the data-community attribute.
    handler = js.split("querySelectorAll('[data-community]')", 1)[1].split("});", 1)[0]
    assert "state.selected = null" in handler, (
        "focusing a community must drop the node selection"
    )


def test_a_community_focus_also_dims_the_edges():
    """Nodes alone were community-aware; the links kept the old selection lit."""
    js = _graph_script()
    edges = js.split("function drawEdges(", 1)[1].split("\n}", 1)[0]
    assert "highlightCommunity" in edges, (
        "drawEdges must respect the community focus, not only the selection"
    )
