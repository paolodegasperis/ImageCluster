from __future__ import annotations

import csv
import html
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from .runtime_dirs import HTML_EXPORTS_DIR, OUTPUT_DIR, PROJECTIONS_DIR, ROOT, SESSIONS_DIR
from .version import APP_VERSION


def projection_catalog() -> list[dict[str, Any]]:
    items = []
    for path in sorted(PROJECTIONS_DIR.glob("*.tsv"), key=lambda p: p.stat().st_mtime, reverse=True):
        rows = read_projection_rows(path)
        meta = projection_metadata(path, rows)
        meta.update({
            "name": path.name,
            "relative_path": path.relative_to(ROOT).as_posix(),
            "size": path.stat().st_size,
            "modified_at": path.stat().st_mtime,
        })
        items.append(meta)
    return items


def cluster_report(projection: str, representatives: int = 5) -> dict[str, Any]:
    path = safe_projection_path(projection)
    rows = read_projection_rows(path)
    meta = projection_metadata(path, rows)
    clustered = [row for row in rows if _has_number(row.get("cluster"))]
    if not rows:
        return {"projection": path.relative_to(ROOT).as_posix(), "total_images": 0, "clustering_available": False, "message": "Projection file is empty."}
    if not clustered:
        return {
            **meta,
            "projection": path.relative_to(ROOT).as_posix(),
            "total_images": len(rows),
            "clustering_available": False,
            "message": "This projection does not contain cluster data. Generate a projection with clustering enabled to create a cluster report.",
            "clusters": [],
        }

    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in clustered:
        groups[int(float(row["cluster"]))].append(row)

    summaries = []
    for cluster_id in sorted(groups):
        group = groups[cluster_id]
        reps = representative_rows(group, representatives)
        folders = Counter(_subfolder(row.get("relative_path", "")) for row in group)
        summaries.append({
            "cluster": cluster_id,
            "count": len(group),
            "percentage": round(len(group) / len(rows) * 100, 2),
            "dominant_subfolder": folders.most_common(1)[0][0] if folders else "",
            "filename_examples": [row.get("filename", "") for row in group[:representatives]],
            "representative_images": [
                {
                    "filename": row.get("filename", ""),
                    "relative_path": row.get("relative_path", ""),
                    "x": _float_or_none(row.get("x")),
                    "y": _float_or_none(row.get("y")),
                }
                for row in reps
            ],
        })

    return {
        **meta,
        "projection": path.relative_to(ROOT).as_posix(),
        "total_images": len(rows),
        "clustering_available": True,
        "cluster_k": _first_number(clustered, "cluster_k") or len(groups),
        "cluster_score": _first_number(clustered, "cluster_score"),
        "cluster_method": clustered[0].get("cluster_method") or "kmeans",
        "clusters": summaries,
    }


def compare_projections(projections: list[str], top_k: int = 5) -> dict[str, Any]:
    paths = [safe_projection_path(item) for item in projections]
    datasets = []
    for path in paths:
        rows = read_projection_rows(path)
        keyed = {_row_key(row): row for row in rows if _row_key(row)}
        datasets.append({"path": path, "rows": rows, "keyed": keyed, "metadata": projection_metadata(path, rows)})

    key_sets = [set(item["keyed"]) for item in datasets]
    common = set.intersection(*key_sets) if key_sets else set()
    missing = {
        item["path"].relative_to(ROOT).as_posix(): sorted(set.union(*key_sets) - set(item["keyed"]))[:200] if key_sets else []
        for item in datasets
    }
    comparisons = []
    if len(datasets) >= 2:
        baseline = datasets[0]
        for other in datasets[1:]:
            comparisons.append(_pairwise_projection_comparison(baseline, other, common, top_k))

    return {
        "projection_count": len(datasets),
        "common_images_count": len(common),
        "missing_images_by_projection": missing,
        "projections": [
            {
                **item["metadata"],
                "relative_path": item["path"].relative_to(ROOT).as_posix(),
                "image_count": len(item["rows"]),
                "cluster_count": len({row.get("cluster") for row in item["rows"] if _has_number(row.get("cluster"))}),
            }
            for item in datasets
        ],
        "pairwise": comparisons,
        "warning": "Projection coordinates from different runs are not directly axis-aligned; comparison uses shared images and neighborhood structure rather than raw x/y equality.",
    }


def save_session(payload: dict[str, Any], session_id: str | None = None) -> dict[str, Any]:
    session_id = session_id or f"session-{time.strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
    now = _iso_now()
    existing = load_session(session_id, missing_ok=True) or {}
    data = {
        **existing,
        **payload,
        "session_id": session_id,
        "app_version": APP_VERSION,
        "created_at": existing.get("created_at", now),
        "updated_at": now,
    }
    path = SESSIONS_DIR / f"{session_id}.json"
    _atomic_write_json(path, data)
    return {**data, "relative_path": path.relative_to(ROOT).as_posix(), "warnings": validate_session_references(data)}


def list_sessions() -> list[dict[str, Any]]:
    sessions = []
    for path in sorted(SESSIONS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        sessions.append({
            "session_id": data.get("session_id", path.stem),
            "name": data.get("name") or data.get("session_id", path.stem),
            "created_at": data.get("created_at"),
            "updated_at": data.get("updated_at"),
            "active_projection": data.get("active_projection"),
            "relative_path": path.relative_to(ROOT).as_posix(),
        })
    return sessions


def load_session(session_id: str, missing_ok: bool = False) -> dict[str, Any] | None:
    path = SESSIONS_DIR / f"{_safe_name(session_id)}.json"
    if not path.exists():
        if missing_ok:
            return None
        raise FileNotFoundError("Session not found")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["warnings"] = validate_session_references(data)
    return data


def delete_session(session_id: str) -> bool:
    path = SESSIONS_DIR / f"{_safe_name(session_id)}.json"
    if not path.exists():
        return False
    path.unlink()
    return True


def export_html_package(payload: dict[str, Any]) -> dict[str, Any]:
    projection = payload.get("projection") or payload.get("active_projection")
    if not projection:
        raise ValueError("Projection path is required.")
    path = safe_projection_path(projection)
    rows = read_projection_rows(path)
    meta = projection_metadata(path, rows)
    export_id = f"html-{time.strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
    export_dir = HTML_EXPORTS_DIR / export_id
    data_dir = export_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    projection_json = {"metadata": meta, "rows": rows}
    session_state = payload.get("session") or {}
    _atomic_write_json(data_dir / "projection.json", projection_json)
    _atomic_write_json(data_dir / "session.json", session_state)
    html_text = render_export_html(meta, rows, session_state)
    (export_dir / "index.html").write_text(html_text, encoding="utf-8")
    return {
        "export_id": export_id,
        "index_path": (export_dir / "index.html").relative_to(ROOT).as_posix(),
        "included_assets": ["data/projection.json", "data/session.json"],
        "warnings": ["Package mode references original image paths such as ../img/...; keep the export folder near the project if thumbnails are needed."],
    }


def read_projection_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def safe_projection_path(value: str) -> Path:
    requested = (ROOT / value).resolve()
    root = PROJECTIONS_DIR.resolve()
    if requested == root or root in requested.parents:
        if not requested.exists():
            raise FileNotFoundError("Projection file not found")
        return requested
    raise ValueError("Projection path must be inside output/projections/.")


def projection_metadata(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    sidecar = path.with_suffix(".json")
    meta = {}
    if sidecar.exists():
        try:
            meta.update(json.loads(sidecar.read_text(encoding="utf-8")))
        except Exception:
            pass
    first = rows[0] if rows else {}
    return {
        "model_key": meta.get("model_key") or first.get("model_key"),
        "model_family": meta.get("model_family") or first.get("model_family"),
        "embedding_model": meta.get("model_label") or first.get("embedding_model"),
        "model_id": meta.get("model_id") or first.get("model_id"),
        "pretrained": meta.get("pretrained") or first.get("pretrained"),
        "provider": meta.get("provider") or first.get("provider"),
        "reducer": meta.get("reducer") or first.get("reducer"),
        "image_count": meta.get("image_count") or len(rows),
        "created_at": meta.get("created_at") or time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(path.stat().st_mtime)),
        "clustering": meta.get("clustering") or {
            "applied": any(_has_number(row.get("cluster")) for row in rows),
            "k": _first_number(rows, "cluster_k"),
            "score": _first_number(rows, "cluster_score"),
            "method": next((row.get("cluster_method") for row in rows if row.get("cluster_method")), None),
        },
    }


def representative_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    numeric = [row for row in rows if _has_number(row.get("x")) and _has_number(row.get("y"))]
    if not numeric:
        return rows[:limit]
    cx = sum(float(row["x"]) for row in numeric) / len(numeric)
    cy = sum(float(row["y"]) for row in numeric) / len(numeric)
    return sorted(numeric, key=lambda row: (float(row["x"]) - cx) ** 2 + (float(row["y"]) - cy) ** 2)[:limit]


def render_export_html(meta: dict[str, Any], rows: list[dict[str, Any]], session: dict[str, Any]) -> str:
    data = json.dumps({"metadata": meta, "rows": rows, "session": session}, ensure_ascii=False)
    title = html.escape(session.get("name") or "ClusterIMG-V-52 export")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #151515; color: #f2f2f2; }}
    header {{ padding: 18px 22px; border-bottom: 1px solid #333; }}
    canvas {{ display: block; width: 100vw; height: calc(100vh - 112px); background: #555; }}
    .meta {{ color: #bbb; font-size: 13px; }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <div class="meta">ClusterIMG-V-52 {APP_VERSION} · {html.escape(str(meta.get("embedding_model") or meta.get("model_key") or ""))} · {html.escape(str(meta.get("reducer") or ""))} · {len(rows)} images</div>
  </header>
  <canvas id="plot"></canvas>
  <script>
    const payload = {data};
    const canvas = document.getElementById('plot');
    const ctx = canvas.getContext('2d');
    function resize() {{ canvas.width = innerWidth; canvas.height = Math.max(320, innerHeight - 112); draw(); }}
    function draw() {{
      const rows = payload.rows.filter(r => Number.isFinite(Number(r.x)) && Number.isFinite(Number(r.y)));
      ctx.fillStyle = '#555'; ctx.fillRect(0, 0, canvas.width, canvas.height);
      if (!rows.length) return;
      const xs = rows.map(r => Number(r.x)); const ys = rows.map(r => Number(r.y));
      const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
      const colors = ['#ff6b6b','#4ecdc4','#ffe66d','#a29bfe','#fd79a8','#74b9ff','#55efc4','#fab1a0'];
      for (const r of rows) {{
        const x = 50 + ((Number(r.x) - minX) / ((maxX - minX) || 1)) * (canvas.width - 100);
        const y = canvas.height - 50 - ((Number(r.y) - minY) / ((maxY - minY) || 1)) * (canvas.height - 100);
        const c = Number(r.cluster);
        ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fillStyle = Number.isFinite(c) ? colors[Math.max(0, c - 1) % colors.length] : '#f2f2f2';
        ctx.fill();
      }}
    }}
    addEventListener('resize', resize); resize();
  </script>
</body>
</html>"""


def validate_session_references(data: dict[str, Any]) -> list[str]:
    warnings = []
    projection = data.get("active_projection")
    if projection and not (ROOT / projection).exists():
        warnings.append(f"Referenced projection is missing: {projection}")
    search = data.get("semantic_search", {}).get("results_path") if isinstance(data.get("semantic_search"), dict) else None
    if search and not (ROOT / search).exists():
        warnings.append(f"Referenced search result file is missing: {search}")
    return warnings


def _pairwise_projection_comparison(a: dict[str, Any], b: dict[str, Any], common: set[str], top_k: int) -> dict[str, Any]:
    overlaps = []
    stable = []
    for key in sorted(common):
        neigh_a = _nearest_keys(key, a["keyed"], top_k)
        neigh_b = _nearest_keys(key, b["keyed"], top_k)
        union = neigh_a | neigh_b
        score = len(neigh_a & neigh_b) / len(union) if union else 0.0
        overlaps.append(score)
        stable.append({"relative_path": key, "neighbor_overlap": round(score, 4)})
    stable.sort(key=lambda item: item["neighbor_overlap"], reverse=True)
    return {
        "a": a["path"].relative_to(ROOT).as_posix(),
        "b": b["path"].relative_to(ROOT).as_posix(),
        "nearest_neighbor_overlap": {
            "top_k": top_k,
            "mean": round(sum(overlaps) / len(overlaps), 4) if overlaps else None,
            "median": round(sorted(overlaps)[len(overlaps) // 2], 4) if overlaps else None,
        },
        "most_stable": stable[:10],
        "least_stable": list(reversed(stable[-10:])),
        "cluster_membership_difference": _cluster_difference(a["keyed"], b["keyed"], common),
    }


def _nearest_keys(key: str, keyed: dict[str, dict[str, Any]], top_k: int) -> set[str]:
    row = keyed.get(key)
    if not row or not _has_number(row.get("x")) or not _has_number(row.get("y")):
        return set()
    x = float(row["x"])
    y = float(row["y"])
    distances = []
    for other_key, other in keyed.items():
        if other_key == key or not _has_number(other.get("x")) or not _has_number(other.get("y")):
            continue
        distances.append((math.hypot(float(other["x"]) - x, float(other["y"]) - y), other_key))
    return {item[1] for item in sorted(distances)[:top_k]}


def _cluster_difference(a: dict[str, dict[str, Any]], b: dict[str, dict[str, Any]], common: set[str]) -> dict[str, Any]:
    comparable = [key for key in common if _has_number(a[key].get("cluster")) and _has_number(b[key].get("cluster"))]
    changed = [key for key in comparable if int(float(a[key]["cluster"])) != int(float(b[key]["cluster"]))]
    return {"comparable": len(comparable), "changed": len(changed), "changed_percentage": round(len(changed) / len(comparable) * 100, 2) if comparable else None}


def _row_key(row: dict[str, Any]) -> str:
    return row.get("relative_path") or row.get("filename") or ""


def _subfolder(relative_path: str) -> str:
    parts = relative_path.replace("\\", "/").split("/")
    return "/".join(parts[:-1]) if len(parts) > 1 else ""


def _first_number(rows: list[dict[str, Any]], key: str) -> float | int | None:
    for row in rows:
        value = _float_or_none(row.get(key))
        if value is not None:
            return int(value) if float(value).is_integer() else value
    return None


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _has_number(value: Any) -> bool:
    return _float_or_none(value) is not None


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in value)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
