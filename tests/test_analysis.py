from pathlib import Path

from backend.analysis import cluster_report, compare_projections, projection_catalog, save_session, load_session


def write_projection(path: Path, model_key: str, reducer: str, clustered: bool = True) -> None:
    fields = ["filename", "relative_path", "x", "y", "model_key", "model_family", "embedding_model", "reducer"]
    if clustered:
        fields += ["cluster", "cluster_k", "cluster_score", "cluster_method"]
    rows = [
        ["a.jpg", "img/a.jpg", "0", "0", model_key, "OpenCLIP", model_key, reducer, "1", "2", "0.5", "kmeans"],
        ["b.jpg", "img/b.jpg", "1", "0", model_key, "OpenCLIP", model_key, reducer, "1", "2", "0.5", "kmeans"],
        ["c.jpg", "img/c.jpg", "0", "1", model_key, "OpenCLIP", model_key, reducer, "2", "2", "0.5", "kmeans"],
    ]
    if not clustered:
        rows = [row[:8] for row in rows]
    path.write_text("\t".join(fields) + "\n" + "\n".join("\t".join(row) for row in rows), encoding="utf-8")


def test_cluster_report_from_projection(tmp_path, monkeypatch):
    import backend.analysis as analysis

    monkeypatch.setattr(analysis, "ROOT", tmp_path)
    projection_dir = tmp_path / "output" / "projections"
    projection_dir.mkdir(parents=True)
    monkeypatch.setattr(analysis, "PROJECTIONS_DIR", projection_dir)
    projection = projection_dir / "a.tsv"
    write_projection(projection, "model_a", "umap", clustered=True)

    report = cluster_report("output/projections/a.tsv")

    assert report["clustering_available"] is True
    assert report["cluster_k"] == 2
    assert len(report["clusters"]) == 2


def test_projection_comparison(tmp_path, monkeypatch):
    import backend.analysis as analysis

    monkeypatch.setattr(analysis, "ROOT", tmp_path)
    projection_dir = tmp_path / "output" / "projections"
    projection_dir.mkdir(parents=True)
    monkeypatch.setattr(analysis, "PROJECTIONS_DIR", projection_dir)
    write_projection(projection_dir / "a.tsv", "model_a", "umap", clustered=True)
    write_projection(projection_dir / "b.tsv", "model_b", "tsne", clustered=True)

    result = compare_projections(["output/projections/a.tsv", "output/projections/b.tsv"])

    assert result["common_images_count"] == 3
    assert result["projection_count"] == 2
    assert result["pairwise"]


def test_session_save_load(tmp_path, monkeypatch):
    import backend.analysis as analysis

    monkeypatch.setattr(analysis, "ROOT", tmp_path)
    sessions_dir = tmp_path / "output" / "sessions"
    monkeypatch.setattr(analysis, "SESSIONS_DIR", sessions_dir)

    saved = save_session({"name": "Test", "filters": {"filename": "a"}})
    loaded = load_session(saved["session_id"])

    assert loaded["name"] == "Test"
    assert loaded["filters"]["filename"] == "a"
