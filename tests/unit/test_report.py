from pathlib import Path

from rlattack.report import render_dashboard, write_dashboard_report


def test_dashboard_html_embeds_safe_offline_data() -> None:
    html = render_dashboard({"label": "</script><unsafe>"})

    assert "<!doctype html>" in html
    assert "window.__RLATTACK_API__=false" in html
    assert "\\u003c/script>" in html
    assert "Run experiment" in html


def test_dashboard_html_can_enable_api_and_write_report(tmp_path: Path) -> None:
    data = {"config": {"seed": 4}}
    output = write_dashboard_report(data, tmp_path / "nested" / "report.html")
    api_html = render_dashboard(data, api_enabled=True)

    assert output.is_absolute()
    assert output.exists()
    assert "window.__RLATTACK_API__=true" in api_html
