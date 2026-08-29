from pathlib import Path

from rlattack.report import (
    render_dashboard,
    render_transfer_report,
    write_dashboard_report,
    write_transfer_report,
)


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


def test_transfer_report_is_self_contained() -> None:
    data = {
        "policy": "ppo",
        "reference": "small/easy",
        "seeds": [1, 2],
        "stages": [
            {
                "agent_name": "small/easy",
                "success_rate": 1.0,
                "detection_rate": 0.0,
                "mean_steps": 20.0,
                "std_steps": 1.0,
                "mean_reward": 5.0,
                "reward_ci_low": 4.0,
                "reward_ci_high": 6.0,
            }
        ],
        "comparisons": [],
        "conditions": [["Defender", "passive"]],
        "note": "paired across classes",
    }

    html = render_transfer_report(data)

    assert "__RLATTACK_TRANSFER__" in html
    assert "small/easy" in html
    assert "http://" not in html.replace("http://www.w3.org", "")


def test_transfer_report_is_written_to_disk(tmp_path: Path) -> None:
    output = write_transfer_report({"policy": "greedy"}, tmp_path / "nested" / "transfer.html")

    assert output.exists()
    assert output.name == "transfer.html"


def test_report_declares_responsive_rules_for_narrow_viewports() -> None:
    """Long condition labels and the CI column overflowed a 320px viewport.

    A browser is not available in CI, so this guards the CSS rules that fixed it.
    """

    html = render_dashboard({"config": {}})

    assert ".panel-head { display:flex" in html
    assert "flex-wrap:wrap" in html
    assert "white-space:normal" in html
    assert "@media (max-width:720px)" in html
    assert ".wide-only { display:none }" in html
