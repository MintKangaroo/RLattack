import rlattack


def test_package_version() -> None:
    assert rlattack.__version__ == "0.2.0"


def test_package_documents_simulation_scope() -> None:
    assert "simulated attack graphs" in (rlattack.__doc__ or "")
