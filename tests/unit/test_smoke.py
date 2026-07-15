"""Basic project configuration tests."""

from vitality_engagement import __version__


def test_package_version() -> None:
    """Verify that the package can be imported."""
    assert __version__ == "0.1.0"
