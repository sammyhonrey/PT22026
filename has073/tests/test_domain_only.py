"""Tests for the helper that removes the path (used in the domain-only experiment)."""
import pytest

from train import domain_only


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://en.wikipedia.org/wiki/Machine_learning", "en.wikipedia.org"),
        ("github.com/anthropics/anthropic-sdk-python", "github.com"),
        ("example.com/page?x=1#top", "example.com"),
        ("example.com?x=1", "example.com"),
        ("paypa1-secure-login.xyz", "paypa1-secure-login.xyz"),
    ],
)
def test_domain_only_keeps_just_the_host(url, expected):
    assert domain_only(url) == expected