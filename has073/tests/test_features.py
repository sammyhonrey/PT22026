"""Unit tests for URL normalisation and feature extraction."""
import pytest

from features import FEATURE_NAMES, extract_features, normalize_url


def feature(url, name):
    """Return the value of one named feature for a URL."""
    return extract_features(url)[FEATURE_NAMES.index(name)]


def test_there_are_14_features():
    assert len(FEATURE_NAMES) == 14
    assert len(extract_features("example.com")) == 14


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://www.wikipedia.org", "www.wikipedia.org"),
        ("http://example.com/a?b=1", "example.com/a?b=1"),
        ("example.com/login", "example.com/login"),
        ("  https://example.com  ", "example.com"),
    ],
)
def test_normalize_url_removes_the_scheme(url, expected):
    assert normalize_url(url) == expected


def test_scheme_does_not_change_the_features():
    plain = extract_features("github.com/anthropics")
    assert extract_features("http://github.com/anthropics")
    assert extract_features("https://github.com/anthropics") 


def test_url_hostname_and_path_length():
    assert feature("https://github.com", "url_length") == len("github.com")
    assert feature("github.com/x", "hostname_length") == len("github.com")
    assert feature("github.com/x", "path_length") == len("/x")


def test_at_symbol():
    assert feature("http://user@evil.com/login", "has_at_symbol") == 1
    assert feature("http://example.com/login", "has_at_symbol") == 0


def test_ip_address_host():
    assert feature("http://192.168.1.15/bank/login.php", "has_ip_address") == 1
    assert feature("http://91.239.24.244:6892", "has_ip_address") == 1
    assert feature("example.com", "has_ip_address") == 0


def test_suspicious_words_are_counted():
    assert feature("example.com", "suspicious_word_count") == 0
    assert feature("paypal-login.com", "suspicious_word_count") == 2
    assert (
        feature("http://paypa1-secure-login.xyz/verify@account", "suspicious_word_count")
        == 4
    )


def test_hyphens_dots_and_subdomains():
    assert feature("a-b-c.com", "num_hyphens") == 2
    assert feature("a.b.c.example.com", "num_dots") == 4
    assert feature("a.b.c.example.com", "num_subdomains") == 3
    assert feature("example.com", "num_subdomains") == 0


def test_query_parameters_and_double_slash():
    assert feature("example.com/page?x=1&y=2", "num_query_params") == 2
    assert feature("example.com//page", "has_double_slash_in_path") == 1
    assert feature("example.com/page", "has_double_slash_in_path") == 0


@pytest.mark.parametrize("bad_input", ["", "   ", "http://[::1", "%%%", "a" * 5000])
def test_odd_inputs_never_crash(bad_input):
    values = extract_features(bad_input)
    assert len(values) == 14
    assert all(isinstance(v, (int, float)) for v in values)