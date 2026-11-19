import re
from urllib.parse import urlparse

SUSPICIOUS_WORDS = [
    "login", "verify", "secure", "account", "update",
    "bank", "paypal", "signin", "confirm", "password",
]

FEATURE_NAMES = [
    "url_length", "hostname_length", "path_length", "num_dots",
    "num_hyphens", "num_underscores", "num_digits", "num_special_chars",
    "num_subdomains", "has_at_symbol", "has_ip_address",
    "suspicious_word_count", "has_double_slash_in_path", "num_query_params",
]


def normalize_url(url: str) -> str:
    """Remove the scheme (http://, https://) because the dataset has almost none."""
    url = str(url).strip()
    return re.sub(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", "", url)


def extract_features(url: str) -> list:
    """Convert one URL string into a numeric feature list."""
    url = normalize_url(url)

    try:
        parsed = urlparse("http://" + url) 
        host = parsed.hostname or ""
        path = parsed.path or ""
        query = parsed.query or ""
    except ValueError:
        host, path, query = "", "", ""

    lower = url.lower()
    host_parts = [p for p in host.split(".") if p]

    return [
        len(url),
        len(host),
        len(path),
        url.count("."),
        url.count("-"),
        url.count("_"),
        sum(c.isdigit() for c in url),
        len(re.findall(r"[^A-Za-z0-9]", url)),
        max(len(host_parts) - 2, 0),
        int("@" in url),
        int(bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host))),
        sum(w in lower for w in SUSPICIOUS_WORDS),
        int("//" in path),
        len([q for q in query.split("&") if q]),
    ]