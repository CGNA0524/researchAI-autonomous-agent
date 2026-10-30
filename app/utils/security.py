"""
Security validation, SSRF defense, and untrusted web content sanitization.
"""

import ipaddress
import re
import socket
from urllib.parse import urlparse
from typing import Tuple
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Prompt injection signatures to neutralize in retrieved content
INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+instructions", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(an?|the)", re.IGNORECASE),
    re.compile(r"system\s*prompt\s*:", re.IGNORECASE),
    re.compile(r"<\s*\|im_start\|\s*>", re.IGNORECASE),
    re.compile(r"<\s*\|im_end\|\s*>", re.IGNORECASE),
    re.compile(r"\[\s*INST\s*\]", re.IGNORECASE),
    re.compile(r"forget\s+everything\s+you\s+know", re.IGNORECASE),
]


class SecurityValidator:
    """Security utilities for input validation, SSRF protection, and content safety."""

    @staticmethod
    def validate_url_safe_for_ssrf(url: str) -> Tuple[bool, str]:
        """
        Validates that a URL does not target localhost, private subnets, or cloud metadata services (SSRF protection).
        Returns (is_safe: bool, reason: str).
        """
        try:
            parsed = urlparse(url)
            if parsed.scheme.lower() not in ("http", "https"):
                return False, f"Unsupported scheme '{parsed.scheme}': only http and https are allowed."

            hostname = parsed.hostname
            if not hostname:
                return False, "Missing hostname in URL."

            # Check explicit blocked hostnames
            lower_host = hostname.lower()
            if lower_host in ("localhost", "127.0.0.1", "0.0.0.0", "::1", "metadata.google.internal"):
                return False, f"Access to restricted hostname '{hostname}' is blocked."

            # DNS resolution to check underlying IP addresses
            try:
                addr_info = socket.getaddrinfo(hostname, None)
            except socket.gaierror as e:
                return False, f"DNS resolution failed for hostname '{hostname}': {str(e)}"

            for item in addr_info:
                ip_str = item[4][0]
                ip = ipaddress.ip_address(ip_str)

                if ip.is_loopback:
                    return False, f"Loopback address '{ip}' is blocked."
                if ip.is_private:
                    return False, f"Private network address '{ip}' is blocked."
                if ip.is_link_local:
                    return False, f"Link-local address '{ip}' is blocked."
                if ip.is_multicast:
                    return False, f"Multicast address '{ip}' is blocked."
                if ip.is_reserved:
                    return False, f"Reserved address '{ip}' is blocked."
                if str(ip) == "169.254.169.254":
                    return False, "Cloud metadata endpoint is strictly prohibited."

            return True, "URL is safe"
        except Exception as e:
            return False, f"URL validation failed with error: {str(e)}"

    @staticmethod
    def sanitize_untrusted_content(text: str, source_url: str) -> str:
        """
        Sanitizes retrieved web text and wraps it in safe XML delimiters
        with explicit instructions to neutralize prompt injection attacks.
        """
        if not text:
            return ""

        # Flag and defang common prompt injection attacks
        sanitized_text = text
        for pattern in INJECTION_PATTERNS:
            if pattern.search(sanitized_text):
                logger.warning(
                    f"Possible prompt injection pattern detected in content from {source_url}. Defanging match."
                )
                sanitized_text = pattern.sub("[POTENTIAL_INJECTION_DEFANGED]", sanitized_text)

        # Wrap in explicit isolation tags
        safe_wrapped = (
            f'<untrusted_web_evidence source="{source_url}">\n'
            f"{sanitized_text.strip()}\n"
            f"</untrusted_web_evidence>"
        )
        return safe_wrapped
