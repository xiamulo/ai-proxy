from __future__ import annotations

import socket


def is_port_in_use(port: int) -> bool:
    checks = [("127.0.0.1", socket.AF_INET)]
    if socket.has_ipv6:
        checks.append(("::1", socket.AF_INET6))

    for host, family in checks:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            try:
                if sock.connect_ex((host, port)) == 0:
                    return True
            except OSError:
                continue
    return False
