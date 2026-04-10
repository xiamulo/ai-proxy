from __future__ import annotations

from enum import StrEnum


class ErrorCode(StrEnum):
    UNKNOWN = "unknown"
    CONFIG_INVALID = "config_invalid"
    NETWORK_ERROR = "network_error"
    REMOTE_ERROR = "remote_error"
    NO_VERSION = "no_version"
    FILE_NOT_FOUND = "file_not_found"
    PERMISSION_DENIED = "permission_denied"
    PORT_IN_USE = "port_in_use"
    BACKUP_DIR_MISSING = "backup_dir_missing"
    NO_BACKUPS = "no_backups"


__all__ = ["ErrorCode"]
