from __future__ import annotations

from modules.runtime.error_codes import ErrorCode
from modules.runtime.operation_result import OperationResult

_DEFAULT_MESSAGES: dict[ErrorCode, str] = {
    ErrorCode.NETWORK_ERROR: "网络异常",
    ErrorCode.REMOTE_ERROR: "远程服务异常",
    ErrorCode.NO_VERSION: "未解析到版本号",
    ErrorCode.BACKUP_DIR_MISSING: "未找到备份文件夹",
    ErrorCode.NO_BACKUPS: "未找到任何备份",
    ErrorCode.CONFIG_INVALID: "配置无效",
    ErrorCode.FILE_NOT_FOUND: "文件不存在",
    ErrorCode.PERMISSION_DENIED: "权限不足",
    ErrorCode.PORT_IN_USE: "端口已被占用",
    ErrorCode.UNKNOWN: "发生未知错误",
}


def describe_result(result: OperationResult, default_message: str) -> str:
    if result.message:
        return result.message
    if result.code and result.code in _DEFAULT_MESSAGES:
        return _DEFAULT_MESSAGES[result.code]
    return default_message


__all__ = ["describe_result"]
