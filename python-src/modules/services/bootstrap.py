from __future__ import annotations

from dataclasses import dataclass

from modules.runtime.resource_manager import ResourceManager
from modules.runtime.thread_manager import ThreadManager
from modules.services.config_service import ConfigStore


@dataclass(frozen=True)
class AppContext:
    resource_manager: ResourceManager
    thread_manager: ThreadManager
    config_store: ConfigStore
    config_file: str


def build_app_context() -> AppContext:
    resource_manager = ResourceManager()
    thread_manager = ThreadManager()
    config_file = resource_manager.get_user_config_file()
    config_store = ConfigStore(config_file)
    return AppContext(
        resource_manager=resource_manager,
        thread_manager=thread_manager,
        config_store=config_store,
        config_file=config_file,
    )
