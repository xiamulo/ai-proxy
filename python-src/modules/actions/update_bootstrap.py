from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from modules.actions import update_actions


@dataclass(frozen=True)
class UpdateBootstrapDeps:
    window: Any
    log: Any
    thread_manager: Any
    check_button: Any
    app_display_name: str
    app_version: str
    repo: str
    default_font: Any
    update_service: Any
    update_dialog: Any
    messagebox: Any
    create_tkinterweb_html_widget: Any
    program_resource_dir: str


def configure_update_controller(
    controller: update_actions.UpdateCheckController,
    deps: UpdateBootstrapDeps,
) -> None:
    controller.configure(
        update_actions.UpdateCheckDeps(
            window=deps.window,
            log=deps.log,
            thread_manager=deps.thread_manager,
            check_button=deps.check_button,
            app_display_name=deps.app_display_name,
            app_version=deps.app_version,
            repo=deps.repo,
            default_font=deps.default_font,
            update_service=deps.update_service,
            update_dialog=deps.update_dialog,
            messagebox=deps.messagebox,
            create_tkinterweb_html_widget=deps.create_tkinterweb_html_widget,
            program_resource_dir=deps.program_resource_dir,
        )
    )


def build_update_controller(deps: UpdateBootstrapDeps) -> update_actions.UpdateCheckController:
    controller = update_actions.UpdateCheckController(state=update_actions.UpdateCheckState())
    configure_update_controller(controller, deps)
    return controller
