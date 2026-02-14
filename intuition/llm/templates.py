"""TemplateEngine — Jinja2-based prompt template management."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

_DEFAULT_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


class TemplateEngine:
    def __init__(self, template_dir: str | Path | None = None) -> None:
        directory = Path(template_dir) if template_dir else _DEFAULT_DIR
        self._env = Environment(
            loader=FileSystemLoader(str(directory)),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=False,
        )

    def render(self, template_name: str, **kwargs: Any) -> str:
        tmpl = self._env.get_template(template_name)
        return tmpl.render(**kwargs)

    def list_templates(self) -> list[str]:
        return self._env.list_templates()
