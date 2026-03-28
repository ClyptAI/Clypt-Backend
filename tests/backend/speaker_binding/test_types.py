import importlib.util
import sys
import types
import typing
from pathlib import Path

from typing_extensions import NotRequired as ExtensionsNotRequired


def test_speaker_binding_types_import_without_typing_notrequired(tmp_path):
    module_path = Path(__file__).resolve().parents[3] / "backend" / "speaker_binding" / "types.py"
    fake_typing = types.ModuleType("typing")
    for name in dir(typing):
        if name == "NotRequired":
            continue
        setattr(fake_typing, name, getattr(typing, name))

    fake_typing_extensions = types.ModuleType("typing_extensions")
    fake_typing_extensions.NotRequired = ExtensionsNotRequired

    prior_typing = sys.modules.get("typing")
    prior_typing_extensions = sys.modules.get("typing_extensions")
    sys.modules["typing"] = fake_typing
    sys.modules["typing_extensions"] = fake_typing_extensions
    try:
        spec = importlib.util.spec_from_file_location("_speaker_binding_types_py310", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        if prior_typing is None:
            sys.modules.pop("typing", None)
        else:
            sys.modules["typing"] = prior_typing
        if prior_typing_extensions is None:
            sys.modules.pop("typing_extensions", None)
        else:
            sys.modules["typing_extensions"] = prior_typing_extensions

    assert hasattr(module, "DiarizedSpan")
    assert hasattr(module, "ScheduledSpan")
