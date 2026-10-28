"""Use the real AstrBot SDK when present; otherwise stub only its host interfaces."""
import importlib.util
import logging
import os
import sys
import types
from pathlib import Path


def load_plugin_package():
    root = Path(__file__).resolve().parents[1]
    if "hakimi_memory" not in sys.modules:
        spec = importlib.util.spec_from_file_location("hakimi_memory", root / "__init__.py", submodule_search_locations=[str(root)])
        module = importlib.util.module_from_spec(spec)
        sys.modules["hakimi_memory"] = module
        spec.loader.exec_module(module)
    if os.environ.get("HAKIMI_TEST_STUB_SDK") != "1" and importlib.util.find_spec("astrbot") is not None:
        return
    if "astrbot.api" in sys.modules:
        return

    def decorator(*args, **kwargs):
        return lambda function: function

    class Star:
        def __init__(self, context, config=None):
            self.context = context

    class TextPart:
        def __init__(self, text):
            self.text = text
            self._no_save = False

        def mark_as_temp(self):
            self._no_save = True
            return self

    modules = {}
    for name in ["astrbot", "astrbot.api", "astrbot.api.event", "astrbot.api.star", "astrbot.core", "astrbot.core.agent", "astrbot.core.agent.message", "astrbot.core.star", "astrbot.core.star.star_tools"]:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = modules[name] = module
    modules["astrbot.api"].AstrBotConfig = dict
    modules["astrbot.api"].logger = logging.getLogger("hakimi-test")
    modules["astrbot.api.event"].AstrMessageEvent = object
    modules["astrbot.api.event"].filter = types.SimpleNamespace(
        EventMessageType=types.SimpleNamespace(ALL=0), event_message_type=decorator,
        on_llm_request=decorator, after_message_sent=decorator, llm_tool=decorator, command=decorator)
    modules["astrbot.api.star"].Context = object
    modules["astrbot.api.star"].Star = Star
    modules["astrbot.api.star"].register = decorator
    modules["astrbot.core.agent.message"].TextPart = TextPart
    modules["astrbot.core.star.star_tools"].StarTools = types.SimpleNamespace(get_data_dir=lambda name: root / "data" / name)
