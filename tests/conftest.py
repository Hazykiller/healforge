"""Allow pure-logic tests to run in offline environments without openai installed."""

import importlib.util
import sys
import types


if importlib.util.find_spec("openai") is None:
    sys.modules["openai"] = types.SimpleNamespace(OpenAI=object)
