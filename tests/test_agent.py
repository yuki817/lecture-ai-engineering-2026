"""エージェントの配線テスト。

TestModel は実際のLLMを呼ばない偽モデルで、登録されたツールを全部呼んでから
ダミーの回答を返す。APIキーも利用枠も使わずに、
「ツールが登録され、呼ばれ、結果がモデルに戻るか」を確かめられる。
"""

from pydantic_ai import DeferredToolRequests
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models.test import TestModel

from app import tools
from app.agent import agent, resolve_approval_requests

BASE_TOOLS = {"search_glossary", "get_lecture_summary"}


def _parts(result, kind):
    return [p for m in result.all_messages() for p in m.parts if isinstance(p, kind)]


def test_base_tools_are_registered_and_called():
    model = TestModel()
    with agent.override(model=model):
        result = agent.run_sync("LoRAとは？")

    offered = {t.name for t in model.last_model_request_parameters.function_tools}
    assert BASE_TOOLS <= offered  # 演習でツールを足しても壊れないよう「含む」で判定

    called = {p.tool_name for p in _parts(result, ToolCallPart)}
    assert BASE_TOOLS <= called
    # 承認が必要なツール（requires_approval=True）は、実行されずに承認待ちになるので戻り値が無い
    waiting = len(result.output.approvals) if isinstance(result.output, DeferredToolRequests) else 0
    assert len(_parts(result, ToolReturnPart)) + waiting == len(_parts(result, ToolCallPart))
    assert result.output


def test_glossary_result_reaches_the_model():
    model = TestModel(call_tools=["search_glossary"])
    with agent.override(model=model):
        result = agent.run_sync("RAGとは？")
    returns = _parts(result, ToolReturnPart)
    assert returns and returns[0].tool_name == "search_glossary"


def test_save_learning_log_waits_for_approval(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", tmp_path)
    model = TestModel(call_tools=["save_learning_log"])
    with agent.override(model=model):
        result = agent.run_sync("今日の学びを記録に保存してください。")

    assert isinstance(result.output, DeferredToolRequests)
    assert [call.tool_name for call in result.output.approvals] == ["save_learning_log"]
    assert not list(tmp_path.iterdir())


def test_approved_learning_log_is_saved(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", tmp_path)
    monkeypatch.setattr("builtins.input", lambda _: "y")
    model = TestModel(call_tools=["save_learning_log"])
    with agent.override(model=model):
        pending = agent.run_sync("今日の学びを記録に保存してください。")
    assert isinstance(pending.output, DeferredToolRequests)

    result = resolve_approval_requests(agent, pending, TestModel(call_tools=[]))

    assert isinstance(result.output, str)
    assert len(list(tmp_path.glob("*.md"))) == 1


def test_denied_learning_log_is_not_saved(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", tmp_path)
    monkeypatch.setattr("builtins.input", lambda _: "n")
    model = TestModel(call_tools=["save_learning_log"])
    with agent.override(model=model):
        pending = agent.run_sync("今日の学びを記録に保存してください。")
    assert isinstance(pending.output, DeferredToolRequests)

    result = resolve_approval_requests(agent, pending, TestModel(call_tools=[]))

    assert isinstance(result.output, str)
    assert not list(tmp_path.iterdir())
