"""M2: command-bar agent loop invariants with a scripted provider (network-free)."""
import asyncio
import json
import unittest

from application.ai.agent_conversation import AgentConversation
from application.ai.agent_events import RunOutcome
from application.ai.agent_loop import AgentLoop, ExecOutcome, LoopHooks, LoopPolicy
from application.ai.agent_wire import AgentUsage
from application.ai.budget import SessionBudget
from application.ai.errors import ProviderMalformedStreamError, ProviderOfflineError
from application.ai.limits import AgentRunLimits
from application.ai.providers.mock import ScriptedAgentProvider, scripted_turn
from application.commands import Surface, default_registry


def use(call_id, name, arguments=None):
    return {"type": "tool_use", "id": call_id, "name": name, "input": {} if arguments is None else arguments}


def text(value):
    return {"type": "text", "text": value}


THINK = {"type": "thinking", "thinking": "plan", "signature": "sig=="}


class FakeExecutor:
    def __init__(self, fail=()):
        self.registry = default_registry()
        self.calls = []
        self.fail = set(fail)

    def spec_for(self, tool):
        return self.registry.by_tool(tool) if self.registry.has_tool(tool) else None

    async def execute(self, tool, arguments, cancel):
        self.calls.append((tool, dict(arguments)))
        if tool in self.fail:
            return ExecOutcome(False, {"ok": False, "code": "command_failed", "message": "boom"})
        return ExecOutcome(True, {"ok": True, "tool": tool})


class Flag:
    cancellation_requested = False


def tools():
    return tuple(default_registry().tool_definitions(Surface.AGENT_INAPP))


def make(script, *, executor=None, hooks=None, policy=None, limits=None, session=None, conversation=None):
    provider = ScriptedAgentProvider(script)
    executor = executor or FakeExecutor()
    events = []
    hooks = hooks or LoopHooks()
    original = hooks.emit
    hooks.emit = lambda event: (events.append(event), original(event))
    loop = AgentLoop(provider, executor, model="claude-opus-5-5", tools=tools(), hooks=hooks,
                     policy=policy, limits=limits, session_budget=session, conversation=conversation)
    return loop, provider, executor, events


def results_of(message):
    return [block for block in message["content"] if block.get("type") == "tool_result"]


class LoopTests(unittest.TestCase):
    def test_parallel_calls_one_result_message_in_order_and_verbatim_replay(self):
        loop, provider, executor, events = make([
            scripted_turn(THINK, use("t1", "go_to_time", {"time_s": 2.0}),
                          use("t2", "set_view", {"left_sidebar": False})),
            scripted_turn(text("Done.")),
        ])
        result = asyncio.run(loop.run("hide the left sidebar and go to 2 s", {"time": 0}))
        self.assertIs(result.outcome, RunOutcome.COMPLETED)
        self.assertEqual(result.text, "Done.")
        self.assertEqual([call[0] for call in executor.calls], ["go_to_time", "set_view"])
        second = provider.requests[1].messages
        self.assertEqual(second[-1]["role"], "user")
        self.assertEqual([r["tool_use_id"] for r in results_of(second[-1])], ["t1", "t2"])
        # thinking block replayed byte-identical, policy system message precedes the assistant turn
        self.assertEqual(json.dumps(second[2]["content"][0], sort_keys=True), json.dumps(THINK, sort_keys=True))
        self.assertEqual(second[1]["role"], "system")
        self.assertIn("<ghostgui_state>", second[0]["content"][-1]["text"])
        self.assertIn("tool_finished", [event.kind for event in events])

    def test_request_prefix_is_append_only_across_runs(self):
        loop, provider, _, _ = make([scripted_turn(text("a")), scripted_turn(text("b"))])
        asyncio.run(loop.run("one"))
        asyncio.run(loop.run("two"))
        first, second = provider.requests
        self.assertEqual(json.dumps(first.messages[0], sort_keys=True), json.dumps(second.messages[0], sort_keys=True))
        self.assertEqual(first.tools, second.tools)
        self.assertEqual([t["name"] for t in first.tools], sorted(t["name"] for t in first.tools))

    def test_invalid_input_and_unknown_tool_are_errors(self):
        loop, provider, executor, _ = make([
            scripted_turn(use("t1", "go_to_time", "not-json"), use("t2", "rm_rf")),
            scripted_turn(text("ok")),
        ])
        asyncio.run(loop.run("x"))
        blocks = results_of(provider.requests[1].messages[-1])
        self.assertTrue(all(block["is_error"] for block in blocks))
        self.assertIn("INVALID_JSON", blocks[0]["content"][0]["text"])
        self.assertEqual(executor.calls, [])

    def test_denial_skips_later_mutating_calls(self):
        async def deny(*args):
            return "deny"

        loop, provider, executor, _ = make([
            scripted_turn(use("t1", "history_undo"), use("t2", "set_view", {"left_sidebar": True}),
                          use("t3", "get_recent_actions")),
            scripted_turn(text("ok")),
        ], hooks=LoopHooks(confirm=deny))
        asyncio.run(loop.run("undo"))
        self.assertEqual([call[0] for call in executor.calls], ["get_recent_actions"])
        blocks = results_of(provider.requests[1].messages[-1])
        self.assertIn("user_declined", blocks[0]["content"][0]["text"])
        self.assertIn("skipped", blocks[1]["content"][0]["text"])

    def test_allow_for_run_skips_second_confirmation(self):
        asked = []

        async def allow(*args):
            asked.append(args[0].tool)
            return "run"

        loop, _, executor, _ = make([
            scripted_turn(use("t1", "history_undo")), scripted_turn(use("t2", "history_undo")),
            scripted_turn(text("ok"))], hooks=LoopHooks(confirm=allow))
        asyncio.run(loop.run("undo twice"))
        self.assertEqual(asked, ["history_undo"])
        self.assertEqual(len(executor.calls), 2)

    def test_stop_synthesises_results_and_keeps_history_valid(self):
        flag = Flag()

        class StopExecutor(FakeExecutor):
            async def execute(self, tool, arguments, cancel):
                flag.cancellation_requested = True
                return await super().execute(tool, arguments, cancel)

        loop, provider, executor, _ = make([
            scripted_turn(use("t1", "go_to_time", {"time_s": 1}), use("t2", "go_to_time", {"time_s": 2})),
            scripted_turn(text("next run"))], executor=StopExecutor())
        result = asyncio.run(loop.run("x", cancel=flag))
        self.assertIs(result.outcome, RunOutcome.STOPPED)
        self.assertEqual(len(executor.calls), 1)
        wire = loop.conversation.wire()
        self.assertIn("Cancelled by the user", json.dumps(results_of(wire[-1])[1]))
        asyncio.run(loop.run("again"))
        self.assertEqual(len(provider.requests), 2)

    def test_provider_cancel_mid_stream(self):
        from application.ai.errors import ProviderCancelledError
        loop, _, _, _ = make([ProviderCancelledError("stop")])
        self.assertIs(asyncio.run(loop.run("x")).outcome, RunOutcome.STOPPED)
        self.assertEqual(loop.conversation.last_role, "user")

    def test_budget_stop_and_notice(self):
        expensive = AgentUsage(input_tokens=200_000, output_tokens=10_000, served_model="claude-opus-5-5")
        loop, provider, _, _ = make([
            scripted_turn(use("t1", "get_recent_actions"), usage=expensive),
            scripted_turn(use("t2", "get_recent_actions"), usage=expensive),
            scripted_turn(text("never"))], limits=AgentRunLimits(max_run_cost_usd=1.2))
        result = asyncio.run(loop.run("x"))
        self.assertIs(result.outcome, RunOutcome.BUDGET)
        self.assertEqual(len(provider.requests), 2)
        self.assertIn("Budget notice", json.dumps(provider.requests[1].messages[-1]))

    def test_session_cap_blocks_new_runs(self):
        session = SessionBudget(cap_usd=0.0)
        loop, provider, _, _ = make([scripted_turn(text("x"))], session=session)
        self.assertIs(asyncio.run(loop.run("x")).outcome, RunOutcome.BUDGET)
        self.assertEqual(provider.requests, [])

    def test_repeated_identical_failure_aborts(self):
        loop, provider, _, _ = make([
            scripted_turn(use("a", "set_view", {"left_sidebar": True})),
            scripted_turn(use("b", "set_view", {"left_sidebar": True})),
            scripted_turn(text("never"))], executor=FakeExecutor(fail={"set_view"}))
        self.assertIs(asyncio.run(loop.run("x")).outcome, RunOutcome.TOOL_ERRORS)
        self.assertEqual(len(provider.requests), 2)

    def test_max_tokens_with_tool_use_is_reissued_not_appended(self):
        truncated = scripted_turn(use("t1", "go_to_time", {"time_s": 1}), stop_reason="max_tokens")
        loop, provider, executor, _ = make([truncated, scripted_turn(text("ok"))])
        asyncio.run(loop.run("x"))
        self.assertEqual(provider.requests[1].max_tokens, 128_000)
        self.assertEqual(executor.calls, [])
        self.assertEqual(len(loop.conversation), 3)  # user, system, assistant

    def test_malformed_stream_reissued(self):
        loop, provider, _, _ = make([ProviderMalformedStreamError("bad"), scripted_turn(text("ok"))])
        self.assertIs(asyncio.run(loop.run("x")).outcome, RunOutcome.COMPLETED)
        self.assertEqual(len(provider.requests), 2)

    def test_refusal_and_pause_turn(self):
        refusal = scripted_turn(stop_reason="refusal", stop_details={"category": "cyber"})
        loop, _, _, _ = make([refusal])
        result = asyncio.run(loop.run("x"))
        self.assertIs(result.outcome, RunOutcome.REFUSED)
        self.assertEqual(result.refusal_category, "cyber")
        loop, provider, _, _ = make([scripted_turn(text("..."), stop_reason="pause_turn"), scripted_turn(text("done"))])
        self.assertIs(asyncio.run(loop.run("x")).outcome, RunOutcome.COMPLETED)
        self.assertEqual(len(provider.requests), 2)

    def test_context_exceeded_and_provider_errors(self):
        loop, _, _, _ = make([scripted_turn(stop_reason="model_context_window_exceeded")])
        self.assertIs(asyncio.run(loop.run("x")).outcome, RunOutcome.CONTEXT_FULL)
        loop, _, _, events = make([ProviderOfflineError("offline")])
        result = asyncio.run(loop.run("x"))
        self.assertIs(result.outcome, RunOutcome.ERROR)
        self.assertIn("offline", result.message)

    def test_plan_mode_defers_and_replays(self):
        loop, _, executor, _ = make([
            scripted_turn(use("t1", "set_view", {"left_sidebar": False}), use("t2", "get_recent_actions")),
            scripted_turn(text("Plan ready."))], policy=LoopPolicy(plan_mode=True))
        result = asyncio.run(loop.run("hide sidebar"))
        self.assertIs(result.outcome, RunOutcome.PLAN_READY)
        self.assertEqual([call[0] for call in executor.calls], ["get_recent_actions"])
        self.assertEqual(result.deferred[0].tool, "set_view")
        summary = asyncio.run(loop.execute_plan(result.deferred))
        self.assertEqual(summary[0]["status"], "ok")
        self.assertIn("Approved plan results", loop.pending_note)

    def test_ask_user(self):
        async def answer(question, options):
            return options[0]

        loop, provider, _, _ = make([
            scripted_turn(use("q", "ask_user", {"question": "Which hand?", "options": ["left", "right"]})),
            scripted_turn(text("ok"))], hooks=LoopHooks(ask=answer))
        asyncio.run(loop.run("raise a hand"))
        block = results_of(provider.requests[1].messages[-1])[0]
        self.assertEqual(json.loads(block["content"][0]["text"])["answer"], "left")

    def test_key_in_prompt_blocked(self):
        loop, provider, _, _ = make([scripted_turn(text("x"))])
        result = asyncio.run(loop.run("use sk-ant-api03-abcdefghijklmnop"))
        self.assertIs(result.outcome, RunOutcome.ERROR)
        self.assertEqual(provider.requests, [])

    def test_conversation_placement_rules(self):
        conversation = AgentConversation("m")
        with self.assertRaises(ValueError):
            conversation.add_system("x")
        conversation.add_user([text("hi")])
        conversation.add_system("policy")
        with self.assertRaises(ValueError):
            conversation.add_user([text("again")])
        with self.assertRaises(ValueError):
            AgentLoop(ScriptedAgentProvider(), FakeExecutor(), model="other", tools=(), conversation=conversation)


if __name__ == "__main__":
    unittest.main()
