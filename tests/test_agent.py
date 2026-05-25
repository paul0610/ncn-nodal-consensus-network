"""Phase 2 — Agent tests: skills, executor, memory, planner, runner."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from agent.executor import ActionExecutor
from agent.memory import SessionMemory
from agent.planner import ActionPlanner
from agent.runner import AgentRunner
from agent.skill import SkillManager
from core.config_loader import (
    AgentConfig,
    Config,
    GraphConfig,
    KuzuConfig,
    ProviderConfig,
    SwarmConfig,
    SwarmNodeConfig,
)
from core.exceptions import (
    AgentDisabledError,
    CommandTimeoutError,
    DangerousCommandError,
)
from core.models import (
    ActionPlan,
    ActionResult,
    ActionStep,
    AgentResponse,
    GraphNode,
    NodeRole,
)
from graph.kuzu_adapter import KuzuAdapter
from graph.writer import SingleWriter
from providers.ollama import OllamaProvider
from swarm.pool import SwarmPool


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _cfg(tmp_path, agent_enabled=True) -> Config:
    return Config(
        graph=GraphConfig(kuzu=KuzuConfig(path=str(tmp_path / "ag"))),
        swarm=SwarmConfig(
            max_concurrent=5, request_timeout=30,
            nodes=[
                SwarmNodeConfig(provider="ollama", model="m", count=1, role="planner"),
                SwarmNodeConfig(provider="ollama", model="m", count=1, role="critic"),
                SwarmNodeConfig(provider="ollama", model="m", count=1, role="synthesizer"),
            ],
        ),
        agent=AgentConfig(enabled=agent_enabled, command_timeout=5),
    )


def _mock_pool(pool):
    for n in pool.nodes:
        n.provider = OllamaProvider(ProviderConfig(base_url="http://x"))
        n.provider._post = AsyncMock(return_value={"response": "{}", "eval_count": 1})


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

class TestAgentConfig:
    def test_defaults(self):
        cfg = Config()
        assert cfg.agent.enabled is False
        assert cfg.agent.allow_shell is True
        assert cfg.agent.command_timeout == 30

    def test_backward_compat(self):
        cfg = Config()
        assert hasattr(cfg, "agent")


# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------

class TestAgentModels:
    def test_action_step(self):
        s = ActionStep(action_type="shell", command="ls")
        assert s.step_id != ""
        assert s.timeout == 30

    def test_action_result(self):
        r = ActionResult(action_type="shell", command="ls", success=True)
        assert r.exit_code == 0

    def test_action_plan(self):
        p = ActionPlan(goal="test")
        assert p.plan_id != ""
        assert p.steps == []

    def test_agent_response(self):
        r = AgentResponse(goal="test", success=True, summary="done")
        assert r.results == []

    def test_planner_role_exists(self):
        assert NodeRole.PLANNER.value == "planner"


# ------------------------------------------------------------------
# ActionExecutor
# ------------------------------------------------------------------

class TestActionExecutor:
    async def test_shell_echo(self, tmp_path):
        cfg = _cfg(tmp_path)
        exe = ActionExecutor(cfg)
        result = await exe.execute_shell("echo hello")
        assert result.success
        assert "hello" in result.stdout

    async def test_shell_timeout(self, tmp_path):
        cfg = _cfg(tmp_path)
        cfg = cfg.model_copy(update={"agent": AgentConfig(enabled=True, command_timeout=1)})
        exe = ActionExecutor(cfg)
        with pytest.raises(CommandTimeoutError):
            await exe.execute_shell("ping -n 10 127.0.0.1", timeout=1)

    async def test_dangerous_command(self, tmp_path):
        cfg = _cfg(tmp_path)
        exe = ActionExecutor(cfg)
        assert exe.is_dangerous("rm -rf /")
        with pytest.raises(DangerousCommandError):
            await exe.execute_shell("rm -rf /tmp/test")

    async def test_shell_disabled(self, tmp_path):
        cfg = _cfg(tmp_path)
        cfg = cfg.model_copy(update={"agent": AgentConfig(enabled=True, allow_shell=False)})
        exe = ActionExecutor(cfg)
        with pytest.raises(AgentDisabledError):
            await exe.execute_shell("ls")

    async def test_file_read(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")
        cfg = _cfg(tmp_path)
        exe = ActionExecutor(cfg)
        result = await exe.execute_file_read(str(test_file))
        assert result.success
        assert "hello world" in result.stdout

    async def test_file_write(self, tmp_path):
        cfg = _cfg(tmp_path)
        exe = ActionExecutor(cfg)
        target = str(tmp_path / "output.txt")
        result = await exe.execute_file_write(target, "test content")
        assert result.success
        assert Path(target).read_text() == "test content"

    async def test_execute_step(self, tmp_path):
        cfg = _cfg(tmp_path)
        exe = ActionExecutor(cfg)
        step = ActionStep(action_type="shell", command="echo step_test")
        result = await exe.execute_step(step)
        assert result.success
        assert result.step_id == step.step_id


# ------------------------------------------------------------------
# SkillManager
# ------------------------------------------------------------------

class TestSkillManager:
    async def test_add_and_get(self, tmp_path):
        cfg = _cfg(tmp_path)
        adapter = KuzuAdapter(cfg)
        await adapter.initialize()
        writer = SingleWriter(adapter, cfg)
        sm = SkillManager(adapter, cfg, writer)

        nid = await sm.add("list-files", "ls -la", description="List files")
        assert nid != ""

        skill = await sm.get("list-files")
        assert skill is not None
        assert skill.metadata["command"] == "ls -la"
        await adapter.close()

    async def test_list_all(self, tmp_path):
        cfg = _cfg(tmp_path)
        adapter = KuzuAdapter(cfg)
        await adapter.initialize()
        writer = SingleWriter(adapter, cfg)
        sm = SkillManager(adapter, cfg, writer)

        await sm.add("skill-a", "cmd-a")
        await sm.add("skill-b", "cmd-b")
        skills = await sm.list_all()
        assert len(skills) == 2
        await adapter.close()

    async def test_remove(self, tmp_path):
        cfg = _cfg(tmp_path)
        adapter = KuzuAdapter(cfg)
        await adapter.initialize()
        writer = SingleWriter(adapter, cfg)
        sm = SkillManager(adapter, cfg, writer)

        await sm.add("to-remove", "cmd")
        assert await sm.remove("to-remove")
        assert await sm.get("to-remove") is None
        await adapter.close()

    async def test_get_nonexistent(self, tmp_path):
        cfg = _cfg(tmp_path)
        adapter = KuzuAdapter(cfg)
        await adapter.initialize()
        writer = SingleWriter(adapter, cfg)
        sm = SkillManager(adapter, cfg, writer)
        assert await sm.get("nope") is None
        await adapter.close()


# ------------------------------------------------------------------
# SessionMemory
# ------------------------------------------------------------------

class TestSessionMemory:
    async def test_set_and_get(self, tmp_path):
        cfg = _cfg(tmp_path)
        adapter = KuzuAdapter(cfg)
        await adapter.initialize()
        writer = SingleWriter(adapter, cfg)
        mem = SessionMemory(adapter, cfg, writer)

        await mem.set("user_name", "Paul")
        val = await mem.get("user_name")
        assert val == "Paul"
        await adapter.close()

    async def test_update_existing(self, tmp_path):
        cfg = _cfg(tmp_path)
        adapter = KuzuAdapter(cfg)
        await adapter.initialize()
        writer = SingleWriter(adapter, cfg)
        mem = SessionMemory(adapter, cfg, writer)

        await mem.set("color", "blue")
        await mem.set("color", "red")
        assert await mem.get("color") == "red"
        await adapter.close()

    async def test_delete(self, tmp_path):
        cfg = _cfg(tmp_path)
        adapter = KuzuAdapter(cfg)
        await adapter.initialize()
        writer = SingleWriter(adapter, cfg)
        mem = SessionMemory(adapter, cfg, writer)

        await mem.set("temp", "value")
        assert await mem.delete("temp")
        assert await mem.get("temp") is None
        await adapter.close()

    async def test_list_all(self, tmp_path):
        cfg = _cfg(tmp_path)
        adapter = KuzuAdapter(cfg)
        await adapter.initialize()
        writer = SingleWriter(adapter, cfg)
        mem = SessionMemory(adapter, cfg, writer)

        await mem.set("a", "1")
        await mem.set("b", "2")
        entries = await mem.list_all()
        assert len(entries) == 2
        await adapter.close()


# ------------------------------------------------------------------
# ActionPlanner
# ------------------------------------------------------------------

class TestActionPlanner:
    async def test_plan_generation(self, tmp_path):
        cfg = _cfg(tmp_path)
        adapter = KuzuAdapter(cfg)
        await adapter.initialize()
        writer = SingleWriter(adapter, cfg)
        pool = SwarmPool(cfg)
        _mock_pool(pool)

        plan_json = json.dumps({"steps": [
            {"action_type": "shell", "command": "echo hello", "description": "test", "timeout": 5}
        ]})
        for n in pool.nodes:
            if n.role == NodeRole.PLANNER:
                n.provider._post = AsyncMock(
                    return_value={"response": plan_json, "eval_count": 1}
                )

        sm = SkillManager(adapter, cfg, writer)
        mem = SessionMemory(adapter, cfg, writer)
        planner = ActionPlanner(cfg, pool, sm, mem)

        plan = await planner.plan("say hello")
        assert len(plan.steps) == 1
        assert plan.steps[0].command == "echo hello"
        await adapter.close()

    async def test_verify_plan(self, tmp_path):
        cfg = _cfg(tmp_path)
        adapter = KuzuAdapter(cfg)
        await adapter.initialize()
        writer = SingleWriter(adapter, cfg)
        pool = SwarmPool(cfg)
        _mock_pool(pool)

        approve_json = json.dumps({"approve": True, "concerns": []})
        for n in pool.nodes:
            if n.role == NodeRole.CRITIC:
                n.provider._post = AsyncMock(
                    return_value={"response": approve_json, "eval_count": 1}
                )

        sm = SkillManager(adapter, cfg, writer)
        mem = SessionMemory(adapter, cfg, writer)
        planner = ActionPlanner(cfg, pool, sm, mem)

        plan = ActionPlan(goal="test", steps=[
            ActionStep(action_type="shell", command="echo hi")
        ])
        approved, concerns = await planner.verify_plan(plan)
        assert approved
        await adapter.close()


# ------------------------------------------------------------------
# AgentRunner (full cycle)
# ------------------------------------------------------------------

class TestAgentRunner:
    async def test_disabled(self, tmp_path):
        cfg = _cfg(tmp_path, agent_enabled=False)
        from core.orchestrator import Orchestrator

        orch = Orchestrator(cfg)
        await orch.start()
        runner = AgentRunner(orch)
        with pytest.raises(AgentDisabledError):
            await runner.start()
        await orch.stop()

    async def test_full_cycle(self, tmp_path):
        cfg = _cfg(tmp_path)
        from core.orchestrator import Orchestrator

        class MockEmbedder:
            async def embed(self, t): return [0.1] * 10
            async def embed_batch(self, t): return [[0.1] * 10 for _ in t]

        orch = Orchestrator(cfg, embedder=MockEmbedder())
        await orch.start()

        pool = orch.swarm_pool
        plan_json = json.dumps({"steps": [
            {"action_type": "shell", "command": "echo agent_test", "description": "test echo", "timeout": 5}
        ]})
        approve_json = json.dumps({"approve": True, "concerns": []})

        for n in pool.nodes:
            n.provider = OllamaProvider(ProviderConfig(base_url="http://x"))
            if n.role == NodeRole.PLANNER:
                n.provider._post = AsyncMock(return_value={"response": plan_json, "eval_count": 1})
            elif n.role == NodeRole.CRITIC:
                n.provider._post = AsyncMock(return_value={"response": approve_json, "eval_count": 1})
            else:
                n.provider._post = AsyncMock(return_value={"response": "Agent executed successfully.", "eval_count": 1})

        runner = AgentRunner(orch)
        await runner.start()
        resp = await runner.run("say hello")

        assert isinstance(resp, AgentResponse)
        assert resp.success
        assert len(resp.results) == 1
        assert "agent_test" in resp.results[0].stdout

        # Memory should have been updated
        val = await runner.memory.get("last_goal")
        assert val == "say hello"

        # Action log should be in graph
        stats = await orch.graph.get_stats()
        assert stats.total_nodes > 0

        await orch.stop()


# ------------------------------------------------------------------
# CLI commands exist
# ------------------------------------------------------------------

class TestCLIAgent:
    def test_agent_help(self):
        from interface.cli.app import app
        from typer.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(app, ["agent", "--help"])
        assert result.exit_code == 0
        assert "run" in result.output

    def test_skill_help(self):
        from interface.cli.app import app
        from typer.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(app, ["skill", "--help"])
        assert result.exit_code == 0
        assert "add" in result.output

    def test_memory_help(self):
        from interface.cli.app import app
        from typer.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(app, ["memory", "--help"])
        assert result.exit_code == 0
        assert "show" in result.output
