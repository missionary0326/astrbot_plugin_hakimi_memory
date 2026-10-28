import asyncio
import json
import shutil
import sqlite3
import sys
import tempfile
import time
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from bootstrap import load_plugin_package
load_plugin_package()
from hakimi_memory.store import Store, day_at


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(self.temp.name)
        self.scope = "example:FriendMessage:10001|test"

    def tearDown(self):
        self.store.db.close()
        self.temp.cleanup()

    def source(self, pid="1", text="我喜欢喝咖啡", role="user", scope=None):
        return self.store.record(scope or self.scope, pid, role, text, settle=0)

    def fact(self, sid, text="用户喜欢咖啡"):
        return self.store.add_item(self.scope, 3, text, [sid], "user", "饮品偏好")

    def test_raw_saved_before_api_and_idempotent(self):
        sid = self.source()
        self.assertEqual(sid, self.source())
        self.assertEqual(self.store.status(self.scope)["sources"], 1)
        self.assertEqual(self.store.source_rows([sid])[0]["text"], "我喜欢喝咖啡")

    def test_l0_l1_do_not_wait_for_summary(self):
        self.source()
        self.store.index_ready_sources()
        self.store.index_ready_sources()
        self.assertEqual(self.store.status(self.scope)["layers"], {"0": 1, "1": 1})

    def test_semantic_search_scope_and_restart(self):
        a = self.source()
        item = self.fact(a)
        self.store.save_vector(item, "test", [.9, .1])
        other = self.source("2", "他喜欢茶", scope="other")
        other_item = self.store.add_item("other", 1, "茶", [other])
        self.store.save_vector(other_item, "test", [1, 0])
        self.store.db.close()
        self.store = Store(self.temp.name)
        result = self.store.search(self.scope, "饮料", [1, 0], "test")
        self.assertEqual([r["id"] for r in result], [item])

    def test_keyword_two_character_fallback(self):
        sid = self.source()
        self.fact(sid)
        self.assertTrue(self.store.search(self.scope, "咖啡"))

    def test_current_fact_replaced_with_history_retained(self):
        old = self.fact(self.source())
        new = self.fact(self.source("2", "现在不喝咖啡了"), "用户现在不喝咖啡")
        self.assertEqual(self.store.db.execute("SELECT active FROM items WHERE id=?", (old,)).fetchone()[0], 0)
        self.assertEqual(self.store.db.execute("SELECT active FROM items WHERE id=?", (new,)).fetchone()[0], 1)

    def test_no_unknown_evidence_or_cross_scope(self):
        with self.assertRaises(ValueError):
            self.fact("missing")
        sid = self.source(scope="other")
        with self.assertRaises(ValueError):
            self.fact(sid)

    def test_reject_assistant_as_user_fact_and_all_or_nothing(self):
        user = self.source()
        bot = self.source("2", "你大概喜欢咖啡吧", "assistant")
        payload = {"memories": [
            {"layer": 2, "text": "谈到咖啡", "source_ids": [user]},
            {"layer": 3, "text": "用户喜欢咖啡", "source_ids": [bot], "subject": "user", "attribute": "喜好", "certainty": "explicit"}]}
        with self.assertRaises(ValueError):
            self.store.commit_summary(self.scope, payload, [user, bot])
        self.assertEqual(self.store.status(self.scope)["layers"], {})
        self.assertEqual(len(self.store.pending_sources(self.scope)), 2)

    def test_reject_inferred_and_identity_facts(self):
        sid = self.source()
        for subject, certainty in [("user", "inferred"), ("bot", "explicit")]:
            with self.assertRaises(ValueError):
                self.store.commit_summary(self.scope, {"memories": [{"layer": 3, "text": "猜测", "source_ids": [sid], "subject": subject, "attribute": "身份", "certainty": certainty}]}, [sid])

    def test_summary_commit_marks_only_batch_processed(self):
        sid = self.source()
        other = self.source("2", "明天再聊")
        self.store.commit_summary(self.scope, {"memories": [{"layer": 2, "text": "用户聊过咖啡", "source_ids": [sid]}]}, [sid])
        self.assertEqual([r["id"] for r in self.store.pending_sources(self.scope)], [other])

    def test_forget_clears_derived_vector_diary_and_replay(self):
        sid = self.source()
        item = self.fact(sid)
        self.store.save_vector(item, "test", [1, 0])
        self.store.save_diary(self.scope, day_at(time.time()), "今天聊到咖啡", [sid])
        self.store.export(self.scope)
        self.assertGreater(self.store.forget(self.scope, iid=item), 0)
        self.assertFalse(self.store.search(self.scope, "咖啡", [1, 0], "test"))
        self.assertIsNone(self.source())
        self.assertFalse(list((self.store.vault / "04_日记").glob("*.md")))

    def test_forgotten_backup_cannot_restore_memory(self):
        sid = self.source()
        item = self.fact(sid)
        copy = Path(self.temp.name) / "old.sqlite3"
        with closing(sqlite3.connect(copy)) as out:
            self.store.db.backup(out)
        self.store.forget(self.scope, iid=item)
        self.store.db.close()
        with closing(sqlite3.connect(copy)) as src, closing(sqlite3.connect(Path(self.temp.name) / "memory.sqlite3")) as dst:
            src.backup(dst)
        self.store = Store(self.temp.name)
        self.assertFalse(self.store.search(self.scope, "咖啡"))

    def test_jobs_survive_restart_and_retry_backoff(self):
        sid = self.source()
        jid = self.store.enqueue("summary", self.scope, {"source_ids": [sid]})
        self.store.next_job()
        self.store.db.close()
        self.store = Store(self.temp.name)
        job = self.store.next_job()
        self.assertEqual(job["id"], jid)
        self.store.finish_job(jid, "RateLimitError")
        self.assertIsNone(self.store.next_job())

    def test_growth_needs_three_turns_twenty_replies_and_rollback(self):
        refs = [self.source(str(i), "喜欢你偶尔吐槽") for i in range(3)]
        proposal = [{"attribute": "吐槽尺度", "value": "偶尔轻吐槽", "reason": "多次明确表达", "source_ids": refs}]
        self.store.accept_growth(self.scope, proposal)
        self.assertEqual(self.store.growth_text(self.scope), "")
        for i in range(20):
            self.source("bot" + str(i), "好呀", "assistant")
        with self.store.db:
            self.store.db.execute("UPDATE sources SET turn_id='same-turn' WHERE role='user'")
        self.store.accept_growth(self.scope, proposal)
        self.assertEqual(self.store.growth_text(self.scope), "")
        with self.store.db:
            self.store.db.execute("UPDATE sources SET turn_id=platform_id WHERE role='user'")
        self.store.accept_growth(self.scope, proposal)
        self.assertIn("轻吐槽", self.store.growth_text(self.scope))
        self.assertTrue(self.store.undo_growth(self.scope))
        self.assertEqual(self.store.growth_text(self.scope), "")

    def test_secret_redaction_and_model_switch(self):
        sid = self.source(text="api_key=sk-abcdefghijklmnopqrstuvwxyz")
        self.assertNotIn("abcdefghijklmnopqrstuvwxyz", self.store.source_rows([sid])[0]["text"])
        item = self.fact(sid)
        self.store.save_vector(item, "old", [1, 0])
        self.assertEqual(self.store.missing_vectors("new")[0]["id"], item)


class PluginTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from hakimi_memory.main import HakimiMemory
        self.temp = tempfile.TemporaryDirectory()
        self.cfg = {"data_dir": self.temp.name, "persona_id": "test", "summary_provider_id": "test-model", "diary_provider_id": "diary-model", "enabled": True, "target_umo": "example:FriendMessage:10001", "owner_uid": "10001"}
        self.context = SimpleNamespace(get_config=lambda **kw: {}, llm_generate=AsyncMock())
        self.plugin = HakimiMemory(self.context, self.cfg)

    async def asyncTearDown(self):
        self.plugin.store.db.close()
        self.temp.cleanup()

    def event(self, text="我喜欢喝茶", mid="msg1", proactive=False):
        extras = {}
        event = SimpleNamespace(unified_msg_origin="example:FriendMessage:10001",
                                message_obj=SimpleNamespace(message_id=("proactive:" if proactive else "") + mid, raw_message={}, timestamp=time.time()),
                                get_sender_id=lambda: "10001", get_message_str=lambda: text,
                                get_messages=lambda: [], get_extra=lambda k,d=None: extras.get(k,d),
                                set_extra=lambda k,v: extras.__setitem__(k,v),
                                get_result=lambda: SimpleNamespace(chain=[SimpleNamespace(text="好，我记住了")]))
        return event

    async def test_synthetic_user_is_not_recorded(self):
        event = self.event("模拟用户提示", proactive=True)
        await self.plugin.capture_raw(event)
        self.assertEqual(self.plugin.store.status(self.plugin.scope())["sources"], 0)
        await self.plugin.after_sent(event)
        rows = self.plugin.store.db.execute("SELECT role,kind FROM sources").fetchall()
        self.assertEqual([tuple(r) for r in rows], [("assistant", "proactive")])

    async def test_interrupted_response_not_recorded(self):
        event = self.event()
        event.set_extra("continuous_message_turn_id", "turn")
        await self.plugin.after_sent(event)
        self.assertEqual(self.plugin.store.status(self.plugin.scope())["sources"], 0)
        event.set_extra("continuous_message_committed", True)
        await self.plugin.after_sent(event)
        await self.plugin.after_sent(event)
        self.assertEqual(self.plugin.store.status(self.plugin.scope())["sources"], 1)

    async def test_injection_temp_and_no_repeated_injection(self):
        event = self.event()
        sid = self.plugin.store.record(self.plugin.scope(), "1", "user", "我喜欢喝茶", settle=0)
        self.plugin.store.add_item(self.plugin.scope(), 3, "用户喜欢喝茶", [sid], "user", "饮品偏好")
        self.plugin.retrieve = AsyncMock(return_value=[])
        req = SimpleNamespace(prompt="你好", extra_user_content_parts=[])
        await self.plugin.on_request(event, req)
        await self.plugin.on_request(event, req)
        self.assertEqual(len(req.extra_user_content_parts), 1)
        self.assertTrue(req.extra_user_content_parts[0]._no_save)

    async def test_embedding_failure_keeps_keyword_search(self):
        sid = self.plugin.store.record(self.plugin.scope(), "1", "user", "咖啡", settle=0)
        self.plugin.store.add_item(self.plugin.scope(), 1, "咖啡", [sid])
        self.plugin.embed = AsyncMock(side_effect=ConnectionError())
        self.assertTrue(await self.plugin.retrieve(self.plugin.scope(), "咖啡"))

    async def test_note_incremental_edit_delete_never_becomes_user_fact(self):
        path = self.plugin.store.vault / "手工笔记" / "test.md"
        path.write_text("资料中的第三人喜欢咖啡", encoding="utf-8")
        self.plugin.scan_notes(self.plugin.scope())
        self.plugin.scan_notes(self.plugin.scope())
        self.assertEqual(self.plugin.store.status(self.plugin.scope())["sources"], 1)
        self.assertEqual(self.plugin.store.pending_sources(self.plugin.scope()), [])
        path.write_text("资料已更新", encoding="utf-8")
        self.plugin.scan_notes(self.plugin.scope())
        self.assertEqual(self.plugin.store.status(self.plugin.scope())["sources"], 1)
        path.unlink()
        self.plugin.scan_notes(self.plugin.scope())
        self.assertEqual(self.plugin.store.status(self.plugin.scope())["sources"], 0)

    async def test_summary_and_diary_use_independent_providers(self):
        scope = self.plugin.scope()
        sid = self.plugin.store.record(scope, "1", "user", "喜欢茶", settle=0)
        self.context.llm_generate.return_value = SimpleNamespace(completion_text=json.dumps({"memories": [{"layer": 3, "text": "用户喜欢茶", "source_ids": [sid], "subject": "user", "attribute": "饮品偏好", "certainty": "explicit"}]}))
        jid = self.plugin.store.enqueue("summary", scope, {"source_ids": [sid]})
        await self.plugin.execute_job(self.plugin.store.next_job())
        self.assertEqual(self.context.llm_generate.call_args.kwargs["chat_provider_id"], "test-model")
        self.context.llm_generate.return_value = SimpleNamespace(completion_text="今天和他聊到了茶，感觉挺开心。")
        self.plugin.store.enqueue("diary", scope, {"source_ids": [sid], "day": day_at(time.time())})
        await self.plugin.execute_job(self.plugin.store.next_job())
        self.assertEqual(self.context.llm_generate.call_args.kwargs["chat_provider_id"], "diary-model")
        self.assertTrue(list((self.plugin.store.vault / "04_日记").glob("*.md")))

    async def test_scope_rejects_other_user(self):
        event = self.event()
        event.unified_msg_origin = "example:FriendMessage:other"
        await self.plugin.capture_raw(event)
        self.assertEqual(self.plugin.store.status(self.plugin.scope())["sources"], 0)

    async def test_default_and_incomplete_configuration_do_not_capture(self):
        event = self.event()
        self.cfg.pop("enabled")
        await self.plugin.capture_raw(event)
        self.cfg["enabled"] = True
        self.cfg["owner_uid"] = ""
        await self.plugin.capture_raw(event)
        self.assertEqual(self.plugin.store.status(self.plugin.scope())["sources"], 0)

    async def test_persona_mismatch_and_group_are_rejected(self):
        event = self.event()
        self.context.get_config = lambda **kw: {"agent_runner":{"config":{"persona":{"persona_id":"other"}}}}
        self.assertFalse(self.plugin.allowed(event))
        self.context.get_config = lambda **kw: {}
        self.cfg["target_umo"] = event.unified_msg_origin = "example:GroupMessage:10001"
        self.assertFalse(self.plugin.allowed(event))

    async def test_api_slow_does_not_hold_up_source_indexing(self):
        gate = asyncio.Event()
        entered = asyncio.Event()
        async def slow_api(**kwargs):
            entered.set()
            await gate.wait()
            return SimpleNamespace(completion_text='{"memories":[]}')
        self.context.llm_generate = slow_api
        self.plugin.resolve_model = AsyncMock()
        self.plugin.embed = AsyncMock(side_effect=lambda texts: [[1.0,0.0] for _ in texts])
        sid = self.plugin.store.record(self.plugin.scope(),"old","user","先前消息",settle=0)
        self.plugin.store.enqueue("summary",self.plugin.scope(),{"source_ids":[sid]})
        worker = asyncio.create_task(self.plugin.run_worker())
        jobs = asyncio.create_task(self.plugin.run_jobs())
        try:
            await asyncio.wait_for(entered.wait(),timeout=3)
            new = self.plugin.store.record(self.plugin.scope(),"new","user","API慢时的新消息",settle=0)
            await asyncio.sleep(2.2)
            found = self.plugin.store.db.execute("SELECT 1 FROM items WHERE layer=1 AND sources LIKE ?", ('%'+new+'%',)).fetchone()
            self.assertIsNotNone(found)
            self.assertFalse(gate.is_set())
        finally:
            for task in (worker,jobs):
                task.cancel()
            await asyncio.gather(worker,jobs,return_exceptions=True)

    async def test_explicit_correction_queues_without_ten_turns(self):
        self.plugin.store.record(self.plugin.scope(),"correction","user","更正一下，现在不喝咖啡了",settle=0)
        self.plugin.schedule(self.plugin.scope())
        job = self.plugin.store.next_job()
        self.assertIsNotNone(job)
        self.assertEqual(job["kind"],"summary")


if __name__ == "__main__":
    unittest.main()
