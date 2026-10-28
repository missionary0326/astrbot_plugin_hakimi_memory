"""Local evidence store. Ollama computes vectors; this database owns persistence."""
import array
import hashlib
import json
import math
import re
import sqlite3
import time
from contextlib import closing, nullcontext
from datetime import datetime, timedelta, timezone
from pathlib import Path

TZ = timezone(timedelta(hours=8))
LAYERS = {0: "00_场景锚点", 1: "01_原文切片", 2: "02_事件时间线", 3: "03_当前事实"}
GROWTH_KEYS = {"称呼习惯", "吐槽尺度", "口头禅", "兴趣表达"}


def day_at(stamp):
    return datetime.fromtimestamp(stamp, TZ).strftime("%Y-%m-%d")


def ident(*values):
    return hashlib.sha256("\0".join(map(str, values)).encode()).hexdigest()[:24]


def clean(text):
    text = re.sub(r"\b(?:sk|AIza)[-_a-zA-Z0-9]{16,}\b", "[密钥已隐藏]", str(text))
    return re.sub(r"(?i)((?:api[_ -]?key|password|密码|token)\s*[:=：]\s*)[^\s，。;]+", r"\1[已隐藏]", text)


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.vault = self.root / "Obsidian"
        (self.vault / "手工笔记").mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "memory.sqlite3")
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
        PRAGMA journal_mode=WAL;
        PRAGMA foreign_keys=ON;
        CREATE TABLE IF NOT EXISTS sources(
          id TEXT PRIMARY KEY, scope TEXT NOT NULL, platform_id TEXT NOT NULL,
          role TEXT NOT NULL, kind TEXT NOT NULL, text TEXT NOT NULL,
          stamp REAL NOT NULL, day TEXT NOT NULL, ready REAL NOT NULL,
          processed INTEGER NOT NULL DEFAULT 0, forgotten INTEGER NOT NULL DEFAULT 0,
          turn_id TEXT NOT NULL DEFAULT '',
          UNIQUE(scope, platform_id));
        CREATE INDEX IF NOT EXISTS source_scope ON sources(scope, stamp);
        CREATE TABLE IF NOT EXISTS items(
          id TEXT PRIMARY KEY, scope TEXT NOT NULL, layer INTEGER NOT NULL,
          text TEXT NOT NULL, subject TEXT NOT NULL DEFAULT '', attribute TEXT NOT NULL DEFAULT '',
          sources TEXT NOT NULL, stamp REAL NOT NULL, active INTEGER NOT NULL DEFAULT 1,
          confidence REAL NOT NULL DEFAULT 1);
        CREATE INDEX IF NOT EXISTS item_scope ON items(scope, layer, active);
        CREATE VIRTUAL TABLE IF NOT EXISTS item_fts USING fts5(id UNINDEXED, text, tokenize='trigram');
        CREATE TABLE IF NOT EXISTS vectors(
          item_id TEXT PRIMARY KEY REFERENCES items(id) ON DELETE CASCADE,
          model TEXT NOT NULL, dim INTEGER NOT NULL, vector BLOB NOT NULL);
        CREATE TABLE IF NOT EXISTS jobs(
          id TEXT PRIMARY KEY, kind TEXT NOT NULL, scope TEXT NOT NULL, payload TEXT NOT NULL,
          attempts INTEGER NOT NULL DEFAULT 0, next_at REAL NOT NULL DEFAULT 0,
          status TEXT NOT NULL DEFAULT 'pending', error TEXT NOT NULL DEFAULT '');
        CREATE TABLE IF NOT EXISTS diaries(
          scope TEXT NOT NULL, day TEXT NOT NULL, text TEXT NOT NULL, sources TEXT NOT NULL,
          revision INTEGER NOT NULL DEFAULT 1, stamp REAL NOT NULL,
          PRIMARY KEY(scope, day));
        CREATE TABLE IF NOT EXISTS growth(
          id INTEGER PRIMARY KEY AUTOINCREMENT, scope TEXT NOT NULL, attribute TEXT NOT NULL,
          value TEXT NOT NULL, previous TEXT NOT NULL, reason TEXT NOT NULL,
          sources TEXT NOT NULL, stamp REAL NOT NULL, active INTEGER NOT NULL DEFAULT 1);
        CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)
        self.db.execute("UPDATE jobs SET status='pending' WHERE status='running'")
        self.db.commit()
        ledger = self.root / "遗忘记录.json"
        if ledger.exists():
            for sid in json.loads(ledger.read_text("utf-8")):
                row = self.db.execute("SELECT scope,platform_id FROM sources WHERE id=? AND forgotten=0", (sid,)).fetchone()
                if row:
                    self.forget(row["scope"], platform_id=row["platform_id"])

    def setting(self, key, default=None):
        row = self.db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_setting(self, key, value):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO settings VALUES(?,?)", (key, json.dumps(value)))

    def record(self, scope, platform_id, role, text, kind="chat", stamp=None, settle=4):
        text = clean(text).strip()
        if not text:
            return None
        stamp = stamp or time.time()
        sid = ident(scope, platform_id)
        row = self.db.execute("SELECT text,forgotten FROM sources WHERE id=?", (sid,)).fetchone()
        if row:
            return None if row["forgotten"] else sid
        with self.db:
            self.db.execute("INSERT INTO sources(id,scope,platform_id,role,kind,text,stamp,day,ready) VALUES(?,?,?,?,?,?,?,?,?)",
                            (sid, scope, str(platform_id), role, kind, text, stamp, day_at(stamp), time.time() + settle))
        return sid

    def source_rows(self, ids):
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        return [dict(r) for r in self.db.execute(f"SELECT * FROM sources WHERE id IN ({marks}) AND forgotten=0 ORDER BY stamp", ids)]

    def add_item(self, scope, layer, text, source_ids, subject="", attribute="", confidence=1, stamp=None, transaction=True):
        text = clean(text).strip()
        source_ids = sorted(set(source_ids))
        rows = self.source_rows(source_ids)
        if not text or not source_ids or len(rows) != len(source_ids) or any(r["scope"] != scope for r in rows):
            raise ValueError("Memory evidence must exist in the same scope")
        iid = ident(scope, layer, subject, attribute, text, *source_ids)
        stamp = stamp or max(r["stamp"] for r in rows)
        with self.db if transaction else nullcontext():
            if layer == 3:
                self.db.execute("UPDATE items SET active=0 WHERE scope=? AND layer=3 AND subject=? AND attribute=? AND id<>?",
                                (scope, subject, attribute, iid))
            self.db.execute("INSERT OR IGNORE INTO items VALUES(?,?,?,?,?,?,?,?,1,?)",
                            (iid, scope, layer, text, subject, attribute, json.dumps(source_ids), stamp, confidence))
            self.db.execute("DELETE FROM item_fts WHERE id=?", (iid,))
            self.db.execute("INSERT INTO item_fts VALUES(?,?)", (iid, text))
        return iid

    def index_ready_sources(self):
        rows = self.db.execute("""SELECT s.* FROM sources s WHERE forgotten=0 AND ready<=?
          AND NOT EXISTS(SELECT 1 FROM items i WHERE i.layer=1 AND i.scope=s.scope
          AND EXISTS(SELECT 1 FROM json_each(i.sources) WHERE value=s.id))
          ORDER BY stamp LIMIT 30""", (time.time(),)).fetchall()
        for row in rows:
            text = row["text"]
            for n, start in enumerate(range(0, len(text), 720)):
                part = text[max(0, start - 100): start + 720]
                self.add_item(row["scope"], 1, f"{row['role']}：{part}", [row["id"]], stamp=row["stamp"])
            if row["kind"] in {"chat", "manual"} and row["role"] == "user":
                self.add_item(row["scope"], 0, f"{datetime.fromtimestamp(row['stamp'], TZ):%m-%d %H:%M} 用户话题：{text[:100]}", [row["id"]])
        return len(rows)

    def missing_vectors(self, model, limit=8):
        return [dict(r) for r in self.db.execute("""SELECT i.id,i.text FROM items i
          LEFT JOIN vectors v ON v.item_id=i.id WHERE i.active=1 AND i.layer>0
          AND (v.item_id IS NULL OR v.model<>?) ORDER BY i.layer=1 DESC,i.stamp DESC LIMIT ?""", (model, limit))]

    def save_vector(self, iid, model, vector):
        if not vector or not all(isinstance(x, (float, int)) and math.isfinite(x) for x in vector):
            raise ValueError("Invalid embedding")
        norm = math.sqrt(sum(x * x for x in vector))
        if not norm:
            raise ValueError("Zero embedding")
        vector = [x / norm for x in vector]
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO vectors VALUES(?,?,?,?)", (iid, model, len(vector), array.array("f", vector).tobytes()))

    def search(self, scope, query, vector=None, model="", limit=6):
        ranks = {}
        terms = re.findall(r"[\w\u4e00-\u9fff]+", query.lower())
        terms = [t[:24] for t in terms if len(t) >= 2][:12]
        if terms:
            expression = " OR ".join('"' + t.replace('"', '') + '"' for t in terms if len(t) >= 3)
            lexical = []
            if expression:
                lexical = self.db.execute("""SELECT i.* FROM item_fts f JOIN items i ON i.id=f.id
                    WHERE item_fts MATCH ? AND i.scope=? AND i.active=1 ORDER BY rank LIMIT 24""", (expression, scope)).fetchall()
            # Two-character Chinese terms are not indexed by the trigram tokenizer.
            if len(lexical) < 6:
                clauses = " OR ".join("text LIKE ? ESCAPE '\\'" for _ in terms)
                likes = ["%" + t.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%" for t in terms]
                extra = self.db.execute(f"SELECT * FROM items WHERE scope=? AND active=1 AND ({clauses}) ORDER BY stamp DESC LIMIT 24", [scope, *likes]).fetchall()
                seen = {r["id"] for r in lexical}
                lexical = [*lexical, *(r for r in extra if r["id"] not in seen)]
            for index, row in enumerate(lexical):
                ranks[row["id"]] = [1 / (60 + index + 1), dict(row)]
        if vector:
            norm = math.sqrt(sum(x * x for x in vector))
            vector = [x / norm for x in vector] if norm else []
            scored = []
            for row in self.db.execute("SELECT i.*,v.vector,v.dim FROM items i JOIN vectors v ON i.id=v.item_id WHERE i.scope=? AND i.active=1 AND v.model=?", (scope, model)):
                if row["dim"] != len(vector):
                    continue
                values = array.array("f")
                values.frombytes(row["vector"])
                score = sum(x * y for x, y in zip(values, vector))
                if score >= .35:
                    data = dict(row)
                    del data["vector"]
                    del data["dim"]
                    scored.append((score, data))
            for index, (_, row) in enumerate(sorted(scored, key=lambda t: t[0], reverse=True)[:24]):
                entry = ranks.setdefault(row["id"], [0, row])
                entry[0] += 1 / (60 + index + 1)
        result, used_sources, length = [], set(), 0
        for score, row in sorted(ranks.values(), key=lambda t: t[0] + (.006 if t[1]['layer'] == 3 else 0), reverse=True):
            source_ids = set(json.loads(row["sources"]))
            if source_ids <= used_sources or length + len(row["text"]) > 5000:
                continue
            result.append(row)
            length += len(row["text"])
            used_sources.update(source_ids)
            if len(result) >= limit:
                break
        return result

    def pending_sources(self, scope, limit=40):
        return [dict(r) for r in self.db.execute("""SELECT * FROM sources WHERE scope=? AND processed=0
          AND forgotten=0 AND ready<=? AND kind IN ('chat','manual') ORDER BY stamp LIMIT ?""", (scope, time.time(), limit))]

    def enqueue(self, kind, scope, payload):
        jid = ident(kind, scope, json.dumps(payload, sort_keys=True))
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO jobs(id,kind,scope,payload) VALUES(?,?,?,?)", (jid, kind, scope, json.dumps(payload)))
        return jid

    def next_job(self):
        row = self.db.execute("SELECT * FROM jobs WHERE status='pending' AND next_at<=? ORDER BY kind='summary' DESC,rowid LIMIT 1", (time.time(),)).fetchone()
        if not row:
            return None
        with self.db:
            self.db.execute("UPDATE jobs SET status='running' WHERE id=?", (row["id"],))
        return dict(row)

    def finish_job(self, jid, error=None):
        with self.db:
            if error:
                row = self.db.execute("SELECT attempts FROM jobs WHERE id=?", (jid,)).fetchone()
                count = row[0] + 1
                self.db.execute("UPDATE jobs SET status='pending',attempts=?,next_at=?,error=? WHERE id=?",
                                (count, time.time() + min(3600, 60 * 2 ** min(count, 6)), str(error)[:80], jid))
            else:
                self.db.execute("UPDATE jobs SET status='done',error='' WHERE id=?", (jid,))

    def commit_summary(self, scope, payload, allowed_ids):
        rows = self.source_rows(allowed_ids)
        if len(rows) != len(allowed_ids):
            raise ValueError("Evidence changed during extraction")
        user_ids = {r["id"] for r in rows if r["role"] == "user" and r["kind"] in {"chat", "manual"}}
        allowed = set(allowed_ids)
        for entry in payload.get("memories", []):
            refs = entry.get("source_ids", [])
            layer = entry.get("layer")
            if layer not in (0, 2, 3) or not refs or not set(refs) <= allowed:
                raise ValueError("Invalid extracted memory or source")
            if layer == 3 and (not set(refs) <= user_ids or entry.get("certainty") != "explicit"):
                raise ValueError("Current facts require explicit user evidence")
            if layer == 3 and (not entry.get("subject") or not entry.get("attribute")):
                raise ValueError("Facts need a stable subject and attribute")
            if layer == 3 and entry.get("subject") == "bot":
                raise ValueError("Core persona is not an extracted user fact")
            confidence = float(entry.get("confidence", .8))
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("Invalid confidence")
            if len(str(entry.get("text", ""))) > 1000:
                raise ValueError("Oversized memory")
        with self.db:
            for entry in payload.get("memories", [])[:18]:
                self.add_item(scope, entry["layer"], entry["text"], entry["source_ids"], entry.get("subject", ""), entry.get("attribute", ""), float(entry.get("confidence", .8)), transaction=False)
            if allowed_ids:
                marks = ",".join("?" * len(allowed_ids))
                self.db.execute(f"UPDATE sources SET processed=1 WHERE id IN ({marks})", allowed_ids)
        return len(payload.get("memories", []))

    def accept_growth(self, scope, proposals):
        if self.setting("growth_paused", False):
            return
        today = day_at(time.time())
        existing = self.db.execute("SELECT stamp FROM growth WHERE scope=? ORDER BY id DESC LIMIT 1", (scope,)).fetchone()
        if existing and day_at(existing[0]) == today:
            return
        turns = self.db.execute("SELECT COUNT(*) FROM sources WHERE scope=? AND role='assistant' AND kind='chat' AND forgotten=0", (scope,)).fetchone()[0]
        if turns < 20:
            return
        for proposal in proposals:
            key = proposal.get("attribute")
            refs = list(set(proposal.get("source_ids", [])))
            evidence = self.source_rows(refs)
            if key not in GROWTH_KEYS or len(evidence) < 3 or any(r["scope"] != scope or r["role"] != "user" or r["kind"] != "chat" for r in evidence):
                continue
            if len({r["turn_id"] or r["platform_id"] for r in evidence}) < 3:
                continue
            value = clean(proposal.get("value", ""))[:200]
            old = self.db.execute("SELECT value FROM growth WHERE scope=? AND attribute=? AND active=1 ORDER BY id DESC LIMIT 1", (scope, key)).fetchone()
            if not value or (old and old[0] == value):
                continue
            with self.db:
                self.db.execute("UPDATE growth SET active=0 WHERE scope=? AND attribute=?", (scope, key))
                self.db.execute("INSERT INTO growth(scope,attribute,value,previous,reason,sources,stamp) VALUES(?,?,?,?,?,?,?)",
                                (scope, key, value, old[0] if old else "", clean(proposal.get("reason", ""))[:300], json.dumps(refs), time.time()))
            break

    def growth_text(self, scope):
        return "\n".join(f"{r['attribute']}：{r['value']}" for r in self.db.execute("SELECT * FROM growth WHERE scope=? AND active=1 ORDER BY id", (scope,)))[:600]

    def undo_growth(self, scope):
        row = self.db.execute("SELECT * FROM growth WHERE scope=? AND active=1 ORDER BY id DESC LIMIT 1", (scope,)).fetchone()
        if not row:
            return False
        with self.db:
            self.db.execute("UPDATE growth SET active=0 WHERE id=?", (row["id"],))
            old = self.db.execute("SELECT id FROM growth WHERE scope=? AND attribute=? AND id<? ORDER BY id DESC LIMIT 1", (scope, row["attribute"], row["id"])).fetchone()
            if old:
                self.db.execute("UPDATE growth SET active=1 WHERE id=?", (old[0],))
        return True

    def diary_sources(self, scope, day):
        return [dict(r) for r in self.db.execute("SELECT * FROM sources WHERE scope=? AND day=? AND forgotten=0 AND kind IN ('chat','proactive','manual') ORDER BY stamp", (scope, day))]

    def save_diary(self, scope, day, text, refs):
        if len(self.source_rows(refs)) != len(refs):
            raise ValueError("Diary evidence changed")
        with self.db:
            self.db.execute("""INSERT INTO diaries VALUES(?,?,?,?,1,?) ON CONFLICT(scope,day)
              DO UPDATE SET text=excluded.text,sources=excluded.sources,revision=diaries.revision+1,stamp=excluded.stamp""",
                            (scope, day, clean(text), json.dumps(refs), time.time()))

    def forget(self, scope, iid=None, platform_id=None):
        refs = []
        if iid:
            row = self.db.execute("SELECT sources FROM items WHERE id=? AND scope=?", (iid, scope)).fetchone()
            refs = json.loads(row[0]) if row else []
        if platform_id:
            row = self.db.execute("SELECT id FROM sources WHERE scope=? AND platform_id=?", (scope, str(platform_id))).fetchone()
            refs = [row[0]] if row else []
        if not refs:
            return 0
        affected = []
        ref_set = set(refs)
        with self.db:
            for row in self.db.execute("SELECT id,sources FROM items WHERE scope=?", (scope,)).fetchall():
                if ref_set & set(json.loads(row["sources"])):
                    affected.append(row["id"])
                    self.db.execute("DELETE FROM item_fts WHERE id=?", (row["id"],))
                    self.db.execute("DELETE FROM items WHERE id=?", (row["id"],))
            for sid in refs:
                self.db.execute("UPDATE sources SET text='',forgotten=1,processed=1 WHERE id=?", (sid,))
            for table in ("growth", "diaries"):
                for row in self.db.execute(f"SELECT rowid,sources FROM {table} WHERE scope=?", (scope,)).fetchall():
                    if ref_set & set(json.loads(row["sources"])):
                        self.db.execute(f"DELETE FROM {table} WHERE rowid=?", (row["rowid"],))
            for row in self.db.execute("SELECT id,payload FROM jobs WHERE scope=? AND status<>'done'", (scope,)).fetchall():
                if ref_set & set(json.loads(row["payload"]).get("source_ids", [])):
                    self.db.execute("DELETE FROM jobs WHERE id=?", (row["id"],))
        # Restore tooling must apply this source-ID ledger before rebuilding any index.
        ledger = self.root / "遗忘记录.json"
        previous = json.loads(ledger.read_text("utf-8")) if ledger.exists() else []
        ledger.write_text(json.dumps(sorted(set(previous) | set(refs))), encoding="utf-8")
        self.export(scope)
        return len(affected)

    def export(self, scope):
        for layer, folder in LAYERS.items():
            directory = self.vault / folder
            directory.mkdir(exist_ok=True)
            keep = set()
            for row in self.db.execute("SELECT * FROM items WHERE scope=? AND layer=? AND active=1", (scope, layer)):
                filename = row["id"] + ".md"
                keep.add(filename)
                body = f"---\nmemory_id: {row['id']}\nlayer: {layer}\nsource_ids: {row['sources']}\n---\n\n{row['text']}\n"
                path = directory / filename
                if not path.exists() or path.read_text("utf-8") != body:
                    path.write_text(body, encoding="utf-8")
            for path in directory.glob("*.md"):
                if path.name not in keep:
                    path.unlink()
        diary_dir = self.vault / "04_日记"
        diary_dir.mkdir(exist_ok=True)
        days = set()
        for row in self.db.execute("SELECT * FROM diaries WHERE scope=?", (scope,)):
            name = row["day"] + ".md"
            days.add(name)
            (diary_dir / name).write_text(f"---\ntype: role_diary\nrevision: {row['revision']}\nsource_ids: {row['sources']}\n---\n\n{row['text']}\n", encoding="utf-8")
        for path in diary_dir.glob("*.md"):
            if path.name not in days:
                path.unlink()
        (self.vault / "05_成长记录.md").write_text(self.growth_text(scope) or "尚无自动成长记录。", encoding="utf-8")
        (self.vault / "README.md").write_text("# 哈基米记忆\n\n00～05 是自动导出，修改会被覆盖。将这个目录作为 Obsidian 仓库打开即可阅读。\n\n自己的 Markdown 笔记放入 `手工笔记`，每分钟增量检查；仅作为外部资料，不自动成为个人事实或日记经历。\n\n原始消息及可恢复的索引保存在上一级 memory.sqlite3。日记是角色感想，不能当作已确认事实。\n", encoding="utf-8")

    def backup(self):
        directory = self.root / "备份"
        directory.mkdir(exist_ok=True)
        today = day_at(time.time())
        target = directory / (today + ".sqlite3")
        if not target.exists():
            with closing(sqlite3.connect(target)) as out:
                self.db.backup(out)
        for path in directory.glob("*.sqlite3"):
            if path.stem < (datetime.now(TZ) - timedelta(days=7)).strftime("%Y-%m-%d"):
                path.unlink()

    def status(self, scope):
        layers = {str(r[0]): r[1] for r in self.db.execute("SELECT layer,COUNT(*) FROM items WHERE scope=? AND active=1 GROUP BY layer", (scope,))}
        return {"sources": self.db.execute("SELECT COUNT(*) FROM sources WHERE scope=? AND forgotten=0", (scope,)).fetchone()[0],
                "layers": layers, "vectors": self.db.execute("SELECT COUNT(*) FROM vectors v JOIN items i ON i.id=v.item_id WHERE i.scope=? AND i.active=1", (scope,)).fetchone()[0],
                "pending_jobs": self.db.execute("SELECT COUNT(*) FROM jobs WHERE scope=? AND status<>'done'", (scope,)).fetchone()[0]}
