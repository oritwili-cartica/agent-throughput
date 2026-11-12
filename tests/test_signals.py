"""Tests for the session signals (skill, plugin, MCP server, output style names).

Fixtures are tiny synthetic transcripts with fake ids. Sentinel strings stand in
for conversation content that must never reach the metadata or the SQLite file.
"""
import json
import os
import shutil
import tempfile
import unittest

from throughput import ingest, schema
from throughput.adapters import claude_code
from throughput.types import SourceFile

TS = "2026-09-01T10:00:00.000Z"


def _write(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _call(mid, line=None, blocks=None, **extra):
    rec = {"type": "assistant", "timestamp": TS, "sessionId": "s1", "cwd": "/work/proj",
           "uuid": line or f"u-{mid}",
           "message": {"id": mid, "model": "claude-sonnet-5",
                       "usage": {"input_tokens": 1, "output_tokens": 1},
                       "content": blocks or [{"type": "text", "text": "ok"}]}}
    rec.update(extra)
    return rec


def _skill_block(name, args=None):
    inp = {"skill": name}
    if args is not None:
        inp["args"] = args
    return {"type": "tool_use", "id": "t1", "name": "Skill", "input": inp}


def _style(name, uuid):
    return {"type": "attachment", "uuid": uuid, "timestamp": TS,
            "attachment": {"type": "output_style", "style": name}}


class SignalCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="signals-test-")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.path = os.path.join(self.tmp, "claude", "projects", "-work-proj", "s1.jsonl")

    def parse(self, records):
        _write(self.path, records)
        st = os.stat(self.path)
        return claude_code.parse(SourceFile("claude_code", self.path, st.st_size, st.st_mtime_ns, {}))[0]

    def metadata(self, records):
        return self.parse(records).metadata


class SignalSourceTests(SignalCase):
    def test_nothing_used_still_records_version(self):
        self.assertEqual(self.metadata([_call("m1")]), {"signals_version": 1})

    def test_attribution_skill(self):
        md = self.metadata([_call("m1", attributionSkill="gsd-execute-phase")])
        self.assertEqual(md["skills"], {"gsd-execute-phase": 1})

    def test_skill_tool_use_on_later_line_of_same_call(self):
        md = self.metadata([
            _call("m1", line="u1", attributionSkill="caveman"),
            _call("m1", line="u2", blocks=[_skill_block("caveman")]),
        ])
        self.assertEqual(md["skills"], {"caveman": 1})

    def test_two_skills_on_one_call_count_each(self):
        md = self.metadata([
            _call("m1", line="u1", attributionSkill="a"),
            _call("m1", line="u2", blocks=[_skill_block("b")]),
        ])
        self.assertEqual(md["skills"], {"a": 1, "b": 1})

    def test_plugins_counted_per_call(self):
        md = self.metadata([_call(f"m{i}", attributionPlugin="p") for i in range(3)])
        self.assertEqual(md["plugins"], {"p": 3})

    def test_mcp_servers_counted_per_call(self):
        md = self.metadata([_call("m1", attributionMcpServer="linear"),
                            _call("m2", attributionMcpServer="linear")])
        self.assertEqual(md["mcp_servers"], {"linear": 2})

    def test_output_styles_sorted_and_distinct(self):
        md = self.metadata([_style("caveman-terse", "a1"), _style("Explanatory", "a2"),
                            _style("caveman-terse", "a3")])
        self.assertEqual(md["output_styles"], ["Explanatory", "caveman-terse"])

    def test_existing_metadata_keys_kept(self):
        md = self.metadata([_call("m1", attributionSkill="x"),
                            {"type": "cost-state", "totalCostUSD": 1.0, "modelUsage": {}, "startTime": 1}])
        self.assertEqual(md["cost_state"]["totalCostUSD"], 1.0)
        self.assertEqual(md["signals_version"], 1)


class SignalNameTests(SignalCase):
    def test_names_are_cleaned(self):
        records = [_call("m1", attributionSkill=5), _call("m2", attributionSkill="  x  "),
                   _call("m3", attributionSkill=""), _call("m4", attributionSkill="   "),
                   _call("m5", attributionSkill="bad\x07name"), _call("m6", attributionSkill=["x"])]
        self.assertEqual(self.metadata(records)["skills"], {"x": 1})

    def test_long_name_truncated_and_collisions_merge(self):
        base = "a" * 64
        md = self.metadata([_call("m1", attributionSkill="a" * 100),
                            _call("m2", attributionSkill=base + "zzz")])
        self.assertEqual(md["skills"], {base: 2})

    def test_cap_keeps_highest_counts_then_name(self):
        records = []
        for i in range(60):
            name = f"s{i:02d}"
            for n in range(2 if i < 5 else 1):
                records.append(_call(f"m-{name}-{n}", attributionSkill=name))
        skills = self.metadata(records)["skills"]
        self.assertEqual(len(skills), 50)
        self.assertTrue(all(skills[f"s{i:02d}"] == 2 for i in range(5)))
        self.assertIn("s49", skills)
        self.assertNotIn("s50", skills)

    def test_byte_budget_drops_lowest_counts_first(self):
        records = []
        for key in ("attributionSkill", "attributionPlugin", "attributionMcpServer"):
            for i in range(50):
                name = f"{key[-3:]}{i:02d}" + "é" * 58
                for n in range(3 if i == 0 else 1):
                    records.append(_call(f"m-{key}-{i}-{n}", **{key: name}))
        md = self.metadata(records)
        self.assertLess(len(json.dumps(md, sort_keys=True)), 32 * 1024)
        self.assertLess(sum(len(md[key]) for key in ("skills", "plugins", "mcp_servers")), 150)
        for key in ("skills", "plugins", "mcp_servers"):
            self.assertTrue(any(c == 3 for c in md[key].values()))


class SignalPrivacyTests(SignalCase):
    def records(self):
        return [
            {"type": "user", "timestamp": TS, "sessionId": "s1", "cwd": "/work/proj",
             "message": {"role": "user", "content": "SENTINEL-USER-TEXT"}},
            _call("m1", line="u1", attributionSkill="gsd-plan-phase", blocks=[
                {"type": "text", "text": "SENTINEL-ASSISTANT-TEXT"},
                _skill_block("gsd-plan-phase", args="SENTINEL-SKILL-ARGS"),
                {"type": "tool_use", "id": "t2", "name": "Bash", "input": {"command": "echo SENTINEL-BASH-INPUT"}},
            ]),
            {"type": "user", "timestamp": TS, "sessionId": "s1", "cwd": "/work/proj",
             "message": {"role": "user", "content": [
                 {"type": "tool_result", "tool_use_id": "t2", "content": "SENTINEL-TOOL-RESULT"}]}},
        ]

    def test_no_sentinel_in_parsed_metadata(self):
        ps = self.parse(self.records())
        self.assertEqual(ps.metadata["skills"], {"gsd-plan-phase": 1})
        self.assertNotIn("SENTINEL", json.dumps(ps.metadata))

    def test_no_sentinel_in_sqlite_bytes(self):
        _write(self.path, self.records())
        db = os.path.join(self.tmp, "usage.sqlite3")
        conn = schema.connect(db)
        schema.migrate(conn)
        ingest.ingest(conn, {"claude_code": os.path.join(self.tmp, "claude", "projects"),
                             "codex": os.path.join(self.tmp, "none-codex"),
                             "kimi": os.path.join(self.tmp, "none-kimi")})
        stored = conn.execute("SELECT raw_metadata_json FROM sessions").fetchone()[0]
        self.assertIn("gsd-plan-phase", stored)
        conn.close()
        for name in os.listdir(self.tmp):
            if name.startswith("usage.sqlite3"):
                with open(os.path.join(self.tmp, name), "rb") as fh:
                    self.assertNotIn(b"SENTINEL", fh.read())


if __name__ == "__main__":
    unittest.main()
