"""
noticing.py: the part of Kept that decides when to reach out.

The phone (or a Shortcut) reports a signal:  POST /signal {"kind": "walk", "detail": "started"}
The model sees the signal plus context and picks ONE action:
    stay_quiet  - do nothing
    mark        - quietly keep a card for tonight's reading
    nudge       - send one short line to the phone
Every decision is saved with its reason:      GET /decisions

Hard limits live in code, not in the prompt. When a nudge is not allowed
(quiet hours, daily budget spent) the model is not even given the nudge tool.
"""
import json
import os
import threading
import uuid
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests
from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel
from smolagents import LiteLLMModel, Tool, ToolCallingAgent

HERE = os.path.dirname(os.path.abspath(__file__))
SECRET = os.environ.get("KEPT_SECRET", "")
DATA_DIR = os.environ.get("DATA_DIR", ".")
ZONE = ZoneInfo(os.environ.get("KEPT_TZ", "America/New_York"))
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "")      # empty = nudges are logged only

MOMENTS = os.path.join(DATA_DIR, "moments.jsonl")
TOMORROW = os.path.join(DATA_DIR, "tomorrow.json")
DECISIONS = os.path.join(DATA_DIR, "decisions.jsonl")
CONTEXT = os.path.join(DATA_DIR, "context.json")
NOTES = os.path.join(DATA_DIR, "notes.txt")

# ---- hard limits (change these, not the prompt) ----
BUDGET = 2            # nudges per day
QUIET_BEFORE = 8      # no nudges before 8 am
QUIET_AFTER = 21      # no nudges from 9 pm
COOLDOWN_MIN = 10     # the model is asked at most once per this many minutes
ANSWER_MIN = 15       # a capture this soon after a nudge counts as answering it
CONTEXT_HOURS = 3     # older signals are dropped from what the model sees

model = LiteLLMModel(model_id="anthropic/claude-haiku-4-5-20251001")
lock = threading.Lock()
router = APIRouter()


# ---------- small helpers ----------

def now():
    return datetime.now(ZONE).replace(tzinfo=None)

def stamp(t):
    return t.isoformat(timespec="seconds")

def parse(s):
    return datetime.fromisoformat(s)

def clock(t):
    return t.strftime("%a %I:%M %p").replace(" 0", " ")

def ago(t, then):
    mins = int((t - then).total_seconds() // 60)
    if mins < 1:
        return "just now"
    if mins < 60:
        return f"{mins} min ago"
    return f"{mins // 60} h {mins % 60} min ago"

def rules(name):
    with open(os.path.join(HERE, name)) as f:
        return f.read()

def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

def write_rows(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

def append_row(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")

def read_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default

def read_notes():
    if not os.path.exists(NOTES):
        return ""
    with open(NOTES) as f:
        return f.read().strip()

def check(secret):
    if SECRET and secret != SECRET:
        raise HTTPException(status_code=401, detail="wrong secret")


# ---------- the action space: one tool per action ----------

class Choice(Tool):
    """Each action writes into a shared box. Only the first choice counts."""
    output_type = "string"

    def __init__(self, box):
        super().__init__()
        self.box = box

    def choose(self, **decision):
        if "action" in self.box:
            return "Already decided. Call final_answer with the word done."
        self.box.update(decision)
        return "Recorded. Call final_answer with the word done."


class StayQuiet(Choice):
    name = "stay_quiet"
    description = "Do nothing. This is the right choice most of the time."
    inputs = {"reason": {"type": "string", "description": "One plain sentence on why."}}

    def forward(self, reason: str) -> str:
        return self.choose(action="quiet", reason=reason)


class Mark(Choice):
    name = "mark"
    description = ("Quietly keep this moment as a card for tonight's reading. "
                   "The person is not contacted.")
    inputs = {
        "reason": {"type": "string", "description": "One plain sentence on why."},
        "title": {"type": "string", "description":
                  "A short card title naming one concrete thing from the signals, "
                  "like 'The River Walk'. Never a feeling."},
    }

    def forward(self, reason: str, title: str) -> str:
        return self.choose(action="mark", reason=reason, title=title)


class Nudge(Choice):
    name = "nudge"
    description = ("Send one short line to the person's phone inviting them to keep "
                   "this moment. Costs one of today's nudges.")
    inputs = {
        "reason": {"type": "string", "description": "One plain sentence on why."},
        "line": {"type": "string", "description":
                 "At most 8 words, shown on the lock screen. No personal details, "
                 "no names of places or people."},
    }

    def forward(self, reason: str, line: str) -> str:
        return self.choose(action="nudge", reason=reason, line=line)


class KeepNotes(Tool):
    """Given to the night reading. This is how the agent carries what it learned
    about HOW to ask into tomorrow. It never stores what moments were about."""
    name = "keep_notes"
    description = ("Save short notes on how to ask this person and when reaching "
                   "out lands well. Replaces yesterday's notes.")
    inputs = {"notes": {"type": "string", "description":
                        "At most five short lines. Only how to ask and when to "
                        "reach out. Never what their moments were about."}}
    output_type = "string"

    def __init__(self, session=None):
        super().__init__()

    def forward(self, notes: str) -> str:
        lines = [l.strip() for l in notes.strip().splitlines() if l.strip()][:5]
        with open(NOTES, "w") as f:
            f.write("\n".join(lines) + "\n")
        return "Notes kept."


# ---------- the observation space: what the model is shown ----------

def outcome(row, t):
    if row.get("action") != "nudge":
        return ""
    if row.get("answered"):
        return " (they answered)"
    if t - parse(row["time"]) > timedelta(minutes=ANSWER_MIN):
        return " (ignored)"
    return " (waiting)"

def why_no_nudge(t, rows):
    """Returns a reason if nudging is off limits right now, else an empty string."""
    if t.hour < QUIET_BEFORE or t.hour >= QUIET_AFTER:
        return "it is quiet hours"
    today = stamp(t)[:10]
    used = sum(1 for r in rows if r.get("action") == "nudge" and r["time"].startswith(today))
    if used >= BUDGET:
        return f"today's {BUDGET} nudges are used"
    return ""

def observations(t, kind, detail, context, rows, blocked):
    today = stamp(t)[:10]
    out = [f"Time: {t.strftime('%A')} {clock(t)[4:]}", f"New signal: {kind}: {detail}".rstrip(": ")]

    recent = []
    for k, v in context.items():
        if k == kind:
            continue
        then = parse(v["time"])
        if t - then <= timedelta(hours=CONTEXT_HOURS):
            recent.append(f"- {k}: {v['detail']} ({ago(t, then)})")
    out.append("Other recent signals:\n" + ("\n".join(recent) if recent else "- none"))

    if blocked:
        out.append(f"Nudging is not available because {blocked}. You can stay quiet or mark.")
    else:
        used = sum(1 for r in rows if r.get("action") == "nudge" and r["time"].startswith(today))
        out.append(f"Nudges left today: {BUDGET - used} of {BUDGET}")

    kept = sum(1 for m in read_rows(MOMENTS) if m.get("time", "").startswith(today))
    out.append(f"Cards already kept today: {kept}")

    dawn = read_json(TOMORROW, {})
    if dawn.get("text"):
        out.append(f"The card they set for themselves last night: \"{dawn['text']}\"")

    past = []
    for r in rows[-8:]:
        s = r["signal"]
        past.append(f"- {clock(parse(r['time']))}, {s['kind']}: {s['detail']} -> "
                    f"{r['action']}{outcome(r, t)}. {r['reason']}")
    out.append("Your recent decisions:\n" + ("\n".join(past) if past else "- none yet"))

    notes = read_notes()
    if notes:
        out.append("Your notes on this person:\n" + notes)
    return "\n\n".join(out)


# ---------- the decision ----------

def send(line):
    if not NTFY_TOPIC:
        return "simulated"
    try:
        r = requests.post(f"https://ntfy.sh/{NTFY_TOPIC}", data=line.encode("utf-8"),
                          headers={"Title": "Kept"}, timeout=10)
        return "sent" if r.ok else f"failed ({r.status_code})"
    except Exception as e:
        return f"failed ({e})"

def decide(kind, detail, dry=False, at=""):
    with lock:
        t = parse(at) if (dry and at) else now()
        context = read_json(CONTEXT, {})
        rows = read_rows(DECISIONS)
        row = {"id": uuid.uuid4().hex[:8], "time": stamp(t),
               "signal": {"kind": kind, "detail": detail}, "by": "model", "answered": None}

        last = next((r for r in reversed(rows) if r.get("by") == "model"), None)
        cooling = last and t - parse(last["time"]) < timedelta(minutes=COOLDOWN_MIN)

        if cooling and not dry:
            row.update(by="rule", action="quiet",
                       reason=f"Already decided in the last {COOLDOWN_MIN} minutes.")
        else:
            blocked = why_no_nudge(t, rows)
            seen = observations(t, kind, detail, context, rows, blocked)
            box = {}
            tools = [StayQuiet(box), Mark(box)] + ([] if blocked else [Nudge(box)])
            try:
                agent = ToolCallingAgent(tools=tools, model=model,
                                         instructions=rules("noticing_rules.txt"), max_steps=3)
                agent.run(seen)
            except Exception as e:
                if "action" not in box:
                    box.update(action="quiet", reason=f"The decision failed: {e}")
            if "action" not in box:
                box.update(action="quiet", reason="No decision was made.")
            row.update(box)
            row["saw"] = seen

        if dry:
            row["dry"] = True
            return row

        context[kind] = {"detail": detail, "time": stamp(t)}
        with open(CONTEXT, "w") as f:
            json.dump(context, f)

        if row["action"] == "mark":
            append_row(MOMENTS, {
                "id": row["id"], "time": row["time"], "source": "noticed",
                "title": row.get("title", "An Unmarked Moment"), "suit": "",
                "kept": f"Noticed: {kind}, {detail}".rstrip(", ") + f", {clock(t)[4:]}.",
                "senses": [], "turns": []})
        if row["action"] == "nudge":
            row["delivery"] = send(row.get("line", ""))

        append_row(DECISIONS, row)
        return row


# ---------- what the other two modes are told ----------

def for_capture():
    """Appended to the capture prompt. Also records that a nudge was answered."""
    text = ""
    with lock:
        t = now()
        rows = read_rows(DECISIONS)
        for r in reversed(rows):
            fresh = t - parse(r["time"]) <= timedelta(minutes=ANSWER_MIN)
            if r.get("action") == "nudge" and not r.get("answered") and fresh:
                r["answered"] = True
                write_rows(DECISIONS, rows)
                text += ("\n\nThey are answering a nudge you sent "
                         f"{ago(t, parse(r['time']))}. You sent it because: {r['reason']} "
                         f"You wrote: \"{r.get('line', '')}\". "
                         "Let your first question fit that, still under ten words.")
                break
    notes = read_notes()
    if notes:
        text += "\n\nYour notes on how to ask this person:\n" + notes
    return text

def for_reading():
    """Appended to the reading prompt: today's decisions and yesterday's notes."""
    t = now()
    today = stamp(t)[:10]
    rows = [r for r in read_rows(DECISIONS)
            if r["time"].startswith(today) and r.get("by") == "model"]
    text = ""
    if rows:
        text += "\n\nWhat you decided today:\n" + "\n".join(
            f"- {clock(parse(r['time']))}, {r['signal']['kind']}: {r['signal']['detail']} -> "
            f"{r['action']}{outcome(r, t)}. {r['reason']}" for r in rows)
    notes = read_notes()
    if notes:
        text += "\n\nYour notes from before:\n" + notes
    return text


# ---------- endpoints ----------

class Signal(BaseModel):
    kind: str             # walk, place, scroll, focus, music, ...
    detail: str = ""      # "started", "arrived at the river", "opened TikTok"
    dry: bool = False     # true = decide and show, but save and send nothing
    at: str = ""          # dry runs only: pretend it is this time, e.g. 2026-10-05T14:30:00

@router.post("/signal")
def signal(s: Signal, x_kept_secret: str = Header(default="")):
    check(x_kept_secret)
    return decide(s.kind.strip().lower(), s.detail.strip(), s.dry, s.at)

@router.get("/decisions")
def decisions(x_kept_secret: str = Header(default="")):
    check(x_kept_secret)
    t = now()
    return [{"time": r["time"], "signal": r["signal"], "by": r["by"],
             "action": r["action"] + outcome(r, t), "reason": r["reason"],
             "line": r.get("line", ""), "title": r.get("title", "")}
            for r in read_rows(DECISIONS)[-50:]]

@router.get("/notes")
def notes(x_kept_secret: str = Header(default="")):
    check(x_kept_secret)
    return {"notes": read_notes()}

@router.post("/import")
async def import_moments(request: Request, x_kept_secret: str = Header(default="")):
    """Adds moments sent as JSONL text. Skips any whose id is already in the deck."""
    check(x_kept_secret)
    body = (await request.body()).decode("utf-8")
    have = {m.get("id") for m in read_rows(MOMENTS)}
    added = 0
    for line in body.splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        rid = row.get("id")
        if rid and rid in have:
            continue
        append_row(MOMENTS, row)
        have.add(rid)
        added += 1
    return {"added": added, "total": len(read_rows(MOMENTS))}