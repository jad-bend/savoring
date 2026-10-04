import json, os, queue, random, threading, uuid
from datetime import datetime
from zoneinfo import ZoneInfo
import requests
from fastapi import FastAPI, Header, HTTPException, Response
from pydantic import BaseModel
import noticing
from smolagents import ToolCallingAgent, LiteLLMModel, Tool

HERE = os.path.dirname(os.path.abspath(__file__))
SECRET = os.environ.get("KEPT_SECRET", "")
DATA_DIR = os.environ.get("DATA_DIR", ".")
ZONE = ZoneInfo(os.environ.get("KEPT_TZ", "America/New_York"))
MOMENTS = os.path.join(DATA_DIR, "moments.jsonl")
READINGS = os.path.join(DATA_DIR, "readings.jsonl")
TOMORROW = os.path.join(DATA_DIR, "tomorrow.json")

def rules(name):
    """Read a prompt file fresh, so edits apply to the next session without a restart."""
    with open(os.path.join(HERE, name)) as f:
        return f.read()

STOP = object()
ENDED = ("The user ended the session. Do not say anything else. "
         "Save or finish if you can, then stop.")
DONE = {"say": "", "done": True}
TONES = {"curious", "gentle", "warm", "thoughtful", "quiet"}
TONE_HELP = "How to say it. One of: curious, gentle, warm, thoughtful, quiet."
tone_of = {}   # the tone chosen for each spoken line, looked up when the app asks for audio

def load(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

def append(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")

def now():
    return datetime.now(ZONE).replace(tzinfo=None).isoformat(timespec="seconds")

def make_card(position, m):
    return {"position": position, "id": m["id"], "title": m["title"], "suit": m["suit"],
            "body": m["kept"], "time": m["time"]}

class Session:
    def __init__(self):
        self.to_user = queue.Queue()    # lines the agent wants spoken
        self.from_user = queue.Queue()  # what the user said back
        self.turns = []                 # the whole exchange
        self.pending = []               # cards waiting to turn over with the next line
        self.cards = {}                 # cards turned over so far, by position
        self.closed = False
        self.saved = False

    def close(self):
        self.closed = True
        self.from_user.put(STOP)

    def turn_over(self, card):
        self.pending.append(card)
        self.cards[card["position"]] = card

    def speak(self, line, listen, tone):
        if self.closed:
            return ENDED
        tone = tone.strip().lower()
        tone_of[line] = tone if tone in TONES else "curious"
        out = {"say": line, "done": False, "listen": listen}
        if self.pending:
            out["cards"], self.pending = self.pending, []
        self.to_user.put(out)
        reply = self.from_user.get()
        if reply is STOP:
            return ENDED
        self.turns.append({"agent": line, "tone": tone_of[line], "user": reply if listen else None})
        return reply if listen else "said"

class SessionTool(Tool):
    def __init__(self, session):
        super().__init__()
        self.session = session

class Ask(SessionTool):
    name = "ask"
    description = "Speak one short question to the user and return what they say back."
    inputs = {
        "line": {"type": "string", "description": "The question to speak."},
        "tone": {"type": "string", "description": TONE_HELP},
    }
    output_type = "string"

    def forward(self, line: str, tone: str) -> str:
        return self.session.speak(line, True, tone)

class Tell(SessionTool):
    name = "tell"
    description = "Speak one short line to the user without waiting for an answer."
    inputs = {
        "line": {"type": "string", "description": "The line to speak."},
        "tone": {"type": "string", "description": TONE_HELP},
    }
    output_type = "string"

    def forward(self, line: str, tone: str) -> str:
        s = self.session
        if any(c["position"] == "today" for c in s.pending):
            return "Do not open with a statement. Use ask, with a question about one concrete detail."
        return s.speak(line, False, tone)

class SaveMoment(SessionTool):
    name = "save_moment"
    description = (
        "Save the moment once the user has said what they want to keep. "
        "Suits: marveling = awe at something in the world; "
        "thanksgiving = gratitude for something or someone; "
        "basking = pride in something they did; "
        "luxuriating = physical pleasure or comfort."
    )
    inputs = {
        "kept_detail": {"type": "string", "description": "The one detail the user wants to keep, in their exact words."},
        "title": {"type": "string", "description": "A card title naming one concrete thing, like 'The Robin's Nest'."},
        "suit": {"type": "string", "description": "One of: marveling, thanksgiving, basking, luxuriating."},
        "senses": {"type": "string", "description": "Senses the user described, comma separated, from: sight, sound, touch, smell, taste, body."},
    }
    output_type = "string"

    def forward(self, kept_detail: str, title: str, suit: str, senses: str) -> str:
        s = self.session
        if s.saved:
            return "already saved"
        append(MOMENTS, {
            "id": uuid.uuid4().hex[:8],
            "time": now(),
            "source": "press",
            "title": title,
            "suit": suit.strip().lower(),
            "kept": kept_detail,
            "senses": [x.strip().lower() for x in senses.split(",") if x.strip()],
            "turns": s.turns,
        })
        s.saved = True
        return "saved"

class GetDeck(SessionTool):
    name = "get_deck"
    description = "List the user's kept moments: today's, and older ones."
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        today = datetime.now(ZONE).date().isoformat()
        brief = lambda m: {k: m.get(k) for k in ("id", "time", "title", "suit", "kept", "senses")}
        rows = load(MOMENTS)
        return json.dumps({
            "today": [brief(m) for m in rows if m["time"].startswith(today)],
            "older": [brief(m) for m in rows if not m["time"].startswith(today)],
        })

class Deal(SessionTool):
    name = "deal"
    description = ("Shuffle and deal tonight's cards. Give candidate moment ids for each "
                   "position; one is drawn at random for each. Both cards turn over with "
                   "your first question. Returns the two moments that were drawn.")
    inputs = {
        "today_ids": {"type": "string", "description": "Comma-separated ids of today's moments."},
        "echo_ids": {"type": "string", "description": "Comma-separated ids of related older moments. May be empty."},
    }
    output_type = "string"

    def forward(self, today_ids: str, echo_ids: str) -> str:
        rows = {m["id"]: m for m in load(MOMENTS)}
        def ids(text):
            return [i.strip() for i in text.split(",") if i.strip() in rows]
        todays = ids(today_ids)
        if not todays:
            return "None of those ids exist. Call get_deck and use ids from it."
        t = random.choice(todays)
        others = [i for i in ids(echo_ids) if i != t] or [i for i in rows if i != t]
        e = random.choice(others) if others else None
        s = self.session
        s.turn_over(make_card("today", rows[t]))
        if e:
            s.turn_over(make_card("echo", rows[e]))
        def detail(m):
            return {"title": m["title"], "kept": m["kept"], "suit": m["suit"],
                    "senses": m["senses"], "time": m["time"],
                    "said": [x["user"] for x in m.get("turns", []) if x.get("user")]}
        return json.dumps({"today": detail(rows[t]), "echo": detail(rows[e]) if e else None})

class ShowTomorrow(SessionTool):
    name = "show_tomorrow"
    description = "Turn over the Dawn card. It turns as your next line is spoken."
    inputs = {"text": {"type": "string", "description": "One small, concrete thing to look for or do tomorrow."}}
    output_type = "string"

    def forward(self, text: str) -> str:
        self.session.turn_over({"position": "tomorrow", "title": "", "suit": "", "body": text})
        return "The card will turn over with your next line."

class FinishReading(SessionTool):
    name = "finish_reading"
    description = "Record the reading once the Dawn card is settled."
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        s = self.session
        if s.saved:
            return "already saved"
        append(READINGS, {"time": now(), "cards": s.cards, "turns": s.turns})
        if "tomorrow" in s.cards:
            with open(TOMORROW, "w") as f:
                json.dump({"set": now(), "text": s.cards["tomorrow"]["body"], "noticed": None}, f)
        s.saved = True
        return "saved"

model = LiteLLMModel(model_id="anthropic/claude-haiku-4-5-20251001")
current = None
lock = threading.Lock()

def run(s, tools, prompt, task, steps):
    try:
        agent = ToolCallingAgent(tools=[t(s) for t in tools], model=model,
                                 instructions=prompt, max_steps=steps)
        agent.run(task)
    finally:
        s.closed = True
        s.to_user.put(DONE)

def begin(tools, prompt, task, steps):
    global current
    with lock:
        if current and not current.closed:
            current.close()
        current = s = Session()
    threading.Thread(target=run, args=(s, tools, prompt, task, steps), daemon=True).start()
    return s.to_user.get()

def check(secret: str):
    if SECRET and secret != SECRET:
        raise HTTPException(status_code=401)

app = FastAPI()
app.include_router(noticing.router)

class Reply(BaseModel):
    text: str

@app.post("/press")
def press(x_kept_secret: str = Header(default="")):
    check(x_kept_secret)
    return begin([Ask, SaveMoment], rules("capture_rules.txt") + noticing.for_capture(),
                 "The user just pressed the button. Guide one capture.", 10)

@app.post("/reading")
def reading(x_kept_secret: str = Header(default="")):
    check(x_kept_secret)
    return begin([Ask, Tell, GetDeck, Deal, ShowTomorrow, FinishReading, noticing.KeepNotes],
                 rules("reading_rules.txt") + noticing.for_reading(), "It is night. Run tonight's reading.", 40)

@app.post("/reply")
def reply(r: Reply, x_kept_secret: str = Header(default="")):
    check(x_kept_secret)
    s = current
    if s is None or s.closed:
        return DONE
    s.from_user.put(r.text)
    return s.to_user.get()

@app.post("/stop")
def stop(x_kept_secret: str = Header(default="")):
    check(x_kept_secret)
    s = current
    if s and not s.closed:
        s.close()
    return DONE

@app.get("/moments")
def moments(x_kept_secret: str = Header(default="")):
    check(x_kept_secret)
    return load(MOMENTS)

@app.get("/tomorrow")
def tomorrow(x_kept_secret: str = Header(default="")):
    check(x_kept_secret)
    if not os.path.exists(TOMORROW):
        return {}
    with open(TOMORROW) as f:
        return json.load(f)

# --- Voice: turn a line into audio with ElevenLabs ---

ELEVEN_KEY = os.environ.get("ELEVEN_API_KEY", "")
ELEVEN_VOICE = os.environ.get("ELEVEN_VOICE_ID", "")
ELEVEN_MODEL = os.environ.get("ELEVEN_MODEL", "eleven_v4_turbo")
PLAIN_MODEL = "eleven_multilingual_v2"
TAGGED = {"eleven_v4_turbo", "eleven_v3_conversational", "eleven_v3"}  # models that take [cues]

def synthesize(text, model_id):
    return requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE}",
        params={"output_format": "mp3_44100_128"},
        headers={"xi-api-key": ELEVEN_KEY},
        json={"text": text, "model_id": model_id,
              "voice_settings": {"stability": 0.40, "similarity_boost": 0.80}},
        timeout=30,
    )

class Line(BaseModel):
    text: str

@app.post("/voice")
def voice(line: Line, x_kept_secret: str = Header(default="")):
    check(x_kept_secret)
    if not (ELEVEN_KEY and ELEVEN_VOICE):
        raise HTTPException(status_code=503, detail="No voice configured")
    tone = tone_of.get(line.text, "curious")
    cued = f"[calm] [{tone}] {line.text}" if ELEVEN_MODEL in TAGGED else line.text
    r = synthesize(cued, ELEVEN_MODEL)
    if r.status_code != 200 and ELEVEN_MODEL != PLAIN_MODEL:
        print(f"voice: {ELEVEN_MODEL} refused ({r.status_code}): {r.text[:200]}")
        print(f"voice: falling back to {PLAIN_MODEL}")
        r = synthesize(line.text, PLAIN_MODEL)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=r.text[:200])
    return Response(content=r.content, media_type="audio/mpeg")