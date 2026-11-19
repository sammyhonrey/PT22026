"""Conformance corpus for RiskPilot's grammar (src/assistant.py).

One row per sentence an analyst would actually type, with the answer it must
get. It is deliberately not organised by the parser's rules -- it is organised
by *intent and phrasing*, so a gap in the grammar shows up as a failing row
instead of as a bug report from the person using it.

Three properties are checked, and they are the three that broke in real use:
  * the sentence is read as the request it is;
  * a question about closing gets an answer that agrees with its own advice
    ("Yes, I would." in front of "I'd call this confirmed fraud", never
    "Not yet.");
  * a second instruction in the same message is never carried out silently.
"""
from types import SimpleNamespace as N

import pytest

from src.assistant import RiskPilot, _topic, parse
from src.cases import CaseStore
from src.proposals import ProposalStore
from src.tools import Toolkit

# --------------------------------------------------------------------------- #
# The corpus: (sentence, family, acceptable intents, acceptable topics)
# --------------------------------------------------------------------------- #
CASES: list[tuple[str, str, tuple[str, ...], tuple[str, ...]]] = []


def add(family, kinds, topics, *messages):
    for message in messages:
        CASES.append((message, family, tuple(kinds), tuple(topics)))


Q = "question"

# -- explanation -------------------------------------------------------------
add("why", [Q], ("why",),
    "why was it flagged?", "why did this trigger an alert?", "why is this suspicious?",
    "explain the alert", "what's the reason for the alert?", "why did it come to me?",
    "why this case?", "what is the reason?", "why now?",
    "why was this sent for review?", "why is the risk so high?")
add("why", [Q], ("why", ""),
    "what made you flag this?", "what triggered it?", "what stands out here?")

# -- recommendation and judgement --------------------------------------------
add("recommend", [Q], ("recommend",),
    "what should I do?", "what do you recommend?", "what would you do?",
    "should I escalate this?", "your call?", "what's your advice?", "next steps?",
    "what should I do with this one?", "do you think I should close it as fraud?",
    "should I close it?", "should I mark it as false positive?",
    "should I close it as confirmed fraud?", "should I close it as a false positive?")
add("judge", [Q], ("judge",),
    "is it fraud?", "how bad is it?", "should I worry about this?",
    "is it safe to approve?", "does this look like fraud?")
add("judge", [Q], ("judge", ""),
    "is this a false positive?")

# -- the three fields --------------------------------------------------------
add("resolution", [Q], ("resolution",),
    "what's the resolution?", "should I change the resolution?",
    "what verdict should I record?", "what does the resolution say?",
    "is closing it the right call?", "should this be closed as fraud?")
add("resolution", [Q], ("vocabulary",),
    "what's the difference between closing and resolving?")
add("status", [Q], ("status",),
    "what's the status?", "is it still open?", "what priority is it?",
    "is it closed?", "what state is it in?")
add("history", [Q], ("history",),
    "what changed?", "when was it last updated?",
    "has anyone worked on it?", "who touched this case?")

# -- the transaction ---------------------------------------------------------
add("transaction", [Q], ("transaction",),
    "how much was the transaction?", "what was the amount?", "who sent the money?",
    "what was the receiver's balance?", "show me the transaction details",
    "what time was it?", "what type of transfer is it?")
add("related", [Q], ("related",),
    "any related transactions?", "are there similar cases?", "is it linked to anything?",
    "same account activity?", "any look-alikes?", "is this part of a ring?")
add("related", [Q], ("related", ""),
    "what else looks like it?")

# -- policy and score --------------------------------------------------------
add("policy", [Q], ("policy",),
    "did the policy block it?", "what threshold applies?", "what did the policy decide?",
    "was it approved or reviewed?", "what's the auto-block threshold?")
add("policy", [Q], ("policy", "why"),
    "why was it blocked?", "why did the system approve it?")
add("risk", [Q], ("risk",),
    "what's the risk?", "what's the score?", "how risky is it?", "explain the score")

# -- notes and the manual ----------------------------------------------------
add("notes", [Q], ("notes",),
    "what do the notes say?", "any comments on this case?", "show me the notes",
    "did anyone leave a comment?", "what did you write in your notes?")
add("help", ["help"], ("",), "help", "what can you do?", "commands")
add("help", ["help", Q], ("",), "how do I use you?")

# -- social turns ------------------------------------------------------------
add("social", ["greeting"], ("",), "hi", "hello", "good morning", "hey")
add("social", ["ack"], ("",),
    "thanks", "thank you!", "ok", "got it", "perfect, thanks", "ok thanks", "ok then")

# -- stating a verdict -------------------------------------------------------
add("resolution", ["resolution"], ("",),
    "mark it as false positive", "mark it as confirmed fraud", "mark this case as fraud",
    "it's a false positive", "this is confirmed fraud", "flag it as fraudulent",
    "classify it as a false positive", "treat it as fraud",
    "set the resolution to confirmed fraud", "change the resolution to false positive",
    "mark it fraud please", "please mark this as false positive",
    "can you mark this as false positive?", "could you classify this as fraud",
    "clearly a false positive")
add("resolution", ["unknown"], ("",),
    "the customer confirmed it is fraud", "it was fraud")

# -- closing -----------------------------------------------------------------
add("close", ["close_fraud"], ("",),
    "close it as fraud", "close as confirmed fraud", "close it as fraud and make it urgent")
add("close", ["close_false_positive"], ("",),
    "close it as a false positive", "close this as false positive")
add("close", ["close"], ("",),
    "close it", "close the case", "mark it closed", "mark this case as closed",
    "close it out")

# -- priority and reopening --------------------------------------------------
add("priority", ["priority"], ("",),
    "make it urgent", "set priority to high", "raise the priority to medium",
    "bump it to high", "escalate it", "escalate this case", "lower the priority to low",
    "mark it urgent", "this should be high priority", "change the priority to low")
add("reopen", ["reopen"], ("",), "reopen it", "reopen the case", "reopen this case please")

# -- dictating a note --------------------------------------------------------
add("note", ["note"], ("",),
    "add this to the note: customer called", "note: customer confirmed",
    "add a note: the customer called",
    "write a note that the customer confirmed the transfer",
    "append to the notes: called", "put this in the note: test",
    "add this note: customer says legit", "note that the customer called",
    "add to the note customer says it was fraud")

# -- recording the assistant's own reading -----------------------------------
add("adopt_note", ["adopt_note"], ("",),
    "add your note", "save the suggested note", "add the note you suggested",
    "save your suggestion as a note", "put your findings in the note",
    "write up your findings", "record what you found")

# -- a bare answer to a question it just asked -------------------------------
add("answer", ["confirm"], ("",), "yes", "do it", "go ahead", "do the needful")
add("answer", ["decline"], ("",), "no", "don't close it", "don't mark it as fraud",
    "no need to close it")

# -- typing fast -------------------------------------------------------------
add("typo", ["close_fraud"], ("",), "clsoe it as fraud")
add("typo", ["resolution"], ("",), "mark it as fals positive", "mark it as fraudulant")
add("typo", ["close"], ("",), "clsoe it")
add("typo", ["priority"], ("",), "escalte", "set prioritty to high")
add("typo", ["reopen"], ("",), "reopn the case")
add("typo", ["note"], ("",), "add this to the noet: x")

# -- politeness --------------------------------------------------------------
add("polite", ["close_fraud"], ("",), "please close it as fraud")
add("polite", ["close_false_positive"], ("",), "hey, can you close it as false positive")
add("polite", ["resolution"], ("",), "ok, mark it as confirmed fraud")
add("polite", [Q], ("recommend",), "so, what do you recommend?")
add("polite", [Q], ("why",), "and why?")

# -- outside the grammar, and it must stay outside ---------------------------
add("unknown", ["unknown"], ("",),
    "hmm", "asdfgh", "i think it's fraud", "this looks like fraud",
    "looks like a false positive to me")


# --------------------------------------------------------------------------- #
# A case to talk about
# --------------------------------------------------------------------------- #
class Investigation:
    """The shape `InvestigationAgent.investigate` returns, built by hand."""

    agent = "Investigator"

    def __init__(self, kind="escalate", risk=99.2, decision="review"):
        self.risk = risk
        self.policy_decision = decision
        self.evidence = [N(source="live feed")]
        self.explanation = N(reasons_up=[N(text="Amount 1,258,316 (typical 171,704)"),
                                         N(text="Receiver balance 0 (typical 558,465)")],
                             reasons_down=[N(text="Receiving account was empty")])
        self.related = None
        self.gaps = ()
        self.assessment = N(rule="risk 99.2 is at or above the review threshold")
        self.recommendation = N(kind=kind, rule="R3", proposed_priority="urgent",
                                requires_override=False, label=kind, rationale="risk is high")
        self.suggested_note = "Cash out 477,067 at hour 03 -- risk 99.2, review."
        self.transaction = {"id": 986, "type": "CASH_OUT", "amount": 477_066.97, "hour": 3,
                            "oldbalanceOrg": 1_000_000.0, "oldbalanceDest": 0.0}


class World:
    def __init__(self, tmp_path, kind="escalate", decision="review"):
        self.cases = CaseStore(tmp_path / "cases.db")
        self.queue = ProposalStore(tmp_path / "proposals.db")
        self.case = self.cases.create(txn_id=986, risk=95.1, decision="review",
                                     amount=477_066.97, tx_type="CASH_OUT", source="live",
                                     explanation="Receiver balance - Receiver balance 0")
        self.tools = Toolkit(cases=self.cases)
        self.pilot = RiskPilot(self.tools, proposals=self.queue)
        self.pilot.agent.investigate = lambda case_id: Investigation(kind, decision=decision)

    def say(self, message):
        return self.pilot.reply(self.case.case_id, message)


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


# --------------------------------------------------------------------------- #
# 1. It reads the sentence as the request it is
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("message, family, kinds, topics", CASES,
                         ids=[f"{c[1]}:{c[0][:34]}" for c in CASES])
def test_the_grammar_reads_what_an_analyst_would_type(message, family, kinds, topics):
    intent = parse(message)
    assert intent.kind in kinds, f"{family}: {message!r} read as {intent.kind!r}"
    if intent.kind == "question" and topics != ("",):
        topic = _topic(message)
        assert topic in topics, f"{family}: {message!r} answered about {topic!r}"


# --------------------------------------------------------------------------- #
# 2. A close question never contradicts its own advice
# --------------------------------------------------------------------------- #
CLOSE_QUESTIONS = (
    "should I close it?", "should I close this?",
    "should I close it as confirmed fraud?", "should I close it as a false positive?",
    "do you think I should close it as fraud?", "why should I close it as confirmed fraud?",
)


@pytest.mark.parametrize("message", CLOSE_QUESTIONS)
@pytest.mark.parametrize("kind, wording", [
    ("close_fraud", "Yes, I would."),
    ("close_false_positive", "Yes, I would."),
    ("escalate", "Not yet."),
])
def test_a_close_question_agrees_with_the_advice_behind_it(tmp_path, message, kind, wording):
    reply = World(tmp_path, kind=kind).say(message)
    assert reply.text.startswith(wording), reply.text
    assert not reply.text.startswith("Yes, I would. I'd make it"), "advice and prefix disagree"


# --------------------------------------------------------------------------- #
# 3. A second instruction is never silent
# --------------------------------------------------------------------------- #
COMPOUND = (
    "add this to the note: confirmed fraud, close it as confirmed fraud",
    "mark it as fraud and close it and make it urgent",
    "close it as false positive then reopen",
    "add your note and close it as confirmed fraud",
    "set priority to high and close it",
)


@pytest.mark.parametrize("message", COMPOUND)
def test_a_second_instruction_is_said_back_not_swallowed(world, message):
    reply = world.say(message)
    assert "I also heard" in reply.text, reply.text
    assert "one change per message" in reply.text
    # one action, and only one: a note, or a single pending proposal
    if reply.action["kind"] == "proposal":
        assert world.queue.counts()["pending"] == 1


def test_two_halves_of_one_close_are_not_a_second_instruction(world):
    # "mark it false positive, then close it" is a single request; echoing a
    # phantom second instruction here would train the analyst to ignore echoes.
    reply = world.say("mark it false positive, then close it")
    assert "I also heard" not in reply.text, reply.text
    assert reply.intent == "close_false_positive"


@pytest.mark.parametrize("message", [
    "add this to the note: customer will close the account tomorrow",
    "add this to the note: called the customer, they confirmed the transfer",
])
def test_a_note_that_merely_mentions_closing_is_only_a_note(world, message):
    reply = world.say(message)
    assert reply.action["kind"] == "note"
    assert "I also heard" not in reply.text, reply.text


def test_a_note_dictated_inside_quotes_keeps_the_words_without_the_quotes(world):
    reply = world.say('add this note: "This is a confirmed fraud"')
    assert reply.action == {"kind": "note", "note": "This is a confirmed fraud"}


# --------------------------------------------------------------------------- #
# 4. A bare yes or no files nothing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("message", ["yes", "do it", "go ahead"])
def test_yes_asks_which_change_instead_of_picking_one(world, message):
    reply = world.say(message)
    assert reply.intent == "confirm"
    assert reply.action["kind"] == "clarify"
    assert world.queue.counts()["all"] == 0
    assert "close it as confirmed fraud" in reply.text      # tells them how


@pytest.mark.parametrize("message", ["no", "don't close it", "no need to close it"])
def test_no_changes_nothing_and_says_so(world, message):
    before = world.cases.get(world.case.case_id)
    reply = world.say(message)
    assert reply.intent == "decline"
    assert "nothing changes" in reply.text.lower()
    assert world.cases.get(world.case.case_id) == before
    assert world.queue.counts()["all"] == 0
