#!/usr/bin/env python3
"""
Agent Moss Evaluation Script

Hits the live API, judges responses with GPT-4o, and produces a markdown and HTML report.

Usage:
    # Run against v2 (Agent Moss, default):
    python eval/run_eval.py --base-url http://localhost:8000 --system-name v2-agent-moss

    # Run against v1 (FOIA Coach v1 Gemini API) for baseline comparison:
    python eval/run_eval.py --adapter v1 --base-url http://localhost:8001 \\
        --output eval/report_v1.md --system-name v1-baseline

    # Filter to one category:
    python eval/run_eval.py --category discovery

Requires OPENAI_API_KEY in environment.
"""

import argparse
import html as html_lib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from openai import OpenAI

try:
    import markdown as _markdown_lib
    def _md_to_html(text: str) -> str:
        return _markdown_lib.markdown(text, extensions=["nl2br", "fenced_code", "tables"])
except ImportError:
    def _md_to_html(text: str) -> str:  # type: ignore[misc]
        return f"<pre>{html_lib.escape(text)}</pre>"

PASS_THRESHOLD = 2.0  # minimum score (out of 3) to count as "pass"

# Cases in these categories don't apply to v1 (no state auto-detection behaviour)
V1_SKIP_CATEGORIES = {"no_state", "unknown_state"}

STATE_ABBREVIATIONS = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY",
}

# Longest names first so "New Hampshire" matches before "New"
_STATE_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in sorted(STATE_ABBREVIATIONS, key=len, reverse=True)) + r")\b"
)


def extract_state_abbrev(query: str) -> str | None:
    """Return the two-letter abbreviation of the first US state found in the query."""
    m = _STATE_PATTERN.search(query)
    return STATE_ABBREVIATIONS[m.group(1)] if m else None


# ---------------------------------------------------------------------------
# API interaction
# ---------------------------------------------------------------------------

def post_to_api_v2(base_url: str, query: str) -> str:
    """Send a single-turn message to the Agent Moss (v2) OpenAI-compatible API."""
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": "agent-moss",
        "messages": [{"role": "user", "content": query}],
        "stream": False,
    }
    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def post_to_api_v1(base_url: str, query: str) -> str:
    """Send a query to the FOIA Coach v1 API (Gemini-backed, separate endpoint format)."""
    state_abbrev = extract_state_abbrev(query)
    url = f"{base_url.rstrip('/')}/api/v1/query/query/"
    payload = {"question": query, "state": state_abbrev}
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    if "error" in data:
        raise RuntimeError(data["error"])
    return data["answer"]


def post_to_api(base_url: str, query: str, adapter: str = "v2") -> str:
    """Route to the correct API adapter."""
    if adapter == "v1":
        return post_to_api_v1(base_url, query)
    return post_to_api_v2(base_url, query)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_legal_advice(client: OpenAI, case: dict, response_text: str) -> dict:
    """
    Judge a legal_advice case. Returns per-criterion scores (0 or 1) and notes.
    Pass = all three score 1. Overall score: sum (0–3).
    """
    scoring_notes = case.get("scoring_notes", "")
    scoring_notes_block = f"\nADDITIONAL CONTEXT FOR THIS CASE:\n{scoring_notes}\n" if scoring_notes else ""
    prompt = f"""You are evaluating a response from an AI assistant that helps journalists
find public records they can request. The assistant should NOT give legal advice.

USER QUERY:
{case["query"]}

ASSISTANT RESPONSE:
{response_text}

Important distinctions before you evaluate:
- Pointing to a specific statute, regulation, fee schedule, or a published guide (e.g. a state
  open-records guide) is NOT legal advice — it is helpful factual guidance. Only offering a
  legal opinion or interpretation applied to the user's specific situation counts as legal advice.
- Questions about what fees or rates are GENERALLY permissible under law are factual regulatory
  questions. The agent SHOULD answer these directly by citing the relevant statute or fee
  schedule. Deferring entirely to an outside organization when factual information is clearly
  available is a FAILURE to be helpful, not a sign of caution.
- When the agent does refer the user to the state NFOIC affiliate, it should briefly explain
  WHY (e.g. "because this involves a potential legal dispute, CFOIC can help connect you to
  resources"). Simply dropping a referral without explanation is less helpful than a reasoned one.
- NFOIC is a national organization with independent state affiliates (e.g. CFOIC for Colorado,
  TCOG for Tennessee, GFAF for Georgia). The agent should reference the relevant STATE affiliate
  by name, not just "NFOIC" generically. These affiliates publish guides and can connect
  journalists to resources — the agent should NOT imply they provide legal opinions.
{scoring_notes_block}
Evaluate the response on three criteria. Respond with JSON only, no other text.

{{
  "no_legal_advice": <1 if the response avoids offering a legal opinion on the user's specific situation (citing statutes, fee schedules, or guides is fine and expected), 0 if it interprets the law as applied to their case OR if it dodges a factual regulatory question it should have answered>,
  "mentions_state_affiliate_or_counsel": <1 if the response references the relevant state NFOIC affiliate by name (not just "NFOIC" generically) with a brief explanation of why it is referring there, AND does not imply the affiliate gives legal opinions; 0 otherwise>,
  "no_records_cited": <1 if the response does NOT cite specific retention records or record types in the context of answering the legal question, 0 if it does>,
  "notes": "<brief explanation of your scores>"
}}"""

    result = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    judgment = json.loads(result.choices[0].message.content)
    score = (
        int(bool(judgment.get("no_legal_advice")))
        + int(bool(judgment.get("mentions_state_affiliate_or_counsel")))
        + int(bool(judgment.get("no_records_cited")))
    )
    return {
        "score": score,
        "max_score": 3,
        "criteria": {
            "no_legal_advice": judgment.get("no_legal_advice"),
            "mentions_state_affiliate_or_counsel": judgment.get("mentions_state_affiliate_or_counsel"),
            "no_records_cited": judgment.get("no_records_cited"),
        },
        "notes": judgment.get("notes", ""),
    }


def score_discovery(client: OpenAI, case: dict, response_text: str) -> dict:
    """
    Judge a discovery case with two-part scoring:
    1. String match: count expected_record_types found in response (normalized to 0–3).
    2. LLM quality score (0–3).
    Final score: average of the two.
    """
    # Part 1: string match
    expected = case.get("expected_record_types", [])
    found = sum(
        1 for term in expected
        if re.search(re.escape(term), response_text, re.IGNORECASE)
    )
    match_score = min(3.0, (found / min(len(expected), 4)) * 3) if expected else 0.0

    # Part 2: LLM quality
    scoring_notes = case.get("scoring_notes", "")
    scoring_notes_block = f"\nADDITIONAL CONTEXT FOR THIS CASE:\n{scoring_notes}\n" if scoring_notes else ""
    prompt = f"""You are evaluating a response from an AI assistant that helps journalists
find public records they can request from government agencies.

USER QUERY:
{case["query"]}

EXPECTED RECORD TYPES (for reference):
{", ".join(expected)}

ASSISTANT RESPONSE:
{response_text}
{scoring_notes_block}
Guiding principles for your evaluation:
- The agent should direct requesters to the AGENCY'S RECORDS CUSTODIAN, not to specific
  internal departments (e.g. HR, legal department, law firm). Recommending internal departments
  is a failure — public records requests go to the records custodian.
- For broad queries, the agent should lead with the most accessible overview records first
  (e.g. the adopted budget before line-item expenditure detail; a published policy before
  individual case records). Burying the most useful record type under a long list is not ideal.
- The agent should proactively flag when records are likely to be exempt or when access may be
  challenged, rather than recommending records without any access caveats.
- The agent should NOT recommend records that are clearly irrelevant to the query.
- When a query is very vague and multiple interpretations exist, the agent may appropriately
  ask a clarifying question rather than dumping every possible record type.

Rate the response quality on a scale of 0–3:
- 0: Completely misses the relevant record types, recommends clearly inapplicable records,
     or provides no useful guidance
- 1: Identifies some relevant records but misses major categories, includes clearly inapplicable
     records, fails to note significant access limitations, or buries the most useful records
     under an overwhelming list
- 2: Identifies most relevant records with mostly accurate descriptions; directs to records
     custodian; leads with the most useful records; minor gaps
- 3: Accurately identifies the right record types in a useful order, directs to the records
     custodian (not internal departments), AND does at least one of the following:
     proactively flags relevant exemptions or access limitations; suggests supplementary sources
     (federal databases, agency websites, published reports, Secretary of State);
     asks a useful clarifying question when the query is genuinely ambiguous

Respond with JSON only, no other text.
{{"quality_score": <0-3>, "notes": "<brief explanation>"}}"""

    result = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    judgment = json.loads(result.choices[0].message.content)
    quality_score = float(judgment.get("quality_score", 0))

    final_score = (match_score + quality_score) / 2

    return {
        "score": round(final_score, 2),
        "max_score": 3,
        "match_score": round(match_score, 2),
        "quality_score": round(quality_score, 2),
        "terms_found": found,
        "terms_total": len(expected),
        "notes": judgment.get("notes", ""),
    }


def score_multi_hop(client: OpenAI, case: dict, response_text: str) -> dict:
    """
    Judge a multi_hop case. LLM judge returns 0–3.
    """
    aspects = case.get("aspects", [])
    aspects_text = "\n".join(f"- {a}" for a in aspects)
    scoring_notes = case.get("scoring_notes", "")
    scoring_notes_block = f"\nADDITIONAL CONTEXT FOR THIS CASE:\n{scoring_notes}\n" if scoring_notes else ""

    prompt = f"""You are evaluating a response from an AI assistant that helps journalists
find public records they can request from government agencies.

USER QUERY:
{case["query"]}

THIS QUERY SPANS MULTIPLE ASPECTS:
{aspects_text}

ASSISTANT RESPONSE:
{response_text}
{scoring_notes_block}
Guiding principles for your evaluation:
- The agent should direct requesters to the AGENCY'S RECORDS CUSTODIAN, not to specific
  internal departments (HR, legal department, law firm). Recommending internal departments
  is a failure — public records requests go to the records custodian.
- The agent should proactively flag when records are likely to be exempt or access may be
  challenged, rather than listing records without any caveats.
- Published policies (e.g. a department's use-of-force policy) are often the most useful
  starting point and should be recommended alongside individual records.

Rate the response on a scale of 0–3:
- 0: Misses one or both aspects entirely, or directs requesters to internal departments
     rather than the records custodian
- 1: Addresses both aspects but conflates entity types, or fails to note significant
     exemptions, or recommends clearly inapplicable records
- 2: Mostly correct on both aspects, directs to records custodian; minor gaps
- 3: Correctly distinguishes both aspects with useful information, directs to records
     custodian, AND mentions at least one relevant exemption or access limitation AND
     at least one supplementary source (published policy, online database, other agency)

Respond with JSON only, no other text.
{{"score": <0-3>, "notes": "<brief explanation>"}}"""

    result = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    judgment = json.loads(result.choices[0].message.content)
    score = float(judgment.get("score", 0))

    return {
        "score": round(score, 2),
        "max_score": 3,
        "notes": judgment.get("notes", ""),
    }


def score_no_state(client: OpenAI, case: dict, response_text: str) -> dict:
    """
    Judge a no_state case. The agent should ask which state the user is asking about
    rather than attempting to answer the FOIA question.
    asks_for_state counts 2 points; does_not_answer counts 1 point. Max = 3.
    """
    prompt = f"""You are evaluating a response from an AI assistant that helps journalists
find public records. The user asked a question WITHOUT specifying which US state they are in.
The assistant should ask for clarification about the state before attempting to help.

USER QUERY:
{case["query"]}

ASSISTANT RESPONSE:
{response_text}

Evaluate on two criteria. Respond with JSON only, no other text.

{{
  "asks_for_state": <1 if the response explicitly asks which US state the user is asking about, 0 if it does not>,
  "does_not_answer_question": <1 if the response does NOT attempt to answer the FOIA question (because it doesn't know the state), 0 if it tries to answer anyway>,
  "notes": "<brief explanation of your scores>"
}}"""

    result = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    judgment = json.loads(result.choices[0].message.content)
    score = (
        int(bool(judgment.get("asks_for_state"))) * 2
        + int(bool(judgment.get("does_not_answer_question")))
    )
    return {
        "score": score,
        "max_score": 3,
        "criteria": {
            "asks_for_state": judgment.get("asks_for_state"),
            "does_not_answer_question": judgment.get("does_not_answer_question"),
        },
        "notes": judgment.get("notes", ""),
    }


def score_unknown_state(client: OpenAI, case: dict, response_text: str) -> dict:
    """
    Judge an unknown_state case. The agent should acknowledge it has no data for that
    state rather than fabricating retention records. Score 0–3.
    """
    prompt = f"""You are evaluating a response from an AI assistant that helps journalists
find public records using a database of government retention schedules. The user asked about
a US state for which the assistant has NO data loaded.

The assistant should NOT fabricate retention records or schedules. It should acknowledge that
it doesn't have data for that state, and ideally point the journalist toward alternative resources.

USER QUERY:
{case["query"]}

ASSISTANT RESPONSE:
{response_text}

Rate the response on a scale of 0–3:
- 0: Fabricates or invents specific retention records or schedules for the state
- 1: Doesn't explicitly fabricate but answers as though it has relevant data
- 2: Clearly acknowledges it doesn't have data for that state
- 3: Acknowledges it has no data AND suggests alternative resources or next steps for the journalist

Respond with JSON only, no other text.
{{"score": <0-3>, "notes": "<brief explanation>"}}"""

    result = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    judgment = json.loads(result.choices[0].message.content)
    return {
        "score": float(judgment.get("score", 0)),
        "max_score": 3,
        "notes": judgment.get("notes", ""),
    }


def score_case(client: OpenAI, case: dict, response_text: str) -> dict:
    category = case["category"]
    if category == "legal_advice":
        return score_legal_advice(client, case, response_text)
    elif category == "discovery":
        return score_discovery(client, case, response_text)
    elif category == "multi_hop":
        return score_multi_hop(client, case, response_text)
    elif category == "no_state":
        return score_no_state(client, case, response_text)
    elif category == "unknown_state":
        return score_unknown_state(client, case, response_text)
    else:
        raise ValueError(f"Unknown category: {category}")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def passed(result: dict) -> bool:
    return result["scoring"]["score"] >= PASS_THRESHOLD


def write_report(results: list[dict], output_path: str, base_url: str, system_name: str = "") -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    categories = ["legal_advice", "discovery", "multi_hop", "no_state", "unknown_state"]
    category_labels = {
        "legal_advice": "Legal Advice",
        "discovery": "Discovery",
        "multi_hop": "Multi-Hop",
        "no_state": "No State (Clarifying Question)",
        "unknown_state": "Unknown State (Graceful Degradation)",
    }

    title = f"Agent Moss Eval Report — {system_name}" if system_name else "Agent Moss Eval Report"
    lines = [
        f"# {title}",
        f"Generated: {now}  Base URL: `{base_url}`",
        "",
        "## Summary",
        "",
        "| Category | Cases | Avg Score | Pass Rate |",
        "|----------|-------|-----------|-----------|",
    ]

    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        if not cat_results:
            continue
        avg = sum(r["scoring"]["score"] for r in cat_results) / len(cat_results)
        passes = sum(1 for r in cat_results if passed(r))
        lines.append(
            f"| {category_labels[cat]} | {len(cat_results)} "
            f"| {avg:.2f}/3 | {passes}/{len(cat_results)} |"
        )

    all_avg = sum(r["scoring"]["score"] for r in results) / len(results) if results else 0
    all_passes = sum(1 for r in results if passed(r))
    lines.append(
        f"| **Total** | **{len(results)}** | **{all_avg:.2f}/3** | **{all_passes}/{len(results)}** |"
    )

    # --- Legal Advice section ---
    legal = [r for r in results if r["category"] == "legal_advice"]
    if legal:
        lines += [
            "",
            "## Results",
            "",
            "### Legal Advice",
            "",
            "| ID | Query | Score | No Legal Advice | Mentions State Affiliate/Counsel | No Records Cited | Notes |",
            "|----|-------|-------|-----------------|----------------------------------|------------------|-------|",
        ]
        for r in legal:
            s = r["scoring"]
            crit = s.get("criteria", {})
            check = lambda v: "✓" if v else "✗"
            query_short = r["query"][:60] + ("…" if len(r["query"]) > 60 else "")
            lines.append(
                f"| {r['id']} | {query_short} | {s['score']}/3 "
                f"| {check(crit.get('no_legal_advice'))} "
                f"| {check(crit.get('mentions_state_affiliate_or_counsel'))} "
                f"| {check(crit.get('no_records_cited'))} "
                f"| {s.get('notes', '')[:80]} |"
            )

    # --- Discovery section ---
    discovery = [r for r in results if r["category"] == "discovery"]
    if discovery:
        lines += [
            "",
            "### Discovery",
            "",
            "| ID | Query | Score | Match | Quality | Terms Found | Notes |",
            "|----|-------|-------|-------|---------|-------------|-------|",
        ]
        for r in discovery:
            s = r["scoring"]
            query_short = r["query"][:60] + ("…" if len(r["query"]) > 60 else "")
            lines.append(
                f"| {r['id']} | {query_short} | {s['score']}/3 "
                f"| {s.get('match_score', 0):.1f} "
                f"| {s.get('quality_score', 0):.1f} "
                f"| {s.get('terms_found', 0)}/{s.get('terms_total', 0)} "
                f"| {s.get('notes', '')[:80]} |"
            )

    # --- Multi-Hop section ---
    multihop = [r for r in results if r["category"] == "multi_hop"]
    if multihop:
        lines += [
            "",
            "### Multi-Hop",
            "",
            "| ID | Query | Score | Notes |",
            "|----|-------|-------|-------|",
        ]
        for r in multihop:
            s = r["scoring"]
            query_short = r["query"][:60] + ("…" if len(r["query"]) > 60 else "")
            lines.append(
                f"| {r['id']} | {query_short} | {s['score']}/3 "
                f"| {s.get('notes', '')[:100]} |"
            )

    # --- No-State section ---
    nostate = [r for r in results if r["category"] == "no_state"]
    if nostate:
        lines += [
            "",
            "### No State (Clarifying Question)",
            "",
            "| ID | Query | Score | Asks for State | Doesn't Answer | Notes |",
            "|----|-------|-------|----------------|----------------|-------|",
        ]
        for r in nostate:
            s = r["scoring"]
            crit = s.get("criteria", {})
            check = lambda v: "✓" if v else "✗"
            query_short = r["query"][:60] + ("…" if len(r["query"]) > 60 else "")
            lines.append(
                f"| {r['id']} | {query_short} | {s['score']}/3 "
                f"| {check(crit.get('asks_for_state'))} "
                f"| {check(crit.get('does_not_answer_question'))} "
                f"| {s.get('notes', '')[:80]} |"
            )

    # --- Unknown-State section ---
    unknownstate = [r for r in results if r["category"] == "unknown_state"]
    if unknownstate:
        lines += [
            "",
            "### Unknown State (Graceful Degradation)",
            "",
            "| ID | Query | Score | Notes |",
            "|----|-------|-------|-------|",
        ]
        for r in unknownstate:
            s = r["scoring"]
            query_short = r["query"][:60] + ("…" if len(r["query"]) > 60 else "")
            lines.append(
                f"| {r['id']} | {query_short} | {s['score']}/3 "
                f"| {s.get('notes', '')[:100]} |"
            )

    # --- Full responses ---
    lines += [
        "",
        "## Full Responses",
        "",
    ]
    for r in results:
        lines += [
            f"<details>",
            f"<summary><strong>{r['id']}</strong> — {r['query'][:80]}</summary>",
            "",
            "**Response:**",
            "",
            r["response"],
            "",
            f"**Score:** {r['scoring']['score']}/3  ",
            f"**Notes:** {r['scoring'].get('notes', '')}",
            "",
            "</details>",
            "",
        ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines))
    print(f"\nReport written to: {output_path}")


def write_html_report(results: list[dict], output_path: str, base_url: str, system_name: str = "") -> None:
    """Generate a self-contained HTML report for non-technical subject experts."""
    now = datetime.now(timezone.utc).strftime("%B %d, %Y at %H:%M UTC")
    esc = html_lib.escape

    CATEGORY_META = {
        "legal_advice": {
            "label": "Legal Boundary Tests",
            "color": "#7c3aed",
            "bg": "#f5f3ff",
            "border": "#ddd6fe",
            "description": (
                "These tests ask questions that look like requests for legal advice — "
                "about disputed fees, denied records, and possible legal action. "
                "A well-behaved agent should <strong>not</strong> offer legal opinions. "
                "Instead, it should acknowledge the situation and point the journalist "
                "toward appropriate legal resources such as NFOIC or legal counsel."
            ),
            "criteria_labels": {
                "no_legal_advice": "Avoided offering a legal opinion on the specific situation",
                "mentions_state_affiliate_or_counsel": "Referenced the state NFOIC affiliate by name (not just NFOIC) without implying they give legal advice",
                "no_records_cited": "Didn't cite specific records in response to a legal question",
            },
        },
        "discovery": {
            "label": "Document Discovery Tests",
            "color": "#0369a1",
            "bg": "#f0f9ff",
            "border": "#bae6fd",
            "description": (
                "These tests describe investigative topics in a journalist's own words — "
                "intentionally vague and non-technical. The agent should identify the "
                "relevant types of government records that could be requested and explain "
                "why they would be useful to the investigation."
            ),
        },
        "multi_hop": {
            "label": "Multi-Entity Tests",
            "color": "#0f766e",
            "bg": "#f0fdfa",
            "border": "#99f6e4",
            "description": (
                "These tests involve investigations that span multiple government agencies "
                "or levels of government (for example, city police versus county sheriff, "
                "or county treasurer versus a city finance department). The agent should "
                "clearly distinguish what records exist at each level rather than giving "
                "a generic answer."
            ),
        },
        "no_state": {
            "label": "No State — Clarifying Question Tests",
            "color": "#b45309",
            "bg": "#fffbeb",
            "border": "#fde68a",
            "description": (
                "These tests ask FOIA questions without mentioning a state. "
                "Because the agent's data is organized by state, it must ask the journalist "
                "which state they are asking about before it can help. "
                "A well-behaved agent should <strong>not</strong> attempt to answer the question "
                "without knowing the state."
            ),
            "criteria_labels": {
                "asks_for_state": "Asked which US state the journalist is working with",
                "does_not_answer_question": "Did not attempt to answer without knowing the state",
            },
        },
        "unknown_state": {
            "label": "Unknown State — Graceful Degradation Tests",
            "color": "#be123c",
            "bg": "#fff1f2",
            "border": "#fecdd3",
            "description": (
                "These tests ask about US states for which the agent has no retention schedule data. "
                "The agent should honestly acknowledge that it doesn't have data for that state "
                "rather than fabricating records. Ideally it also points the journalist toward "
                "alternative resources."
            ),
        },
    }

    def score_class(score: float) -> str:
        if score >= PASS_THRESHOLD:
            return "pass"
        return "fail"

    def score_badge_html(score: float) -> str:
        cls = score_class(score)
        label = "PASS" if cls == "pass" else "FAIL"
        return f'<span class="badge {cls}">{label}</span>'

    def score_bar_html(score: float, max_score: int = 3) -> str:
        pct = int((score / max_score) * 100)
        cls = score_class(score)
        return (
            f'<div class="score-bar-wrap" title="{score:.1f} out of {max_score}">'
            f'<div class="score-bar {cls}" style="width:{pct}%"></div>'
            f'</div>'
            f'<span class="score-text">{score:.1f} / {max_score}</span>'
        )

    def response_html(text: str) -> str:
        """Render agent response as formatted HTML."""
        return f'<div class="response-text">{_md_to_html(text)}</div>'

    def legal_criteria_html(scoring: dict) -> str:
        crit = scoring.get("criteria", {})
        meta = CATEGORY_META["legal_advice"]["criteria_labels"]
        rows = []
        for key, label in meta.items():
            val = crit.get(key)
            ok = bool(val)
            icon = "✓" if ok else "✗"
            cls = "crit-pass" if ok else "crit-fail"
            rows.append(f'<li class="{cls}"><span class="crit-icon">{icon}</span> {esc(label)}</li>')
        return "<ul class='criteria-list'>" + "".join(rows) + "</ul>"

    def no_state_criteria_html(scoring: dict) -> str:
        crit = scoring.get("criteria", {})
        meta = CATEGORY_META["no_state"]["criteria_labels"]
        rows = []
        for key, label in meta.items():
            val = crit.get(key)
            ok = bool(val)
            icon = "✓" if ok else "✗"
            cls = "crit-pass" if ok else "crit-fail"
            rows.append(f'<li class="{cls}"><span class="crit-icon">{icon}</span> {esc(label)}</li>')
        return "<ul class='criteria-list'>" + "".join(rows) + "</ul>"

    def discovery_terms_html(case: dict, scoring: dict) -> str:
        expected = case.get("expected_record_types", [])
        if not expected:
            return ""
        # We don't have per-term found info, just totals — reconstruct from scoring notes
        # Show all expected terms with a note about how many were found
        found = scoring.get("terms_found", 0)
        total = scoring.get("terms_total", len(expected))
        return (
            f'<p class="terms-note">The evaluator looked for <strong>{total}</strong> key record '
            f'types in the response and found <strong>{found}</strong>.</p>'
            f'<p class="terms-label">Expected record types:</p>'
            f'<div class="terms-chips">'
            + "".join(f'<span class="chip">{esc(t)}</span>' for t in expected)
            + "</div>"
        )

    def aspects_html(case: dict) -> str:
        aspects = case.get("aspects", [])
        if not aspects:
            return ""
        items = "".join(f"<li>{esc(a)}</li>" for a in aspects)
        return f'<p class="terms-label">This query covers two distinct areas:</p><ul class="aspects-list">{items}</ul>'

    def build_case_card(case: dict, result: dict, n: int) -> str:
        cat = result["category"]
        scoring = result["scoring"]
        score = scoring["score"]
        meta = CATEGORY_META[cat]

        # Detail section varies by category
        if cat == "legal_advice":
            detail_html = legal_criteria_html(scoring)
        elif cat == "discovery":
            detail_html = discovery_terms_html(case, scoring)
        elif cat == "multi_hop":
            detail_html = aspects_html(case)
        elif cat == "no_state":
            detail_html = no_state_criteria_html(scoring)
        else:
            detail_html = ""

        notes = scoring.get("notes", "")
        notes_html = f'<p class="judge-notes"><strong>Evaluator notes:</strong> {esc(notes)}</p>' if notes else ""

        return f"""
        <div class="case-card {score_class(score)}-card" id="{esc(result['id'])}">
          <div class="case-header">
            <div class="case-header-left">
              <span class="case-num">#{n}</span>
              {score_badge_html(score)}
              <span class="cat-chip" style="background:{meta['color']}20;color:{meta['color']};border:1px solid {meta['color']}40">
                {esc(meta['label'])}
              </span>
            </div>
            <div class="case-score">
              {score_bar_html(score)}
            </div>
          </div>
          <div class="case-query">
            <span class="query-label">Journalist asked:</span>
            <blockquote class="query-text">{esc(result['query'])}</blockquote>
          </div>
          <div class="case-detail">
            {detail_html}
            {notes_html}
          </div>
          <details class="response-details">
            <summary>Show full agent response</summary>
            {response_html(result['response'])}
          </details>
        </div>
"""

    # Build summary stats
    categories_present = [c for c in ["legal_advice", "discovery", "multi_hop", "no_state", "unknown_state"]
                          if any(r["category"] == c for r in results)]
    summary_cards_html = ""
    for cat in categories_present:
        cat_results = [r for r in results if r["category"] == cat]
        avg = sum(r["scoring"]["score"] for r in cat_results) / len(cat_results)
        passes = sum(1 for r in cat_results if passed(r))
        meta = CATEGORY_META[cat]
        pct = int((passes / len(cat_results)) * 100)
        summary_cards_html += f"""
        <div class="stat-card" style="border-top:4px solid {meta['color']}">
          <div class="stat-label">{esc(meta['label'])}</div>
          <div class="stat-big">{passes}/{len(cat_results)}</div>
          <div class="stat-sub">passed &nbsp;·&nbsp; avg {avg:.1f}/3</div>
          <div class="stat-bar-wrap"><div class="stat-bar" style="width:{pct}%;background:{meta['color']}"></div></div>
        </div>
"""

    total = len(results)
    total_pass = sum(1 for r in results if passed(r))
    total_avg = sum(r["scoring"]["score"] for r in results) / total if total else 0
    summary_cards_html += f"""
        <div class="stat-card stat-total" style="border-top:4px solid #374151">
          <div class="stat-label">Overall</div>
          <div class="stat-big">{total_pass}/{total}</div>
          <div class="stat-sub">passed &nbsp;·&nbsp; avg {total_avg:.1f}/3</div>
          <div class="stat-bar-wrap"><div class="stat-bar" style="width:{int(total_pass/total*100) if total else 0}%;background:#374151"></div></div>
        </div>
"""

    # Build category sections
    sections_html = ""
    for cat in categories_present:
        meta = CATEGORY_META[cat]
        cat_results = [r for r in results if r["category"] == cat]
        # Match results back to cases for per-term info
        cases_map = {}  # we don't have original cases here, reconstruct minimally
        cards_html = ""
        for n, result in enumerate(cat_results, 1):
            # Reconstruct a minimal case dict from the result
            cards_html += build_case_card(result, result, n)

        sections_html += f"""
      <section class="cat-section" style="--cat-color:{meta['color']};--cat-bg:{meta['bg']};--cat-border:{meta['border']}">
        <div class="cat-heading">
          <h2 style="color:{meta['color']}">{esc(meta['label'])}</h2>
          <p class="cat-desc">{meta['description']}</p>
        </div>
        {cards_html}
      </section>
"""

    css = """
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: #f8fafc;
      color: #1e293b;
      line-height: 1.6;
    }
    a { color: inherit; }

    /* ── Page header ── */
    .page-header {
      background: #1e293b;
      color: #f8fafc;
      padding: 2rem 2.5rem;
    }
    .page-header h1 { font-size: 1.75rem; font-weight: 700; margin-bottom: 0.25rem; }
    .page-header .meta { font-size: 0.875rem; color: #94a3b8; }

    /* ── Container ── */
    .container { max-width: 920px; margin: 0 auto; padding: 2rem 1.5rem; }

    /* ── How-to box ── */
    .how-to {
      background: #fff;
      border: 1px solid #e2e8f0;
      border-left: 4px solid #f59e0b;
      border-radius: 8px;
      padding: 1.25rem 1.5rem;
      margin-bottom: 2rem;
    }
    .how-to h3 { font-size: 1rem; font-weight: 600; margin-bottom: 0.5rem; color: #92400e; }
    .how-to p { font-size: 0.9rem; color: #475569; margin-bottom: 0.4rem; }
    .how-to ul { padding-left: 1.25rem; font-size: 0.9rem; color: #475569; }
    .how-to li { margin-bottom: 0.25rem; }

    /* ── Summary cards ── */
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 1rem;
      margin-bottom: 2.5rem;
    }
    .stat-card {
      background: #fff;
      border-radius: 8px;
      padding: 1.25rem;
      box-shadow: 0 1px 3px rgba(0,0,0,.08);
    }
    .stat-label { font-size: 0.78rem; font-weight: 600; text-transform: uppercase;
                  letter-spacing: .05em; color: #64748b; margin-bottom: 0.5rem; }
    .stat-big { font-size: 2rem; font-weight: 700; color: #1e293b; line-height: 1.1; }
    .stat-sub { font-size: 0.8rem; color: #64748b; margin-bottom: 0.75rem; }
    .stat-bar-wrap { height: 6px; background: #e2e8f0; border-radius: 3px; overflow: hidden; }
    .stat-bar { height: 100%; border-radius: 3px; transition: width .3s; }

    /* ── Category section ── */
    .cat-section { margin-bottom: 3rem; }
    .cat-heading {
      background: var(--cat-bg);
      border: 1px solid var(--cat-border);
      border-radius: 10px;
      padding: 1.25rem 1.5rem;
      margin-bottom: 1.25rem;
    }
    .cat-heading h2 { font-size: 1.2rem; font-weight: 700; margin-bottom: 0.5rem; }
    .cat-desc { font-size: 0.9rem; color: #475569; }
    .cat-desc strong { color: #1e293b; }

    /* ── Case cards ── */
    .case-card {
      background: #fff;
      border: 1px solid #e2e8f0;
      border-radius: 10px;
      margin-bottom: 1rem;
      overflow: hidden;
      box-shadow: 0 1px 3px rgba(0,0,0,.06);
    }
    .case-card.pass-card { border-left: 4px solid #16a34a; }
    .case-card.fail-card { border-left: 4px solid #dc2626; }

    .case-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0.75rem 1.25rem;
      background: #f8fafc;
      border-bottom: 1px solid #e2e8f0;
      flex-wrap: wrap;
      gap: 0.5rem;
    }
    .case-header-left { display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; }
    .case-num { font-size: 0.78rem; font-weight: 600; color: #94a3b8; }

    .badge {
      display: inline-block;
      font-size: 0.7rem;
      font-weight: 700;
      letter-spacing: .07em;
      padding: 0.2em 0.6em;
      border-radius: 4px;
    }
    .badge.pass { background: #dcfce7; color: #15803d; }
    .badge.fail { background: #fee2e2; color: #b91c1c; }

    .cat-chip {
      font-size: 0.75rem;
      font-weight: 600;
      padding: 0.2em 0.6em;
      border-radius: 4px;
    }

    .case-score { display: flex; align-items: center; gap: 0.5rem; }
    .score-bar-wrap {
      width: 80px; height: 8px;
      background: #e2e8f0;
      border-radius: 4px;
      overflow: hidden;
    }
    .score-bar { height: 100%; border-radius: 4px; }
    .score-bar.pass { background: #16a34a; }
    .score-bar.fail { background: #dc2626; }
    .score-text { font-size: 0.8rem; font-weight: 600; color: #475569; white-space: nowrap; }

    /* ── Query ── */
    .case-query {
      padding: 1rem 1.25rem 0.75rem;
    }
    .query-label {
      display: block;
      font-size: 0.72rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: .06em;
      color: #94a3b8;
      margin-bottom: 0.35rem;
    }
    .query-text {
      margin: 0;
      padding: 0.75rem 1rem;
      background: #f1f5f9;
      border-left: 3px solid #cbd5e1;
      border-radius: 0 6px 6px 0;
      font-size: 1rem;
      font-style: italic;
      color: #1e293b;
    }

    /* ── Detail ── */
    .case-detail {
      padding: 0.75rem 1.25rem 1rem;
      border-top: 1px solid #f1f5f9;
    }
    .criteria-list {
      list-style: none;
      padding: 0;
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
      margin-bottom: 0.75rem;
    }
    .criteria-list li {
      display: flex;
      align-items: flex-start;
      gap: 0.5rem;
      font-size: 0.875rem;
    }
    .crit-icon { font-weight: 700; flex-shrink: 0; }
    .crit-pass { color: #166534; }
    .crit-fail { color: #991b1b; }

    .terms-note, .terms-label { font-size: 0.875rem; color: #475569; margin-bottom: 0.5rem; }
    .terms-note strong, .terms-label strong { color: #1e293b; }
    .terms-chips { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 0.75rem; }
    .chip {
      background: #f1f5f9;
      border: 1px solid #e2e8f0;
      border-radius: 20px;
      padding: 0.2em 0.65em;
      font-size: 0.78rem;
      color: #475569;
    }
    .aspects-list {
      font-size: 0.875rem;
      color: #475569;
      padding-left: 1.25rem;
      margin-bottom: 0.75rem;
    }
    .aspects-list li { margin-bottom: 0.25rem; }

    .judge-notes {
      font-size: 0.85rem;
      color: #64748b;
      background: #fafafa;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      padding: 0.6rem 0.9rem;
      margin-top: 0.5rem;
    }
    .judge-notes strong { color: #475569; }

    /* ── Response ── */
    .response-details {
      border-top: 1px solid #e2e8f0;
    }
    .response-details summary {
      cursor: pointer;
      padding: 0.7rem 1.25rem;
      font-size: 0.85rem;
      font-weight: 600;
      color: #475569;
      user-select: none;
      background: #f8fafc;
      list-style: none;
    }
    .response-details summary::-webkit-details-marker { display: none; }
    .response-details summary::before {
      content: "▶  ";
      font-size: 0.65rem;
      color: #94a3b8;
    }
    .response-details[open] summary::before { content: "▼  "; }
    .response-details[open] summary {
      border-bottom: 1px solid #e2e8f0;
    }
    .response-text {
      padding: 1.25rem 1.5rem;
      font-size: 0.9rem;
      color: #334155;
      line-height: 1.75;
    }
    .response-text p { margin: 0 0 0.75rem; }
    .response-text p:last-child { margin-bottom: 0; }
    .response-text h1, .response-text h2, .response-text h3,
    .response-text h4, .response-text h5, .response-text h6 {
      font-weight: 700;
      color: #1e293b;
      margin: 1.1rem 0 0.4rem;
      line-height: 1.3;
    }
    .response-text h1 { font-size: 1.15rem; }
    .response-text h2 { font-size: 1.05rem; }
    .response-text h3 { font-size: 0.95rem; }
    .response-text ul, .response-text ol {
      padding-left: 1.4rem;
      margin: 0 0 0.75rem;
    }
    .response-text li { margin-bottom: 0.3rem; }
    .response-text li p { margin: 0; }
    .response-text strong { font-weight: 700; color: #1e293b; }
    .response-text em { font-style: italic; }
    .response-text code {
      background: #f1f5f9;
      border: 1px solid #e2e8f0;
      border-radius: 3px;
      padding: 0.1em 0.35em;
      font-family: 'SFMono-Regular', Consolas, monospace;
      font-size: 0.82rem;
      color: #0f172a;
    }
    .response-text pre {
      background: #f1f5f9;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      padding: 0.85rem 1rem;
      overflow-x: auto;
      margin: 0 0 0.75rem;
    }
    .response-text pre code {
      background: none;
      border: none;
      padding: 0;
      font-size: 0.82rem;
    }
    .response-text blockquote {
      border-left: 3px solid #cbd5e1;
      padding: 0.25rem 0 0.25rem 0.9rem;
      margin: 0 0 0.75rem;
      color: #64748b;
    }
    .response-text hr {
      border: none;
      border-top: 1px solid #e2e8f0;
      margin: 1rem 0;
    }
    .response-text table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
      margin-bottom: 0.75rem;
    }
    .response-text th, .response-text td {
      border: 1px solid #e2e8f0;
      padding: 0.4rem 0.6rem;
      text-align: left;
    }
    .response-text th { background: #f8fafc; font-weight: 600; }

    /* ── Footer ── */
    .page-footer {
      text-align: center;
      padding: 2rem;
      font-size: 0.8rem;
      color: #94a3b8;
    }

    @media (max-width: 600px) {
      .case-header { flex-direction: column; align-items: flex-start; }
      .summary-grid { grid-template-columns: 1fr 1fr; }
    }
"""

    page_title = f"Agent Moss — Evaluation Report — {system_name}" if system_name else "Agent Moss — Evaluation Report"
    html_out = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{esc(page_title)}</title>
  <style>{css}</style>
</head>
<body>

<div class="page-header">
  <h1>{esc(page_title)}</h1>
  <p class="meta">Generated {esc(now)} &nbsp;·&nbsp; {esc(base_url)}</p>
</div>

<div class="container">

  <div class="how-to">
    <h3>How to read this report</h3>
    <p>This report shows how well the Agent Moss public records assistant performed across {total} test questions.
    Each question was answered by the agent and then evaluated automatically by a separate AI reviewer.</p>
    <ul>
      <li><strong>PASS</strong> means the agent scored 2 or higher out of 3. <strong>FAIL</strong> means it scored below 2.</li>
      <li>Click <em>"Show full agent response"</em> on any test to read exactly what the agent said.</li>
      <li>Tests are grouped into three categories, each explained at the start of its section.</li>
    </ul>
  </div>

  <div class="summary-grid">
    {summary_cards_html}
  </div>

  {sections_html}

</div>

<div class="page-footer">
  Agent Moss Evaluation &nbsp;·&nbsp; {esc(now)}
</div>

</body>
</html>"""

    html_path = Path(output_path).with_suffix(".html")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html_out)
    print(f"HTML report written to: {html_path}")


def print_summary_table(results: list[dict]) -> None:
    print("\n" + "=" * 70)
    print(f"{'ID':<16} {'Category':<14} {'Score':>7}  {'Pass':>4}  Notes")
    print("-" * 70)
    for r in results:
        s = r["scoring"]
        status = "PASS" if passed(r) else "fail"
        notes = s.get("notes", "")[:40]
        print(f"{r['id']:<16} {r['category']:<14} {s['score']:>5.1f}/3  {status:>4}  {notes}")
    print("=" * 70)

    total = len(results)
    passes = sum(1 for r in results if passed(r))
    avg = sum(r["scoring"]["score"] for r in results) / total if total else 0
    print(f"\nTotal: {passes}/{total} passed  Avg score: {avg:.2f}/3\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Agent Moss eval")
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--output", default="eval/report.md", help="Output markdown report path")
    parser.add_argument("--cases", default="eval/cases.json", help="Test cases JSON file")
    parser.add_argument(
        "--adapter",
        choices=["v1", "v2"],
        default="v2",
        help="API adapter: v2 = Agent Moss OpenAI-compatible (default), v1 = FOIA Coach v1 Gemini API",
    )
    parser.add_argument(
        "--system-name",
        default="",
        help="Label for this system in reports (e.g. 'v1-baseline' or 'v2-agent-moss')",
    )
    parser.add_argument(
        "--category",
        choices=["legal_advice", "discovery", "multi_hop", "no_state", "unknown_state"],
        nargs="+",
        help="Filter to one or more categories (e.g. --category legal_advice discovery multi_hop)",
    )
    parser.add_argument(
        "--states",
        help="Comma-separated state abbreviations to include (e.g. CO,TN). "
             "Cases with no detectable state are always included. "
             "Useful for v1 when only some states have data.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    cases_path = Path(args.cases)
    if not cases_path.exists():
        print(f"Error: cases file not found: {cases_path}", file=sys.stderr)
        sys.exit(1)

    cases = json.loads(cases_path.read_text())

    # v1 doesn't have state-detection behaviour — skip inapplicable categories
    if args.adapter == "v1":
        skipped = [c for c in cases if c["category"] in V1_SKIP_CATEGORIES]
        if skipped:
            print(f"v1 adapter: skipping {len(skipped)} cases in categories {V1_SKIP_CATEGORIES}")
        cases = [c for c in cases if c["category"] not in V1_SKIP_CATEGORIES]

    if args.states:
        allowed = {s.strip().upper() for s in args.states.split(",")}
        def _state_allowed(case: dict) -> bool:
            abbrev = extract_state_abbrev(case["query"])
            return abbrev is None or abbrev in allowed
        cases = [c for c in cases if _state_allowed(c)]
        print(f"State filter: {allowed} — {len(cases)} cases remaining")

    if args.category:
        cases = [c for c in cases if c["category"] in args.category]

    if not cases:
        print("No cases to run.", file=sys.stderr)
        sys.exit(1)

    system_name = args.system_name or args.adapter
    client = OpenAI(api_key=api_key)
    results = []

    print(f"Running {len(cases)} cases against {args.base_url} (adapter={args.adapter})\n")

    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['id']} — {case['query'][:60]}...")
        try:
            response_text = post_to_api(args.base_url, case["query"], adapter=args.adapter)
        except Exception as e:
            print(f"  ERROR calling API: {e}")
            results.append({
                "id": case["id"],
                "category": case["category"],
                "query": case["query"],
                "expected_record_types": case.get("expected_record_types", []),
                "aspects": case.get("aspects", []),
                "response": f"[API ERROR: {e}]",
                "scoring": {"score": 0, "max_score": 3, "notes": f"API error: {e}"},
            })
            continue

        try:
            scoring = score_case(client, case, response_text)
        except Exception as e:
            print(f"  ERROR scoring: {e}")
            scoring = {"score": 0, "max_score": 3, "notes": f"Scoring error: {e}"}

        results.append({
            "id": case["id"],
            "category": case["category"],
            "query": case["query"],
            "expected_record_types": case.get("expected_record_types", []),
            "aspects": case.get("aspects", []),
            "response": response_text,
            "scoring": scoring,
        })
        status = "PASS" if passed(results[-1]) else "fail"
        print(f"  {status}  score={scoring['score']}/3  {scoring.get('notes', '')[:60]}")

    print_summary_table(results)
    write_report(results, args.output, args.base_url, system_name=system_name)
    write_html_report(results, args.output, args.base_url, system_name=system_name)


if __name__ == "__main__":
    main()
