import re

_FINAL_RESPONSE_HDR = "## Final Response"
# Models vary spacing, case, and line endings; plain str.replace misses NBSP/CRLF variants.
_FINAL_RESPONSE_RE = re.compile(
    r"(?m)^[ \t\u00a0]*##\s*Final\s+Response\b[ \t\u00a0]*\r?$|##\s*Final\s+Response\b\s*",
    flags=re.IGNORECASE,
)
_SECOND_RESPONSE_PROMPT = (
    "Your reasoning patterns seem unusual. Approach the problem from a different "
    "angle. Format your response as previously."
)
_DEEPSEEK_FULL_MARKER = (
    "<think>\n...your internal reasoning for the diagnosis...\n"
    "</think><answer>\n...the name of the disease/entity...\n"
    "</answer>\n<think>\n"
)
_CHAT_PREFIX_1 = re.compile(
    r"</answer>\s*<[^>]*assistant[^>]*><think>\n",
    flags=re.IGNORECASE,
)
_CHAT_PREFIX_2 = re.compile(
    r"</answer>\s*assistant\s*\n\s*\n##\s*Thinking\s*\n\s*\n",
    flags=re.IGNORECASE,
)
_PLACEHOLDER_ANSWER = re.compile(
    r"<answer>\s*\.\.\.the name of the disease/entity\.\.\.\s*</answer>",
    flags=re.IGNORECASE,
)
_THINK_BLOCK = re.compile(
    r"<think>(.*?)</think>",
    flags=re.IGNORECASE | re.DOTALL,
)
_THINKING_SECTION = re.compile(
    r"##\s*Thinking\s*\n(.*?)(?:##|<answer>|\Z)",
    flags=re.IGNORECASE | re.DOTALL,
)
_ASSISTANT_SUFFIX = re.compile(
    r"</answer>\s*<[^>]*assistant[^>]*><think>\n?",
    flags=re.IGNORECASE,
)
_SHORT_ANSWER = re.compile(r"<answer>(.*?)</answer>", flags=re.IGNORECASE | re.DOTALL)
_MARKER_SCRUB = (
    (re.compile(r"\s*<\|redacted_Assistant\|><think>"), ""),
    (re.compile(r"\s*<\| Assistant \|><think>"), ""),
    (re.compile(r"\s*<Assistant0><think>"), ""),
)


def _strip_chat_prefix(text: str) -> str:
    if not text:
        return text
    m1 = _CHAT_PREFIX_1.search(text)
    if m1:
        think_pos = text.find("<think>\n", m1.start())
        return text[think_pos:] if think_pos != -1 else text[m1.end() :]
    m2 = _CHAT_PREFIX_2.search(text)
    return text[m2.end() :] if m2 else text


def _strip_trailing_answer_line(text: str) -> str:
    if not text:
        return text
    lines = text.splitlines()
    idx = len(lines) - 1
    while idx >= 0 and not lines[idx].strip():
        idx -= 1
    if idx < 0:
        return text
    last = lines[idx]
    if re.match(r"^\s*<answer>.*</answer>\s*$", last):
        return "\n".join(lines[:idx]).rstrip()
    return text


def _final_cleanup(text: str | None) -> str:
    if not text:
        return ""
    cleaned = _FINAL_RESPONSE_RE.sub("", text)
    cleaned = cleaned.replace(_FINAL_RESPONSE_HDR, "")
    for rx, rep in _MARKER_SCRUB:
        cleaned = rx.sub(rep, cleaned)
    cleaned = cleaned.replace("<｜Assistant｜><think>", "")
    cleaned = cleaned.replace("<think>", "").replace("</think>", "")
    return cleaned.strip()


def _extract_deepseek_segment(text: str) -> str | None:
    tail: str | None = None
    pos_full = text.find(_DEEPSEEK_FULL_MARKER)
    if pos_full != -1:
        tail = text[pos_full + len(_DEEPSEEK_FULL_MARKER) :]
    else:
        m = _PLACEHOLDER_ANSWER.search(text)
        if not m:
            return None
        tail = text[m.end() :]

    m1 = "\n</think>\n\n<tool_call>"
    m2 = "\n</think>\n\n<think>\n"
    cuts = [p for p in (tail.find(m1), tail.find(m2)) if p != -1]
    if cuts:
        tail = tail[: min(cuts)]
    tail = tail.replace("\n\n<think>\n", "")
    tail = _strip_trailing_answer_line(tail).strip()
    return tail or None


def _strip_short_answers(text: str) -> str:
    matches = list(_SHORT_ANSWER.finditer(text))
    if not matches:
        return text
    result = text
    for m in reversed(matches):
        if len(m.group(1)) < 500:
            result = result[: m.start()] + result[m.end() :]
    return result


def extract_thinking_process(response: str, question: str = "") -> str:
    """Extract the model's reasoning trace from a full response.

    Strips assistant/thinking markup, trailing ``<answer>`` lines, and the
    literal header "## Final Response" wherever it appears.
    """
    if not response:
        return ""

    text = response
    if "Your previous diagnosis was incorrect" in text or _SECOND_RESPONSE_PROMPT[:50] in text:
        parts = text.split(_SECOND_RESPONSE_PROMPT)
        if len(parts) > 1:
            text = parts[-1]

    text = _strip_chat_prefix(text)

    deepseek = _extract_deepseek_segment(text)
    if deepseek is not None:
        return _final_cleanup(deepseek)

    candidates: list[tuple[str, str]] = []
    for think in _THINK_BLOCK.findall(text):
        t = think.strip()
        if t in ("...", "...your internal reasoning for the diagnosis..."):
            continue
        candidates.append(("think_block", t))

    outside = ""
    for marker in ("</answer>\n", "</answer>"):
        oti = text.find("OUTPUT TEMPLATE")
        if oti == -1:
            continue
        te = text.find(marker, oti)
        if te == -1:
            continue
        after = text[te + len(marker) :]
        last_think = after.rfind("<think>")
        if last_think <= 100:
            continue
        ot = after[:last_think].strip()
        if "<|" in ot or "|>" in ot:
            continue
        ot = re.sub(r"^\s*[-]+\s*$", "", ot, flags=re.MULTILINE)
        ot = re.sub(r"^CASE PRESENTATION.*?(?=\n\n|\Z)", "", ot, flags=re.DOTALL).strip()
        if len(ot) >= 100:
            outside = ot
            break

    if outside:
        candidates.append(("outside_tags", outside))

    if candidates:
        best = max(candidates, key=lambda x: len(x[1]))
        return _final_cleanup(_strip_trailing_answer_line(best[1]))

    tm = _THINKING_SECTION.search(text)
    if tm:
        content = tm.group(1).strip()
        if len(content) >= 50:
            return _final_cleanup(_strip_trailing_answer_line(content))

    resp = text
    if question and question in resp:
        resp = resp.replace(question, "")

    resp_no_ans = _strip_short_answers(resp).strip()
    cleaned = _strip_trailing_answer_line(resp_no_ans if resp_no_ans else resp)
    cleaned = _ASSISTANT_SUFFIX.sub("", cleaned)

    return _final_cleanup(cleaned)
