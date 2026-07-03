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
# A dangling </answer> (or <answer>) closer followed by a short single-line bare answer at the
# very end — the middle of QwQ's "</answer>\nPTSD\n</answer>" sandwich once the outer closer is
# peeled. Lets the peel loop also drop the bare "PTSD" answer text between two dangling closers.
_BARE_ANSWER_TAIL = re.compile(r"(?:</?answer>)\s*[^<>\n]{1,160}\Z", flags=re.IGNORECASE)
# Terminal special/EOS tokens now retained under skip_special_tokens=False decode. These sit
# AFTER the final "<answer>...</answer>" (or at the very end when generation was truncated), so
# stripping any run of them off the tail lets the downstream "</answer>" trailing-strip fire.
# Covers deepseek fullwidth EOS/pad "<｜end▁of▁sentence｜>" plus the ASCII EOS variants; the
# generic "<|...|>" arm mops up padding tokens (e.g. huatuo "<|end_of_text|>") after the EOS.
_TRAILING_SPECIAL = re.compile(
    r"(?:\s*(?:<｜end▁of▁sentence｜>|<\|(?:eot_id|end_of_text|endoftext|im_end|return|end)\|>))+\s*\Z"
)
_MARKER_SCRUB = (
    (re.compile(r"\s*<\|redacted_Assistant\|><think>"), ""),
    (re.compile(r"\s*<\| Assistant \|><think>"), ""),
    (re.compile(r"\s*<Assistant0><think>"), ""),
)


def _strip_chat_prefix(text: str) -> str:
    if not text:
        return text
    # Anchor on the LAST assistant boundary. Some deepseek-distill traces emit a doubled
    # "<｜Assistant｜><think>" where the first one echoes the prompt's OUTPUT TEMPLATE
    # ("...the name of the disease/entity...</answer>"); taking the first match leaks that
    # template tail into the reasoning, so use the last (where real generation begins).
    matches = list(_CHAT_PREFIX_1.finditer(text))
    if matches:
        m1 = matches[-1]
        think_pos = text.find("<think>\n", m1.start())
        return text[think_pos:] if think_pos != -1 else text[m1.end() :]
    m2 = _CHAT_PREFIX_2.search(text)
    return text[m2.end() :] if m2 else text


def _strip_trailing_answer_line(text: str) -> str:
    if not text:
        return text
    # Peel trailing answer markup off the reasoning, one tag at a time from the end:
    #   * a balanced <answer>...</answer> block (content may be long/multi-line) -> drop it whole;
    #   * a dangling </answer> closer with no matching opener -> drop just the closer, then also
    #     drop a short bare answer line that sat right after another answer tag.
    # This handles the plain final answer AND QwQ's spurious dangling closer, e.g.
    # "...diagnosis.\n</answer>\n<answer>\nDx\n</answer>" or "...likely.\n</answer>\nPTSD\n</answer>".
    # Matching the opener via rfind (nearest, with no </answer> between) rather than the string's
    # first <answer> avoids the old bug of eating real reasoning back to an earlier quoted tag.
    s = text.rstrip()
    original = s
    while s.endswith("</answer>"):
        close = s.rfind("</answer>")
        opener = s.rfind("<answer>", 0, close)
        if opener != -1 and "</answer>" not in s[opener + len("<answer>") : close]:
            s = s[:opener].rstrip()          # balanced block
        else:
            s = s[:close].rstrip()           # dangling closer
            mb = _BARE_ANSWER_TAIL.search(s)  # + a short bare answer wedged before it
            if mb:
                bare = re.match(r"</?answer>", s[mb.start() :])
                s = s[: mb.start() + bare.end()].rstrip()
    return s if s != original else text


def _final_cleanup(text: str | None) -> str:
    if not text:
        return ""
    cleaned = _FINAL_RESPONSE_RE.sub("", text)
    cleaned = cleaned.replace(_FINAL_RESPONSE_HDR, "")
    for rx, rep in _MARKER_SCRUB:
        cleaned = rx.sub(rep, cleaned)
    cleaned = cleaned.replace("<｜Assistant｜><think>", "")
    # Strip Llama-3 chat scaffolding: huatuo leaks
    # "<|start_header_id|>assistant<|end_header_id|>\n\n## Thinking" between the prompt's
    # answer-placeholder and the real reasoning. Remove the full header (incl. role word),
    # any remaining <|...|> special tokens, then a leading "## Thinking" header.
    cleaned = re.sub(r"<\|start_header_id\|>.*?<\|end_header_id\|>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<\|[^>]*\|>", "", cleaned)
    # Huatuo doubles/echoes the turn header in several variants — e.g.
    # "<|start_header_id|>assistant<|end_header_id|>\n\nassistant\n\n## Thinking", or a leftover
    # "</answer>assistant\n\n## Thinking". The reasoning reliably begins right after a near-leading
    # "## Thinking", so cut to it; then strip any residual bare leading "assistant".
    cleaned = re.sub(r"^.{0,40}?##\s*Thinking\s*\n+", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"^\s*assistant\b\s*", "", cleaned)
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

    # qwq-32b emits "</think>\n\n\n\n<tool_call>" (4 newlines) where the deepseek-R1 distills
    # use 2. Match the boundary whitespace-tolerantly so the tool_call / second-think block is
    # cut for all variants instead of leaking <tool_call>/answer tags into the reasoning.
    cut1 = re.search(r"\n</think>\s*<tool_call>", tail)
    cut2 = re.search(r"\n</think>\s*<think>\n", tail)
    cuts = [m.start() for m in (cut1, cut2) if m]
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


# gpt-oss native "harmony" format puts the chain-of-thought in an "analysis" channel and the
# user-facing answer in a separate "final" channel, so the analysis channel IS the CoT.
# Under uniform skip_special_tokens=True decode (our chosen setting), the `<|channel|>` /
# `<|message|>` / `<|end|>` / `<|return|>` markers are stripped but the channel-name plain
# text remains as glued concatenations at the channel boundaries:
#   "<|start|>assistant<|channel|>analysis<|message|>"      -> "assistantanalysis"
#   "<|end|><|start|>assistant<|channel|>final<|message|>"  -> "assistantfinal"
# Those glued strings (no space between the two words) only arise at the channel boundaries,
# so they're reliable text markers. Reasoning lives between "assistantanalysis" and
# "assistantfinal" (or to end if the generation hit the token cap before closing analysis).
_HARMONY_TEXT = re.compile(r"assistantanalysis(.*?)assistantfinal", flags=re.DOTALL)
_HARMONY_TEXT_OPEN = re.compile(r"assistantanalysis(.*)\Z", flags=re.DOTALL)  # fallback: no boundary present
# Token form (skip_special_tokens=False decode): the channel markers survive verbatim as
# "<|channel|>analysis<|message|>...<|end|>", so the analysis-channel CoT is the span from
# "<|channel|>analysis<|message|>" up to the first following boundary (<|end|> / next
# <|channel|> for the final channel / <|return|> / end-of-text on truncation).
_HARMONY_TOK = re.compile(
    r"<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|<\|channel\|>|<\|return\|>|\Z)",
    flags=re.DOTALL,
)


def extract_thinking_process(response: str, question: str = "") -> str:
    """Extract the model's reasoning trace from a full response.

    Strips assistant/thinking markup, trailing ``<answer>`` lines, and the
    literal header "## Final Response" wherever it appears.
    """
    if not response:
        return ""

    # Drop any terminal EOS/pad tokens (present under skip_special_tokens=False decode) so the
    # trace ends on "</answer>" and the downstream trailing-answer strip fires. No-op on legacy
    # skip_special_tokens=True files, which carry no trailing special tokens.
    response = _TRAILING_SPECIAL.sub("", response)

    # gpt-oss harmony: the "analysis" channel holds the CoT. Two on-disk forms are supported —
    # the token form "<|channel|>analysis<|message|>...<|end|>" (skip_special_tokens=False) and
    # the legacy glued form "assistantanalysis"..."assistantfinal" (skip_special_tokens=True),
    # where the channel markers were stripped and the channel names collapsed onto "assistant".
    if "<|channel|>analysis<|message|>" in response or "assistantanalysis" in response:
        m = (
            _HARMONY_TOK.search(response)
            or _HARMONY_TEXT.search(response)
            or _HARMONY_TEXT_OPEN.search(response)
        )
        if m is not None:
            return _final_cleanup(m.group(1))

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
