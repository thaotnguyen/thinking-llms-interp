import re


def extract_thinking_process(response: str, question: str = "") -> str:
    """Extract the model's reasoning trace from a full response.

    Strategy (in priority order):
    1. DeepSeek-style OUTPUT TEMPLATE with placeholder
       - Remove everything up to and including
         "<answer>\n...the name of the disease/entity...\n</answer>".
       - Remove anything after "\n</think>\n\n<tool_call>" (inclusive).
       - Remove anything after "\n</think>\n\n<think>\n" (inclusive).
       - Remove all occurrences of "\n\n<think>\n".
       - The remainder is returned as the reasoning trace.
    2. General <think>...</think> blocks – use the longest non-placeholder block.
    3. Reasoning outside <think> tags between the OUTPUT TEMPLATE answer and the
       last <think> block (gpt-oss-20b-style traces).
    4. "## Thinking" markdown heading section.
    5. Fallback: remove the question (if provided) and short <answer>...</answer>
       blocks, and return the remainder.
    """
    if not response:
        return ""

    full_response = response

    # Handle second responses where the model is asked to re-answer
    SECOND_RESPONSE_PROMPT = (
        "Your reasoning patterns seem unusual. Approach the problem from a different "
        "angle. Format your response as previously."
    )

    text_to_search = full_response
    if (
        "Your previous diagnosis was incorrect" in full_response
        or SECOND_RESPONSE_PROMPT[:50] in full_response
    ):
        parts = full_response.split(SECOND_RESPONSE_PROMPT)
        if len(parts) > 1:
            text_to_search = parts[-1]

    # -------------------------------------------------
    # Pre-rule: strip chat-style prefixes before assistant reasoning
    # -------------------------------------------------
    def _strip_chat_prefix(text: str) -> str:
        if not text:
            return text

        # Pattern 1: variants of "</answer><...Assistant...><think>\n"
        m1 = re.search(
            r"</answer>\s*<[^>]*assistant[^>]*><think>\n",
            text,
            flags=re.IGNORECASE,
        )
        if m1:
            # Keep from the <think> onwards (so generic <think> extraction still works)
            think_pos = text.find("<think>\n", m1.start())
            if think_pos != -1:
                return text[think_pos:]
            return text[m1.end():]

        # Pattern 2: "</answer>assistant\n\n## Thinking\n\n" — drop everything before this
        m2 = re.search(
            r"</answer>\s*assistant\s*\n\s*\n##\s*Thinking\s*\n\s*\n",
            text,
            flags=re.IGNORECASE,
        )
        if m2:
            return text[m2.end() :]

        return text

    text_to_search = _strip_chat_prefix(text_to_search)

    # -------------------------------------------------
    # Final cleanup helper applied before every return
    # -------------------------------------------------
    def _final_cleanup(text: str | None) -> str:
        if not text:
            return ""

        cleaned = text

        # Remove any remaining assistant chat markers
        cleaned = re.sub(r"\s*<｜Assistant｜><think>", "", cleaned)
        cleaned = re.sub(r"\s*<\| Assistant \|><think>", "", cleaned)
        cleaned = re.sub(r"\s*<Assistant0><think>", "", cleaned)
        cleaned = cleaned.replace("<｜Assistant｜><think>", "")

        # Drop any bare think tags that might still be present
        cleaned = cleaned.replace("<think>", "").replace("</think>", "")

        return cleaned.strip()

    # Helper: drop a trailing line that's purely an <answer>...</answer>
    def _strip_trailing_answer_line(text: str) -> str:
        if not text:
            return text
        lines = text.splitlines()
        # Find last non-empty line
        idx = len(lines) - 1
        while idx >= 0 and not lines[idx].strip():
            idx -= 1
        if idx < 0:
            return text
        last = lines[idx]
        if re.match(r"^\s*<answer>.*</answer>\s*$", last):
            new_lines = lines[:idx]
            return "\n".join(new_lines).rstrip()
        return text

    # ------------------------------------------------------------------
    # 1) DeepSeek-style OUTPUT TEMPLATE handling (requested custom logic)
    # ------------------------------------------------------------------
    def _extract_deepseek_segment(text: str) -> str | None:
        # First, handle the exact full template + follow-up <think> marker, if present.
        # This matches the literal string:
        # "<think>\n...your internal reasoning for the diagnosis...\n</think><answer>\n"
        # "...the name of the disease/entity...\n</answer>\n<think>\n"
        full_marker = (
            "<think>\n...your internal reasoning for the diagnosis...\n"  # noqa: E501
            "</think><answer>\n...the name of the disease/entity...\n"  # noqa: E501
            "</answer>\n<think>\n"
        )

        tail: str | None = None
        pos_full = text.find(full_marker)
        if pos_full != -1:
            # Remove everything up to and including this full marker
            tail = text[pos_full + len(full_marker) :]
        else:
            # Otherwise, fall back to just the placeholder <answer> block.
            m = re.search(
                r"<answer>\s*\.\.\.the name of the disease/entity\.\.\.\s*</answer>",
                text,
                flags=re.IGNORECASE,
            )
            if not m:
                return None

            # Remove everything before (and including) the placeholder answer block
            tail = text[m.end() :]

        # 2 & 3) Trim anything starting at </think> followed by tool_call or another <think>
        marker1 = "\n</think>\n\n<tool_call>"
        marker2 = "\n</think>\n\n<think>\n"
        cut_points = [p for p in (tail.find(marker1), tail.find(marker2)) if p != -1]
        if cut_points:
            tail = tail[: min(cut_points)]

        # 4) Remove all explicit think open tags of the form "\n\n<think>\n"
        tail = tail.replace("\n\n<think>\n", "")

        # Apply trailing-answer cleanup and final strip
        tail = _strip_trailing_answer_line(tail).strip()
        return tail or None

    deepseek_segment = _extract_deepseek_segment(text_to_search)
    if deepseek_segment is not None:
        return _final_cleanup(deepseek_segment)

    # ----------------------------------------------
    # 2) Generic <think>...</think> based extraction
    # ----------------------------------------------
    reasoning_candidates: list[tuple[str, str]] = []

    think_matches = re.findall(
        r"<think>(.*?)</think>", text_to_search, flags=re.IGNORECASE | re.DOTALL
    )

    # Filter out known template placeholders but keep even short real reasoning
    for think in think_matches:
        think_clean = think.strip()
        if think_clean in ["...", "...your internal reasoning for the diagnosis..."]:
            continue
        reasoning_candidates.append(("think_block", think_clean))

    # ------------------------------------------------------
    # 3) Reasoning outside <think> tags (OUTPUT TEMPLATE use)
    # ------------------------------------------------------
    template_markers = ["</answer>\n", "</answer>"]
    outside_reasoning = ""
    for marker in template_markers:
        output_template_idx = text_to_search.find("OUTPUT TEMPLATE")
        if output_template_idx == -1:
            continue
        search_start = output_template_idx
        template_end_idx = text_to_search.find(marker, search_start)
        if template_end_idx == -1:
            continue

        after_template = text_to_search[template_end_idx + len(marker) :]
        # Find the LAST <think> block start in the remaining text
        last_think_start = after_template.rfind("<think>")
        if last_think_start <= 100:
            continue

        outside_text = after_template[:last_think_start].strip()

        # Skip Huatuo-style chat template markers
        if "<|" in outside_text or "|>" in outside_text:
            continue

        # Remove separator lines and echoed case text
        outside_text = re.sub(
            r"^\s*[-]+\s*$", "", outside_text, flags=re.MULTILINE
        )
        outside_text = re.sub(
            r"^CASE PRESENTATION.*?(?=\n\n|\Z)",
            "",
            outside_text,
            flags=re.DOTALL,
        )
        outside_text = outside_text.strip()
        if len(outside_text) >= 100:
            outside_reasoning = outside_text
            break

    if outside_reasoning:
        reasoning_candidates.append(("outside_tags", outside_reasoning))

    # If we collected any candidates, return the longest by length
    if reasoning_candidates:
        best = max(reasoning_candidates, key=lambda x: len(x[1]))
        cleaned_best = _strip_trailing_answer_line(best[1])
        return _final_cleanup(cleaned_best)

    # ---------------------------------
    # 4) "## Thinking" markdown block
    # ---------------------------------
    thinking_match = re.search(
        r"##\s*Thinking\s*\n(.*?)(?:##|<answer>|\Z)",
        text_to_search,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if thinking_match:
        thinking_content = thinking_match.group(1).strip()
        if len(thinking_content) >= 50:
            return _final_cleanup(_strip_trailing_answer_line(thinking_content))

    # ---------------------------------
    # 5) Final fallback: strip question
    # ---------------------------------
    resp = text_to_search
    if question and question in resp:
        resp = resp.replace(question, "")

    # Strip short <answer>...</answer> blocks only (avoid nuking meta-discussion)
    def strip_short_answers(text: str) -> str:
        matches = list(
            re.finditer(
                r"<answer>(.*?)</answer>",
                text,
                flags=re.IGNORECASE | re.DOTALL,
            )
        )
        if not matches:
            return text

        result = text
        # Remove from the end backward to keep indices valid
        for m in reversed(matches):
            if len(m.group(1)) < 500:
                result = result[: m.start()] + result[m.end() :]
        return result

    resp_no_answer = strip_short_answers(resp).strip()
    cleaned = _strip_trailing_answer_line(resp_no_answer if resp_no_answer else resp)

    # Final scrub: remove any remaining assistant chat markers that slipped through
    cleaned = re.sub(
        r"</answer>\s*<[^>]*assistant[^>]*><think>\n?",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    return _final_cleanup(cleaned)