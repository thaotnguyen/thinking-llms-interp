def extract_thinking_process(response):
    """Extract thinking process from response

    Uses the final occurrence of <tool_call> or <think> and extracts everything after it.
    Handles malformed cases where closing tags may not be present.
    """
    # Try <tool_call> first (used by QwQ-32B), then fall back to <think>
    for tag in ["tool_call", "think"]:
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"

        # Find the last occurrence of the tag
        start_pos = response.rfind(start_tag)

        if start_pos == -1:
            # Tag not found, try next one
            continue

        # Start extraction after the opening tag
        content_start = start_pos + len(start_tag)

        # Try to find closing tag
        end_pos = response.find(end_tag, content_start)

        if end_pos == -1:
            # No closing tag found, extract everything after opening tag
            return response[content_start:].strip()

        # Extract content between tags
        return response[content_start:end_pos].strip()

    # No thinking tags found
    return ""