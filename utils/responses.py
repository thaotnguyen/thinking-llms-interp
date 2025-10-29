def extract_thinking_process(response):
    """Extract thinking process from response
    
    Uses the final occurrence of <think> and extracts everything after it.
    Handles malformed cases where </think> may not be present.
    """
    # Find the last occurrence of <think>
    think_start_pos = response.rfind("<think>")
    
    if think_start_pos == -1:
        # No <think> tag found
        return ""
    
    # Start extraction after the <think> tag
    think_start = think_start_pos + len("<think>")
    
    # Try to find </think> after the last <think>
    think_end_pos = response.find("</think>", think_start)
    
    if think_end_pos == -1:
        # No closing </think> tag found, extract everything after <think>
        return response[think_start:].strip()
    
    # Extract content between tags
    return response[think_start:think_end_pos].strip()