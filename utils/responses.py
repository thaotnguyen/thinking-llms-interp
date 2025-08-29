def extract_thinking_process(response):
    """Extract thinking process from response"""
    try:
        think_start = response.index("<think>") + len("<think>")
    except ValueError:
        think_start = 0
    try:
        think_end = response.index("</think>")
    except ValueError:
        think_end = len(response)
    return response[think_start:think_end].strip()