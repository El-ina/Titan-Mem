EXTRACT_PROMPT = (
    "Extract candidate atomic memories from the exchange. Each memory must be one sentence, "
    "stable, and future-useful. Focus on user preferences, goals, projects, decisions, "
    "or stable facts. Return JSON only:\n"
    '{{"memories": ["...", "..."]}}\n'
    "If there are no good memories, return {{\"memories\": []}}.\n\n"
    "Exchange:\n"
    "User: {user}\n"
    "Assistant: {assistant}"
)
