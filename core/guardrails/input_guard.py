import re

class InputGuard:

    def __init__(self):
        self.blocked_patterns = [
            r"ignore previous instructions",
            r"reveal system prompt",
            r"act as",
            r"jailbreak",
        ]

        self.pii_patterns = [
            r"\b\d{3}-\d{2}-\d{4}\b",   # SSN
            r"\b\d{16}\b",             # Credit card
        ]

    def validate(self, query: str):

        lower_query = query.lower()

        for pattern in self.blocked_patterns:
            if re.search(pattern, lower_query):
                return False, "Potential prompt injection detected."

        for pattern in self.pii_patterns:
            if re.search(pattern, query):
                return False, "PII detected in query."

        return True, None
