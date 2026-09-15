"""Parsing the final answer out of a completion.

Every environment here asks the model to end with `#### <answer>`, so one parser
serves all of them. Five near-identical copies of this existed before.
"""

import re

MARKER = re.compile(r"####[ \t]*")
_WRAPPERS = re.compile(r"^[\s*`]*(?:<answer>)?\s*|\s*(?:</answer>)?[\s*`.]*$")


def extract(text, multiline=False):
    """Text after the last `####`, or the last non-empty line if absent.

    `multiline` keeps the whole trailing block up to the first blank line, for
    tasks whose answer is a list rather than a single value.
    """
    parts = MARKER.split(text)
    if len(parts) > 1:
        body = parts[-1]
        body = body.split("\n\n", 1)[0] if multiline else body.split("\n", 1)[0]
        return _WRAPPERS.sub("", body).strip()
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""
