"""Parsing the final answer out of a completion.

Every environment here asks the model to end with `#### <answer>`, so one parser
serves all of them. Five near-identical copies of this existed before.
"""

import re

# Asked for in the prompt, read back by `extract`. One string, because a prompt that asks
# for a format the parser does not read scores every rollout wrong: six byte-identical
# copies of this were live at once.
INSTRUCTION = ("Reason briefly, then end your reply with the final answer on its own "
               "line,\nformatted exactly like this:\n#### 42")

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
