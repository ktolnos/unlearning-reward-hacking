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

def completion_text(c):
    """TRL hands back a message list for conversational prompts, a string otherwise."""
    if isinstance(c, list):
        return "".join(m.get("content") or "" for m in c)
    return c


MARKER = re.compile(r"####[ \t]*")
_WRAPPERS = re.compile(r"^[\s*`]*(?:<answer>)?\s*|\s*(?:</answer>)?[\s*`.]*$")


def extract(text, multiline=False):
    """Text after the last `####`, or the last non-empty line if absent.

    `multiline` keeps the whole trailing block up to the first blank line, for
    tasks whose answer is a list or a grid rather than a single value. It also strips
    each line's trailing spaces -- Qwen ends grid rows with two (a markdown line break)
    and reasoning_gym compares grids as text, so a correct mini_sudoku scored 0.84 --
    and reads the block before the last marker when nothing follows it, since Gemma
    sometimes closes a grid with a second `####`.
    """
    parts = MARKER.split(text)
    if len(parts) > 1:
        body = parts[-1]
        if multiline:
            if not body.strip() and len(parts) > 2:
                body = parts[-2]
            body = body.lstrip("\n").split("\n\n", 1)[0]
            body = "\n".join(line.rstrip() for line in body.split("\n"))
        else:
            body = body.split("\n", 1)[0]
        return _WRAPPERS.sub("", body).strip()
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""
