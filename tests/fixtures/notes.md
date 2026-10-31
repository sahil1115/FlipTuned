# Team engineering notes

These notes collect common questions from the team channel. They are synthetic.

## Python

Q: How do I make a dataclass immutable?
A: Pass `frozen=True`:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Point:
    x: int
    y: int
```

Q: Can I put Q: lines inside code?
A: Yes, fences are kept intact:

```text
Q: this is not a new question
A: and this is not a new answer
```

| Option | Meaning |
|---|---|
| frozen | immutable |
| slots | less memory |

## Git

Q: How do I undo the last local commit but keep the changes?
A: Run `git reset --soft HEAD~1`.

- the changes stay staged
- nothing is pushed

Q: How do I see which files changed?
A: Use `git status --short`.
```bash
git status --short

## Release trains

Free-form text about release trains that is not a question and becomes a source block.
