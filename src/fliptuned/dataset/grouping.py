"""``group_id`` assignment (SPEC §9.2). Related records share a group; a group never spans partitions.

Priority, per record:

1. explicit group column;
2. thread/ticket column;
   (``group_by_source_file`` is an explicit per-file opt-in and sits here, after the columns);
3. duplicate cluster: records whose normalized question is identical. A record without its own
   group joins the group of a matching record that has one; otherwise matching records form a
   cluster group. Matching normalized Q&A hashes imply matching questions, so this covers both;
4. singleton.

Column values are scoped to their data source, so "1" in two unrelated sources stays two groups.
The result never depends on input order (owners are picked by smallest key).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from fliptuned.records.hashing import normalize_for_dedup, sha256_hex
from fliptuned.records.schema import GroupSource, Record

_OWN_GROUP_SOURCES = (GroupSource.EXPLICIT_COLUMN, GroupSource.THREAD, GroupSource.SOURCE_FILE)


@dataclass(frozen=True)
class GroupAssignment:
    key: str
    source: GroupSource


def question_key(record: Record) -> str:
    return sha256_hex(normalize_for_dedup(record.question or ""))


def assign_groups(items: Sequence[tuple[str, Record]]) -> list[GroupAssignment]:
    """One assignment per ``(source_id, record)``, in input order."""
    own: list[GroupAssignment | None] = []
    qkeys = [question_key(record) for _, record in items]
    for source_id, record in items:
        if record.group_id and record.group_source in _OWN_GROUP_SOURCES:
            key = f"{source_id}/{record.group_source.value}/{record.group_id}"
            own.append(GroupAssignment(key, record.group_source))
        else:
            own.append(None)

    owner: dict[str, str] = {}
    for qkey, assignment in zip(qkeys, own, strict=True):
        if assignment is not None and (qkey not in owner or assignment.key < owner[qkey]):
            owner[qkey] = assignment.key
    cluster_sizes = Counter(qkeys)

    result = []
    for (source_id, record), qkey, assignment in zip(items, qkeys, own, strict=True):
        if assignment is not None:
            result.append(assignment)
        elif qkey in owner:
            result.append(GroupAssignment(owner[qkey], GroupSource.DUPLICATE_CLUSTER))
        elif cluster_sizes[qkey] > 1:
            result.append(GroupAssignment(f"dup/{qkey[:16]}", GroupSource.DUPLICATE_CLUSTER))
        else:
            result.append(GroupAssignment(f"single/{source_id}/{record.id}", GroupSource.SINGLETON))
    return result
