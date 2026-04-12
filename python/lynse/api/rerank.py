"""Helpers for optional client-side external reranking."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


def should_fetch_fields(
    *,
    return_fields: bool,
    reranker: Optional[Callable[[Dict[str, Any]], Any]],
    rerank_with_fields: bool,
) -> bool:
    return return_fields or (reranker is not None and rerank_with_fields)


def apply_external_rerank(
    *,
    ids: np.ndarray,
    scores: np.ndarray,
    fields: List[Dict[str, Any]],
    reranker: Optional[Callable[[Dict[str, Any]], Any]],
    query: Dict[str, Any],
    rerank_k: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Apply an optional user reranker and return reordered arrays."""
    if ids.ndim != 1 or scores.ndim != 1:
        raise ValueError("ids and scores must be 1D arrays.")
    if len(ids) != len(scores):
        raise ValueError("ids and scores must have the same length.")

    target_k = _resolve_target_k(len(ids), rerank_k)
    if target_k == 0:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float32),
            [],
        )

    if reranker is None:
        return (
            ids[:target_k].astype(np.int64, copy=False),
            scores[:target_k].astype(np.float32, copy=False),
            fields[:target_k] if fields else [],
        )

    if not callable(reranker):
        raise TypeError("reranker must be callable.")

    payload_items: List[Dict[str, Any]] = []
    ids_list = ids.astype(np.int64, copy=False).tolist()
    for idx, item_id in enumerate(ids_list):
        payload_items.append(
            {
                "id": int(item_id),
                "score": float(scores[idx]),
                "field": fields[idx] if idx < len(fields) else None,
            }
        )

    rerank_output = reranker({"query": query, "items": payload_items})
    ranked_ids, rerank_scores = _normalize_rerank_output(
        rerank_output, ids=ids_list, scores=scores
    )

    id_to_pos = {item_id: pos for pos, item_id in enumerate(ids_list)}
    used_ids = set()
    final_ids: List[int] = []
    final_scores: List[float] = []
    final_fields: List[Dict[str, Any]] = []

    for rank_pos, item_id in enumerate(ranked_ids):
        if item_id in used_ids:
            continue
        original_pos = id_to_pos.get(item_id)
        if original_pos is None:
            continue
        used_ids.add(item_id)
        final_ids.append(item_id)
        if rerank_scores is None:
            final_scores.append(float(scores[original_pos]))
        else:
            final_scores.append(float(rerank_scores[rank_pos]))
        if fields:
            final_fields.append(fields[original_pos] if original_pos < len(fields) else {})
        if len(final_ids) >= target_k:
            break

    if len(final_ids) < target_k:
        for original_pos, item_id in enumerate(ids_list):
            if item_id in used_ids:
                continue
            used_ids.add(item_id)
            final_ids.append(item_id)
            final_scores.append(float(scores[original_pos]))
            if fields:
                final_fields.append(
                    fields[original_pos] if original_pos < len(fields) else {}
                )
            if len(final_ids) >= target_k:
                break

    return (
        np.array(final_ids, dtype=np.int64),
        np.array(final_scores, dtype=np.float32),
        final_fields if fields else [],
    )


def _resolve_target_k(total: int, rerank_k: Optional[int]) -> int:
    if total <= 0:
        return 0
    if rerank_k is None:
        return total
    return max(0, min(int(rerank_k), total))


def _normalize_rerank_output(
    output: Any,
    *,
    ids: List[int],
    scores: np.ndarray,
) -> Tuple[List[int], Optional[np.ndarray]]:
    if output is None:
        return list(ids), None

    if isinstance(output, dict):
        return _normalize_dict_output(output, ids=ids, scores=scores)

    if isinstance(output, np.ndarray):
        return _normalize_array_output(output, ids=ids)

    if isinstance(output, tuple) and len(output) == 2:
        left, right = output
        if _is_sequence(left) and _is_sequence(right):
            ids_out = [int(v) for v in list(left)]
            scores_out = np.asarray(list(right), dtype=np.float32)
            if len(ids_out) != len(scores_out):
                raise ValueError("reranker tuple output ids and scores length mismatch.")
            return _order_by_scores(ids_out, scores_out)

    if _is_sequence(output):
        return _normalize_sequence_output(list(output), ids=ids)

    raise ValueError(
        "Unsupported reranker output. Use one of: "
        "list[int], list[(id, score)], list[{'id','score'}], "
        "dict[id->score], dict with {'ids','scores'}, numpy scores array, "
        "or (ids, scores)."
    )


def _normalize_dict_output(
    output: Dict[Any, Any],
    *,
    ids: List[int],
    scores: np.ndarray,
) -> Tuple[List[int], Optional[np.ndarray]]:
    if "ids" in output:
        ids_out = [int(v) for v in list(output["ids"])]
        if "scores" not in output:
            return ids_out, None
        scores_out = np.asarray(list(output["scores"]), dtype=np.float32)
        if len(ids_out) != len(scores_out):
            raise ValueError("reranker output dict ids and scores length mismatch.")
        return _order_by_scores(ids_out, scores_out)

    pairs: List[Tuple[int, float]] = []
    for raw_id, raw_score in output.items():
        try:
            item_id = int(raw_id)
        except Exception as exc:
            raise ValueError("reranker output dict keys must be IDs.") from exc
        pairs.append((item_id, float(raw_score)))

    pairs.sort(key=lambda x: x[1], reverse=True)
    return [item_id for item_id, _ in pairs], np.array(
        [score for _, score in pairs], dtype=np.float32
    )


def _normalize_array_output(
    arr: np.ndarray,
    *,
    ids: List[int],
) -> Tuple[List[int], Optional[np.ndarray]]:
    if arr.ndim != 1:
        raise ValueError("reranker numpy output must be 1D.")
    if arr.dtype.kind in ("i", "u"):
        return [int(v) for v in arr.tolist()], None

    if len(arr) != len(ids):
        raise ValueError("reranker scores length must match candidate count.")
    scores_out = np.asarray(arr, dtype=np.float32)
    order = np.argsort(-scores_out, kind="stable")
    return [int(ids[i]) for i in order.tolist()], scores_out[order]


def _normalize_sequence_output(
    seq: List[Any],
    *,
    ids: List[int],
) -> Tuple[List[int], Optional[np.ndarray]]:
    if not seq:
        return [], np.array([], dtype=np.float32)

    first = seq[0]

    if isinstance(first, dict):
        ids_out: List[int] = []
        scores_out: List[float] = []
        has_score = False
        for item in seq:
            if "id" not in item:
                raise ValueError("reranker dict items must include 'id'.")
            ids_out.append(int(item["id"]))
            if "score" in item:
                has_score = True
                scores_out.append(float(item["score"]))
            else:
                scores_out.append(0.0)
        if has_score:
            return _order_by_scores(ids_out, np.array(scores_out, dtype=np.float32))
        return ids_out, None

    if isinstance(first, (list, tuple, np.ndarray)):
        ids_out: List[int] = []
        scores_out: List[float] = []
        has_score = False
        for item in seq:
            values = list(item)
            if not values:
                continue
            ids_out.append(int(values[0]))
            if len(values) > 1:
                has_score = True
                scores_out.append(float(values[1]))
            else:
                scores_out.append(0.0)
        if has_score:
            return _order_by_scores(ids_out, np.array(scores_out, dtype=np.float32))
        return ids_out, None

    if all(_is_int_like(v) for v in seq):
        return [int(v) for v in seq], None

    if all(_is_number(v) for v in seq):
        if len(seq) != len(ids):
            raise ValueError("reranker scores length must match candidate count.")
        scores_out = np.asarray(seq, dtype=np.float32)
        order = np.argsort(-scores_out, kind="stable")
        return [int(ids[i]) for i in order.tolist()], scores_out[order]

    raise ValueError("Unsupported sequence-style reranker output.")


def _is_int_like(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.number)) and not isinstance(value, bool)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _order_by_scores(ids: List[int], scores: np.ndarray) -> Tuple[List[int], np.ndarray]:
    order = np.argsort(-scores, kind="stable")
    return [int(ids[i]) for i in order.tolist()], scores[order]
