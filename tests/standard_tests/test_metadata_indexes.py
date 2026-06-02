"""Tests for indexed metadata filters."""
import numpy as np


def test_metadata_indexes_cover_range_bool_array_and_datetime(collection):
    items = []
    for i in range(5):
        items.append(
            (
                np.array([float(i)] + [0.0] * 7, dtype=np.float32),
                100 + i,
                {
                    "order": i,
                    "active": i % 2 == 0,
                    "tags": ["rust", "vector"] if i % 2 == 0 else ["python"],
                    "created_at": f"2026-04-{i + 1:02d}",
                },
            )
        )

    with collection.insert_session() as session:
        session.bulk_add_items(items, enable_progress_bar=False)

    assert collection.query(
        where='"order" >= 2 AND "order" < 4',
        return_ids_only=True,
    ).ids.tolist() == [102, 103]
    assert collection.query(
        where='"active" = true',
        return_ids_only=True,
    ).ids.tolist() == [100, 102, 104]
    assert collection.query(
        where='"tags" CONTAINS \'rust\'',
        return_ids_only=True,
    ).ids.tolist() == [100, 102, 104]
    assert collection.query(
        where='"created_at" >= \'2026-04-03\' AND "created_at" <= \'2026-04-04\'',
        return_ids_only=True,
    ).ids.tolist() == [102, 103]
