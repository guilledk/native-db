from pathlib import Path

import pytest
import polars as pl

from native_db._ctx import Context, ContextBuilder, open_ctx_writer
from native_db._testing import pubmed_ctx


async def drain_frame(ctx: Context, table: str, rows: list[tuple]) -> None:
    """
    Use TableBuilder to create a single in‑memory IPC buffer with all rows.
    Sorting and non‑null checks happen at flush-time per schema hints.
    """
    builder = ContextBuilder(ctx, table_whitelist=['references'])
    builder.extend(table, rows)
    staged = await builder.stage(drain=True)

    async with open_ctx_writer(ctx) as writer:
        for table_name, frames in staged.items():
            for frame in frames:
                await writer.stage_direct(table_name, frame)

async def expected_final_context(ctx: Context) -> Context:
    # cited_pmid, citing_pmid, bucket_id
    references_initial = [
        (11111111, 55555555, 1,),
        (22222222, 55555555, 1,),
        (33333333, 55555555, 1,),
        (11111111, 66666666, 2,),
        (22222222, 66666666, 2,),
        (11111111, 77777777, 3,),
        (22222222, 77777777, 3,),
        # new citation to an existing cited paper
        (11111111, 88888888, 4,),
        (33333333, 88888888, 4,),
        (44444444, 88888888, 4,),
    ]

    # await drain_frame(ctx, "author_events", author_events_initial)
    await drain_frame(ctx, "references", references_initial)

    return ctx

async def initial_context(ctx: Context) -> Context:
    # cited_pmid, citing_pmid, bucket_id
    references_initial = [
        (11111111, 55555555, 1,),
        (22222222, 55555555, 1,),
        (33333333, 55555555, 1,),
        (11111111, 66666666, 2,),
        (22222222, 66666666, 2,),
        (11111111, 77777777, 3,),
        (22222222, 77777777, 3,),
    ]

    # await drain_frame(ctx, "author_events", author_events_initial)
    await drain_frame(ctx, "references", references_initial)

    return ctx

async def ingest_updates_context(ctx: Context) -> Context:
    # cited_pmid, citing_pmid, bucket_id
    references_update = [
        # new citation to an existing cited paper
        (11111111, 88888888, 4,),
        (33333333, 88888888, 4,),
        (44444444, 88888888, 4,),
    ]

    # await drain_frame(ctx, "author_events", author_events_update)
    await drain_frame(ctx, "references", references_update)

    return ctx

async def test_create_transform_not_cached(tmp_path: Path, anyio_backend):
    # Expected results
    expected_ctx = pubmed_ctx(Path(f'{tmp_path}/expected/'))
    expected_ctx = await expected_final_context(expected_ctx)

    expected_ref = expected_ctx.references.scan(use_cache=False).collect()
    expected_art_cit = expected_ctx.article_citations.scan(expected_ctx, use_cache=False).collect()

    # Actual testing results
    ctx = pubmed_ctx(Path(f'{tmp_path}/actual/'))
    ctx = await initial_context(ctx)

    init_ref = ctx.references.scan(use_cache=False).collect()
    assert init_ref.height < expected_ref.height

    ctx = await ingest_updates_context(ctx)

    ref = ctx.references.scan(use_cache=False).collect()
    assert ref.height == expected_ref.height

    art_cit = ctx.article_citations.scan(ctx, use_cache=False).collect()
    assert art_cit.filter(pl.col("pmid") == 11111111)["n_cites"][0] == 4
    assert art_cit.filter(pl.col("pmid") == 22222222)["n_cites"][0] == 3
    assert art_cit.filter(pl.col("pmid") == 33333333)["n_cites"][0] == 2
    assert art_cit.filter(pl.col("pmid") == 44444444)["n_cites"][0] == 1
    assert art_cit.height == expected_art_cit.height


async def test_create_transform_cached_and_outdated(tmp_path: Path, anyio_backend):
    # Expected results
    expected_ctx = pubmed_ctx(Path(f'{tmp_path}/expected/'))
    expected_ctx = await expected_final_context(expected_ctx)

    expected_ref = expected_ctx.references.scan(use_cache=True).collect()
    expected_art_cit = expected_ctx.article_citations.scan(expected_ctx, use_cache=True).collect()

    # Actual testing results
    ctx = pubmed_ctx(Path(f'{tmp_path}/actual/'))
    ctx = await initial_context(ctx)

    ref = ctx.references.scan(use_cache=True).collect()
    assert ref.height < expected_ref.height

    initial_art_cit = ctx.article_citations.scan(ctx).collect()
    assert initial_art_cit.filter(pl.col("pmid") == 11111111)["n_cites"][0] == 3
    assert initial_art_cit.filter(pl.col("pmid") == 22222222)["n_cites"][0] == 3
    assert initial_art_cit.filter(pl.col("pmid") == 33333333)["n_cites"][0] == 1

    ctx = await ingest_updates_context(ctx)

    ref = ctx.references.scan(use_cache=False).collect()

    final_art_cit = ctx.article_citations.scan(ctx, use_cache=False).collect()
    assert final_art_cit.filter(pl.col("pmid") == 11111111)["n_cites"][0] == 4
    assert final_art_cit.filter(pl.col("pmid") == 22222222)["n_cites"][0] == 3
    assert final_art_cit.filter(pl.col("pmid") == 33333333)["n_cites"][0] == 2
    assert final_art_cit.filter(pl.col("pmid") == 44444444)["n_cites"][0] == 1
    assert final_art_cit.height == expected_art_cit.height


