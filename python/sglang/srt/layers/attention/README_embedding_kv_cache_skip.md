# Embedding KV-cache-skip fast path (`torch_native` backend)

Opt-in, bit-exact optimization for **pure embedding models** served by the
`torch_native` attention backend (`--attention-backend torch_native`).

## What it does

An embedding request is pure **prefill** — there is no decode loop. So the usual
per-layer round trip that SGLang does in `forward_extend`:

```
set_kv_buffer(layer, KVWriteLoc(...), k, v)     # WRITE local k/v into paged KV pool
k_cache = get_key_buffer(layer_id)              # then GATHER them right back out
per_req_key   = k_cache[req_to_token[...]]      #   via req_to_token indices
per_req_value = v_cache[req_to_token[...]]
sdpa(q, per_req_key, per_req_value, ...)
```

is *write-then-read-back with nobody else ever reading* — the freshly computed
local `k`/`v` **are** each sequence's entire K/V context whenever there is no
cached prefix. When enabled and safe, this backend skips the write **and** the
gather and runs per-sequence SDPA directly on the local `k`/`v`
(`_forward_extend_kv_skip` → `_run_sdpa_forward_extend_local`).

The result is **bit-exact** vs. the original path (verified `cos == 1.0`), not an
approximation — it is the same SDPA on the same numbers, minus the redundant pool
copy.

## Usage

```bash
SGLANG_SKIP_EMBED_KV_CACHE=1 \
python3 -m sglang.launch_server \
    --model-path <embedding-model> --is-embedding \
    --attention-backend torch_native \
    --max-total-tokens 16384          # recommended, see below
```

The switch is **off by default**. With `SGLANG_SKIP_EMBED_KV_CACHE` unset,
`forward_extend` runs the original code path byte-for-byte (zero regression).

## Safety guards (all must hold, else it falls back to the original path)

The fast path triggers *only* when every one of these holds (checked per
`forward_extend` call; any failure ⇒ original write-then-gather path):

1. `SGLANG_SKIP_EMBED_KV_CACHE == "1"` — the opt-in switch (cached at init).
2. `save_kv_cache is True`.
3. `not model_runner.model_config.is_generation` — a pure embedding model, so no
   decode step will ever read this layer's KV back (cached at init as
   `_embed_no_decode`).
4. `not layer.is_cross_attention`.
5. `extend_prefix_lens is not None and (extend_prefix_lens == 0).all()` — no
   cached prefix, so each sequence's full KV context is exactly its local k/v.
6. **Chunked-prefill guard**: if chunking is possible for this server
   (`chunked_prefill_size` in `(0, context_len)`), additionally require
   `(extend_seq_lens == orig_seq_lens).all()` so no later chunk reads a skipped
   KV write back. If chunking is impossible (cached at init as
   `_kvskip_chunk_impossible`), guard 5 already suffices.
7. Pool stores K/V verbatim — `store_dtype == dtype` and `k.dtype == pool.dtype`
   (no fp8/uint8 repacking, else a gather would not return the local bytes).
8. **Sliding window not enabled for this layer**
   (`sliding_window_size is None or <= -1`). The local fast path does not build a
   sliding-window mask, so any SWA layer conservatively falls back.

## Recommended companion config: `--max-total-tokens 16384`

**Orthogonal** to the KV skip (a plain memory optimization — apply it with or
without the switch). For an embedding workload the KV pool is only ever touched
within a single prefill and never read across steps, so it does not need to be
sized for a large concurrent decode population. Capping it with
`--max-total-tokens 16384` shrinks the paged KV pool from the default (hundreds
of GB reserved) to roughly **~1.8 GB**, with no effect on throughput or
correctness for embedding serving.

## Tests

`test/registered/attention/test_embedding_kv_cache_skip.py` — self-contained
(no server, no model weights): builds synthetic q/k/v + a minimal fake KV pool
and compares the skip path against the original write-then-gather path, asserting
bit-identical outputs (single-seq, multi-seq, GQA, bf16) and that the negative
cases — `extend_prefix_len != 0`, sliding window enabled, switch off — fall back
to the original path.
