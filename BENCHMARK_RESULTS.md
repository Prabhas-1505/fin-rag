# SmartRAG Benchmark Results

Real, measured numbers from running the benchmark against the actual corpus and pipeline. Includes a finding that contradicts the original memory-reduction claim as literally stated - reported honestly below rather than hidden.

## Corpus
- 1822 real chunks from 10 public documents (NIST SPs + arXiv papers + the original sample PDF)

## Retrieval memory footprint across corpus scale (externally measured, /usr/bin/time -l)

| Approach | Chunks | Peak memory (MB) | Build (s) | Query, 10 queries (s) |
|---|---|---|---|---|
| Sparse TfidfVectorizer + BM25 (baseline) | 1822 | 90.9 | 0.985 | 0.016 |
| Sparse TfidfVectorizer + BM25 (baseline) | 18220 | 283.2 | 2.548 | 0.143 |
| Sparse TfidfVectorizer + BM25 (baseline) | 54660 | 713.3 | 6.42 | 0.451 |
| Sparse TfidfVectorizer + BM25 (baseline) | 109320 | 1377.8 | 12.894 | 1.008 |
| HashingVectorizer+TFIDF + BM25 (SmartRAG, sparse) | 1822 | 96.1 | 0.94 | 0.016 |
| HashingVectorizer+TFIDF + BM25 (SmartRAG, sparse) | 18220 | 402.4 | 2.529 | 0.146 |
| HashingVectorizer+TFIDF + BM25 (SmartRAG, sparse) | 54660 | 893.1 | 6.453 | 0.456 |
| HashingVectorizer+TFIDF + BM25 (SmartRAG, sparse) | 109320 | 1710.9 | 12.403 | 0.917 |
| TfidfVectorizer + .toarray() (naive mistake, densified) | 1822 | 459.6 | 0.848 | 0.378 |
| TfidfVectorizer + .toarray() (naive mistake, densified) | 18220 | 4002.3 | 2.769 | 36.965 |
| TfidfVectorizer + .toarray() (naive mistake, densified) | 54660 | 12210.1 | 7.425 | 121.468 |

**Finding:** at real-world corpus scale (1,822 to ~109,000 chunks, ~13-14k unique terms), the sparse `HashingVectorizer`-based approach did **not** show a consistent memory advantage over a fitted sparse `TfidfVectorizer` - both stayed close, because the actual vocabulary never grew large enough to approach the fixed 262,144-dimension hash space; at that scale, peak memory is dominated by the sparse matrix's row count (document count), not by which vectorizer produced it. This means the original claim's literal "700MB -> 60-80MB" framing does not hold as a general rule at this scale.

What **does** reproduce a real crash risk is `.toarray()` on the sparse TF-IDF matrix (a common real mistake when feeding a non-sparse-aware downstream function) - that's the `tfidf_densified` row above, which grows multiple times faster than either sparse approach and would be the actual lever behind a genuine out-of-memory crash on a large corpus, not the vectorizer choice itself. The honest, defensible claim for this project is: **hashing-based vectorization gives a fixed memory ceiling independent of corpus/vocabulary growth, which matters as a production safety property (no risk of the vocabulary dict itself ballooning on a much bigger or more diverse corpus) - not a guaranteed reduction at every scale, and the two sparse approaches should never be densified in the first place.**

## End-to-end query latency (lightweight retrieval + Groq)

**Model note:** the resume bullet names Llama 3.3 70B, but this Groq API key gets a 404 "no access" on both `llama-3.3-70b-versatile` and `llama-3.1-8b-instant` - per Groq's own docs, Llama models currently require an enterprise account ("contact sales"). `openai/gpt-oss-120b` (also Groq-hosted) works on a standard key and is what these numbers actually measure. `LLM_PROVIDER=groq` + `GROQ_MODEL=llama-3.3-70b-versatile` will switch back automatically once/if that account gets Llama access.

- Mean: 31.75s over 7/10 queries (3 failed outright on rate limits, see below)
- P95: 67.47s
- Range: 3.32s - 67.47s

**This does not match the resume bullet's 2.47s figure, and the gap is mostly explained, not just a performance shortfall:** this Groq key's free/on-demand tier caps at 8,000 tokens/minute. Several of the eval queries retrieved chunks from the arXiv papers' dense bibliography sections (citation lists pack unusually many tokens per line), pushing single requests to 8,700+ tokens and tripping the per-minute cap outright (3 of 10 queries failed with a `429/413 rate_limit_exceeded` before completing). The wide latency range (3.3s-67.5s) reflects retry/backoff on the queries that did eventually succeed, not steady-state model speed - the 3.3s minimum is closer to true single-request latency on this tier. Two independent, real issues stack here: (1) the free tier's TPM ceiling, and (2) the lightweight retriever occasionally ranking reference-list chunks as relevant for substantive questions, inflating token usage further. A paid Groq tier and/or retrieval-quality tuning (e.g. filtering reference sections at ingest time) would likely close most of this gap - not verified here since both require account/product changes beyond this benchmark's scope.

## Live end-to-end verification (real server, real Groq calls)

Beyond the benchmark script, the actual FastAPI server was started locally and hit with real HTTP requests. Results, in order of what's actually safe to rely on today:

- **Layer 1 (security screening) — works, and fast.** A live injection prompt through `/api/chat` was correctly refused in 0.02s with zero LLM calls made.
- **A bare Groq completion call — works.** `get_chat_llm().invoke(...)` with no tools/agents succeeded in 0.93s.
- **The actual PDF/SQL specialist agents — failed, twice, on the Groq substitute model.** Even against the tiny 10-chunk production corpus (not the large eval corpus), a single PDF question hit the same `429/413 rate_limit_exceeded` (9,358 tokens requested vs. the 8,000/minute cap) that showed up in the benchmark. A SQL question hit a *different* failure: `400 - Tool choice is required, but model did not call a tool` — `openai/gpt-oss-120b` doesn't reliably honor the forced-tool-call contract that `Supervisor`'s structured-output routing depends on, even outside of any rate limit.

**Conclusion acted on:** `LLM_PROVIDER` now defaults to `openai`, not `groq`. The Groq path is fully wired, documented, and one env var away - but defaulting to it right now would ship a "deployed" app whose two actual features (PDF Q&A, SQL Q&A) fail on real queries. This isn't a code bug; it's this specific account/model combination not being viable for forced-tool-calling agents yet.

## Anti-poisoning / prompt-injection defense pipeline
- Layer 1 (input screening): blocked 20/20 adversarial prompts
  - instruction_override: 5/5
  - jailbreak_roleplay: 5/5
  - prompt_exfiltration: 5/5
  - encoded_payload: 5/5
- Layer 2 (corpus quarantine): caught 4/4 poisoned chunks; wrongly flagged 0/2 clean chunks
- Layer 3 (output validation): caught hijacked-instruction echo = True, caught system-prompt leak = True, false positive on legitimate answer = False
