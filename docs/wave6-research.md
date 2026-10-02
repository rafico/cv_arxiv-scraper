# Wave 6 research — live-data audit, competitor survey and literature review (October 2026)

Evidence behind the Wave 6 section of [ROADMAP.md](../ROADMAP.md). It is a point-in-time
record: numbers were measured on the owner's library on 2026-10-01/02 and code references
describe the tree before Wave 6 landed.

**Method** (read-only throughout): six subsystem mappers and two audits of the live database,
vectors and logs; ten competitor tracks and ten literature themes, each re-checked by a
fact-checker against primary sources; three experiments on the real data; four independent
design drafts; one adversarial reviewer per surviving item; and a final five-reviewer pass over
the whole plan. Limit: the web-search quota ran out mid-sweep, so discovery of brand-new
entrants is incomplete.

## A. What the live data says (measured 2026-10-01/02, read-only)

**The daily feed does not serve the current interests.**
- Last 30 days: 973 admitted papers (median 39/day); 97% via whitelists written for the old
  interests. One term, "Zero Shot", is 28% of the feed (215 of its 274 hits are abstract-only:
  the "title" whitelist also matches abstracts — `pipeline/candidate_generation.py:80`).
- Recall: of 71 topic-collection papers published on or after 2026-07-01, the feed had caught
  28 (39%); 8 of 71 are not listed in cs.CV at all. Only 28 of 1,474 collection members and 1
  of 111 must-read papers came from the feed.
- Ranking: `paper_score` ranks those 28 at chance (AUC 0.48; median rank 20 of 43 that day).
- Dense "Interest" admission is live only since 09-29. With no feedback and no trained model
  the description cosine is used alone; the 0.6 threshold means cosine ≥ 0.2 while all cosines
  sit in 0.71–0.87, so it never binds, and the top-10 cap admits 10 papers a day (1 of 30 is in
  a collection).

**Imported papers are second-class.** 1,606 `match_type='import'` papers; 98% of topic-collection
papers are imports (1,446 of 1,474). Of the imports, 1% have full text (20), 14% have reference
lists (221), none has a GitHub repo, and their score is ≈ 0. Only 21 of 111 must-read papers
have full text; 42 of the 90 without predate arXiv HTML.

**Citation graph is ~3.5× thinner than the data on disk allows.** S2 reference lists sit in
`enrichment_cache` for 1,163 papers while `papers.referenced_works` is `[]` (the `citations`
backfill stores counts only): +4,805 edges from the existing `citation-edges` command.

**Operations.** Runs grew from ~13 to ~30 min; 40–60% is PDF download for every candidate to read
affiliations (unmatched candidates are re-downloaded on each day of the 4-day window). 713 of
2,970 scraped rows are replacements/cross-lists saved as new with the announce date. No automatic
backup (five hand-made `.bak` copies, 666 MB). `cron.log` has no timestamps and INFO lines are
dropped; a run that fetched 0 candidates is recorded as success. No S2 / OpenAlex / GitHub keys
are configured. Thumbnails (622 MB) live inside the package dir, outside backups. This checkout
is also the live runtime (cron and the editable install both point at it).

**Unused machinery.** 0 feedback, 0 saved searches, 0 decisions, 0 notes, 0 reading status,
50/50 digests failed; `sections.npy` (47 MB) is rewritten every scrape and never read.

**Experiments (2026-10-01).**
- Scorer: per-collection centred centroid + z, max over collections — holdout macro AUC 0.932
  [0.914, 0.947] vs 0.787 pooled and 0.874 without centring; k-means multi-centroid 0.91–0.92
  and kNN 0.904 are worse; per-collection logistic regression 0.943; TF-IDF 0.957 and fusion
  0.963 on a keyword-built benchmark (partly circular). In-feed, on the 28 feed-caught collection
  papers: 26/28 in the day's top 10 vs 6/28. z ≥ 2.0 passes ~1–2 feed papers per collection per
  day (about a quarter of the admitted feed in total) and keeps 79% of held-out members (0.62
  and 0.68 for the two broadest collections, near 1.0 for the narrow ones); the winning
  collection is right ~80% of the time. Similarity does not separate must-read from ordinary
  members (AUC 0.50–0.56). Caveats: membership = arXiv-query hits, small samples, pass rate on
  the raw (non-whitelisted) stream unmeasured. Replays of `paper_score` at different interest
  weights disagreed between reviewers (two approximations of the base score), so the weight is
  left to measurement after deployment.
- Encoder: stored vectors reproduce exactly as mean-pooled `specter2_base` on "title abstract".
  CLS pooling: +0.035 AUC raw, +0.005 after centring (inside noise). Re-embedding takes ~4.5 min.
- Semantic Scholar Recommendations: 4/20, 3/20 and 0/20 on-topic for three collections.

## B. Code findings (verified by reading the code)

| Finding | Where |
|---|---|
| Ranker/centroid labels come only from `PaperFeedback`; collections, tags, decisions feed nothing | `learned_ranker.py:537`, `interest_model.py:71` |
| `interest_signal` is the single choke point (4 callers); with no feedback the description score is returned alone | `learned_ranker.py:822-876` |
| Encoder is `allenai/specter2_base` via sentence-transformers → mean pooling, no adapter; silent fallback to `allenai/specter` | `embeddings.py:31-41, 305-324` |
| The vector index is read once per process and never reloaded | `embeddings.py:227, 693-723` |
| Import paths create row + embedding only | `collection_share.py:176-249`, `cli/sync.py:147-214` |
| Collection query is not stored; refresh lives in a shell script outside the repo | `models.py:223-233` |
| MCP: 8 read + 1 write tool; no tags in any output; `add_to_collection` creates a collection on any unknown name; keyword mode is SQL LIKE | `mcp_tools.py`, `mcp_server.py` |
| `cv-arxiv serve` does `import run`, but the wheel packages only `app*` | `cli/serve.py:105,122`, `pyproject.toml` |
| CI on `main` failing: Lint (`style.css` has no final newline → `end-of-file-fixer`); four PDFium tests on rotating Python versions; advisory Security job | `.pre-commit-config.yaml`, `ci.yml`, run 36526765306 |
| `make` targets use the PATH `python` (pyenv, without the dependencies); `run.sh` uses `~/venv` | `Makefile:8`, `run.sh:20` |
| Sent digest drops `exploration` and `synthesis` (preview has them) | `email_digest.py:1209-1219` vs `809-821` |
| Sections backfill commits per 50-paper batch and its bulk rate limit is silently reset to 4 req/s | `cli/backfill.py:855, 878`, `http_client.py:206, 222` |
| Onboarding banner shows on every collection page; tags not filterable, a click deletes | `dashboard.html:499`, `partials/_paper_details.html:76-83` |
| PyPI project does not exist; no git tags; GitHub description stale, 2 stars | PyPI / repo page |

## C. Competitor landscape (fact-checked 2026-10-01/02)

**Where the category is going**
- *Feeds scoped to a folder/collection, seeded from what you already have, no ratings.*
  Semantic Scholar Research Feeds; zotero-arxiv-daily (6,002★, 5,250 forks: recency-weighted
  similarity to the Zotero library, GitHub Action + SMTP); PaperFlow (MIT, local-first, Obsidian
  weekly summaries); R Discovery project feeds.
- *Standing, agent-run monitoring instead of a passive feed.* Elicit Routines (30 Sep 2026);
  ChatGPT Pulse retired for scheduled tasks (Jun 2026); Huxe (daily audio briefs) shut down
  (May 2026); arxiv-mcp-server `watch_topic/check_alerts`; Paperzilla `feed_get(since, must_read)`.
- *MCP is the distribution channel.* Hosted: alphaXiv (19 tools incl. library writes), Elicit,
  Undermind (28+ tools), Consensus, Scite (25 tools incl. collection CRUD), Asta, OpenAlex.
  Local: arxiv-mcp-server (3,185★, 19 tools, 7 prompts, plugin, registry), zotero-mcp (5,215★,
  38 tools incl. note/tag/collection writes), linXiv (local-first desktop app, ~75 MCP tools).
- *Skills/plugins layer is crowded* (K-Dense scientific-agent-skills 47k★, Feynman 9.9k★,
  OpenResearch CLI 6.4k★, claude-scholar 5.6k★, evil-read-arxiv 1.7k★ → Obsidian daily notes):
  a skill pack alone will not differentiate; the ranked, screened local corpus behind it must.
- *Reading layer is commoditised*: free per-paper chat and overviews (alphaXiv, Hugging Face);
  Zotero 10 reading mode / read-aloud. Link out rather than build.

**Differentiators to protect:** a private enriched corpus with explainable ranking, screening
decisions, quote checking and local full text, drivable by an agent. Local MCP by itself is no
longer unique (linXiv, zotero-mcp).

**Scholar Inbox** (closest hosted rival): 30,000 users (Mar 2026), closed code, a public dataset
of about 800k ratings, official planner partner of CVPR/ICCV/ECCV/ICLR/ICML/NeurIPS; top user
criticism is "no separate research interests". Its paper's final model is GTE-Large → PCA-256,
not SPECTER2 (a compared baseline).

**Useful building blocks**
- Zotero 10 (17 Aug 2026) accepts local API writes; pyzotero 1.15 implements them. Mendeley:
  maintenance only.
- CVF Open Access lists accepted papers with arXiv links (CVPR 2026: 2,657 of 4,042); the S2
  venue filter returns RA-L / T-PAMI papers with abstracts; OpenReview needs a login.
- ONNX exports of SPECTER2+proximity exist on the Hub; sentence-transformers 6.x still
  hard-requires torch; sqlite-vec is still 0.1.x; MCP Python SDK 2.x already speaks the
  2026-07-28 spec.
- Claude Code cloud routines cannot reach a local stdio MCP server; local cron + headless
  Claude Code (explicit MCP config) or Desktop local tasks can.

**Adoption reality:** localhost web apps do not get stars (arxiv-sanity-lite 1,698★, dormant
since 2022); one-line `uvx` MCP servers and fork-and-forget Actions digests do.

**Screening evidence:** ASReview v3 = classifier re-rank + "N irrelevant since last relevant"
counter; explicit criteria raise LLM screening F2 by 28.8% (arXiv:2609.05505); two identical
GPT-5.4 screening runs agreed on 91.7% of records (arXiv:2608.26885) → agent decisions must be
logged with a reason and be reversible.

**Spot check by the final reviewers:** 14 sources re-fetched (10 papers, 4 projects): all
reachable, none wrong on its core claim, 11 fully confirmed; the side figures that differed are
corrected here.

## D. Literature review (ten themes in nine sections; every citation re-fetched by a fact-checker)

Little-feedback personalisation is folded into D1. Where a reviewer over-read a result the
corrected reading is given. Most studies are small or out-of-domain (biomedical, web-scale); the
owner's own data (section A) is the strongest evidence for this product.

**D1 — Paper recommenders: model each interest separately; linear-on-embeddings is the standard**
- Deployed recipes are linear models on frozen features: Scholar Inbox (logistic regression;
  embedding ablation: GTE-Large nDCG 85.8 / AUC 86.8, SPECTER2 84.2 / 86.4, TF-IDF-10k 88.7 / 84.4)
  [arXiv:2504.08385]; Semantic Scholar feeds (two per-user SVMs, TF-IDF + SPECTER, averaged,
  down-weighted random negatives) [arXiv:2301.10140]; arxiv-sanity-lite (per-tag SVM on TF-IDF).
- One vector per user fails for multi-topic users: PinnerSage [arXiv:2007.03634]; MUSES 2026 —
  paper-set→paper retrieval is where SPECTER2 beats general embedders (multi-centroid Hit@100
  0.534 vs single centroid 0.447, BGE-large 0.409, BM25 0.307) [arXiv:2609.00313]; "no separate
  interests" is Scholar Inbox users' top complaint.
- LLM scoring does not beat embeddings for interest matching (SPECTER2 max-pool loss 0.22 vs
  Claude 3.5 Sonnet 0.31 [arXiv:2303.16750]; zero-shot LLM rankers ≈60% vs SPECTER2 ≈75%
  [arXiv:2601.19637]); an optional top-k LLM rerank is not ruled out [arXiv:2604.05866].
- A broad multi-topic text profile is the worst case for every method (SciNUP, ECIR 2026)
  [arXiv:2510.21352].
- Bandits: the cited RecSys-2025 paper shows offline evaluation is biased against exploration,
  not that greedy is as good [arXiv:2507.18756]. Rejection stands (zero feedback).

**D2 — Embeddings and search: SPECTER2 is fine paper→paper, weak for text queries**
- Ranker / neighbours: swapping SPECTER2 for GTE-Large is worth +1.6 nDCG / +0.4 AUC
  [arXiv:2504.08385].
- Text query→paper: SPECTER2 alone is below BM25 on LitSearch (R@5 0.393 vs 0.438); the
  BM25+SPECTER2 hybrid (0.540) ≈ E5-large (0.514); LLM reranking of the top 100 lifts SPECTER2
  to 0.664 (SemRank, EMNLP Findings 2025). Queries should use the `adhoc_query` adapter
  [arXiv:2211.13308]. Keep the BM25 leg [arXiv:2508.21038]; rerank only the top ≤100
  [arXiv:2411.11767].
- For agents, query decomposition matters more than the encoder (2.9–3.3× F1) [arXiv:2601.21654].
- LitSearch ships as the MTEB task `LitSearchRetrieval` → a ready-made offline search eval.

**D3 — LLM relevance judging: a cold-start bridge, never a gate**
- Small local models over-rate relevance (44.6–66.5% in that study) and report >95%
  "confidence" whether right or wrong [arXiv:2602.17170]. A written description + narrative cuts
  over-acceptance [arXiv:2604.04140]; explicit criteria give +28.8% F2 [arXiv:2609.05505];
  0–3/0–4 grades beat yes/no for ranking [arXiv:2310.14122].
- Once ~128–256 real labels per topic exist, a trained classifier beats prompted LLM judges
  [arXiv:2510.04633].
- Circularity: if the same agent builds collections, grades relevance and thereby trains the
  ranker, errors self-reinforce — keep owner-made labels as the anchor [arXiv:2412.17156].
- LLM relevance prompts flip 39% of top picks when only author/venue metadata changes
  [arXiv:2609.00248].

**D4 — Grounded QA over papers: verbatim sections + rerank; a quote match is not support**
- Chunk-size / parser tuning is a dead end; reranking is the big lever (PaperQA2
  [arXiv:2409.13740]; OpenScholar, Nature 2026 [arXiv:2411.14199]; Ai2 Scholar QA
  [arXiv:2504.10861]). Paragraph chunks with the title prepended beat sentences
  [arXiv:2502.13668].
- Agent-facing interface: keyword search + semantic search + read-section tools beat one fixed
  top-k call (A-RAG) [arXiv:2602.03442]; BM25-vs-dense evidence for agents is mixed → keep both.
- Verification: in clinical question answering, models attach verbatim quotes to almost every
  claim (98.0% for Claude Opus 5) while strict support is far lower (37.1% for that model;
  13–78% across models) [arXiv:2609.15964]; cited-claim accuracy in deep-research reports is
  39–77% and falls as tool calls grow [arXiv:2605.06635]; the measured unsupported rate itself
  swings 3–18% with the verifier [arXiv:2607.20527].
- Figures: text and page-image retrieval tie at 166-paper scale; captions carry most of the
  signal [arXiv:2602.17687, 2407.09413].

**D5 — Review automation: per-paper, human-checkable steps only**
- Active-learning ordering always beats random; no evidence below ~240 records
  [doi:10.1007/s41060-025-00777-0]. "Stop after N consecutive excludes" missed 95% recall in 39%
  of trials [doi:10.1186/s13643-020-01521-4].
- LLM screening on CS/SE reviews is inconsistent (recall 0.28–0.86 [arXiv:2507.19027]; median
  MCC 0.32 [arXiv:2609.30298]); LLM self-confidence is uninformative (AUC 0.53)
  [arXiv:2608.14551] → suggestions with a logged reason, never auto-exclusion.
- LLM-filled comparison tables: ≈50% cell agreement unverified [arXiv:2410.22360, 2504.10284];
  ≈88–91% with quote-linked human verification [arXiv:2404.13765; doi:10.7326/annals-25-00739].
- Survey/related-work generators still fail on coverage [arXiv:2508.20033, 2508.15804].

**D6 — Citation graph: on-demand seed expansion, little else**
- Fused direct + coupling + co-citation is the best seed-based recipe for mature seeds
  [arXiv:2403.09295]; with SPECTER2 the fusion gain is small (NDCG@10 0.668 vs 0.654)
  [arXiv:2609.26218]. Citation expansion roughly doubles agent search recall in PaSa's ablation
  [arXiv:2501.10120]; a bounded seed→expand→prune pipeline beats free-roaming agents
  [arXiv:2608.24809].
- Early-impact prediction is weak (best LLM 68.8% pairwise) [arXiv:2604.17141]; trend
  forecasting is indefensible at ~1k papers/month [arXiv:2609.24921].

**D7 — Parsing, summaries, extraction**
- LLM summaries over-generalise (26–73% of cases); "be accurate" prompts backfire
  [arXiv:2504.00025] → extractive TL;DR stays default; quote author-stated limitations
  [arXiv:2507.02694].
- Crisp categorical fields extract reliably [arXiv:2601.14429]; numeric results only from source
  tables [arXiv:2502.18791, 2409.12656].
- Rule-based PDF parsers, pdfplumber included, all struggle on scientific pages
  [arXiv:2410.09871]; arXiv HTML converts ~75% of papers error-free [arXiv:2605.16562].

**D8 — HCI: explain through the user's own library; keep batches small**
- Library-relative reasons work: randomised trial on 7,038 alert users, CTR 5.8% vs 4.5%
  [arXiv:2204.10254]; PaperWeaver [arXiv:2403.02939]; CiteSee [arXiv:2302.07302]. Caveat:
  explanations can raise acceptance regardless of correctness [arXiv:2006.14779].
- First-run failures decide adoption (return rate 10% vs 53%) [arXiv:2602.23335]; about half of
  feed users quit in week one [arXiv:2601.04253].

**D9 — Agents and tool design**
- Agentic literature search is far from solved (best Recall@100 0.314) [arXiv:2606.20235]; the
  bottleneck is what the tools expose [arXiv:2609.33233].
- Tool design: consolidate, paginate, actionable errors (Anthropic, "Writing effective tools
  for agents", 2025); description edits regress ~17% of cases [arXiv:2602.14878].
- Security: hidden instructions exist in real arXiv manuscripts [arXiv:2507.06185]; notes an
  agent writes after reading poisoned text can steer later sessions (60–89%) [arXiv:2605.15338];
  prompt-level delimiting is bypassable [arXiv:2510.09023] → narrow, typed, logged, reversible
  write tools are the real defence [arXiv:2506.08837].
