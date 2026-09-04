# Who owns the knobs: cache and reasoning control are serving-stack properties (2026-09-03)

**One line:** the capabilities that decide an arm's cost on OpenRouter belong to
whoever *runs the inference*, not to the weights and not to OpenRouter — and
getting that attribution wrong made an open-weight model look 7× more expensive
than it is.

## How this came up

`z-ai/glm-5.3` crashed at step 0 of a SWE-bench cell with
`400 Reasoning is mandatory for this endpoint and cannot be disabled`, because
the harness sends `reasoning: {"effort": "none"}` for `--thinking off`. The
first explanation — *GLM is reasoning-native, so z.ai refuses* — was wrong, and
believing it cost real money: the workaround it implied (drop the field, let the
endpoint choose) ran GLM at full reasoning for $0.085/task.

OpenRouter is a broker, not a host. One model id is served by many independent
hosts, so any capability question has three possible owners: the weights, the
host's serving stack, or OpenRouter's own layer. They are distinguishable by
experiment, which is the point of this note.

Reproduce with `scripts/openrouter_capability_probe.py` (catalog and endpoint
metadata need no key; `--live` sends 32 max_tokens per host, well under a cent).

## Test 1 — is the reasoning refusal the host's?

`z-ai/glm-5.3` is served by **25 hosts**. Pinning each one with
`provider: {only: [host], allow_fallbacks: false}` and sending
`reasoning: {"effort": "none"}`:

```
$ python scripts/openrouter_capability_probe.py --hosts z-ai/glm-5.3 --live
z-ai/glm-5.3: 25 host(s)
  Reka       ... effort=none -> 400 'Reasoning is mandatory ...' (blamed host: None)
  DeepInfra  ... effort=none -> 400 'Reasoning is mandatory ...' (blamed host: None)
  Fireworks  ... effort=none -> 400 'Reasoning is mandatory ...' (blamed host: None)
  Z.AI       ... effort=none -> 400 'Reasoning is mandatory ...' (blamed host: None)
  ... 25/25 identical
```

25 of 25, byte-identical, and `metadata.provider_name` is `null` — OpenRouter
never routed the request. **The refusal is OpenRouter's**, applied from its own
model metadata before a host is chosen. Choosing a different host is not a
workaround; there is nothing downstream to choose.

It is also not the whole knob. Only the *disable* forms are blocked:

| request                          | result                       |
|----------------------------------|------------------------------|
| `reasoning: {effort: "none"}`    | 400, refused before routing  |
| `reasoning: {enabled: false}`    | 400, refused before routing  |
| `reasoning: {effort: "minimal"}` | **accepted, 0 reasoning tokens** |
| `reasoning: {max_tokens: 1}`     | accepted, 0 reasoning tokens |
| `reasoning: {exclude: true}`     | accepted, 18 reasoning tokens (hidden, still billed) |
| field omitted                    | 23–38 reasoning tokens (host default) |

So "reasoning cannot be disabled" is enforced as a policy on the *name* of the
setting, while the behavior it names is reachable by another name. Note
`exclude: true` is a trap for cost accounting: it hides the reasoning from the
response and still bills it.

Consequence for the harness: `--thinking off` now degrades to
`{"effort": "minimal"}` rather than to the endpoint default. Same cell, same
model, same host pool:

| glm-5.3 run       | resolved | cost   | output tok | steps |
|-------------------|----------|--------|------------|-------|
| endpoint default  | ✅        | $0.085 | 3,811      | 14    |
| `effort: minimal` | ✅        | $0.012 | 486        | 7     |

## Test 2 — where does *explicit* cache control come from?

Explicit caching means the client marks a prefix (`cache_control`) and the
server persists that KV state. Persisted memory is a resource someone has to
sell, so a **cache-write price is the tell**: you cannot be charged to park a
prefix you were never allowed to mark.

Across OpenRouter's whole catalog:

```
$ python scripts/openrouter_capability_probe.py --catalog
models with a cache WRITE price: 76 of 426
  anthropic    31
  google       23
  qwen         13
  openai        9
```

The 25 GLM hosts show a cache-*read* price and **no cache-write price on any of
them** — every one runs a vLLM/SGLang-class stack whose automatic prefix caching
is best-effort reuse of shared capacity. There is no product to sell, so there
is no knob and no price. That is why both GLM runs above show 0 cache writes,
and why the minimal-effort run still billed 6,482 fresh input tokens where
Anthropic's explicit breakpoints billed 24.

**It is not an open-weights limitation.** `qwen/qwen3-max` is open weights and
has exactly one host — Alibaba, who owns the serving stack — and sells explicit
caching on it ($0.97/M write, $0.16/M read against $0.78/M input). Same category
of model as GLM, opposite capability, because the serving arrangement differs.

The dividing line is who runs inference:

| serving arrangement                        | caching offered                | example                        |
|--------------------------------------------|--------------------------------|--------------------------------|
| first-party lab, sells cache as a product   | explicit, client-marked, priced| Anthropic, Google, Alibaba/Qwen|
| first-party lab, no client control          | automatic, discounted reads    | OpenAI `gpt-5` (read $0.12/M, no write price) |
| commodity GPU marketplace serving open weights | best-effort prefix reuse    | all 25 GLM hosts               |

## Why it matters for the arm matrix

1. **Reasoning effort belongs in the arm definition.** It moved cost 7× on one
   model — more than the choice of model did ($0.012 GLM vs $0.040 sonnet-5
   direct, both resolved). An arm labelled only "glm-5.3" is under-specified.
2. **Cache control is an arm property too, and it does not come with the model.**
   A commodity-hosted sidekick cannot have its context pinned, so its fresh-input
   share grows with trajectory length — precisely where the expensive tasks are.
   Cheap-per-token open models are most attractive on short trajectories and
   least attractive on long ones. If E6 wants a cheap sidekick *with* cache
   control, a first-party-hosted open model (Qwen via Alibaba) is the shape to
   test, not GLM on a marketplace.
3. **Pin the host on any measured run.** GLM's hosts differ in quantization
   (fp4 vs fp8) at nearly identical prices, and OpenRouter picks for you by
   default. That is an unlogged quality variable inside an arm; `provider.only`
   removes it.
4. **Capability metadata is not evidence.** All 25 hosts advertise `reasoning`
   and `reasoning_effort` as supported parameters, including for the request
   that is refused outright. Parameters are listed as *accepted*, not as
   *honorable*. Only a live call distinguishes them.

## Caveats

Single task (`pallets__flask-5014`), single seed, n=1 per cell; the cost figures
are feasibility checks, not model-quality claims. Prices are OpenRouter's
2026-09-03 metadata snapshot and move. The host probe used 32 max_tokens per
request, which is enough to observe acceptance and reasoning-token counts but
not behavior on a real trajectory.
