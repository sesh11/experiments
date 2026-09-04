"""Probe which OpenRouter capabilities are the model's and which are the host's.

OpenRouter serves one model id from many independent hosts. A capability that
differs per host is a serving-stack property; one that is identical across every
host is imposed above them. Two questions this answers, both of which changed a
cost conclusion in docs/findings/2026-09-03-cache-control-provenance.md:

  --catalog   which models are sold with *explicit* prompt caching (a cache
              WRITE price is the tell — you cannot be charged to park a prefix
              you were not allowed to mark)
  --hosts ID  whether a per-host capability difference exists for one model:
              its endpoints, quantization, cache pricing, and (with a key)
              whether each host accepts `reasoning: {"effort": "none"}`

Catalog and endpoint metadata need no credentials. The live reasoning probe
needs OPENROUTER_API_KEY and costs well under a cent (32 max_tokens per host).

    python scripts/openrouter_capability_probe.py --catalog
    python scripts/openrouter_capability_probe.py --hosts z-ai/glm-5.3 --live
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import urllib.error
import urllib.request

API = "https://openrouter.ai/api/v1"
PER_MILLION = 1_000_000


def _get(path: str) -> dict:
    with urllib.request.urlopen(f"{API}{path}", timeout=60) as response:
        return json.load(response)


def _rate(pricing: dict, key: str) -> float | None:
    """Per-million-token price, or None when the field is absent or zero."""
    value = pricing.get(key)
    return float(value) * PER_MILLION if value and float(value) > 0 else None


def catalog() -> None:
    """Print which model families are sold with explicit caching."""
    models = _get("/models")["data"]
    explicit = [
        model["id"] for model in models
        if _rate(model.get("pricing", {}), "input_cache_write")
    ]
    print(f"models with a cache WRITE price: {len(explicit)} of {len(models)}")
    families = collections.Counter(
        model_id.split("/")[0].lstrip("~") for model_id in explicit
    )
    for family, count in families.most_common():
        print(f"  {family:<12} {count}")


def hosts(model_id: str, live: bool) -> None:
    """Print per-host serving differences for one model id."""
    endpoints = _get(f"/models/{model_id}/endpoints")["data"]["endpoints"]
    print(f"{model_id}: {len(endpoints)} host(s)")
    for endpoint in endpoints:
        pricing = endpoint["pricing"]
        write = _rate(pricing, "input_cache_write")
        read = _rate(pricing, "input_cache_read")
        row = (
            f"  {endpoint['provider_name']:<14} "
            f"quant={str(endpoint.get('quantization')):<8} "
            f"in=${_rate(pricing, 'prompt'):.2f}/M "
            f"cache_read={f'${read:.2f}/M' if read else '-':<9} "
            f"cache_write={f'${write:.2f}/M' if write else '-':<9}"
        )
        if live:
            row += f" effort=none -> {_reasoning_none(model_id, endpoint)}"
        print(row)


def _reasoning_none(model_id: str, endpoint: dict) -> str:
    """Whether one host accepts a request that disables reasoning."""
    body = {
        "model": model_id,
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "Reply with just: ok"}],
        "reasoning": {"effort": "none"},
        "provider": {
            "only": [endpoint["provider_name"]], "allow_fallbacks": False,
        },
    }
    request = urllib.request.Request(
        f"{API}/chat/completions", data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=120):
            return "accepted"
    except urllib.error.HTTPError as error:
        payload = json.loads(error.read() or b"{}").get("error", {})
        named = (payload.get("metadata") or {}).get("provider_name")
        # A null provider_name means OpenRouter refused before routing, so the
        # refusal is not this host's.
        return (
            f"{error.code} {payload.get('message', '')!r} "
            f"(blamed host: {named})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", action="store_true",
                        help="summarize explicit-caching support across models")
    parser.add_argument("--hosts", metavar="MODEL_ID",
                        help="list the hosts serving one model id")
    parser.add_argument("--live", action="store_true",
                        help="with --hosts, send one request per host")
    args = parser.parse_args()
    if args.catalog:
        catalog()
    if args.hosts:
        hosts(args.hosts, args.live)
    if not (args.catalog or args.hosts):
        parser.error("pass --catalog and/or --hosts MODEL_ID")


if __name__ == "__main__":
    main()
