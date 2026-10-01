"""Pure-Python routing pipeline — no LangGraph dependency.

Scoring follows the LLMRouter paper (Feng et al., 2026):

    π* = argmax_π  E[ perf(y | q) − λ · c(τ) ]        (Eq. 1)

For single-turn routing this becomes:

    reward_m = α · perf_normalized(m) − β · cost_normalized(m)

where (α, β) are the performance–cost trade-off weights swept across
five operating points defined in Table 2 / Figure 5 of the paper.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .catalog import ModelProfile, catalog_fingerprint, load_catalogs
from .classifier import Classification, ClassifierError, LocalClassifier
from .config import resolve_catalog_path, default_index_path, default_user_models_path
from .embeddings import DEFAULT_EMBEDDING_MODEL, EmbeddingIndex, QwenEmbeddingEncoder
from .invocation import ModelInvocationError, invoke_selected_model


RoutingMode = Literal[
    "skill_based",
    "quality_leaning",
    "mixed",
    "cost_sensitive",
    "cost_efficient",
]


# The paper's five evaluation sweep points from Table 2 / Figure 5.
# Each maps to (α, β) where the reward is:  α · perf − β · cost.
# «skill_based» is quality-only (λ=0); «cost_efficient» is heavily
# cost-weighted; the three in between are the paper's interior points.
MODE_WEIGHTS: dict[RoutingMode, tuple[float, float]] = {
    "skill_based":     (1.0, 0.0),   # (α, β) = (1.0, 0.0)  — quality only
    "quality_leaning": (0.8, 0.2),   # (α, β) = (0.8, 0.2)
    "mixed":           (0.6, 0.4),   # (α, β) = (0.6, 0.4)
    "cost_sensitive":  (0.4, 0.6),   # (α, β) = (0.4, 0.6)
    "cost_efficient":  (0.2, 0.8),   # (α, β) = (0.2, 0.8)  — cost dominant
}

COMPLEX_WORK_PATTERN = re.compile(
    r"\b(architecture|architect|multi[- ]file|codebase|repository|refactor|migration|migrate|"
    r"redesign|long[- ]horizon|production incident|security audit|threat model|agentic|"
    r"deep analysis|complex|plan an? implementation)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RouterConfig:
    """All tuneable knobs for the routing pipeline."""

    catalog_path: Path | None = None
    user_catalog_path: Path | None = None
    index_path: Path | None = None
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    # Classifier settings
    classifier_backend: Literal["local", "vllm"] = "local"
    classifier_model_path: str | None = None    # local GGUF path override
    classifier_n_ctx: int = 2048
    classifier_n_threads: int | None = None
    classifier_base_url: str = "http://127.0.0.1:8080"  # for vllm backend
    classifier_model: str | None = None                  # for vllm backend
    classifier_timeout: int = 30
    # Target invocation
    target_timeout: int = 90
    candidate_pool: Literal["all", "user"] = "all"
    routing_mode: RoutingMode = "mixed"
    alpha: float | None = None
    beta: float | None = None
    quality_gap_threshold: float = 0.05
    target_max_tokens: int = 1024


# ---------------------------------------------------------------------------
# Routing state (plain dict, replaces LangGraph TypedDict)
# ---------------------------------------------------------------------------

RouterState = dict[str, Any]


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _category_boost(profile: ModelProfile, category: str) -> float:
    """Small deterministic tie-breaker; semantic similarity remains dominant."""
    text = profile.document().lower()
    terms = {
        "code_generation": ("generation", "completion", "instruct", "coding"),
        "bug_fixing": ("debug", "reasoning", "swe-bench", "agentic", "repository"),
        "performance_optimization": ("performance", "efficient", "optimization", "agentic"),
        "api_help": ("api", "tool-use", "agentic", "integration"),
        "programming": ("general-purpose", "coding", "chat", "instruction"),
    }
    matches = sum(term in text for term in terms.get(category, ()))
    return min(matches * 0.01, 0.03)


def mode_weights(
    mode: RoutingMode,
    alpha_override: float | None = None,
    beta_override: float | None = None,
) -> tuple[float, float]:
    """Return (α, β) weights for the paper's reward formula."""
    try:
        alpha, beta = MODE_WEIGHTS[mode]
    except KeyError as error:
        raise ValueError(f"Unsupported routing mode: {mode}") from error
    if alpha_override is not None:
        alpha = alpha_override
    if beta_override is not None:
        beta = beta_override
    return alpha, beta


def _normalize(values: list[float], *, flat_value: float) -> list[float]:
    """Min-max normalize a candidate-pool value without fabricating a preference."""
    lower, upper = min(values), max(values)
    if math.isclose(lower, upper):
        return [flat_value] * len(values)
    return [(value - lower) / (upper - lower) for value in values]


def estimate_request_cost_usd(
    profile: ModelProfile, input_tokens: int, output_tokens: int
) -> float:
    """Estimate the request cost from the catalog's USD-per-million token rates."""
    return (
        profile.input_price * max(0, input_tokens)
        + profile.output_price * max(0, output_tokens)
    ) / 1_000_000


# ---------------------------------------------------------------------------
# Pipeline stages (pure functions operating on a dict state)
# ---------------------------------------------------------------------------


def _embed_query(encoder: QwenEmbeddingEncoder, state: RouterState) -> None:
    embedding = encoder.encode_query(state["user_query"])
    state["query_embedding"] = embedding.astype(float).tolist()


def _classify_query(classifier: Any, state: RouterState) -> None:
    try:
        classification: Classification = classifier.classify(state["user_query"])
        state["classifier_category"] = classification.category
        state["classifier_raw_response"] = classification.raw_response
        state["classifier_model"] = classification.model
    except ClassifierError as error:
        # Similarity routing can still work during a temporary classifier outage.
        state["classifier_category"] = "programming"
        state["classifier_error"] = str(error)


def _rank_models(
    profiles: list[ModelProfile],
    index: EmbeddingIndex,
    state: RouterState,
) -> None:
    """Score every candidate with the paper's Eq. 1 for single-turn routing."""
    category = state["classifier_category"]
    scores = index.cosine_scores(state["query_embedding"])

    # ── capability-fit proxy: g(E_q, E_m) + category boost ────────
    raw_capability_fit = [
        cosine_score + _category_boost(profile, category)
        for profile, cosine_score in zip(profiles, scores, strict=True)
    ]

    # ── cost term: estimated USD cost for this request ─────────────
    input_tokens = state["estimated_input_tokens"]
    output_tokens = state["estimated_output_tokens"]
    request_costs = [
        estimate_request_cost_usd(profile, input_tokens, output_tokens)
        for profile in profiles
    ]

    # ── min-max normalise both to [0, 1] ──────────────────────────
    perf_normalized = _normalize(raw_capability_fit, flat_value=1.0)
    cost_normalized = _normalize(request_costs, flat_value=0.0)

    # ── (α, β) from the named mode or explicit CLI overrides ──────
    alpha, beta = mode_weights(
        state["routing_mode"],
        alpha_override=state.get("alpha"),
        beta_override=state.get("beta"),
    )

    # ── Paper Eq. 1:  reward = α · perf − β · cost ────────────────
    ranked: list[dict[str, Any]] = []
    for profile, cosine_score, raw_q, perf_n, cost_usd, cost_n in zip(
        profiles,
        scores,
        raw_capability_fit,
        perf_normalized,
        request_costs,
        cost_normalized,
        strict=True,
    ):
        boost = _category_boost(profile, category)
        reward = alpha * perf_n - beta * cost_n
        ranked.append(
            {
                "catalog_key": profile.catalog_key,
                "similarity": round(cosine_score, 6),
                "category_boost": round(boost, 6),
                "capability_fit_raw": round(raw_q, 6),
                "perf_normalized": round(perf_n, 6),
                "estimated_request_cost_usd": round(cost_usd, 8),
                "cost_normalized": round(cost_n, 6),
                "alpha": alpha,
                "beta": beta,
                "reward": round(reward, 6),
            }
        )
    ranked.sort(
        key=lambda item: (
            item["reward"],
            item["perf_normalized"],
            -item["estimated_request_cost_usd"],
        ),
        reverse=True,
    )
    state["ranked_models"] = ranked


def _choose_group(
    profile_by_key: dict[str, ModelProfile],
    state: RouterState,
) -> None:
    query = state["user_query"]
    advanced = state.get("force_advanced", False) or bool(COMPLEX_WORK_PATTERN.search(query))
    if len(query) > 2_000:
        advanced = True
    if (
        state["routing_mode"] == "skill_based"
        and profile_by_key[state["ranked_models"][0]["catalog_key"]].advanced
    ):
        advanced = True
    state["selection_group"] = "advanced_model_group" if advanced else "standard_model_group"
    state["selection_reason"] = (
        "forced by caller" if state.get("force_advanced") else
        ("advanced semantic match or complex request" if advanced else "standard request")
    )


def _select_model(
    profile_by_key: dict[str, ModelProfile],
    state: RouterState,
) -> None:
    """Pick the best candidate within the group, applying the quality-gap rule."""
    advanced = state["selection_group"] == "advanced_model_group"

    if state.get("force_no_gating"):
        matches = state["ranked_models"]
    else:
        matches = [
            item
            for item in state["ranked_models"]
            if profile_by_key[item["catalog_key"]].advanced is advanced
        ]
        if not matches:
            matches = state["ranked_models"]

    # ── Quality-gap cost-aware decision rule ──────────────────────
    threshold = state.get("quality_gap_threshold", 0.05)
    best = matches[0]
    selection = best
    if threshold > 0 and len(matches) > 1:
        for candidate in matches[1:]:
            perf_gap = best["perf_normalized"] - candidate["perf_normalized"]
            if (
                perf_gap < threshold
                and candidate["estimated_request_cost_usd"]
                < best["estimated_request_cost_usd"]
            ):
                selection = candidate
                break

    profile = profile_by_key[selection["catalog_key"]]
    selected = profile.public_dict() | {
        "similarity": selection["similarity"],
        "perf_normalized": selection["perf_normalized"],
        "reward": selection["reward"],
        "category_boost": selection["category_boost"],
        "estimated_request_cost_usd": selection["estimated_request_cost_usd"],
        "cost_normalized": selection["cost_normalized"],
        "alpha": selection["alpha"],
        "beta": selection["beta"],
    }
    state["selected_model"] = selected


def _invoke_target(
    profile_by_key: dict[str, ModelProfile],
    config: RouterConfig,
    state: RouterState,
) -> None:
    if state.get("route_only"):
        return
    profile = profile_by_key[state["selected_model"]["catalog_key"]]
    try:
        response = invoke_selected_model(
            profile,
            state["user_query"],
            state["classifier_category"],
            timeout=config.target_timeout,
            max_tokens=state["estimated_output_tokens"],
        )
        state["model_response"] = response
    except ModelInvocationError as error:
        state["invocation_error"] = str(error)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class CodingRouter:
    """Main entry point: build the index and route coding requests.

    Usage::

        from coding_router import CodingRouter

        router = CodingRouter()
        result = router.route("Fix this race condition in my Python worker", route_only=True)
        print(result["selected_model"])
    """

    def __init__(self, config: RouterConfig | None = None) -> None:
        self.config = config or RouterConfig()
        cfg = self.config

        catalog_path = resolve_catalog_path(cfg.catalog_path)
        user_catalog_path = cfg.user_catalog_path or default_user_models_path()
        index_path = cfg.index_path or default_index_path()

        self.profiles = load_catalogs(
            catalog_path,
            user_catalog_path,
            candidate_pool=cfg.candidate_pool,
        )
        self.profile_by_key = {p.catalog_key: p for p in self.profiles}

        # Embedding encoder + index
        self.encoder = QwenEmbeddingEncoder(cfg.embedding_model)
        self.index, self.index_rebuilt = EmbeddingIndex.ensure(
            path=index_path,
            profiles=self.profiles,
            fingerprint=catalog_fingerprint(
                (catalog_path, user_catalog_path or Path("<none>")),
                f"{cfg.embedding_model}\0pool={cfg.candidate_pool}",
            ),
            encoder=self.encoder,
        )

        # Classifier
        if cfg.classifier_backend == "vllm":
            from .classifier import VLLMClassifier
            self.classifier = VLLMClassifier(
                base_url=cfg.classifier_base_url,
                model=cfg.classifier_model,
                timeout=cfg.classifier_timeout,
            )
        else:
            self.classifier = LocalClassifier(
                model_path=cfg.classifier_model_path,
                n_ctx=cfg.classifier_n_ctx,
                n_threads=cfg.classifier_n_threads,
            )

        # Start the dashboard server automatically in the background
        import socket
        port = 3000
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            in_use = s.connect_ex(('127.0.0.1', port)) == 0

        if not in_use:
            import subprocess
            import sys
            try:
                subprocess.Popen(
                    [sys.executable, "-m", "coding_router", "dashboard", "--port", str(port)],
                    start_new_session=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                import time
                time.sleep(0.2)  # brief pause to let it bind
            except Exception:
                pass

        print(f"  ✓ Dashboard available at http://127.0.0.1:{port} (or http://localhost:{port})")

    def route(
        self,
        query: str,
        *,
        route_only: bool = False,
        force_advanced: bool = False,
        force_no_gating: bool = False,
        routing_mode: RoutingMode | None = None,
        alpha: float | None = None,
        beta: float | None = None,
        quality_gap_threshold: float | None = None,
        estimated_output_tokens: int | None = None,
    ) -> RouterState:
        """Run the full routing pipeline and return the final state dict."""
        if not query.strip():
            raise ValueError("The user query cannot be empty.")
        mode = routing_mode or self.config.routing_mode
        if mode not in MODE_WEIGHTS:
            raise ValueError(f"Unsupported routing mode: {mode}")
        output_tokens = estimated_output_tokens or self.config.target_max_tokens
        if output_tokens <= 0:
            raise ValueError("Estimated output tokens must be positive.")

        effective_alpha = alpha if alpha is not None else self.config.alpha
        effective_beta = beta if beta is not None else self.config.beta
        gap = quality_gap_threshold if quality_gap_threshold is not None else self.config.quality_gap_threshold

        state: RouterState = {
            "user_query": query,
            "route_only": route_only,
            "force_advanced": force_advanced,
            "force_no_gating": force_no_gating,
            "routing_mode": mode,
            "estimated_input_tokens": max(1, math.ceil(len(query) / 4)),
            "estimated_output_tokens": output_tokens,
            "quality_gap_threshold": gap,
        }
        if effective_alpha is not None:
            state["alpha"] = effective_alpha
        if effective_beta is not None:
            state["beta"] = effective_beta

        # ── Pipeline stages ──────────────────────────────────────
        _embed_query(self.encoder, state)
        _classify_query(self.classifier, state)
        _rank_models(self.profiles, self.index, state)
        _choose_group(self.profile_by_key, state)
        _select_model(self.profile_by_key, state)
        _invoke_target(self.profile_by_key, self.config, state)

        return state
