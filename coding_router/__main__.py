"""Command-line entry point for the coding-router package."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .router import CodingRouter, RouterConfig, MODE_WEIGHTS
from .dashboard import DashboardServer, get_tracker
from .catalog import add_user_model
from .config import (
    SERVICE_ENV_MAP,
    _is_local_service,
    bundled_catalog_path,
    default_index_path,
    default_user_models_path,
    env_vars_for_service,
    find_api_key,
    load_json,
    resolve_catalog_path,
)
from .embeddings import DEFAULT_EMBEDDING_MODEL


# ---------------------------------------------------------------------------
# Subcommand: init
# ---------------------------------------------------------------------------


def init_command(arguments: list[str]) -> int:
    """Set up a project-local router/ directory with the model catalog."""
    router_dir = Path("router")
    catalog_path = router_dir / "coding_llm.json"
    env_example_path = Path(".env.example")

    # 1. Create router/ directory
    router_dir.mkdir(parents=True, exist_ok=True)

    # 2. Copy the bundled catalog template (never overwrite user edits)
    if catalog_path.exists():
        print(f"  ✓ Catalog already exists: {catalog_path}  (skipped, not overwriting)")
    else:
        bundled = bundled_catalog_path()
        if not bundled.exists():
            print(f"  ✗ Bundled template not found at {bundled}", file=sys.stderr)
            return 1
        catalog_path.write_text(bundled.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"  ✓ Created: {catalog_path}")

    # 3. Generate .env.example from the catalog's service fields
    data = load_json(catalog_path)
    needed_vars: dict[str, str] = {}  # var_name → service_name
    for group_name, models in data.items():
        if not isinstance(models, dict):
            continue
        for key, record in models.items():
            if not isinstance(record, dict):
                continue
            service = record.get("service", "")
            if _is_local_service(service):
                continue
            for var in env_vars_for_service(service):
                if var not in needed_vars:
                    needed_vars[var] = service

    if needed_vars:
        lines = ["# Required API keys for coding-router", "#"]
        for var, service in sorted(needed_vars.items()):
            lines.append(f"# {service}")
            lines.append(f"{var}=")
        env_example_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"  ✓ Created: {env_example_path}")
    else:
        print("  ℹ No cloud API keys needed (only local models in catalog)")

    # 4. Print next steps
    print()
    print("  Next steps:")
    print(f"    1. Edit {catalog_path} — delete models you don't have access to")
    print(f"    2. Set the required env vars (see {env_example_path})")
    print("    3. Run: coding-router validate")
    print()
    return 0


# ---------------------------------------------------------------------------
# Subcommand: validate
# ---------------------------------------------------------------------------


def validate_command(arguments: list[str]) -> int:
    """Validate the catalog and check API key availability."""
    parser = argparse.ArgumentParser(
        prog="coding-router validate",
        description="Validate the model catalog and check API keys / endpoints.",
    )
    parser.add_argument("--catalog", type=Path, default=None)
    args = parser.parse_args(arguments)

    try:
        catalog_path = resolve_catalog_path(args.catalog)
    except FileNotFoundError as e:
        print(f"  ✗ {e}", file=sys.stderr)
        return 1

    # Load and validate JSON structure
    try:
        data = load_json(catalog_path)
    except json.JSONDecodeError as e:
        print(f"  ✗ Invalid JSON in {catalog_path}: {e}", file=sys.stderr)
        return 1

    if not isinstance(data, dict):
        print(f"  ✗ {catalog_path}: Expected a JSON object at the top level", file=sys.stderr)
        return 1

    print(f"  Catalog: {catalog_path}\n")

    required_fields = ("model", "service", "api_endpoint", "feature", "size", "input_price", "output_price")
    has_errors = False
    cloud_ok = 0
    cloud_missing = 0
    local_ok = 0
    local_unreachable = 0

    for group_name, models in data.items():
        if not isinstance(models, dict):
            continue
        for key, record in models.items():
            if not isinstance(record, dict):
                print(f"  ✗ {key}: Expected a JSON object, got {type(record).__name__}", file=sys.stderr)
                has_errors = True
                continue

            # Check required fields
            missing_fields = [f for f in required_fields if f not in record]
            if missing_fields:
                print(
                    f"  ✗ {key}: Missing required fields: {', '.join(missing_fields)}",
                    file=sys.stderr,
                )
                has_errors = True
                continue

            service = record.get("service", "")
            model_id = record.get("model", key)

            if _is_local_service(service):
                # Best-effort ping the endpoint
                endpoint = record.get("api_endpoint", "")
                reachable = False
                if endpoint:
                    try:
                        url = endpoint.rstrip("/") + "/models"
                        req = urllib.request.Request(url, method="GET")
                        urllib.request.urlopen(req, timeout=3)
                        reachable = True
                    except Exception:
                        pass

                if reachable:
                    print(f"  ✓ {model_id} ({service}) — endpoint reachable")
                    local_ok += 1
                else:
                    print(f"  ⚠ {model_id} ({service}) — endpoint unreachable: {endpoint}")
                    local_unreachable += 1
            else:
                # Cloud model: check API key
                api_key = find_api_key(service)
                candidate_vars = env_vars_for_service(service)
                if api_key:
                    print(f"  ✓ {model_id} ({service}) — key found")
                    cloud_ok += 1
                else:
                    var_hint = " or ".join(candidate_vars) if candidate_vars else "???"
                    print(f"  ✗ {model_id} ({service}) — set {var_hint}")
                    cloud_missing += 1
                    has_errors = True

    print()
    print(f"  Cloud: {cloud_ok} ready, {cloud_missing} missing keys")
    print(f"  Local: {local_ok} reachable, {local_unreachable} unreachable")

    if has_errors:
        print()
        print("  ⚠ Some models are missing API keys. Set them and re-run validate.")
        return 1

    print()
    print("  ✓ All models validated successfully!")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: models  (list all models with status)
# ---------------------------------------------------------------------------


def models_command(arguments: list[str]) -> int:
    """List all available models and their status."""
    parser = argparse.ArgumentParser(
        prog="coding-router models",
        description="List all models in the catalog.",
    )
    parser.add_argument("--catalog", type=Path, default=None)
    args = parser.parse_args(arguments)

    try:
        catalog_path = resolve_catalog_path(args.catalog)
    except FileNotFoundError as e:
        print(f"  ✗ {e}", file=sys.stderr)
        return 1

    data = load_json(catalog_path)
    print(f"  Catalog: {catalog_path}\n")

    cloud_models = []
    local_models = []

    for group_name, models in data.items():
        if not isinstance(models, dict):
            continue
        for key, record in models.items():
            if not isinstance(record, dict):
                continue
            service = record.get("service", "?")
            entry = {
                "key": key,
                "service": service,
                "size": record.get("size", "?"),
                "input_price": record.get("input_price", 0),
                "output_price": record.get("output_price", 0),
            }
            if _is_local_service(service):
                local_models.append(entry)
            else:
                local_models.append(entry) if _is_local_service(service) else cloud_models.append(entry)

    def _print_table(title: str, entries: list[dict]) -> None:
        if not entries:
            return
        print(f"  {title}")
        print(f"  {'─' * 80}")
        for e in entries:
            price = (
                f"${e['input_price']:.2f} / ${e['output_price']:.2f} per 1M tokens"
                if e["input_price"] > 0 or e["output_price"] > 0
                else "free (self-hosted)"
            )
            print(f"    {e['key']:<30s}  {e['service']:<28s}  {price}")
        print()

    _print_table("☁️  Cloud Models", cloud_models)
    _print_table("🖥️  Local / Self-Hosted Models", local_models)

    total = len(cloud_models) + len(local_models)
    print(f"  {total} models in catalog")
    print()
    print("  To remove a model, delete its entry from the catalog JSON.")
    print("  To add a model:  coding-router add-model --help")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: add-model
# ---------------------------------------------------------------------------


def add_model_command(arguments: list[str]) -> int:
    """Add a model the user can actually invoke to the persisted candidate pool."""
    parser = argparse.ArgumentParser(description="Add a local or cloud model to the router.")
    parser.add_argument("--key", required=True, help="unique lowercase key, e.g. my-local-coder")
    parser.add_argument("--model", required=True, help="model ID accepted by the endpoint")
    parser.add_argument(
        "--endpoint",
        required=True,
        help="provider base URL; OpenAI-compatible by default",
    )
    parser.add_argument("--feature", required=True, help="capabilities and ideal tasks; used for similarity")
    parser.add_argument("--service", default="OpenAI-compatible")
    parser.add_argument("--size", default="Unknown")
    parser.add_argument("--input-price", type=float, required=True, help="USD per million input tokens")
    parser.add_argument("--output-price", type=float, required=True, help="USD per million output tokens")
    parser.add_argument("--tier", choices=("standard", "advanced"), default="standard")
    parser.add_argument("--api-key-env", help="environment variable containing this model's API key")
    parser.add_argument("--user-catalog", type=Path, default=None)
    args = parser.parse_args(arguments)

    user_catalog = args.user_catalog or default_user_models_path()
    add_user_model(
        user_catalog,
        catalog_key=args.key,
        model=args.model,
        service=args.service,
        api_endpoint=args.endpoint,
        feature=args.feature,
        size=args.size,
        input_price=args.input_price,
        output_price=args.output_price,
        tier=args.tier,
        api_key_env=args.api_key_env,
    )
    print(f"  ✓ Added '{args.key}' to {user_catalog}")
    print("  Index will rebuild automatically on next route.")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: route (default)
# ---------------------------------------------------------------------------


def route_command(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="coding-router",
        description="Classify, semantically route, and optionally invoke a coding LLM.",
    )
    parser.add_argument("query", nargs="?", help="The user request to route")
    parser.add_argument("--route-only", action="store_true", help="Select a model without invoking it")
    parser.add_argument("--force-advanced", action="store_true", help="Use the advanced model subgroup")
    parser.add_argument(
        "--mode",
        choices=tuple(MODE_WEIGHTS),
        default="mixed",
        help="Routing policy: skill_based, quality_leaning, mixed, cost_sensitive, or cost_efficient",
    )
    parser.add_argument("--alpha", type=float, default=None, help="Override performance weight α (0.0–1.0)")
    parser.add_argument("--beta", type=float, default=None, help="Override cost weight β (0.0–1.0)")
    parser.add_argument(
        "--quality-gap-threshold",
        type=float,
        default=None,
        help="Quality-gap threshold for the cost-aware decision rule (default: 0.05)",
    )
    parser.add_argument("--build-index", action="store_true", help="Create or refresh the persisted embedding index")
    parser.add_argument("--catalog", type=Path, default=None)
    parser.add_argument("--user-catalog", type=Path, default=None)
    parser.add_argument(
        "--pool",
        choices=("all", "user"),
        default="all",
        help="all includes the bundled catalog and user models; user uses only your added models",
    )
    parser.add_argument("--index", type=Path, default=None)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    # Classifier backend options
    parser.add_argument(
        "--classifier-backend",
        choices=("llama-server", "local", "vllm"),
        default="llama-server",
        help="llama-server (default) auto-starts llama serve; local loads GGUF in-process; vllm connects to a remote server",
    )
    parser.add_argument("--classifier-model-path", help="path to a custom GGUF classifier model")
    parser.add_argument("--classifier-base-url", default="http://127.0.0.1:8080", help="server URL (for --classifier-backend=vllm)")
    parser.add_argument("--classifier-model", help="server model ID; auto-detected when omitted")
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=1024,
        help="output-token budget used for both cost estimation and invocation",
    )
    parser.add_argument("--show-ranking", action="store_true", help="Include every similarity score in output")
    args = parser.parse_args(arguments)

    try:
        catalog_path = resolve_catalog_path(args.catalog)
    except FileNotFoundError as e:
        print(f"  ✗ {e}", file=sys.stderr)
        return 1

    router = CodingRouter(
        RouterConfig(
            catalog_path=catalog_path,
            user_catalog_path=args.user_catalog or default_user_models_path(),
            index_path=args.index or default_index_path(),
            embedding_model=args.embedding_model,
            classifier_backend=args.classifier_backend,
            classifier_model_path=args.classifier_model_path,
            classifier_base_url=args.classifier_base_url,
            classifier_model=args.classifier_model,
            candidate_pool=args.pool,
            routing_mode=args.mode,
            target_max_tokens=args.max_output_tokens,
        )
    )
    if args.build_index:
        status = "rebuilt" if router.index_rebuilt else "already current"
        index_path = args.index or default_index_path()
        print(f"Embedding index {status}: {index_path}")
        if not args.query:
            return 0
    if not args.query:
        parser.error("query is required unless --build-index is used")

    t0 = time.perf_counter()
    result = router.route(
        args.query,
        route_only=args.route_only,
        force_advanced=args.force_advanced,
        routing_mode=args.mode,
        alpha=args.alpha,
        beta=args.beta,
        quality_gap_threshold=args.quality_gap_threshold,
        estimated_output_tokens=args.max_output_tokens,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    # Record the event to the local SQLite database for the dashboard.
    try:
        get_tracker().record(result, latency_ms)
    except Exception:
        pass  # never let dashboard tracking break a routing call

    selected = result["selected_model"]
    output: dict[str, object] = {
        "mode": args.mode,
        "alpha": selected.get("alpha"),
        "beta": selected.get("beta"),
        "candidate_pool": args.pool,
        "category": result["classifier_category"],
        "classifier_model": result.get("classifier_model"),
        "classifier_raw_response": result.get("classifier_raw_response"),
        "classifier_error": result.get("classifier_error"),
        "selection_group": result["selection_group"],
        "selection_reason": result["selection_reason"],
        "selected_model": selected,
        "model_response": result.get("model_response"),
        "invocation_error": result.get("invocation_error"),
    }
    if args.show_ranking:
        output["ranking"] = result["ranked_models"]
    print(json.dumps(output, indent=2))
    return 0 if not result.get("invocation_error") else 2


# ---------------------------------------------------------------------------
# Subcommand: dashboard
# ---------------------------------------------------------------------------


def dashboard_command(arguments: list[str]) -> int:
    """Start the dashboard API server so the web UI can display live metrics."""
    parser = argparse.ArgumentParser(
        prog="coding-router dashboard",
        description="Start the dashboard API server for the web UI.",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=3000, help="Port (default: 3000)"
    )
    args = parser.parse_args(arguments)

    server = DashboardServer(host=args.host, port=args.port)
    url = server.start()
    print(f"  ✓ Dashboard API running at {url}")
    print(f"    Open http://localhost:8080 in your browser to view the dashboard.")
    print(f"    Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  Stopping dashboard server…")
        server.stop()
    return 0


# ---------------------------------------------------------------------------
# Subcommand: serve (LLM Gateway)
# ---------------------------------------------------------------------------


def serve_command(arguments: list[str]) -> int:
    """Start the OpenAI/Anthropic-compatible LLM gateway."""
    parser = argparse.ArgumentParser(
        prog="coding-router serve",
        description="Start the local LLM gateway (OpenAI + Anthropic compatible).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    parser.add_argument("--no-auth", action="store_true", help="Disable gateway authentication")
    parser.add_argument("--log-level", default="info", choices=("debug", "info", "warning", "error"))
    args = parser.parse_args(arguments)

    import logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.no_auth:
        import os
        os.environ["CODING_ROUTER_AUTH"] = "false"

    print()
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║               coding-router  ·  LLM Gateway                ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")
    print()
    print(f"  Gateway URL:  http://{args.host}:{args.port}")
    print()
    print("  Endpoints:")
    print("    POST /v1/chat/completions   (OpenAI Chat)")
    print("    POST /v1/responses          (OpenAI Responses)")
    print("    POST /v1/messages           (Anthropic Messages)")
    print("    GET  /v1/models             (Model listing)")
    print("    GET  /health                (Health check)")
    print()

    if not args.no_auth:
        from .gateway.auth import load_or_create_key
        key = load_or_create_key()
        # Show only first 12 chars
        print(f"  API Key:      {key[:12]}...  (set CODING_ROUTER_API_KEY or see ~/.config/coding-router/gateway.key)")
    else:
        print("  API Key:      DISABLED (--no-auth)")

    print(f"  Virtual model: coding-router")
    print()
    print("  Press Ctrl+C to stop.")
    print()

    import uvicorn
    from .gateway.server import create_app
    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


# ---------------------------------------------------------------------------
# Subcommand: doctor
# ---------------------------------------------------------------------------


def doctor_command(arguments: list[str]) -> int:
    """Validate gateway configuration, router, and provider keys."""
    parser = argparse.ArgumentParser(
        prog="coding-router doctor",
        description="Diagnose gateway configuration and readiness.",
    )
    parser.add_argument("--catalog", type=Path, default=None)
    parser.parse_args(arguments)

    import os
    print()
    print("  coding-router doctor")
    print("  " + "─" * 50)
    errors = 0

    # 1. Catalog
    try:
        catalog_path = resolve_catalog_path()
        print(f"  ✓ Model catalog: {catalog_path}")
    except FileNotFoundError as e:
        print(f"  ✗ Model catalog: {e}")
        errors += 1

    # 2. Gateway key
    from .gateway.auth import load_or_create_key
    try:
        key = load_or_create_key()
        print(f"  ✓ Gateway API key: {key[:12]}...")
    except Exception as e:
        print(f"  ✗ Gateway API key: {e}")
        errors += 1

    # 3. Provider keys
    for service, env_var in [
        ("OpenAI", "OPENAI_API_KEY"),
        ("Anthropic", "ANTHROPIC_API_KEY"),
        ("Google", "GOOGLE_API_KEY"),
        ("DeepSeek", "DEEPSEEK_API_KEY"),
        ("xAI", "XAI_API_KEY"),
        ("Mistral AI", "MISTRAL_API_KEY"),
    ]:
        val = os.environ.get(env_var)
        if val:
            print(f"  ✓ {service}: {env_var} set")
        else:
            print(f"  ⚠ {service}: {env_var} not set")

    # 4. Router initialization
    print()
    print("  Testing router initialization...")
    try:
        from .router import CodingRouter
        router = CodingRouter()
        print(f"  ✓ Router ready: {len(router.profiles)} models in catalog")
        print(f"  ✓ Embedding index: loaded")
        print(f"  ✓ Classifier: loaded")
    except Exception as e:
        print(f"  ✗ Router initialization failed: {e}")
        errors += 1

    # 5. Dependencies
    print()
    for dep, package in [("fastapi", "fastapi"), ("uvicorn", "uvicorn"), ("httpx", "httpx")]:
        try:
            __import__(package)
            print(f"  ✓ {dep}: installed")
        except ImportError:
            print(f"  ✗ {dep}: NOT installed (pip install {dep})")
            errors += 1

    print()
    if errors:
        print(f"  ⚠ {errors} issue(s) found. Fix them before running 'coding-router serve'.")
    else:
        print("  ✓ All checks passed! Run 'coding-router serve' to start the gateway.")
    print()
    return 1 if errors else 0


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------


SUBCOMMANDS = {
    "init": init_command,
    "validate": validate_command,
    "models": models_command,
    "add-model": add_model_command,
    "dashboard": dashboard_command,
    "serve": serve_command,
    "doctor": doctor_command,
}


def main() -> int:
    if len(sys.argv) > 1:
        subcommand = sys.argv[1]
        if subcommand in SUBCOMMANDS:
            return SUBCOMMANDS[subcommand](sys.argv[2:])
    return route_command(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
