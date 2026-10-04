"""Rich terminal UI helpers for coding-router.

Provides the large ASCII banner, coloured status lines, spinners,
and download progress so every command feels polished and alive.
All rendering uses raw ANSI escape codes (no third-party dependency).
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Iterator


# ---------------------------------------------------------------------------
# ANSI colour / style helpers
# ---------------------------------------------------------------------------

_RESET     = "\033[0m"
_BOLD      = "\033[1m"
_DIM       = "\033[2m"
_ITALIC    = "\033[3m"
_UNDERLINE = "\033[4m"

# Foreground colours (256-colour mode)
def _fg(code: int) -> str:           return f"\033[38;5;{code}m"
def _bg(code: int) -> str:           return f"\033[48;5;{code}m"

CYAN       = _fg(51)
BLUE       = _fg(75)
PURPLE     = _fg(141)
MAGENTA    = _fg(207)
GREEN      = _fg(78)
YELLOW     = _fg(220)
RED        = _fg(203)
WHITE      = _fg(255)
GREY       = _fg(244)
DARK       = _fg(236)
ORANGE     = _fg(214)
TEAL       = _fg(43)
PINK       = _fg(219)

# Gradient shades for the banner (cool blue → electric cyan → violet)
_GRAD = [
    _fg(63), _fg(69), _fg(75), _fg(81), _fg(87),
    _fg(87), _fg(81), _fg(75), _fg(69), _fg(63),
]

_SUPPORTS_COLOR = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def c(color: str, text: str) -> str:
    """Wrap *text* in ANSI colour codes (or passthrough when no TTY)."""
    if not _SUPPORTS_COLOR:
        return text
    return f"{color}{text}{_RESET}"


def b(text: str) -> str:  return c(_BOLD,   text)
def d(text: str) -> str:  return c(_DIM,    text)
def i(text: str) -> str:  return c(_ITALIC, text)


# ---------------------------------------------------------------------------
# Big banner
# ---------------------------------------------------------------------------

_BANNER_LINES = [
    "  ██████╗ ██████╗ ██████╗ ██╗███╗   ██╗ ██████╗ ",
    "  ██╔════╝██╔═══██╗██╔══██╗██║████╗  ██║██╔════╝ ",
    "  ██║     ██║   ██║██║  ██║██║██╔██╗ ██║██║  ███╗",
    "  ██║     ██║   ██║██║  ██║██║██║╚██╗██║██║   ██║",
    "  ╚██████╗╚██████╔╝██████╔╝██║██║ ╚████║╚██████╔╝",
    "   ╚═════╝ ╚═════╝ ╚═════╝ ╚═╝╚═╝  ╚═══╝ ╚═════╝ ",
]

_ROUTER_LINES = [
    "  ██████╗  ██████╗ ██╗   ██╗████████╗███████╗██████╗ ",
    "  ██╔══██╗██╔═══██╗██║   ██║╚══██╔══╝██╔════╝██╔══██╗",
    "  ██████╔╝██║   ██║██║   ██║   ██║   █████╗  ██████╔╝",
    "  ██╔══██╗██║   ██║██║   ██║   ██║   ██╔══╝  ██╔══██╗",
    "  ██║  ██║╚██████╔╝╚██████╔╝   ██║   ███████╗██║  ██║",
    "  ╚═╝  ╚═╝ ╚═════╝  ╚═════╝    ╚═╝   ╚══════╝╚═╝  ╚═╝",
]

_TAGLINE = "Semantic LLM Router · Intelligent Model Selection · Powered by LiteLLM"


def _gradient_line(line: str, grad: list[str], base_color: str) -> str:
    """Apply a horizontal colour gradient to a single ASCII-art line."""
    if not _SUPPORTS_COLOR:
        return line
    words = list(line)
    width = max(len(grad), 1)
    out = []
    for idx, ch in enumerate(words):
        shade = grad[int(idx / max(len(words), 1) * width) % width]
        out.append(f"{_BOLD}{shade}{ch}")
    return "".join(out) + _RESET


def print_banner() -> None:
    """Print the big gradient 'CODING ROUTER' banner with tagline."""
    if not _SUPPORTS_COLOR:
        print("\n  ═══  CODING-ROUTER  ═══\n")
        return

    print()
    # "CODING" in blue→cyan gradient
    coding_grad = [_fg(c) for c in [63, 69, 75, 81, 87, 87, 81, 75, 69, 63]]
    for line in _BANNER_LINES:
        print(_gradient_line(line, coding_grad, BLUE))

    # small spacer then "ROUTER" in magenta→pink gradient
    router_grad = [_fg(c) for c in [129, 135, 141, 147, 207, 207, 147, 141, 135, 129]]
    for line in _ROUTER_LINES:
        print(_gradient_line(line, router_grad, PURPLE))

    # Decorative separator
    sep = "  " + "─" * 56
    print(f"\n{CYAN}{_BOLD}{sep}{_RESET}")
    # Tagline
    print(f"  {GREY}{_ITALIC}{_TAGLINE}{_RESET}")
    print(f"{CYAN}{_BOLD}{sep}{_RESET}\n")


def print_mini_banner(subtitle: str = "") -> None:
    """Compact one-line banner for sub-commands."""
    if not _SUPPORTS_COLOR:
        print(f"\n  ── coding-router  {subtitle} ──\n")
        return
    left  = f"  {_BOLD}{CYAN}coding{_RESET}{_BOLD}{PURPLE}-router{_RESET}"
    mid   = f"  {_fg(240)}·{_RESET}"
    right = f"  {_BOLD}{YELLOW}{subtitle}{_RESET}" if subtitle else ""
    print(f"\n{left}{mid}{right}\n")


# ---------------------------------------------------------------------------
# Status / log lines
# ---------------------------------------------------------------------------

def ok(msg: str) -> None:
    print(f"  {GREEN}{_BOLD}✔{_RESET}  {msg}")

def fail(msg: str) -> None:
    print(f"  {RED}{_BOLD}✘{_RESET}  {msg}", file=sys.stderr)

def warn(msg: str) -> None:
    print(f"  {YELLOW}{_BOLD}⚠{_RESET}  {YELLOW}{msg}{_RESET}")

def info(msg: str) -> None:
    print(f"  {CYAN}{_BOLD}→{_RESET}  {msg}")

def note(msg: str) -> None:
    print(f"  {GREY}{_DIM}ℹ  {msg}{_RESET}")

def section(title: str) -> None:
    """Print a styled section heading."""
    print(f"\n  {PURPLE}{_BOLD}{'━' * 4}  {title}  {'━' * (max(0, 44 - len(title)))}{_RESET}")

def kv(key: str, value: str, *, highlight: bool = False) -> None:
    """Print a labelled key-value pair."""
    val_color = CYAN if highlight else WHITE
    print(f"  {GREY}{key:<22}{_RESET}  {val_color}{_BOLD}{value}{_RESET}")

def rule(char: str = "─", width: int = 56) -> None:
    print(f"  {GREY}{char * width}{_RESET}")


# ---------------------------------------------------------------------------
# Spinner
# ---------------------------------------------------------------------------

_SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


class Spinner:
    """A non-blocking terminal spinner with a status message.

    Usage::

        with Spinner("Loading model weights…") as sp:
            heavy_work()
            sp.set_text("Almost done…")
    """

    def __init__(self, text: str, *, color: str = CYAN) -> None:
        self.text = text
        self.color = color
        self._active = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def set_text(self, text: str) -> None:
        with self._lock:
            self.text = text

    def _run(self) -> None:
        frame_idx = 0
        hide_cursor = "\033[?25l"
        show_cursor = "\033[?25h"
        if _SUPPORTS_COLOR:
            sys.stdout.write(hide_cursor)
            sys.stdout.flush()
        try:
            while self._active:
                frame = _SPINNER_FRAMES[frame_idx % len(_SPINNER_FRAMES)]
                with self._lock:
                    text = self.text
                line = (
                    f"  {self.color}{_BOLD}{frame}{_RESET}  {text}"
                    if _SUPPORTS_COLOR
                    else f"  ... {text}"
                )
                sys.stdout.write(f"\r{line}  ")
                sys.stdout.flush()
                time.sleep(0.08)
                frame_idx += 1
        finally:
            if _SUPPORTS_COLOR:
                sys.stdout.write(show_cursor)
            sys.stdout.write("\r" + " " * 80 + "\r")
            sys.stdout.flush()

    def start(self) -> "Spinner":
        self._active = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._active = False
        if self._thread:
            self._thread.join()

    def __enter__(self) -> "Spinner":
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# Download progress bar
# ---------------------------------------------------------------------------

def download_progress(
    current_bytes: int,
    total_bytes: int,
    *,
    label: str = "Downloading",
    width: int = 36,
) -> None:
    """Overwrite current line with a colour progress bar."""
    if total_bytes <= 0:
        mb = current_bytes / 1_048_576
        sys.stderr.write(f"\r  {CYAN}↓{_RESET}  {label}  {YELLOW}{mb:.1f} MB{_RESET}   ")
        sys.stderr.flush()
        return
    fraction = min(current_bytes / total_bytes, 1.0)
    filled   = int(width * fraction)
    empty    = width - filled
    pct      = int(fraction * 100)
    mb_cur   = current_bytes / 1_048_576
    mb_tot   = total_bytes   / 1_048_576

    bar   = f"{GREEN}{'█' * filled}{_RESET}{GREY}{'░' * empty}{_RESET}"
    stats = f"{YELLOW}{mb_cur:.1f}{_RESET}/{YELLOW}{mb_tot:.1f} MB{_RESET} {GREY}({pct}%){_RESET}"

    sys.stderr.write(f"\r  {CYAN}↓{_RESET}  {label}  [{bar}]  {stats}   ")
    sys.stderr.flush()
    if fraction >= 1.0:
        sys.stderr.write("\n")
        sys.stderr.flush()


# ---------------------------------------------------------------------------
# Routing result pretty-printer
# ---------------------------------------------------------------------------

def print_route_result(result: dict, *, mode: str = "", pool: str = "", latency_ms: float = 0) -> None:
    """Pretty-print a routing result dict to stdout."""
    sel   = result.get("selected_model", {})
    model = sel.get("model", "?")
    cat   = result.get("classifier_category", "?")
    grp   = result.get("selection_group",   "?")
    reason = result.get("selection_reason", "")
    service = sel.get("service", "?")
    score   = sel.get("score",   "")
    response = result.get("model_response")
    err      = result.get("invocation_error")

    section("Routing Decision")
    kv("Selected model",   model,   highlight=True)
    kv("Service",          service)
    kv("Category",         cat)
    kv("Group",            grp)
    if mode:
        kv("Routing mode", mode)
    if pool:
        kv("Candidate pool", pool)
    if score:
        kv("Score",        f"{score:.4f}" if isinstance(score, float) else str(score))
    if latency_ms:
        kv("Latency",      f"{latency_ms:.0f} ms")
    if reason:
        note(reason)

    if err:
        section("Invocation Error")
        fail(err)
    elif response:
        section("Model Response")
        print(f"\n  {WHITE}{response}{_RESET}\n")

    rule()


# ---------------------------------------------------------------------------
# Gateway startup block
# ---------------------------------------------------------------------------

def print_gateway_startup(
    host: str,
    port: int,
    mode: str,
    api_key: str | None,
    model_count: int,
) -> None:
    """Print the styled gateway info block shown when 'serve' starts."""
    print_banner()

    section("Gateway")
    kv("URL",           f"http://{host}:{port}", highlight=True)
    kv("Routing mode",  mode,                    highlight=True)
    kv("Models loaded", str(model_count))
    kv("LLM backend",   "LiteLLM (acompletion)",  highlight=False)
    if api_key:
        kv("API Key",  f"{api_key[:12]}…  (see ~/.config/coding-router/gateway.key)")
    else:
        kv("API Key",  "DISABLED (--no-auth)")

    section("Endpoints")
    for method, path, label in [
        ("POST", "/v1/chat/completions", "OpenAI Chat"),
        ("POST", "/v1/responses",        "OpenAI Responses"),
        ("POST", "/v1/messages",         "Anthropic Messages"),
        ("GET",  "/v1/models",           "Model listing"),
        ("GET",  "/health",              "Health check"),
    ]:
        m_color = GREEN if method == "GET" else BLUE
        print(
            f"  {m_color}{_BOLD}{method:<5}{_RESET}  "
            f"{WHITE}{path:<35}{_RESET}  "
            f"{GREY}{label}{_RESET}"
        )

    section("Controls")
    info(f"Virtual model name:  {c(CYAN, 'coding-router')}")
    info(f"Dashboard:           {c(CYAN, 'http://localhost:3000')}")
    warn("Press Ctrl+C to stop the gateway.")
    rule()
    print()
