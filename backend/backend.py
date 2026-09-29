from __future__ import annotations

import time
import contextvars
import uuid
from input_parser import parse_math
import re as _regex
from threading import RLock

from fastapi import FastAPI
from pydantic import BaseModel
from sympy import (
    symbols, I, exp, Integral, oo, latex, simplify, expand, diff, Derivative, re, im, sqrt, N, factorial,
    Add, Mul, pi, sin, cos, Piecewise, Abs, sign, factor_terms, sympify, S, integrate, tan, sinh, cosh, log, E
, Function, together, fraction, Poly, div, apart, limit, roots, Wild, srepr)
from sympy.functions.special.delta_functions import Heaviside, DiracDelta
from sympy.parsing.sympy_parser import (
    standard_transformations, implicit_multiplication_application
)

# ============================================================
# Fourier Backend (engineering convention, omega real)
#   X(omega) = integral_{-infinity}^{infinity} x(t)e^{-j omega t} dt
#
# User conventions:
#   - Convolution separators are normalized to U+2022 internally
#   - Multiplication uses '*' (or implicit multiplication)
#
# Policy:
#   - Force omega real (avoid complex-omega arg(...) artifacts)
#   - For ANY expression containing trig, first rewrite to complex exponentials.
#     If it becomes a finite sum of pure tones C_k e^{j omega_k t}, return the
#     distribution result via the definition integral:
#         integral e^{-j(omega-omega0)t} dt = 2*pi*delta(omega-omega0)
#   - Otherwise fall back to property rules + integral fallback.
# ============================================================
# (Engineering convention, omega real)
app = FastAPI(title="Fourier Backend ")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://zelaifeng-lab.github.io",   # GitHub Pages frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],   # allow POST/OPTIONS
    allow_headers=["*"],   # allow Content-Type
)


# ---------- middleware: per-request performance summary ----------
from starlette.requests import Request
from starlette.responses import Response
# ---------- middleware: per-request performance summary ----------
from starlette.requests import Request
from starlette.responses import Response

@app.middleware("http")
async def _perf_middleware(request: Request, call_next):
    if not PERF_LOG_ENABLED or request.url.path != "/fourier":
        return await call_next(request)
    req_id = uuid.uuid4().hex[:8]
    token = _PERF_CTX.set(_PerfCtxObj(req_id))
    t0 = time.perf_counter()
    try:
        resp: Response = await call_next(request)
        return resp
    finally:
        total = time.perf_counter() - t0
        ctx = _PERF_CTX.get()
        if ctx is not None and total >= PERF_SLOW_REQUEST_SEC:
            evs = sorted(ctx.events, key=lambda x: x[0], reverse=True)[:12]
            summary = ", ".join([f"{dt:.3f}s {name}" + (f" [{info}]" if info else "") for dt,name,info in evs])
            print(f"[perf {req_id}] total={total:.3f}s; top: {summary}")
        _PERF_CTX.reset(token)


BUILD_ID = "dissertation_revision_20260920"
APART_FULL_DEFAULT = False  # always keep real-field partial fractions (avoid RootSum)

# ===== Omega-real cleanup (avoid Piecewise/arg/RootSum) =====
def _omega_real_cleanup(expr):
    """
    Post-process SymPy outputs assuming omega is real:
    - replace omega/Abs(omega) with sign(omega)
    - drop Piecewise branches that only special-case omega=0 (removable after sign rewrite)
    """
    try:
        expr = expr.subs(omega/Abs(omega), sign(omega))
        expr = expr.subs(-omega/Abs(omega), -sign(omega))
        expr = expr.subs(Abs(omega)/omega, sign(omega))

        # Replace Abs(omega)*omega**(-1) patterns with sign(omega)
        def _abs_over_omega_to_sign(e):
            if not isinstance(e, Mul):
                return e
            args = list(e.args)
            try:
                idx_abs = args.index(Abs(omega))
            except ValueError:
                return e
            idx_inv = None
            for i, a in enumerate(args):
                if i == idx_abs:
                    continue
                if getattr(a, "is_Pow", False) and a.base == omega and a.exp == -1:
                    idx_inv = i
                    break
            if idx_inv is None:
                return e
            newargs = [a for j, a in enumerate(args) if j not in (idx_abs, idx_inv)]
            return Mul(sign(omega), *newargs)

        expr = expr.replace(
            lambda e: isinstance(e, Mul) and e.has(Abs(omega)) and e.has(omega),
            _abs_over_omega_to_sign
        )

        # recursively drop Piecewise(Eq(omega,0), True-default)
        def _strip_pw(e):
            if not isinstance(e, Piecewise):
                return e
            default_expr = None
            has_omega0 = False
            for ex, cond in e.args:
                if cond == True:
                    default_expr = ex
                elif getattr(cond, "is_Equality", False) and ((cond.lhs == omega and cond.rhs == 0) or (cond.rhs == omega and cond.lhs == 0)):
                    has_omega0 = True
            if has_omega0 and default_expr is not None:
                return default_expr
            return e
        expr = expr.replace(lambda e: isinstance(e, Piecewise), _strip_pw)
        return expr
    except Exception:
        return expr


def _simplify_spectrum(expr, **kwargs):
    """Simplify coefficients without expanding known sign distributions into cases."""
    from sympy import Dummy
    expr = sympify(expr)
    protected = {atom: Dummy(real=True) for atom in expr.atoms(sign)}
    return simplify(expr.xreplace(protected), **kwargs).xreplace({v:k for k,v in protected.items()})


# ---------- performance logging ----------
# Enable lightweight timing logs for slow requests / expensive SymPy operations.
PERF_LOG_ENABLED = False
PERF_SLOW_REQUEST_SEC = 0.50   # summarize per-request if total time exceeds this
PERF_EVENT_MIN_SEC = 0.02      # record individual events longer than this

_PERF_CTX = contextvars.ContextVar("perf_ctx", default=None)

class _PerfCtxObj:
    def __init__(self, req_id: str):
        self.req_id = req_id
        self.t0 = time.perf_counter()
        self.events = []  # list of (dt, name, info)

    def add(self, dt: float, name: str, info: str = ""):
        if dt >= PERF_EVENT_MIN_SEC:
            self.events.append((dt, name, info))

def _perf_add(dt: float, name: str, info: str = "") -> None:
    if not PERF_LOG_ENABLED:
        return
    ctx = _PERF_CTX.get()
    if ctx is None:
        return
    ctx.add(dt, name, info)

class _PerfTimer:
    __slots__ = ("name","info","t0")
    def __init__(self, name: str, info: str = ""):
        self.name = name
        self.info = info
        self.t0 = 0.0
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, exc_type, exc, tb):
        _perf_add(time.perf_counter() - self.t0, self.name, self.info)
        return False



t = symbols("t", real=True)
s_sym = symbols("s", real=True)
omega = symbols("omega", real=True)

def _tb_make_steps(*, recognize_lines, strategy_lines, pair_lines, combine_lines, final_expr):
    """
    Build a unified textbook-style step list:
      Identify -> Split/Properties -> Known pairs -> Combine/Simplify -> Final

    All inputs are lists of LaTeX strings (WITHOUT surrounding $$).
    This only changes narration; it must not affect computation.
    """
    steps = []
    steps.append(r"\textbf{Step 1: Identify}")
    steps.extend([s for s in (recognize_lines or []) if s])

    steps.append(r"\textbf{Step 2: Choose properties / decomposition}")
    steps.extend([s for s in (strategy_lines or []) if s])

    steps.append(r"\textbf{Step 3: Apply known transform pairs}")
    steps.extend([s for s in (pair_lines or []) if s])

    steps.append(r"\textbf{Step 4: Combine and simplify}")
    steps.extend([s for s in (combine_lines or []) if s])

    steps.append(r"\textbf{Final Result}")
    steps.append(r"X(\omega)=" + latex(final_expr))
    return steps


# ---------- lightweight caches for expensive SymPy operations ----------
_CACHE_LOCK = RLock()
_CACHE_MAX_SIZE = 512
_TOGETHER_CACHE = {}
_APART_CACHE = {}

def _cache_store(cache, key, value):
    with _CACHE_LOCK:
        if len(cache) >= _CACHE_MAX_SIZE:
            cache.clear()
        cache[key] = value
    return value

def _together_cached(expr):
    key = srepr(expr)
    with _CACHE_LOCK:
        v = _TOGETHER_CACHE.get(key)
        if v is not None:
            return v
    with _PerfTimer("sympy.together", info=str(expr)[:80]):
        v = together(expr)
    return _cache_store(_TOGETHER_CACHE, key, v)

def _apart_cached(expr):
    key = srepr(expr)
    with _CACHE_LOCK:
        v = _APART_CACHE.get(key)
        if v is not None:
            return v
    with _PerfTimer("sympy.apart", info=str(expr)[:80]):
        v = apart(expr, t, full=APART_FULL_DEFAULT)
    return _cache_store(_APART_CACHE, key, v)


def _apart_real_roots(expr):
    """Try real-root partial fractions when sympy.apart(full=False) does not split.

    Returns an Add expression (polynomial part + sum of A_{k}/(t-r)^k) or None.
    Only triggers when denominator has (provably) real roots (including algebraic radicals).
    """
    try:
        num, den = fraction(_together_cached(expr))
        Pn, Pd = Poly(num, t), Poly(den, t)
        if Pd.is_zero:
            return None
        # If already a polynomial, nothing to do
        if Pd.degree() <= 0:
            return None

        # Polynomial long division to ensure a proper rational remainder
        q_poly, r_poly = div(Pn, Pd)
        q_expr = (q_poly.as_expr() if q_poly is not None else 0)
        r_expr = (r_poly.as_expr() if r_poly is not None else num)

        if r_expr == 0:
            return q_expr

        proper = r_expr / den

        # Roots with multiplicities (may include algebraic radicals / RootOf)
        roots_dict = roots(Pd, t)
        if not roots_dict:
            return None

        # Require all roots to be real (best-effort: is_real or numeric check)
        for r, m in roots_dict.items():
            if getattr(r, "is_real", None) is True:
                continue
            if getattr(r, "is_real", None) is False:
                return None
            # fall back to numeric check if possible
            try:
                if abs(complex(N(r))) == float("inf"):
                    return None
                if abs(im(N(r))) > 1e-10:
                    return None
            except Exception:
                # unknown -> be conservative
                return None

        parts = []

        # Build partial fraction terms for each root/multiplicity
        for r, m in roots_dict.items():
            if m == 1:
                Ak = simplify(limit((t - r) * proper, t, r))
                if Ak != 0:
                    parts.append(Ak / (t - r))
            else:
                # repeated root: sum_{k=1..m} A_k/(t-r)^k
                # A_k = 1/(m-k)! * lim_{t->r} d^{m-k}/dt^{m-k} ((t-r)^m * proper)
                base = (t - r) ** m * proper
                for k in range(1, m + 1):
                    deriv_order = m - k
                    expr_k = base
                    if deriv_order > 0:
                        expr_k = diff(expr_k, t, deriv_order)
                    Ak = simplify(limit(expr_k, t, r) / factorial(deriv_order))
                    if Ak != 0:
                        parts.append(Ak / (t - r) ** k)

        if not parts:
            return None

        return q_expr + Add(*parts, evaluate=False)
    except Exception:
        return None

# Rect(x): unit rectangular pulse, rect(x)=1 for |x|<=1/2 else 0
class Rect(Function):
    nargs = 1

class Tri(Function):
    nargs = 1

# PV(x): principal value marker (display only)
class PV(Function):
    nargs = 1

TRANSFORMS = standard_transformations + (implicit_multiplication_application,)

# ---------- parsing / normalization ----------

def _convert_frac_calls(s: str) -> str:
    # Replace FRAC(a,b) -> ((a)/(b)) with nesting support
    out = []
    i = 0
    while i < len(s):
        if s.startswith("FRAC(", i):
            i += 5
            depth = 1
            args = []
            cur = []
            while i < len(s) and depth > 0:
                ch = s[i]
                if ch == "(":
                    depth += 1
                    cur.append(ch)
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        args.append("".join(cur).strip())
                        cur = []
                        i += 1
                        break
                    cur.append(ch)
                elif ch == "," and depth == 1:
                    args.append("".join(cur).strip())
                    cur = []
                else:
                    cur.append(ch)
                i += 1
            if len(args) == 2:
                out.append(f"(({_convert_frac_calls(args[0])})/({_convert_frac_calls(args[1])}))")
            else:
                out.append("FRAC(" + ",".join(args) + ")")
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def _normalize_convolution_symbols(s: str) -> str:
    return (
        (s or "")
        .replace(r"\bullet", "\u2022")
        .replace("\u00b7", "\u2022")
        .replace("\u2219", "\u2022")
        .replace("\u22c6", "\u2022")
        .replace("\u2217", "\u2022")
    )


def _pre_normalize(s: str) -> str:
    s = _normalize_convolution_symbols(s).strip()
    s = s.replace("\uff08", "(").replace("\uff09", ")")
    s = s.replace("\uff0c", ",").replace("\u3001", ",")
    s = s.replace("\u03c0", "pi")
    s = s.replace("\u03c9", "omega")
    s = s.replace("\u2212", "-")

    # users often type '^' for power; sympy uses '**'
    s = s.replace("^", "**")

    # step/delta aliases
    s = s.replace("u(", "Heaviside(")
    s = s.replace("\u03b8(", "Heaviside(")
    s = s.replace("heaviside(", "Heaviside(")

    s = s.replace("\u03b4(", "DiracDelta(")
    s = s.replace("delta(", "DiracDelta(")
    s = s.replace("abs(", "Abs(")

    # frac(a,b)
    s = s.replace("rect(", "Rect(")
    s = s.replace("tri(", "Tri(")
    s = s.replace("frac(", "FRAC(")
    s = _convert_frac_calls(s)
    return s


def _parse_sympy(expr_str: str):
    if len(expr_str) > 2048:
        raise ValueError("Expression is too long (maximum 2048 characters).")
    depth = 0
    for ch in expr_str:
        depth += (ch in '(（') - (ch in ')）')
        if depth < 0 or depth > 32:
            raise ValueError('Invalid or excessively nested parentheses.')
    if depth:
        raise ValueError('Unbalanced parentheses.')
    expr_str = _pre_normalize(expr_str)
    local_dict = {
        "t": t,
        "omega": omega,
        "pi": pi,
        "Heaviside": Heaviside,
        "DiracDelta": DiracDelta,
        "I": I,
        "j": I,
        "exp": exp,
        "sin": sin,
        "cos": cos,
        "Abs": Abs,
        "sign": sign,
        "Rect": Rect,
        "Tri": Tri,
    }
    local_dict.update({"e": E, "i": I, "tan": tan, "sinh": sinh,
                       "cosh": cosh, "sqrt": sqrt, "log": log, "PV": PV})
    expr = parse_math(expr_str, local_dict, TRANSFORMS)
    if expr.has(S.NaN, S.ComplexInfinity, oo, -oo):
        raise ValueError("The input contains an undefined or infinite value.")
    if expr.has(sign) or expr.has(Rect) or expr.has(Tri):
        return expr
    return simplify(expr)


def _strip_outer_parens_once(s: str) -> str:
    s = s.strip()
    if not (s.startswith("(") and s.endswith(")")):
        return s
    depth = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if depth == 0 and i != len(s) - 1:
            return s
    return s[1:-1].strip()


def _split_convolution_top_level(s: str):
    """
    Convolution is normalized to '?' (U+2022) and split only at top level.
    Multiplication must be written as '*'.
    """
    s = _strip_outer_parens_once(_normalize_convolution_symbols(s))
    depth = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if depth != 0:
            continue
        if ch == "\u2022":
            left = s[:i].strip()
            right = s[i + 1 :].strip()
            if left and right:
                return left, right
            return None
    return None


# ---------- engineering Fourier core ----------

def _fourier_def_integral_steps(f):
    integrand = simplify(f * exp(-I * omega * t))
    X_def = Integral(integrand, (t, -oo, oo))
    steps = [
        r"X(\omega)=\int_{-\infty}^{\infty}x(t)\,e^{-j\omega t}\,dt",
        r"x(t)=" + latex(f),
        r"\Rightarrow\;X(\omega)=\int_{-\infty}^{\infty}\left(" + latex(f) + r"\right)e^{-j\omega t}\,dt",
        r"=\int_{-\infty}^{\infty}" + latex(integrand) + r"\,dt",
        ]
    return X_def, integrand, steps


def _fourier_one_sided_steps(g):
    integrand = simplify(g * exp(-I * omega * t))
    X_def = Integral(integrand, (t, 0, oo))
    steps = [
        r"x(t)=g(t)\,u(t),\;u(t)=\mathrm{Heaviside}(t)",
        r"X(\omega)=\int_{0}^{\infty}g(t)\,e^{-j\omega t}\,dt",
        r"g(t)=" + latex(g),
        r"\Rightarrow\;X(\omega)=\int_{0}^{\infty}" + latex(integrand) + r"\,dt",
        ]
    return X_def, integrand, steps


def _doit_or_keep_integral(X_def):
    _record_method("direct_integral")
    try:
        X = X_def.doit()
        X = _simplify_spectrum(X)
        return not X.has(Integral), X
    except Exception:
        return False, X_def


def _piecewise_conditions_latex(pw: Piecewise) -> str:
    parts = []
    for _, cond in pw.args:
        parts.append(latex(cond))
    if not parts:
        return ""
    seen = set()
    uniq = []
    for c in parts:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return r"\text{Piecewise conditions: }" + r"\;\text{or}\;".join(uniq)


# ---------- helpers for pattern rules ----------

def _as_linear_in_t(expr):
    """If expr is a*t + b with a,b independent of t, return (a,b), else None.

    Iron gate: ONLY accept true affine (degree-1 polynomial) in t.
    This prevents mis-detecting expressions like t*(t^2+5t+4) as (a(t))*t + b.
    """
    try:
        if not expr.is_polynomial(t):
            return None
        P = Poly(expr, t)
        if P.degree() != 1:
            return None
        a, b = P.all_coeffs()  # expr = a*t + b
        if a.has(t) or b.has(t):
            return None
        return simplify(a), simplify(b)
    except Exception:
        return None



def _match_t_plus_a(den):
    """Return a such that den == t + a (with coefficient 1), else None."""
    lin = _as_linear_in_t(den)
    if lin is None:
        return None
    a1, b1 = lin
    if simplify(a1 - 1) != 0:
        return None
    return simplify(b1)


def _phase_factor_latex(shift):
    return _format_result_display_latex(latex(exp(I * omega * simplify(shift))))


def _display_in_s(expr):
    return _format_step_display_latex(latex(expand(expr).subs(t, s_sym)))


def _linear_phase_display(func_name: str, a, b, *, var: str = "t") -> str:
    var_sym = t if var == "t" else s_sym
    arg = simplify(a * var_sym + b)
    return rf"\{func_name}\!\left(" + _format_step_display_latex(latex(arg)) + r"\right)"


def _complex_exponential_display(w0, phi) -> str:
    return _format_step_display_latex(latex(exp(I * simplify(w0 * t + phi))))


def _is_definition_heading(s: str) -> bool:
    return "Fourier transform definition" in s


def _is_definition_integral_line(s: str) -> bool:
    return s.startswith(r"X(\omega)=\int_{-\infty}^{\infty}x(t)e^{-j\omega t}\,dt")


def _strip_leading_definition_block(steps: list[str]) -> list[str]:
    if len(steps) >= 3 and _is_definition_heading(steps[0]) and steps[1].startswith("x(t)=") and _is_definition_integral_line(steps[2]):
        return steps[3:]
    return steps


def _drop_nested_definition_blocks(steps: list[str]) -> list[str]:
    cleaned: list[str] = []
    i = 0
    while i < len(steps):
        if i + 2 < len(steps) and _is_definition_heading(steps[i]) and steps[i + 1].startswith("x(t)=") and _is_definition_integral_line(steps[i + 2]):
            i += 3
            continue
        cleaned.append(steps[i])
        i += 1
    return cleaned


def _should_keep_definition_block(steps: list[str]) -> bool:
    joined = "\n".join(steps)
    keep_markers = (
        r"Replace the full integral",
        r"finite interval",
        r"integration range",
        r"Evaluate the",
        r"Closed-form not found",
        r"returned the engineering-definition integral",
    )
    return any(marker in joined for marker in keep_markers)


def _renumber_teaching_steps(steps: list[str]) -> list[str]:
    out: list[str] = []
    n = 1
    pattern = _regex.compile(r"^\\textbf\{Step\s+\d+:\s*(.*?)\}$")
    for s in steps:
        m = pattern.match(s)
        if m:
            out.append(r"\textbf{Step " + str(n) + ": " + m.group(1) + "}")
            n += 1
        else:
            out.append(s)
    return out


def _strip_step_heading(s: str) -> str:
    m = _regex.match(r"^\\textbf\{Step\s+\d+:\s*(.*?)\}$", s)
    if not m:
        return s
    return r"\text{" + m.group(1) + r"}"


def _format_numeric_exponential_order(text: str) -> str:
    def repl(match):
        sign = match.group(1)
        coeff = match.group(2).strip()
        if coeff in ("0", "0.0"):
            return "1"
        if coeff == "1":
            return rf"e^{{{sign}j\omega}}"
        if coeff == "-1":
            opposite = "-" if sign == "+" else ""
            return rf"e^{{{opposite}j\omega}}"
        if coeff.startswith("-"):
            actual = "-" if sign == "+" else ""
            coeff = coeff[1:].strip()
        else:
            actual = "-" if sign == "-" else ""
        return rf"e^{{{actual}{coeff}j\omega}}"

    numeric = r"(-?\d+(?:\.\d+)?|-?\\frac\{[^{}]+\}\{[^{}]+\})"
    text = _regex.sub(rf"e\^\{{([+-])j\\omega\s+({numeric})\}}", repl, text)
    return text


def _step_start_definition(f):
    return [
        r"\textbf{Step 1: According to the Fourier transform definition}",
        r"x(t)=" + _format_step_display_latex(latex(f)),
        r"X(\omega)=\int_{-\infty}^{\infty}x(t)e^{-j\omega t}\,dt",
    ]


def _step_final_result(X):
    return [
        r"\textbf{Final Result}",
        r"X(\omega)=" + latex(X),
    ]


def _format_step_display_latex(text: str) -> str:
    """Use consistent teaching notation in steps: u(t) for steps and j for engineering convention."""
    s = text.replace("\u03c9", r"\omega").replace("\u8805", r"\omega")
    s = _regex.sub(r"\\theta\\left\((.*?)\\right\)", r"u(\1)", s)
    s = s.replace(r"e^{-i", r"e^{-j")
    s = s.replace(r"e^{i", r"e^{j")
    s = s.replace(r"e^{{-i", r"e^{{-j")
    s = s.replace(r"e^{{i", r"e^{{j")
    s = s.replace(r"-i\pi", r"-j\pi")
    s = s.replace(r"- i \pi", r"- j \pi")
    s = s.replace(r" i \pi", r" j \pi")
    s = s.replace(r"i\omega", r"j\omega")
    s = s.replace(r"i \omega", r"j \omega")
    s = s.replace(r"-i", r"-j")
    s = _regex.sub(r"(?<![A-Za-z])i(?![A-Za-z])", "j", s)
    s = s.replace(r"\pj", r"\pi")
    return _format_result_display_latex(s)


def _pv_fraction_latex(term: str) -> str:
    return rf"(\mathrm{{PV}})\frac{{1}}{{{term}}}"


def _replace_pv_fraction_form(text: str, prefix: str, suffix: str) -> str:
    """Convert PV(1/term)-style LaTeX into (PV) 1/term display notation."""
    s = text
    search_from = 0
    while True:
        start = s.find(prefix, search_from)
        if start == -1:
            return s

        term_start = start + len(prefix)
        depth = 0
        term_end = None
        for idx in range(term_start, len(s)):
            ch = s[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                if depth == 0:
                    term_end = idx
                    break
                depth -= 1

        if term_end is None:
            return s

        suffix_start = term_end + 1
        suffix_end = suffix_start + len(suffix)
        if suffix and s[suffix_start:suffix_end] != suffix:
            search_from = term_start
            continue

        term = s[term_start:term_end]
        replacement = _pv_fraction_latex(term)
        if suffix:
            s = s[:start] + replacement + s[suffix_end:]
        else:
            s = s[:start] + replacement + s[term_end + 1:]
        search_from = start + len(replacement)


def _format_result_display_latex(text: str) -> str:
    """Keep math unchanged while making common distribution notation more readable."""
    s = text
    for n in range(1, 10):
        s = s.replace(
            rf"\delta^{{\left( {n} \right)}}\left( \omega \right)",
            rf"\delta^{{({n})}}(\omega)",
        )
    s = s.replace(r"\left|{\omega}\right|", r"|\omega|")
    s = s.replace(r"\operatorname{sign}{\left(\omega \right)}", r"\mathrm{sign}(\omega)")
    s = _replace_pv_fraction_form(s, r"\operatorname{PV}{\left(\frac{1}{", r" \right)}")
    s = _replace_pv_fraction_form(s, r"\mathrm{PV}\!\left(\frac{1}{", r"\right)")
    s = _replace_pv_fraction_form(s, r"\mathrm{PV}\frac{1}{", "")
    s = s.replace(
        r"- \frac{j}{\omega}",
        r"- j (\mathrm{PV})\frac{1}{\omega}",
    )
    s = s.replace(
        r"+ \frac{j}{\omega}",
        r"+ j (\mathrm{PV})\frac{1}{\omega}",
    )
    s = _format_numeric_exponential_order(s)
    s = _regex.sub(r"(?<![A-Za-z])i(?![A-Za-z])", "j", s)
    s = s.replace(
        r"- \frac{j}{\omega}",
        r"- j (\mathrm{PV})\frac{1}{\omega}",
    )
    s = s.replace(
        r"+ \frac{j}{\omega}",
        r"+ j (\mathrm{PV})\frac{1}{\omega}",
    )
    return s



def _try_trig_as_exp_distribution(f):
    """
    Robust trig -> exp -> pure-tone detection.

    Also has a guaranteed path for sin(a*t+b) / cos(a*t+b) to avoid SymPy's arg(omega) Piecewise.

    NOTE: This function is "steps-only refactor": computation is unchanged, only steps text is made
    textbook-like.
    """
    # Guaranteed: sin(a*t+b), cos(a*t+b) -> delta-combination (distribution)
    if f.func in (sin, cos) and len(f.args) == 1:
        lin = _as_linear_in_t(f.args[0])
        if lin is not None:
            a, b = lin  # argument = a*t + b, with a,b independent of t
            if f.func == sin:
                X = _simplify_spectrum(pi / I * (exp(I*b) * DiracDelta(omega - a) - exp(-I*b) * DiracDelta(omega + a)))
                steps = _step_start_definition(f)
                steps += [
                    r"\textbf{Step 2: Identify the sinusoidal phase}",
                    r"\theta=a t+b,\quad a=" + latex(a) + r",\quad b=" + latex(b),
                    r"\textbf{Step 3: Use Euler identity}",
                    r"\sin(\theta)=\frac{e^{j\theta}-e^{-j\theta}}{2j},\;\;\theta=a t+b",
                    r"\textbf{Step 4: Use Fourier transform of complex exponentials}",
                    r"\mathcal{F}\{e^{j\omega_0 t}\}=2\pi\,\delta(\omega-\omega_0)",
                    r"\text{Here the two exponential frequencies are }\omega_0=a\text{ and }\omega_0=-a.",
                    r"\textbf{Step 5: Combine delta functions}",
                    r"X(\omega)=\frac{\pi}{j}\Big(e^{jb}\delta(\omega-a)-e^{-jb}\delta(\omega+a)\Big)",
                    ]
                steps += _step_final_result(X)
                return ("distribution_form", True, X, steps, "", None)
            else:
                X = _simplify_spectrum(pi * (exp(I*b) * DiracDelta(omega - a) + exp(-I*b) * DiracDelta(omega + a)))
                steps = _step_start_definition(f)
                steps += [
                    r"\textbf{Step 2: Identify the sinusoidal phase}",
                    r"\theta=a t+b,\quad a=" + latex(a) + r",\quad b=" + latex(b),
                    r"\textbf{Step 3: Use Euler identity}",
                    r"\cos(\theta)=\frac{e^{j\theta}+e^{-j\theta}}{2},\;\;\theta=a t+b",
                    r"\textbf{Step 4: Use Fourier transform of complex exponentials}",
                    r"\mathcal{F}\{e^{j\omega_0 t}\}=2\pi\,\delta(\omega-\omega_0)",
                    r"\text{Here the two exponential frequencies are }\omega_0=a\text{ and }\omega_0=-a.",
                    r"\textbf{Step 5: Combine delta functions}",
                    r"X(\omega)=\pi\Big(e^{jb}\delta(\omega-a)+e^{-jb}\delta(\omega+a)\Big)",
                    ]
                steps += _step_final_result(X)
                return ("distribution_form", True, X, steps, "", None)

    # General path: rewrite to exp and detect pure tones
    if not (f.has(sin) or f.has(cos)):
        return None

    fe = simplify(expand(f.rewrite(exp)))

    terms = list(Add.make_args(fe))
    pairs = []  # (Ck, wk)

    for term in terms:
        term = simplify(term)

        # Split coefficient independent of t
        coeff, rest = term.as_independent(t, as_Add=False)

        # Collect exponent(s) from exp factors, allowing products of exp(...)
        exp_args = []

        def _collect_exp(x):
            if x.func == exp and len(x.args) == 1:
                exp_args.append(x.args[0])
                return True
            # exp(arg)**(-1) == exp(-arg)
            if x.is_Pow and x.base.func == exp and x.exp == -1:
                exp_args.append(-x.base.args[0])
                return True
            return False

        if rest == 1:
            # constant term -> would transform to delta(omega); but keep this function for pure tones only
            return None

        if isinstance(rest, Mul):
            rest_factors = list(rest.args)
        else:
            rest_factors = [rest]

        nonexp = []
        for fac in rest_factors:
            if not _collect_exp(fac):
                nonexp.append(fac)

        if nonexp:
            # Not a pure tone term
            return None

        total_exp = simplify(sum(exp_args))
        # Expect total_exp = I*(w*t + phi)
        inside = simplify(total_exp / I)
        if inside.has(I):
            return None
        lin = _as_linear_in_t(inside)
        if lin is None:
            return None
        w, phi = lin
        if phi != 0:
            # absorb constant phase into coefficient
            coeff = simplify(coeff * exp(I*phi))
        pairs.append((coeff, w))

    # Build X(omega)=sum 2*pi*Ck*delta(omega-wk)
    X = 0
    for Ck, wk in pairs:
        X += simplify(2*pi*Ck*DiracDelta(omega - wk))
    X = _omega_real_cleanup(X)

    # Textbook-like steps (no internal logs)
    steps = _step_start_definition(f)
    steps += [
        r"\textbf{Step 2: Identify sinusoidal signal}",
        r"\textbf{Step 2: Use Euler identity}",
        r"\text{Rewrite }x(t)\text{ as a sum of complex exponentials.}",
        r"\textbf{Step 3: Use Fourier transform of complex exponentials}",
        r"\mathcal{F}\{e^{j\omega_0 t}\}=2\pi\,\delta(\omega-\omega_0)",
        r"\textbf{Step 4: Combine delta functions}",
        r"X(\omega)=\sum_k 2\pi C_k\,\delta(\omega-\omega_k)",
        r"\textbf{Final Result}",
        r"X(\omega)=" + latex(X),
        ]
    return ("distribution_form", True, X, steps, "", None)
def _rule_linear_over_t2_plus_c(f):
    """
    Known pair:
      (a*t + b)/(t^2 + c), c>0

    F{(a t + b)/(t^2 + c)} =
      pi*exp(-sqrt(c)*Abs(omega))*(b/sqrt(c) - j*a*sign(omega))
    """
    try:
        num, den = fraction(together(f))

        # Normalize denominator to monic quadratic in t by absorbing any
        # t-independent scalar factor into the numerator.
        Pden0 = Poly(den, t)
        if Pden0.degree() != 2:
            return None
        lc = simplify(Pden0.LC())
        if lc.has(t):
            return None
        if lc != 1:
            num = simplify(num / lc)
            den = simplify(den / lc)

        Pden = Poly(den, t)
        a2, a1, a0 = Pden.all_coeffs()
        if simplify(a2 - 1) != 0 or simplify(a1) != 0:
            return None
        c = simplify(a0)
        if c.has(t):
            return None
        if not (c.is_positive or (c.is_Number and float(c) > 0)):
            return None

        Pnum = Poly(num, t)
        if Pnum.degree() > 1:
            return None
        if Pnum.degree() == 1:
            a, b = Pnum.all_coeffs()
        else:
            a = 0
            b = Pnum.all_coeffs()[0]

        c_sqrt = sqrt(c)
        X = pi * exp(-c_sqrt * Abs(omega)) * (b/c_sqrt - I*a*sign(omega))

        steps = _step_start_definition(f)
        steps += [
            r"\textbf{Step 2: Decompose the numerator by linearity}",
            r"\frac{a t + b}{t^2+c}=a\frac{t}{t^2+c}+b\frac{1}{t^2+c}",
            r"\textbf{Step 3: Use the two standard rational transform pairs}",
            r"\text{Known pair: }\frac{1}{t^2+c}\;\xleftrightarrow{\mathcal{F}}\;\frac{\pi}{\sqrt{c}}e^{-\sqrt{c}|\omega|},\;c>0",
            r"\text{And }\frac{t}{t^2+c}\;\xleftrightarrow{\mathcal{F}}\;-j\pi\,\mathrm{sign}(\omega)e^{-\sqrt{c}|\omega|}",
            r"\textbf{Step 4: Identify parameters}",
            r"\text{Here }a=%s,\;b=%s,\;c=%s" % (latex(a), latex(b), latex(c)),
            ]
        X = _omega_real_cleanup(X)
        steps += _step_final_result(X)
        return ("distribution_form", True, X, steps, "", None)
    except Exception:
        return None
def _rule_rational_apart_linearity(f):
    """
    If f is a rational function in t and sympy can decompose it (long division + apart),
    compute FT by linearity term-wise using existing rules.

    NOTE: This function is "steps-only refactor": computation is unchanged, only steps are
    written in a textbook style (no internal rule logs).
    """
    # Fast guard:
    # If f is already a simple term that other rules can handle (e.g. 1/(t+a), 1/(t+a)^2,
    # or (at+b)/(t^2+c)), do NOT run together/div/apart again.
    # HOWEVER: for products of distinct linear factors like 1/((t+a)(t+b)), we MUST allow apart.
    try:
        num0, den0 = fraction(together(f))
        Pn0, Pd0 = Poly(num0, t), Poly(den0, t)

        if Pd0.degree() <= 2 and Pn0.degree() <= 1:
            # Skip expensive apart only for shapes already covered by dedicated rules:
            #   1/(t+a), 1/(t+a)^2, and (at+b)/(t^2+c) / 1/(t^2+a^2) (monic, no t-term).
            # But for general quadratics with real (possibly irrational) roots, we MUST allow decomposition.
            if Pd0.degree() == 1 and Pn0.degree() < Pd0.degree():
                return None
            if Pd0.degree() == 2:
                try:
                    a2, a1, a0 = Pd0.all_coeffs()
                    if simplify(a2 - 1) == 0 and simplify(a1) == 0:
                        # Only allow apart when it truly splits into two distinct linear factors.
                        facs = Pd0.factor_list()[1]
                        degs = [fp.degree() for (fp, _e) in facs]
                        if not (len(facs) == 2 and degs == [1, 1]):
                            return None
                except Exception:
                    pass
    except Exception:
        pass

    try:
        if not getattr(f, "is_rational_function", None) or (not f.is_rational_function(t)):
            return None

        # normalize to a single fraction
        num, den = fraction(together(f))

        # attempt polynomial long division (used for nicer decomposition display)
        f_rem = num/den
        div_line = None
        try:
            P = Poly(num, t)
            Q = Poly(den, t)
            q_poly, r_poly = div(P, Q)
            if q_poly is not None and r_poly is not None and (q_poly.as_expr() != 0):
                div_line = latex(f) + "=" + latex(q_poly.as_expr()) + r"+\frac{" + latex(r_poly.as_expr()) + "}{" + latex(den) + "}"
                f_rem = q_poly.as_expr() + r_poly.as_expr()/den
        except Exception:
            pass

        # partial fractions over reals (keeps irreducible quadratics as needed)
        pf = _apart_cached(f_rem)
        if pf == f_rem and not f_rem.is_Add:
            # sympy.apart(full=False) may refuse to split when roots are irrational.
            # If the denominator has real roots, do a lightweight residue-based partial fraction.
            pf2 = _apart_real_roots(f_rem)
            if pf2 is None or pf2 == f_rem:
                return None
            pf = pf2

        # transform each additive term (computation unchanged)

        terms = pf.as_ordered_terms() if pf.is_Add else [pf]

        X_terms = []
        conditions = []


        # Build term-wise derivations for display (textbook style)

        termwise_steps = []

        for k, term in enumerate(terms, start=1):

            # Compute transform of this term using the same engine (does not change math rules)

            form_k, ok_k, Xk, steps_k, cond_k, err_k = _derive_with_properties(term)


            if not ok_k or err_k or form_k not in {'closed_form', 'distribution_form'}:
                return None
            if cond_k:
                conditions.append(cond_k)
            X_terms.append(Xk)


            termwise_steps.append(r"\textbf{Term %d}:\quad x_{%d}(t)=%s" % (k, k, latex(term)))


            # Include the term's own derivation steps, but avoid repeating global headers/markers.

            for s in (steps_k or []):

                if not s:

                    continue

                ss = str(s).strip()

                if not ss:

                    continue

                if _is_definition_heading(ss):

                    continue

                if ss.startswith("x(t)="):

                    continue

                if _is_definition_integral_line(ss):

                    continue

                if ("=\\mathcal" in ss and "X(" in ss):

                    continue

                if ss.startswith("Method:"):

                    continue

                if "Final Result" in ss:

                    continue

                if ss.startswith(r"\textbf{Step"):

                    if "Use the known or previously derived transform" in ss:
                        termwise_steps.append(r"\text{Use the known or previously derived transform of }g(t)")
                    else:
                        termwise_steps.append(_strip_step_heading(ss))
                    continue

                termwise_steps.append(ss)


            termwise_steps.append(r"\Rightarrow\; X_{%d}(\omega)=%s" % (k, latex(Xk)))


        X = _omega_real_cleanup(Add(*X_terms, evaluate=False))

        # Textbook-like steps
        steps = _step_start_definition(f)
        steps += [
            r"\textbf{Step 2: Apply partial fraction decomposition}",
            ]
        if div_line is not None:
            steps.append(div_line)
        steps.append(latex(f_rem) + "=" + latex(pf))
        steps.append(r"\textbf{Step 3: Use known Fourier transform pairs}")
        steps.extend(termwise_steps)
        steps.append(r"\textbf{Step 4: Combine by linearity}")
        steps.append(r"X(\omega)=\sum_k \mathcal{F}\{t_k\}")
        steps += _step_final_result(X)

        return ("distribution_form", True, X, steps, r",\quad ".join(dict.fromkeys(conditions)), None)
    except Exception:
        return None
def _match_shifted_power(f, n):
    # Match 1/(t+a)^n where n=1 or 2; return a if matched.
    # IMPORTANT: Do NOT match products like 1/((t+a)(t+b)) as 1/(t+a).
    if f.is_Pow and f.exp == -n:
        base = f.base
        lin = _as_linear_in_t(base)
        if lin and lin[0] == 1:
            return simplify(lin[1])

    if isinstance(f, Mul):
        args = list(f.args)
        for i, arg in enumerate(args):
            if arg.is_Pow and arg.exp == -n:
                # Only allow (const) * 1/(t+a)^n. If remaining factors still depend on t,
                # it's not a pure shifted-power term.
                rest = Mul(*[a for j, a in enumerate(args) if j != i])
                if rest.has(t):
                    continue
                base = arg.base
                lin = _as_linear_in_t(base)
                if lin and lin[0] == 1:
                    return simplify(lin[1])
    return None
def _match_heaviside_shift(expr):
    """Return a for Heaviside(t-a), or None if expr is not a unit shifted step."""
    if not _is_heaviside(expr):
        return None
    lin = _as_linear_in_t(expr.args[0])
    if lin is None:
        return None
    a1, b1 = lin
    if simplify(a1 - 1) != 0:
        return None
    return simplify(-b1)


def _heaviside_shift_and_coeff(expr):
    """Return (coeff, a) for coeff*Heaviside(t-a), else None."""
    shift = _match_heaviside_shift(expr)
    if shift is not None:
        return simplify(1), shift
    if isinstance(expr, Mul):
        coeff, rest = expr.as_independent(t, as_Add=False)
        shift = _match_heaviside_shift(rest)
        if shift is not None:
            return simplify(coeff), shift
    return None


def _rule_shifted_heaviside_distribution(f):
    a_shift = _match_heaviside_shift(f)
    if a_shift is None:
        return None

    base = pi*DiracDelta(omega) - I*PV(1/omega)
    X = base if simplify(a_shift) == 0 else exp(-I*omega*a_shift) * base
    steps = _step_start_definition(f)
    steps += [
        r"\textbf{Step 2: Identify the signal as a shifted unit step}",
        r"x(t)=u(t-a),\quad a=" + latex(a_shift),
        r"\textbf{Step 3: Use the known step transform pair}",
        r"\mathcal{F}\{u(t)\}=\pi\delta(\omega)-j\,\mathrm{PV}\!\left(\frac{1}{\omega}\right)",
        r"\text{The PV term appears because }u(t)\text{ is not absolutely integrable over }(-\infty,\infty).",
        r"\textbf{Step 4: Apply the time-shift property}",
        r"\mathcal{F}\{u(t-a)\}=e^{-j\omega a}\left(\pi\delta(\omega)-j\,\mathrm{PV}\!\left(\frac{1}{\omega}\right)\right)",
        r"\text{Substitute }a=" + latex(a_shift),
    ]
    steps += _step_final_result(X)
    return "distribution_form", True, X, steps, "", None


def _rule_finite_step_window(f):
    if not isinstance(f, Add):
        return None
    terms = list(f.args)
    if len(terms) != 2:
        return None

    p0 = _heaviside_shift_and_coeff(terms[0])
    p1 = _heaviside_shift_and_coeff(terms[1])
    if p0 is None or p1 is None:
        return None

    c0, s0 = p0
    c1, s1 = p1
    if simplify(c0 - 1) == 0 and simplify(c1 + 1) == 0:
        a, b = s0, s1
    elif simplify(c1 - 1) == 0 and simplify(c0 + 1) == 0:
        a, b = s1, s0
    else:
        return None

    X = (exp(-I*omega*a) - exp(-I*omega*b)) / (I*omega)
    steps = _step_start_definition(f)
    steps += [
        r"\textbf{Step 2: Identify a finite-duration signal}",
        r"x(t)=u(t-" + latex(a) + r")-u(t-" + latex(b) + r")",
        r"\textbf{Step 3: Determine the nonzero interval}",
        r"x(t)=1\;\;\text{for }t\in[" + latex(a) + "," + latex(b) + r"],\;\;0\text{ otherwise}",
        r"\textbf{Step 4: Replace the full integral by the interval integral}",
        r"X(\omega)=\int_{" + latex(a) + r"}^{" + latex(b) + r"}e^{-j\omega t}\,dt",
        r"\textbf{Step 5: Evaluate the exponential integral}",
        r"\left[\frac{e^{-j\omega t}}{-j\omega}\right]_{" + latex(a) + r"}^{" + latex(b) + r"}",
        r"X(\omega)=\frac{e^{-j\omega " + latex(a) + r"}-e^{-j\omega " + latex(b) + r"}}{j\omega}",
    ]
    steps += _step_final_result(X)
    return "distribution_form", True, X, steps, "", None


def _match_finite_step_window_add(expr):
    if not isinstance(expr, Add):
        return None
    terms = list(expr.args)
    if len(terms) != 2:
        return None

    p0 = _heaviside_shift_and_coeff(terms[0])
    p1 = _heaviside_shift_and_coeff(terms[1])
    if p0 is None or p1 is None:
        return None

    c0, s0 = p0
    c1, s1 = p1
    if simplify(c0 - 1) == 0 and simplify(c1 + 1) == 0:
        return simplify(s0), simplify(s1)
    if simplify(c1 - 1) == 0 and simplify(c0 + 1) == 0:
        return simplify(s1), simplify(s0)
    return None


def _rule_polynomial_finite_step_window(f):
    if not isinstance(f, Mul):
        return None

    window = None
    others = []
    for arg in f.args:
        match = _match_finite_step_window_add(arg)
        if match is not None and window is None:
            window = (arg, match)
        else:
            others.append(arg)
    if window is None:
        return None

    window_expr, (a, b) = window
    p = simplify(Mul(*others)) if others else 1
    if not p.is_polynomial(t):
        return None

    try:
        poly = Poly(p, t)
    except Exception:
        return None

    k = -I*omega

    def _anti_monomial(n, x):
        pieces = []
        for m in range(n + 1):
            pieces.append(((-1)**m) * factorial(n) / factorial(n - m) * (x**(n - m)) / (k**(m + 1)))
        return exp(k*x) * Add(*pieces, evaluate=False)

    antiderivative_parts = []
    X_parts = []
    for (degree,), coeff in poly.terms():
        n = int(degree)
        antiderivative_parts.append(coeff * _anti_monomial(n, t))
        X_parts.append(coeff * (_anti_monomial(n, b) - _anti_monomial(n, a)))
    anti = simplify(Add(*antiderivative_parts, evaluate=False))
    X = _simplify_spectrum(Add(*X_parts, evaluate=False))
    if isinstance(X, Piecewise):
        return None

    steps = _step_start_definition(f)
    steps += [
        r"\textbf{Step 2: Identify a polynomial finite-window signal}",
        r"x(t)=p(t)\left[u(t-a)-u(t-b)\right],\quad p(t)=" + latex(p),
        r"\text{Here }a=" + latex(a) + r",\quad b=" + latex(b),
        r"\textbf{Step 3: Replace the full integral by the finite interval}",
        r"X(\omega)=\int_{" + latex(a) + r"}^{" + latex(b) + r"}p(t)e^{-j\omega t}\,dt",
        r"\textbf{Step 4: Evaluate the finite polynomial integral}",
        r"X(\omega)=\left[" + latex(anti) + r"\right]_{" + latex(a) + r"}^{" + latex(b) + r"}",
    ]
    steps += _step_final_result(X)
    return "closed_form", True, X, steps, "", None


def _rule_distributed_linearity(f):
    if not (isinstance(f, Mul) and any(isinstance(arg, Add) for arg in f.args)):
        return None

    distributed = expand(f)
    if not isinstance(distributed, Add) or distributed == f:
        return None

    terms = list(Add.make_args(distributed))
    if len(terms) < 2 or len(terms) > 12:
        return None

    X_terms = []
    term_data = []
    form = "closed_form"
    conds = []
    for term in terms:
        form_k, ok_k, Xk, _steps_k, cond_k, err_k = _derive_with_properties(term)
        if not ok_k or err_k or isinstance(Xk, Piecewise):
            return None
        latex_xk = latex(Xk)
        if any(token in latex_xk for token in ("Piecewise", "RootSum", "arg", "polar_lift", "meijerg")):
            return None
        if form_k == "distribution_form":
            form = "distribution_form"
        elif form_k == "integral_form" and form != "distribution_form":
            form = "integral_form"
        if cond_k:
            conds.append(cond_k)
        X_terms.append(Xk)
        term_data.append((term, Xk))

    X = _omega_real_cleanup(Add(*X_terms, evaluate=False))
    steps = _step_start_definition(f)
    steps += [
        r"\textbf{Step 2: Distribute multiplication over addition}",
        r"x(t)=" + _format_step_display_latex(latex(distributed)),
        r"\textbf{Step 3: Apply linearity term by term}",
        r"X(\omega)=\sum_k X_k(\omega)",
    ]
    for k, (term, Xk) in enumerate(term_data, start=1):
        steps.append(r"\textbf{Term " + str(k) + r": }x_k(t)=" + _format_step_display_latex(latex(term)))
        steps.append(r"X_" + str(k) + r"(\omega)=" + _format_result_display_latex(latex(Xk)))
    steps += _step_final_result(X)
    return form, True, X, steps, r"\;\;".join(conds), None


def _rule_abs_exponential(f):
    if getattr(f, "func", None) != exp or len(f.args) != 1:
        return None

    coeff, rest = expand(f.args[0]).as_coeff_Mul()
    if rest != Abs(t):
        return None
    if coeff.is_number:
        if coeff >= 0:
            return None
    elif coeff.is_negative is not True:
        return None

    a_val = simplify(-coeff)
    X = 2*a_val/(a_val**2 + omega**2)
    steps = _step_start_definition(f)
    steps += [
        r"\textbf{Step 2: Identify an even two-sided exponential}",
        r"x(t)=e^{-a|t|},\quad a>0,\quad a=" + latex(a_val),
        r"\textbf{Step 3: Split the integral at }t=0",
        r"X(\omega)=\int_{-\infty}^{0}e^{a t}e^{-j\omega t}\,dt+\int_{0}^{\infty}e^{-a t}e^{-j\omega t}\,dt",
        r"\textbf{Step 4: Use the standard two-sided exponential pair}",
        r"\mathcal{F}\{e^{-a|t|}\}=\frac{2a}{a^2+\omega^2},\quad a>0",
        r"\text{Substitute }a=" + latex(a_val),
    ]
    steps += _step_final_result(X)
    return "closed_form", True, X, steps, r"a>0", None


def _extract_single_function_factor(f, func):
    """Return (constant_coeff, func_call) for coeff*func(arg), if present."""
    if getattr(f, "func", None) == func:
        return 1, f
    if isinstance(f, Mul):
        matches = [arg for arg in f.args if getattr(arg, "func", None) == func]
        if len(matches) != 1:
            return None
        core = matches[0]
        coeff = simplify(f / core)
        if coeff.has(t):
            return None
        return coeff, core
    return None


def _positive_linear_scale(a):
    """Return (positive_width, sign(a)) for a real nonzero numeric/symbolic linear coefficient."""
    try:
        a = simplify(a)
        if simplify(a) == 0:
            return None
        if getattr(a, "is_positive", None) is True:
            return simplify(1/a), 1
        if getattr(a, "is_negative", None) is True:
            return simplify(-1/a), -1
        if getattr(a, "is_number", False):
            av = float(N(a))
            if av > 0:
                return simplify(1/a), 1
            if av < 0:
                return simplify(-1/a), -1
    except Exception:
        return None
    return None


def _linear_center_and_width(arg):
    """For arg=a*t+b, return (center c=-b/a, width=1/|a|, sign(a))."""
    lin = _as_linear_in_t(arg)
    if lin is None:
        return None
    a, b = lin
    scale = _positive_linear_scale(a)
    if scale is None:
        return None
    width, sign_a = scale
    center = simplify(-b/a)
    return center, width, sign_a


def _rule_sign_distribution(f):
    extracted = _extract_single_function_factor(f, sign)
    if extracted is None:
        return None
    coeff, core = extracted
    if len(core.args) != 1:
        return None
    params = _linear_center_and_width(core.args[0])
    if params is None:
        return None
    c, _width, sign_a = params
    X = coeff * sign_a * exp(-I*omega*c) * (-2*I) * PV(1/omega)
    X = _omega_real_cleanup(X)
    steps = _step_start_definition(f)
    steps += [
        r"\textbf{Step 2: Identify the shifted sign signal}",
        r"\operatorname{sign}(a(t-c))=\operatorname{sign}(a)\operatorname{sign}(t-c)",
        r"a=" + latex(_as_linear_in_t(core.args[0])[0]) + r",\quad c=" + latex(c),
        r"\textbf{Step 3: Use the sign transform pair}",
        r"\mathcal{F}\{\operatorname{sign}(t)\}=\frac{2}{j\omega}=-2j\,\mathrm{PV}\!\left(\frac{1}{\omega}\right)",
        r"\text{PV appears because }\operatorname{sign}(t)\text{ is interpreted as a distribution.}",
        r"\textbf{Step 4: Apply the time-shift and scale factors}",
        r"\mathcal{F}\{y(t-c)\}=e^{-j\omega c}Y(\omega)",
    ]
    steps += _step_final_result(X)
    return "distribution_form", True, X, steps, "", None


def _rule_rect_distribution(f):
    extracted = _extract_single_function_factor(f, Rect)
    if extracted is None:
        return None
    coeff, core = extracted
    if len(core.args) != 1:
        return None
    params = _linear_center_and_width(core.args[0])
    if params is None:
        return None
    c, width, _sign_a = params
    X = coeff * exp(-I*omega*c) * (2*sin(omega*width/2)/omega)
    X = _omega_real_cleanup(X)
    steps = _step_start_definition(f)
    steps += [
        r"\textbf{Step 2: Identify the shifted and scaled rectangular pulse}",
        r"\operatorname{rect}\!\left(\frac{t-c}{T}\right),\quad c=" + latex(c) + r",\quad T=" + latex(width),
        r"\textbf{Step 3: Use the rectangular-pulse transform pair}",
        r"\mathcal{F}\{\operatorname{rect}(t/T)\}=\frac{2\sin(\omega T/2)}{\omega}",
        r"\textbf{Step 4: Apply the time-shift property}",
        r"\mathcal{F}\{y(t-c)\}=e^{-j\omega c}Y(\omega)",
    ]
    steps += _step_final_result(X)
    return "closed_form", True, X, steps, "", None


def _rule_tri_distribution(f):
    extracted = _extract_single_function_factor(f, Tri)
    if extracted is None:
        return None
    coeff, core = extracted
    if len(core.args) != 1:
        return None
    params = _linear_center_and_width(core.args[0])
    if params is None:
        return None
    c, width, _sign_a = params
    sinc_part = sin(omega*width/2)/(omega*width/2)
    X = coeff * exp(-I*omega*c) * width * sinc_part**2
    X = _omega_real_cleanup(X)
    steps = _step_start_definition(f)
    steps += [
        r"\textbf{Step 2: Identify the shifted and scaled triangular pulse}",
        r"\operatorname{tri}\!\left(\frac{t-c}{T}\right),\quad c=" + latex(c) + r",\quad T=" + latex(width),
        r"\textbf{Step 3: Use the triangular-pulse transform pair}",
        r"\mathcal{F}\{\operatorname{tri}(t/T)\}=T\left(\frac{\sin(\omega T/2)}{\omega T/2}\right)^2",
        r"\textbf{Step 4: Apply the time-shift property}",
        r"\mathcal{F}\{y(t-c)\}=e^{-j\omega c}Y(\omega)",
    ]
    steps += _step_final_result(X)
    return "closed_form", True, X, steps, "", None


def _param_positive_condition(alpha):
    try:
        alpha = simplify(alpha)
        if getattr(alpha, "is_positive", None) is True:
            return ""
        if getattr(alpha, "is_number", False) and float(N(alpha)) > 0:
            return ""
    except Exception:
        pass
    return r"a>0"


def _rule_pv_second_order(f):
    """Distribution rule for PV 1/(a*t+b)^2."""
    try:
        num, den = f.as_numer_denom()
        if simplify(num - 1) != 0:
            return None
        if not (getattr(den, "is_Pow", False) and simplify(den.exp - 2) == 0):
            return None
        lin = _as_linear_in_t(den.base)
        if lin is None:
            return None
        a, b = lin
        if simplify(a) == 0:
            return None
        shift = simplify(b / a)
        X = _simplify_spectrum(-(pi / (a**2)) * Abs(omega) * exp(I*omega*shift))
        X = _omega_real_cleanup(X)
        steps = _step_start_definition(f)
        steps += [
            r"\textbf{Step 2: Interpret the reciprocal square as a principal-value distribution}",
            r"\mathcal{F}\left\{\mathrm{PV}\!\left(\frac{1}{t^2}\right)\right\}=-\pi|\omega|",
            r"\text{PV is required because the time-domain expression has a second-order singularity.}",
            r"\textbf{Step 3: Rewrite the denominator into shifted and scaled form}",
            r"a t+b=a\left(t+\frac{b}{a}\right),\quad a=" + latex(a) + r",\quad b=" + latex(b),
            r"\textbf{Step 4: Apply scaling, shift, and linearity}",
            r"X(\omega)=-\frac{\pi}{a^2}e^{j\omega b/a}|\omega|",
        ]
        steps += _step_final_result(X)
        return ("distribution_form", True, X, steps, "", None)
    except Exception:
        return None


def _rule_sinc_family(f):
    """Known sinc-window pairs, displayed with a frequency-domain rect."""
    try:
        num, den = f.as_numer_denom()
        if getattr(num, "func", None) != sin or len(num.args) != 1:
            return None
        lin = _as_linear_in_t(num.args[0])
        if lin is None:
            return None
        alpha, phase = lin
        if simplify(phase) != 0 or simplify(alpha) == 0:
            return None
        if simplify(den - pi*t) == 0:
            scale = 1
            pair_latex = r"\mathcal{F}\left\{\frac{\sin(a t)}{\pi t}\right\}=\operatorname{rect}\left(\frac{\omega}{2a}\right),\quad a>0"
        elif simplify(den - t) == 0:
            scale = pi
            pair_latex = r"\mathcal{F}\left\{\frac{\sin(a t)}{t}\right\}=\pi\operatorname{rect}\left(\frac{\omega}{2a}\right),\quad a>0"
        else:
            return None
        X = _simplify_spectrum(scale * Rect(omega / (2*alpha)))
        steps = _step_start_definition(f)
        steps += [
            r"\textbf{Step 2: Identify the sinc-type signal}",
            r"x(t)=C\frac{\sin(a t)}{t},\quad a=" + latex(alpha),
            r"\textbf{Step 3: Use the sinc-rect transform pair}",
            pair_latex,
            r"\textbf{Step 4: Substitute the parameter}",
        ]
        steps += _step_final_result(X)
        return ("closed_form", True, X, steps, _param_positive_condition(alpha), None)
    except Exception:
        return None


def _rule_gaussian_family(f):
    """Known Gaussian pair exp(-a*t^2), a>0."""
    try:
        if getattr(f, "func", None) != exp or len(f.args) != 1:
            return None
        arg = expand(f.args[0])
        poly = Poly(arg, t)
        if poly.degree() != 2:
            return None
        c2 = simplify(poly.coeff_monomial(t**2))
        c1 = simplify(poly.coeff_monomial(t))
        c0 = simplify(poly.coeff_monomial(1))
        if simplify(c1) != 0 or simplify(c0) != 0:
            return None
        alpha = simplify(-c2)
        if simplify(alpha) == 0:
            return None
        X = _simplify_spectrum(sqrt(pi/alpha) * exp(-omega**2/(4*alpha)))
        steps = _step_start_definition(f)
        steps += [
            r"\textbf{Step 2: Identify the Gaussian form}",
            r"x(t)=e^{-a t^2},\quad a=" + latex(alpha) + r",\quad a>0",
            r"\textbf{Step 3: Use the Gaussian transform pair}",
            r"\mathcal{F}\{e^{-a t^2}\}=\sqrt{\frac{\pi}{a}}e^{-\omega^2/(4a)},\quad a>0",
            r"\textbf{Step 4: Substitute the parameter}",
        ]
        steps += _step_final_result(X)
        return ("closed_form", True, X, steps, _param_positive_condition(alpha), None)
    except Exception:
        return None


def _rule_two_sided_exponential_parameter(f):
    """Known pair exp(-a*abs(t)), a>0."""
    try:
        if getattr(f, "func", None) != exp or len(f.args) != 1:
            return None
        arg = f.args[0]
        if not arg.has(Abs(t)):
            return None
        alpha = simplify(-arg / Abs(t))
        if alpha.has(t) or simplify(alpha) == 0:
            return None
        X = _simplify_spectrum(2*alpha/(alpha**2 + omega**2))
        steps = _step_start_definition(f)
        steps += [
            r"\textbf{Step 2: Identify the two-sided exponential form}",
            r"x(t)=e^{-a|t|},\quad a=" + latex(alpha) + r",\quad a>0",
            r"\textbf{Step 3: Use the two-sided exponential transform pair}",
            r"\mathcal{F}\{e^{-a|t|}\}=\frac{2a}{a^2+\omega^2},\quad a>0",
            r"\textbf{Step 4: Substitute the parameter}",
        ]
        steps += _step_final_result(X)
        return ("closed_form", True, X, steps, _param_positive_condition(alpha), None)
    except Exception:
        return None


def _iter_subexpressions(expr):
    yield expr
    for arg in getattr(expr, "args", ()):
        yield from _iter_subexpressions(arg)


def _shift_candidates(expr):
    candidates = []
    seen = set()
    for sub in _iter_subexpressions(expr):
        lin = _as_linear_in_t(sub)
        if lin is None:
            continue
        a, b = lin
        if simplify(a) == 0 or simplify(b) == 0:
            continue
        try:
            c = simplify(-b/a)
        except Exception:
            continue
        key = str(c)
        if key not in seen:
            seen.add(key)
            candidates.append(c)
    return candidates


def _shift_score(expr):
    return len(_shift_candidates(expr))


def _rule_generic_time_shift(f):
    """Controlled property wrapper: x(t)=g(t-c) -> e^{-j*w*c}G(w)."""
    try:
        original_score = _shift_score(f)
        if original_score == 0:
            return None
        for c in _shift_candidates(f):
            base = simplify(f.subs(t, t + c))
            if base == f or _shift_score(base) >= original_score:
                continue
            form, ok, G, _, conditions, error = _derive_with_properties(base)
            if not ok or form not in ("closed_form", "distribution_form"):
                continue
            if G.has(DiracDelta):
                continue
            X = _omega_real_cleanup(exp(-I*omega*c) * G)
            steps = _step_start_definition(f)
            steps += [
                r"\textbf{Step 2: Identify a time shift}",
                r"x(t)=g(t-c),\quad c=" + latex(c),
                r"g(t)=" + _format_step_display_latex(latex(base)),
                r"\textbf{Step 3: Use the known or previously derived transform of }g(t)",
                r"G(\omega)=" + latex(G),
                r"\textbf{Step 4: Apply the time-shift property}",
                r"\mathcal{F}\{g(t-c)\}=e^{-j\omega c}G(\omega)",
            ]
            steps += _step_final_result(X)
            return (form, True, X, steps, conditions, error)
    except Exception:
        return None
    return None


def _extract_generic_modulation(f):
    try:
        if getattr(f, "func", None) == exp and len(f.args) == 1:
            arg = f.args[0]
            real_part = simplify(re(arg))
            imag_part = simplify(im(arg))
            lin = _as_linear_in_t(imag_part)
            if lin is not None:
                w0, phi = lin
                if simplify(w0) != 0:
                    base = simplify(exp(real_part))
                    if simplify(base - 1) != 0:
                        return base, simplify(w0), simplify(phi)

        if not isinstance(f, Mul):
            return None
        for factor in f.args:
            if getattr(factor, "func", None) != exp or len(factor.args) != 1:
                continue
            phase = simplify(factor.args[0] / I)
            if phase.has(I):
                continue
            lin = _as_linear_in_t(phase)
            if lin is None:
                continue
            w0, phi = lin
            if simplify(w0) == 0:
                continue
            base = simplify(f / factor)
            if simplify(base - 1) == 0:
                continue
            return base, simplify(w0), simplify(phi)
    except Exception:
        return None
    return None


def _rule_generic_modulation(f):
    """Controlled property wrapper: e^{j*w0*t+j*phi}g(t) -> e^{j*phi}G(w-w0)."""
    try:
        extracted = _extract_generic_modulation(f)
        if extracted is None:
            return None
        base, w0, phi = extracted
        form, ok, G, _, conditions, error = _derive_with_properties(base)
        if not ok or form not in ("closed_form", "distribution_form"):
            return None
        shifted = G.subs(omega, omega - w0)
        if simplify(phi) == 0:
            X = _omega_real_cleanup(shifted)
        else:
            X = _omega_real_cleanup(Mul(exp(I*phi), shifted, evaluate=False))
        steps = _step_start_definition(f)
        steps += [
            r"\textbf{Step 2: Identify a modulation factor}",
            r"x(t)=e^{j(\omega_0 t+\phi)}g(t),\quad \omega_0=" + latex(w0) + r",\quad \phi=" + latex(phi),
            r"g(t)=" + _format_step_display_latex(latex(base)),
            r"\textbf{Step 3: Use the known or previously derived transform of }g(t)",
            r"G(\omega)=" + latex(G),
            r"\textbf{Step 4: Apply the modulation property}",
            r"\mathcal{F}\{e^{j\omega_0 t}g(t)\}=G(\omega-\omega_0)",
            r"\text{The constant phase }e^{j\phi}\text{ is kept as a multiplier.}",
        ]
        steps += _step_final_result(X)
        return (form, True, X, steps, conditions, error)
    except Exception:
        return None


def _rule_time_multiply_closed_form(f):
    """Controlled property wrapper: t*g(t) -> j*dG/dw for non-distribution spectra."""
    try:
        if not isinstance(f, Mul) or not f.has(t):
            return None
        coeff, rest = f.as_independent(t, as_Add=False)
        args = list(Mul.make_args(rest))
        if t not in args:
            return None
        args.remove(t)
        base = simplify(Mul(*args) if args else 1)
        if base == 1:
            return None
        form, ok, G, _, conditions, error = _derive_with_properties(base)
        if not ok or form != "closed_form":
            return None
        if G.has(DiracDelta) or G.has(PV) or G.has(Rect) or G.has(Tri):
            return None
        X = _omega_real_cleanup(simplify(coeff * I * diff(G, omega)))
        steps = _step_start_definition(f)
        steps += [
            r"\textbf{Step 2: Identify multiplication by }t",
            r"x(t)=" + latex(coeff) + r"\,t\,g(t),\quad g(t)=" + _format_step_display_latex(latex(base)),
            r"\textbf{Step 3: Use the known or previously derived transform of }g(t)",
            r"G(\omega)=" + latex(G),
            r"\textbf{Step 4: Apply frequency differentiation}",
            r"\mathcal{F}\{t g(t)\}=j\frac{d}{d\omega}G(\omega)",
        ]
        steps += _step_final_result(X)
        return ("closed_form", True, X, steps, conditions, error)
    except Exception:
        return None


# ---------- main derivation ----------

def _attempt_rule(rule, f):
    paths = _METHOD_PATH.get()
    before = set(paths) if paths is not None else None
    result = rule(f)
    if result is None and paths is not None:
        paths.clear()
        paths.update(before)
    return result


def _derive_with_properties(f):
    """
    Returns: (form, ok, X_expr, steps_latex, conditions_latex, error_or_None)
    form in {"closed_form", "integral_form", "distribution_form", "divergent"}
    """

    # 0) Trig-first policy (user request)

    sign_res = _attempt_rule(_rule_sign_distribution, f)
    if sign_res is not None:
        _record_method("known_pair")
        return sign_res

    rect_res = _attempt_rule(_rule_rect_distribution, f)
    if rect_res is not None:
        _record_method("known_pair")
        return rect_res

    tri_res = _attempt_rule(_rule_tri_distribution, f)
    if tri_res is not None:
        _record_method("known_pair")
        return tri_res

    pv2_res = _attempt_rule(_rule_pv_second_order, f)
    if pv2_res is not None:
        _record_method("known_pair")
        return pv2_res

    sinc_res = _attempt_rule(_rule_sinc_family, f)
    if sinc_res is not None:
        _record_method("known_pair")
        return sinc_res

    gaussian_res = _attempt_rule(_rule_gaussian_family, f)
    if gaussian_res is not None:
        _record_method("known_pair")
        return gaussian_res

    two_sided_exp_res = _attempt_rule(_rule_two_sided_exponential_parameter, f)
    if two_sided_exp_res is not None:
        _record_method("known_pair")
        return two_sided_exp_res

    finite_window_res = _attempt_rule(_rule_finite_step_window, f)
    if finite_window_res is not None:
        _record_method("known_pair")
        return finite_window_res

    poly_window_res = _attempt_rule(_rule_polynomial_finite_step_window, f)
    if poly_window_res is not None:
        _record_method("known_pair")
        return poly_window_res

    shifted_step_res = _attempt_rule(_rule_shifted_heaviside_distribution, f)
    if shifted_step_res is not None:
        _record_method("known_pair")
        return shifted_step_res

    affine_step = _attempt_rule(_rule_affine_step, f)
    if affine_step is not None:
        return affine_step

    modulated_step_res = _attempt_rule(_rule_modulated_step, f)
    if modulated_step_res is not None:
        _record_method("known_pair")
        return modulated_step_res

    abs_exp_res = _attempt_rule(_rule_abs_exponential, f)
    if abs_exp_res is not None:
        _record_method("known_pair")
        return abs_exp_res

    shifted_poly_step_res = _attempt_rule(_rule_shifted_poly_times_step_distribution, f)
    if shifted_poly_step_res is not None:
        _record_method("known_pair")
        return shifted_poly_step_res

    poly_step_res = _attempt_rule(_rule_poly_times_step_distribution, f)

    pv_res = _attempt_rule(_rule_pv_reciprocal, f)
    if pv_res is not None:
        _record_method("known_pair")
        return pv_res

    poly_res = _attempt_rule(_rule_poly_distribution, f)
    if poly_res is not None:
        _record_method("known_pair")
        return poly_res

    if poly_step_res is not None:
        _record_method("known_pair")
        return poly_step_res

    trig_step_res = _attempt_rule(_rule_trig_times_step_distribution, f)
    if trig_step_res is not None:
        _record_method("known_pair")
        return trig_step_res

    trig_res = _attempt_rule(_try_trig_as_exp_distribution, f)
    if trig_res is not None:
        _record_method("known_pair")
        return trig_res


    # --- Known pair: 1/(t^2 + c), c>0 (force omega real; avoid half-branch Piecewise) ---
    # Covers 1/(t^2+1), 1/(t^2+6), 1/(t^2+a^2) (interpreted as c=a^2).
    try:
        num, den = fraction(together(f))
        if simplify(num - 1) == 0:
            P = Poly(den, t)
            if P.degree() == 2:
                a2, a1, a0 = P.all_coeffs()  # a2*t^2 + a1*t + a0
                if simplify(a2 - 1) == 0 and simplify(a1) == 0:
                    c = simplify(a0)
                    # Only apply the integrable case c>0 (e.g. c=1,6,a^2). Otherwise defer to PV rules.
                    c_pos = bool(getattr(c, "is_positive", None))
                    if c_pos or (c.is_Number and float(c) > 0) or (c.is_Pow and c.exp == 2):
                        alpha = simplify(sqrt(c))
                        X = _simplify_spectrum(pi/alpha * exp(-alpha*Abs(omega)))
                        X = _omega_real_cleanup(X)
                        steps = _step_start_definition(f)
                        steps += [
                            r"\textbf{Step 2: Identify a standard rational transform pair}",
                            rf"x(t)=\frac{{1}}{{t^{{2}}+{latex(c)}}},\quad {latex(c)}>0",
                            r"\textbf{Step 3: Rewrite the denominator as }t^2+\alpha^2",
                            rf"\alpha=\sqrt{{{latex(c)}}}={latex(alpha)}",
                            r"\textbf{Step 4: Use the known transform pair}",
                            r"\mathcal{F}\left\{\frac{1}{t^2+\alpha^2}\right\}=\frac{\pi}{\alpha}e^{-\alpha|\omega|},\quad \alpha>0",
                        ]
                        steps += _step_final_result(X)
                        return "distribution_form", True, X, steps, (latex(c)+">0" if c.is_positive is not True else ""), None
    except Exception:
        pass

    quadlin_res = _attempt_rule(_rule_linear_over_t2_plus_c, f)
    if quadlin_res is not None:
        _record_method("known_pair")
        return quadlin_res

    rat_res = _attempt_rule(_rule_rational_apart_linearity, f)
    if rat_res is not None:
        _record_method("property_rule")
        return rat_res

    distributed_res = _attempt_rule(_rule_distributed_linearity, f)
    if distributed_res is not None:
        _record_method("property_rule")
        return distributed_res



    # 0.5) Constant (distribution)
    # x(t)=C  -> X(omega)=2*pi*C*delta(omega)
    if f.free_symbols.isdisjoint({t}):
        X = _simplify_spectrum(2*pi*f*DiracDelta(omega))
        steps = _step_start_definition(f)
        steps += [
            r"\textbf{Step 2: Use the constant transform pair}",
            r"\mathcal{F}\{C\}=2\pi C\,\delta(\omega)",
            r"\text{A constant signal is not absolutely integrable, so the result is interpreted as a distribution.}",
        ]
        steps += _step_final_result(X)
        return "distribution_form", True, X, steps, "", None

    impulse = _attempt_rule(_rule_affine_delta, f)
    if impulse is not None:
        return impulse

    decay = _attempt_rule(_rule_one_sided_decay, f)
    if decay is not None:
        return decay

    # 0.9) exp(-a*Abs(t)) (common integrable)
    if f.func == exp and len(f.args)==1 and f.args[0].has(Abs(t)):
        arg = f.args[0]
        # match -a*Abs(t)
        a_sym = symbols('a_sym', real=True, positive=True)
        m = arg.match(-a_sym*Abs(t))
        if m and a_sym in m:
            a_val = m[a_sym]
            X = _simplify_spectrum(2*a_val/(a_val**2 + omega**2))
            steps = [
                r"x(t)=e^{-a|t|}\ (a>0)",
                r"X(\omega)=\int_{-\infty}^{\infty}e^{-a|t|}e^{-j\omega t}dt=\frac{2a}{a^2+\omega^2}",
                r"X(\omega)=" + latex(X),
                ]
            return "closed_form", True, X, steps, r"a>0", None

    # 0.10) Pure tone exp(I*w0*t + I*phi)
    if f.func == exp and len(f.args)==1:
        inside = simplify(f.args[0]/I)
        if not inside.has(I):
            lin = _as_linear_in_t(inside)
            if lin is not None:
                w0, phi = lin
                X = _simplify_spectrum(2*pi*exp(I*phi)*DiracDelta(omega - w0))
                steps = _step_start_definition(f)
                steps += [
                    r"\textbf{Step 2: Separate the constant phase and the pure tone}",
                    r"x(t)=e^{j\phi}e^{j\omega_0 t},\quad \omega_0=" + latex(w0) + r",\quad \phi=" + latex(phi),
                    r"\textbf{Step 3: Use the pure-tone transform pair}",
                    r"\mathcal{F}\{e^{j\omega_0 t}\}=2\pi\delta(\omega-\omega_0)",
                    r"\textbf{Step 4: Apply the phase constant by linearity}",
                    r"X(\omega)=2\pi e^{j\phi}\delta(\omega-\omega_0)",
                ]
                steps += _step_final_result(X)
                return "distribution_form", True, X, steps, "", None

    # 0.11) t^n * exp(I*w0*t) (frequency differentiation)
    if isinstance(f, Mul):
        # look for exp(I*w0*t) factor and t**n
        exp_factor = None
        tpow = None
        other = []
        for a in f.args:
            if a.func == exp and len(a.args)==1:
                inside = simplify(a.args[0]/I)
                if not inside.has(I):
                    lin = _as_linear_in_t(inside)
                    if lin is not None:
                        w0, phi = lin
                        if phi == 0:
                            exp_factor = w0
                            continue
            if a.is_Pow and a.base == t and a.exp.is_integer and int(a.exp) >= 1:
                tpow = int(a.exp)
                continue
            if a == t:
                tpow = 1
                continue
            other.append(a)
        if exp_factor is not None and tpow is not None:
            coeff = simplify(Mul(*other)) if other else 1
            X = _simplify_spectrum(coeff * 2*pi * (I**tpow) * diff(DiracDelta(omega-exp_factor), omega, tpow))
            steps = [
                r"x(t)=t^n e^{j\omega_0 t}",
                r"\mathcal{F}\{e^{j\omega_0 t}\}=2\pi\delta(\omega-\omega_0)",
                r"\mathcal{F}\{t\,x(t)\}=j\,\frac{d}{d\omega}X(\omega)\Rightarrow\mathcal{F}\{t^n x(t)\}=j^n\frac{d^n}{d\omega^n}X(\omega)",
                r"\Rightarrow\;X(\omega)=2\pi j^n\,\delta^{(n)}(\omega-\omega_0)",
                r"X(\omega)=" + latex(X),
                ]
            return "distribution_form", True, X, steps, "", None

    generic_mod_res = _attempt_rule(_rule_generic_modulation, f)
    if generic_mod_res is not None:
        _record_method("property_rule")
        return generic_mod_res

    generic_shift_res = _attempt_rule(_rule_generic_time_shift, f)
    if generic_shift_res is not None:
        _record_method("property_rule")
        return generic_shift_res

    time_multiply_res = _attempt_rule(_rule_time_multiply_closed_form, f)
    if time_multiply_res is not None:
        _record_method("property_rule")
        return time_multiply_res


    # 1) DiracDelta basics
    if f == DiracDelta(t):
        X = 1
        steps = _step_start_definition(f)
        steps += [
            r"\textbf{Step 2: Apply the sifting property}",
            r"X(\omega)=\int_{-\infty}^{\infty}\delta(t)e^{-j\omega t}\,dt=e^{-j\omega\cdot 0}=1",
        ]
        steps += _step_final_result(X)
        return "distribution_form", True, X, steps, "", None


    # 2) PV rational distributions: 1/(t+a), 1/(t+a)^2
    # Allow an overall constant factor c: F{c*g(t)} = c*G(omega)
    c0 = 1
    f0 = f
    if isinstance(f, Mul):
        const_args = []
        t_args = []
        for _a in f.args:
            if _a.has(t):
                t_args.append(_a)
            else:
                const_args.append(_a)
        if const_args:
            c0 = Mul(*const_args)
            f0 = Mul(*t_args) if t_args else 1
    a1 = _match_shifted_power(f0, 1)
    if a1 is not None:
        X = _simplify_spectrum(c0 * (-I*pi*sign(omega) * exp(I*omega*a1)))
        steps = [
            r"\text{(Distribution)}\;\mathcal{F}\{\mathrm{PV}\tfrac{1}{t}\}=-i\pi\,\mathrm{sign}(\omega)",
            r"\text{Time shift: }\mathcal{F}\{g(t-t_0)\}=e^{-i\omega t_0}G(\omega)",
        ]
        steps.append(r"\text{For this term: }a=" + latex(a1) + r",\;t_0=-a=" + latex(-a1) + r",\;e^{-i\omega t_0}=e^{i\omega a}")

        if c0 != 1:
            steps.append(r"\text{Linearity: }\mathcal{F}\{c\,g(t)\}=c\,G(\omega),\; c=" + latex(c0))
        steps += [
            r"\Rightarrow\;X(\omega)=c\,e^{i\omega a}\left(-i\pi\,\mathrm{sign}(\omega)\right)",
            r"X(\omega)=" + latex(X),
            ]
        return "distribution_form", True, X, steps, "", None

    a2 = _match_shifted_power(f0, 2)
    if a2 is not None:
        X = _simplify_spectrum(c0 * (-pi*omega*sign(omega) * exp(I*omega*a2)))
        steps = [
            r"\frac{d}{dt}\left(\frac{1}{t+a}\right)=-\frac{1}{(t+a)^2}",
            r"\mathcal{F}\left\{\frac{d}{dt}g(t)\right\}=j\omega\,G(\omega)",
            r"\Rightarrow\;\mathcal{F}\left\{\frac{1}{(t+a)^2}\right\}=-j\omega\,\mathcal{F}\left\{\frac{1}{t+a}\right\}",
            r"X(\omega)=" + latex(X),
            ]
        return "distribution_form", True, X, steps, "", None

    # 3) One-sided: g(t)*Heaviside(t)
    if isinstance(f, Mul) and Heaviside(t) in f.args:
        h = Heaviside(t)
        g = simplify(f / h)
        X_def, _, steps = _fourier_one_sided_steps(g)
        ok, X = _doit_or_keep_integral(X_def)
        steps.append(r"X(\omega)=" + latex(X))
        if isinstance(X, Piecewise):
            return "closed_form", True, X, steps, _piecewise_conditions_latex(X), None
        if ok:
            return "closed_form", True, X, steps, r"\text{One-sided integral (via }u(t)\text{).}", None
        return "integral_form", True, X, steps, r"\text{One-sided integral returned (symbolic).}", "Closed-form not found; returned one-sided integral."

    # 4) Linearity / homogeneity
    if isinstance(f, Add):
        _record_method("linearity")
        terms = list(f.args)

        # --- Special teaching-step formatting: finite window u(t-a) - u(t-b) ---
        def _heaviside_shift_and_coeff(expr):
            # returns (coeff, shift) where expr = coeff*Heaviside(t-shift), with a=1 in arg
            coeff = 1
            h = None
            if expr.func == Heaviside and len(expr.args) == 1:
                h = expr
            elif isinstance(expr, Mul):
                c, r = expr.as_independent(t, as_Add=False)
                if c != 1 and (r.func == Heaviside and len(r.args) == 1):
                    coeff = c
                    h = r
            if h is None:
                return None
            lin = _as_linear_in_t(h.args[0])
            if lin is None:
                return None
            a1, b1 = lin  # a1*t + b1
            if a1 != 1:
                return None
            shift = -b1
            return (simplify(coeff), simplify(shift))

        win = None
        if len(terms) == 2:
            p0 = _heaviside_shift_and_coeff(terms[0])
            p1 = _heaviside_shift_and_coeff(terms[1])
            if p0 is not None and p1 is not None:
                c0, s0 = p0
                c1, s1 = p1
                # Match u(t-a) - u(t-b) (coeffs +1 and -1)
                if simplify(c0 - 1) == 0 and simplify(c1 + 1) == 0:
                    win = (s0, s1)
                elif simplify(c1 - 1) == 0 and simplify(c0 + 1) == 0:
                    win = (s1, s0)

        # Compute each term (computation unchanged)
        X_sum = 0
        conds = []
        ok_all = True
        form = "closed_form"
        term_X = []
        for term in terms:
            formk, okk, Xk, _sk, ck, _errk = _derive_with_properties(term)
            ok_all = ok_all and okk
            if formk == "distribution_form":
                form = "distribution_form"
            elif formk == "integral_form" and form != "distribution_form":
                form = "integral_form"
            if ck:
                conds.append(ck)
            term_X.append((term, Xk))
            X_sum += Xk
        X_sum = _simplify_spectrum(X_sum)

        if win is not None:
            a, b = win
            steps = [
                r"\textbf{Step 1: Identify a finite-duration signal}",
                r"x(t)=" + latex(f),
                r"\textbf{Step 2: Determine the nonzero interval}",
                r"x(t)=1\;\;\text{for }t\in[" + latex(a) + "," + latex(b) + r"],\;\;0\text{ otherwise}",
                r"\textbf{Step 3: Write the Fourier transform integral}",
                r"X(\omega)=\int_{" + latex(a) + r"}^{" + latex(b) + r"} e^{-j\omega t}\,dt",
                r"\textbf{Step 4: Evaluate the integral}",
                r"X(\omega)=\frac{e^{-j\omega " + latex(a) + r"}-e^{-j\omega " + latex(b) + r"}}{j\omega}\quad(\text{with distributional interpretation at }\omega=0)",
                r"\textbf{Final Result}",
                r"X(\omega)=" + latex(X_sum),
                ]
            return form, ok_all, X_sum, steps, r"\;\;".join(conds), None if ok_all else "Some terms not closed-form."

        # Generic linearity: concise textbook steps (no internal rule logs)
        steps = [
            r"\textbf{Step 1: Use linearity}",
            r"X(\omega)=\mathcal{F}\{\sum_k x_k(t)\}=\sum_k X_k(\omega)",
        ]
        for k, (term, Xk) in enumerate(term_X, start=1):
            steps.append(r"\textbf{Term " + str(k) + r": }x_k(t)=" + latex(term))
            steps.append(r"X_k(\omega)=" + latex(Xk))
        steps.append(r"\textbf{Final Result}")
        steps.append(r"X(\omega)=" + latex(X_sum))

        return form, ok_all, X_sum, steps, r"\;\;".join(conds), None if ok_all else "Some terms not closed-form."

    # Homogeneity: constant factor
    if isinstance(f, Mul):
        coeff, rest = f.as_independent(t, as_Add=False)
        if coeff != 1:
            _record_method("linearity")
            formG, okG, G, G_steps, condG, errG = _derive_with_properties(rest)
            X = _simplify_spectrum(coeff * G, doit=False)
            steps = [
                r"\text{Use homogeneity: }\mathcal{F}\{C\,g(t)\}=C\,G(\omega)",
                r"x(t)=" + latex(f),
                ]
            steps.extend(G_steps)
            steps.append(r"\Rightarrow\;X(\omega)=" + latex(X))
            return formG, okG, X, steps, condG, errG

    # 5) Fallback: engineering definition integral
    X_def, _, steps = _fourier_def_integral_steps(f)
    ok, X = _doit_or_keep_integral(X_def)
    steps.append(r"X(\omega)=" + latex(X))

    if isinstance(X, Piecewise):
        return "closed_form", True, X, steps, _piecewise_conditions_latex(X), None
    if ok:
        return "closed_form", True, X, steps, "", None

    steps.append(r"\text{Closed-form not found; returned the engineering-definition integral.}")
    return "integral_form", True, X, steps, "", "Closed-form not found; returned integral form."


_METHOD_PATH = contextvars.ContextVar('method_path', default=None)


def _record_method(method):
    paths = _METHOD_PATH.get()
    if paths is not None:
        paths.add(method)


def _is_heaviside(expr):
    """Recognise a single Heaviside, including SymPy's explicit value at zero."""
    return getattr(expr, 'func', None) == Heaviside and len(expr.args) in (1, 2)


def _extract_delta_shift(f):
    """Return the location of a single affine impulse; amplitudes remain separate."""
    pair = _extract_single_function_factor(f, DiracDelta)
    if pair is None:
        return None
    _, core = pair
    lin = _as_linear_in_t(core.args[0])
    if lin is None or lin[0].is_zero is True:
        return None
    return simplify(-lin[1] / lin[0])


def _affine_real_conditions(a, b):
    if a.is_real is False or b.is_real is False or a.is_zero is True:
        return None
    conditions = []
    if a.is_real is not True: conditions.append(latex(a) + r'\in\mathbb{R}')
    if b.is_real is not True: conditions.append(latex(b) + r'\in\mathbb{R}')
    if a.is_nonzero is not True: conditions.append(latex(a) + r'\ne0')
    return r',\quad '.join(conditions)


def _rule_affine_delta(f):
    pair = _extract_single_function_factor(f, DiracDelta)
    center = _extract_delta_shift(f)
    if pair is None or center is None:
        return None
    coeff, core = pair
    a, b = _as_linear_in_t(core.args[0])
    conditions = _affine_real_conditions(a, b)
    if conditions is None: return None
    n = core.args[1] if len(core.args) > 1 else S.Zero
    if n.is_Integer is not True or n < 0: return None
    X = _simplify_spectrum(coeff * (I*omega)**n * exp(-I*omega*center) / (Abs(a)*a**n))
    steps = [
        r'\textbf{Step 1: Identify the impulse location}',
        r't_0=' + latex(center) + r',\quad a=' + latex(a) + r',\quad C=' + latex(coeff),
        r'\textbf{Step 2: Use the sifting property of the delta function}',
        r'\mathcal{F}\{\delta(t-t_0)\}=e^{-j\omega t_0}',
        r'\delta^{(n)}(a(t-t_0))=\frac{\delta^{(n)}(t-t_0)}{|a|a^n}',
        r'\mathcal{F}\{\delta^{(n)}(t-t_0)\}=(j\omega)^n e^{-j\omega t_0},\quad n=' + latex(n),
    ] + _step_final_result(X)
    _record_method('known_pair')
    return 'distribution_form', True, X, steps, conditions, None


def _decay_parts(f):
    factors = list(f.atoms(Heaviside))
    if len(factors) != 1: return None
    h = factors[0]
    if not _is_heaviside(h): return None
    lin = _as_linear_in_t(h.args[0])
    if lin is None: return None
    a, b = lin
    # Unknown step orientation is not silently assumed positive.
    if a.is_positive is True: direction = S.One
    elif a.is_negative is True: direction = -S.One
    else: return None
    center = simplify(-b/a)
    if center.is_real is False: return None
    rest = simplify(f/h)
    coeff, core = rest.as_independent(t, as_Add=False)
    if core.func != exp: return None
    exponent = _as_linear_in_t(core.args[0])
    if exponent is None: return None
    slope, offset = exponent
    return coeff, slope, offset, center, direction


def _has_exp_decay_and_step(f):
    """Recognise a one-sided affine exponential with a possible decay condition."""
    parts = _decay_parts(f)
    if parts is None: return False
    rate = -parts[1]*parts[4]
    return re(rate).is_positive is not False


def _extract_decay_rate(f):
    """Decay rate in the outward coordinate s>=0 at the step edge."""
    parts = _decay_parts(f)
    return simplify(-parts[1]*parts[4]) if parts else None


def _rule_one_sided_decay(f):
    parts = _decay_parts(f)
    if parts is None: return None
    coeff, slope, offset, center, direction = parts
    if not _has_exp_decay_and_step(f):
        if re(-slope*direction).is_negative is True:
            return 'divergent', False, Integral(f*exp(-I*omega*t), (t, -oo, oo)), [], '', 'The signal grows on its unbounded support; the Fourier integral does not converge.'
        return None
    rate = _extract_decay_rate(f)
    condition = '' if re(rate).is_positive is True else latex(re(rate)) + '>0'
    X = _simplify_spectrum(coeff*exp(slope*center+offset-I*omega*center)/(rate+direction*I*omega))
    steps = _step_start_definition(f) + [
        r'\textbf{Step 2: Use the unit step to set the integration range}',
        r'a=' + latex(slope) + r',\quad b=' + latex(offset),
        r't=c+d s,\quad s\ge0,\quad c=' + latex(center) + r',\quad d=' + latex(direction),
        r'\textbf{Step 3: Combine exponential terms}',
        r'X(\omega)=C e^{ac+b-j\omega c}\int_0^\infty e^{-(r+jd\omega)s}\,ds,\quad r=-ad,\quad\Re(r)>0',
        r'\textbf{Step 4: Evaluate the convergent one-sided exponential integral}',
        r'X(\omega)=\frac{C e^{ac+b-j\omega c}}{r+jd\omega}',
    ] + _step_final_result(X)
    _record_method('known_pair')
    return 'closed_form', True, X, steps, condition, None


def _bad_display(text):
    # Match forbidden function names, not ordinary words such as "argument".
    return bool(_regex.search(r'Piecewise|RootSum|polar_lift|meijerg|_rule_|matcher|debug|srepr|\\begin\{cases\}|(?<![A-Za-z])arg(?![A-Za-z])', text or ''))


def _bad_expression(X):
    return any(getattr(node.func, '__name__', '') in
               {'Piecewise', 'RootSum', 'arg', 'polar_lift', 'meijerg'}
               for node in _iter_subexpressions(sympify(X)))


def _finalize_response(f, result, method, *, input_latex=None, definition=None):
    """Single output boundary for normal, convolution and integration paths.

    Reject whole unsuitable candidates, retaining the original defining integral.
    Never strip mathematical clauses to turn a rejected result into a success.
    """
    form, ok, X, steps, conditions, error = result
    X = sympify(X)
    definition = definition if definition is not None else Integral(f*exp(-I*omega*t), (t, -oo, oo))
    unsafe = _bad_expression(X) or any(_bad_display(str(x)) for x in [*steps, conditions])
    nonfinite = not X.has(Integral) and X.has(S.NaN, S.ComplexInfinity, oo, -oo)
    accepted = ok and not error and form in {'closed_form', 'distribution_form'} and not X.has(Integral) and not unsafe and not nonfinite
    if accepted:
        result_text = _format_result_display_latex(_format_pv_reciprocal_result_latex(f) or latex(X))
        teaching = _teaching_steps(f, X, steps)
        if any(_bad_display(x) for x in [result_text, *teaching, conditions or '']):
            accepted = False
    if not accepted:
        form = 'divergent' if form == 'divergent' or nonfinite else 'integral_form'
        X = definition
        result_text = _format_result_display_latex(latex(X))
        teaching = [r'\textbf{Fourier integral representation}',
                    r'X(\omega)=' + result_text,
                    r'\text{An accepted closed form has not been established.}']
        error = error or 'Closed form not established under the supported assumptions.'
        if _bad_display(conditions or ''): conditions = ''
        if _bad_display(result_text):
            # Do not expose a forbidden input function even inside an integral.
            result_text = ''
            teaching = [r'\text{This expression cannot be displayed in the supported notation.}']
    return FourierResponse(build_id=BUILD_ID, ok=accepted,
        input_latex=input_latex if input_latex is not None else _format_step_display_latex(latex(f)),
        result_latex=result_text, steps_latex=teaching, conditions_latex=conditions or '',
        error=None if accepted else error, form=form, method=method)


def _causal_polynomial(f):
    hs = list(f.atoms(Heaviside))
    if len(hs) != 1: return None
    lin = _as_linear_in_t(hs[0].args[0])
    if lin is None or lin[0].is_positive is not True: return None
    c = simplify(-lin[1]/lin[0])
    p = simplify((f/hs[0]).subs(t, t+c))
    if not p.is_polynomial(t): return None
    return c, p


def _convolution_class(f):
    """Conservative sufficient certificates, not a universal convergence oracle."""
    if f == 0: return 'compact'
    if _extract_single_function_factor(f, Rect) or _extract_single_function_factor(f, Tri):
        return 'compact'
    if _match_finite_step_window_add(f) is not None:
        return 'compact'
    if _rule_gaussian_family(f) is not None: return 'smooth_spectrum'
    if _has_exp_decay_and_step(f): return 'smooth_spectrum'
    if _rule_two_sided_exponential_parameter(f) is not None: return 'smooth_spectrum'
    if _rule_abs_exponential(f) is not None: return 'smooth_spectrum'
    # The shipped rational L1 pair and polynomially decaying proper rationals
    # with a denominator provably free of real zeros.
    num, den = fraction(together(f))
    if f.is_rational_function(t):
        try:
            if Poly(den,t).degree()-Poly(num,t).degree() >= 2:
                from sympy import solveset
                if solveset(den, t, domain=S.Reals) == S.EmptySet: return 'l1'
        except Exception:
            pass
    return None


def _convolve(f, g):
    tau = symbols('tau', real=True)
    definition = Integral(Integral(f.subs(t,tau)*g.subs(t,t-tau), (tau,-oo,oo))*exp(-I*omega*t), (t,-oo,oo))
    display = _format_step_display_latex(latex(f)) + r'\star ' + _format_step_display_latex(latex(g))
    Fresult, Gresult = _derive_with_properties(f), _derive_with_properties(g)
    F, G = Fresult[2], Gresult[2]
    conditions = r',\quad '.join(dict.fromkeys(c for c in [Fresult[4], Gresult[4]] if c))
    steps = [r'x(t)=(f\star g)(t)=\int_{-\infty}^{\infty}f(\tau)g(t-\tau)\,d\tau']
    # Compactly supported delta: convolution is a translated derivative, including
    # coefficient, scale and derivative order. No product of distributions needed.
    for impulse, other in [(f,g),(g,f)]:
        pair = _extract_single_function_factor(impulse, DiracDelta)
        c = _extract_delta_shift(impulse)
        if pair is not None and c is not None:
            coeff, d = pair
            a, b = _as_linear_in_t(d.args[0])
            cond = _affine_real_conditions(a,b)
            if cond is None: continue
            n = d.args[1] if len(d.args)>1 else 0
            shifted = coeff*diff(other,t,n).subs(t,t-c)/(Abs(a)*a**n)
            result = _derive_with_properties(shifted)
            steps += [r'\text{A compactly supported impulse gives a translated derivative.}',
                      r'X(\omega)=F(\omega)\,G(\omega)',
                      r'F(\omega)='+latex(F), r'G(\omega)='+latex(G)] + result[3]
            result = (*result[:3], steps, r',\quad '.join(x for x in [conditions,cond,result[4]] if x), result[5])
            return _finalize_response(shifted, result, 'convolution_rule', input_latex=display, definition=definition)
    # Causal polynomial distributions have a well-defined convolution on the half-line.
    # Compute there before transforming; delta^2 and PV^2 are never constructed.
    cp, cq = _causal_polynomial(f), _causal_polynomial(g)
    if cp is not None and cq is not None:
        center = cp[0]+cq[0]
        h = integrate(cp[1].subs(t,tau)*cq[1].subs(t,t-tau),(tau,0,t))
        h = expand(h).subs(t,t-center)*Heaviside(t-center)
        result = _derive_with_properties(h)
        steps += [r'\text{Both supports are bounded below; evaluate the finite time-domain convolution.}',
                  r'x(t)='+latex(h),
                  r'F(\omega)='+latex(F), r'G(\omega)='+latex(G),
                  r'\text{The formal identity }X(\omega)=F(\omega)\,G(\omega)\text{ requires the common causal limiting prescription.}',
                  r'\text{Ordinary products of singular distributions are not used.}'] + result[3]
        return _finalize_response(h, (*result[:3],steps,conditions,result[5]), 'convolution_rule', input_latex=display, definition=definition)
    valid = all(r[1] and not r[5] and r[0] in {'closed_form','distribution_form'}
                and not sympify(r[2]).has(Integral) and not _bad_expression(r[2]) for r in [Fresult,Gresult])
    cf, cg = _convolution_class(f), _convolution_class(g)
    df = sympify(F).has(DiracDelta,PV)
    dg = sympify(G).has(DiracDelta,PV)
    certified = (cf == 'compact' or cg == 'compact' or
                 (cf in {'l1','smooth_spectrum'} and cg in {'l1','smooth_spectrum'}) or
                 (df and cg == 'smooth_spectrum') or (dg and cf == 'smooth_spectrum'))
    if valid and certified and not (df and dg):
        X = _simplify_spectrum(F*G)
        steps += [r'\text{Use the convolution theorem under the stated integrability or smooth-multiplier conditions.}',
                  r'X(\omega)=F(\omega)\,G(\omega)',
                  r'F(\omega)='+latex(F),r'G(\omega)='+latex(G)] + _step_final_result(X)
        return _finalize_response(S.Zero, ('distribution_form' if df or dg else 'closed_form',True,X,steps,conditions,None),
                                  'convolution_rule',input_latex=display,definition=definition)
    result = ('integral_form',False,definition,[],conditions,
              'Convolution convergence or the product of distributions has not been established for these operands.')
    return _finalize_response(S.Zero,result,'convolution_rule',input_latex=display,definition=definition)


# ---------- API models ----------

class FourierRequest(BaseModel):
    expression: str


class FourierResponse(BaseModel):
    build_id: str | None = None
    ok: bool
    input_latex: str
    result_latex: str
    steps_latex: list[str]
    error: str | None = None
    method: str | None = None
    form: str | None = None
    conditions_latex: str | None = None


def _teaching_steps(f, X, steps):
    """Normalize derivation steps for user-facing teaching output."""
    cleaned = []
    for raw in steps or []:
        if raw is None:
            continue
        s = str(raw).strip()
        if not s:
            continue
        s = _format_step_display_latex(s)
        if s.startswith(r"X(\omega)="):
            s = r"X(\omega)=" + _format_result_display_latex(s[len(r"X(\omega)="):])
        cleaned.append(s)

    cleaned = _drop_nested_definition_blocks(cleaned)
    if not _should_keep_definition_block(cleaned):
        cleaned = _strip_leading_definition_block(cleaned)
    cleaned = _renumber_teaching_steps(cleaned)

    result_text = _format_result_display_latex(_format_pv_reciprocal_result_latex(f) or latex(X))
    final_heading = r"\textbf{Final Result}"
    if len(cleaned) >= 2 and cleaned[-2] == final_heading:
        cleaned[-1] = r"X(\omega)=" + result_text
        prefix_end = len(cleaned)-2
    else:
        prefix_end = len(cleaned)
        cleaned += [final_heading, r"X(\omega)=" + result_text]
    for index in range(prefix_end):
        if cleaned[index] == final_heading:
            cleaned[index] = r"\textbf{Intermediate result}"
    return cleaned


@app.post('/fourier', response_model=FourierResponse)
def fourier(req: FourierRequest):
    raw = (req.expression or '').strip()
    if not raw:
        return _input_error('Empty input')
    try:
        # Normalize brackets before finding a top-level convolution separator.
        raw = raw.replace('（','(').replace('）',')')
        conv = _split_convolution_top_level(raw)
        f = _parse_sympy(conv[0] if conv else raw)
        g = _parse_sympy(conv[1]) if conv else None
    except Exception:
        return _input_error('Parser error: check the supported functions, arithmetic and parentheses.')
    token = _METHOD_PATH.set(set())
    try:
        if conv:
            return _convolve(f,g)
        result = _derive_with_properties(f)
        paths = _METHOD_PATH.get()
        if 'direct_integral' in paths: method = 'direct_integral'
        elif 'property_rule' in paths: method = 'property_rule'
        elif 'linearity' in paths: method = 'linearity'
        elif sympify(result[2]).has(DiracDelta,PV) or result[0] == 'distribution_form': method = 'distribution_rule'
        else: method = 'known_pair'
        return _finalize_response(f,result,method)
    except Exception:
        return FourierResponse(build_id=BUILD_ID,ok=False,input_latex='',result_latex='',
            steps_latex=[],conditions_latex='',form='error',method='computation_error',
            error='The expression could not be evaluated under the supported assumptions.')
    finally:
        _METHOD_PATH.reset(token)


def _input_error(message):
    return FourierResponse(build_id=BUILD_ID,ok=False,input_latex='',result_latex='',
        steps_latex=[r'\text{Please enter a supported mathematical expression.}'],
        error=message,form='error',conditions_latex='',method='input_validation')


# ===== Extra rules: modulated step, B-spline, damped oscillation =====

def _rule_modulated_step(f):
    if not (isinstance(f, Mul) and f.has(Heaviside)):
        return None

    h_t = None
    step_shift = None
    for h in f.atoms(Heaviside):
        if len(h.args) < 1:
            continue
        shift = _match_heaviside_shift(h)
        if shift is not None:
            h_t = h
            step_shift = simplify(shift)
            break
    if h_t is None:
        return None

    rest = simplify(f / h_t)
    coeff, exp_part = rest.as_independent(t, as_Add=False)
    if exp_part.func != exp or len(exp_part.args) != 1:
        return None

    phase = simplify(exp_part.args[0] / I)
    if phase.has(I):
        return None
    lin = _as_linear_in_t(phase)
    if lin is None:
        return None

    w0, phi = lin
    c = simplify(step_shift)
    shifted_phase = simplify(w0*c + phi)
    base = pi*DiracDelta(omega - w0) - I*PV(1/(omega - w0))
    X = _simplify_spectrum(coeff * exp(I*shifted_phase) * exp(-I*omega*c) * base)

    if coeff == 1:
        coeff_latex = ""
    elif coeff == -1:
        coeff_latex = "-"
    else:
        coeff_latex = latex(coeff)

    steps = _step_start_definition(f)
    steps += [
        r"\textbf{Step 2: Identify a modulated unit-step signal}",
        r"x(t)=" + coeff_latex + _complex_exponential_display(w0, phi) + r"u(t-c)",
        r"\text{Here }\omega_0=" + latex(w0) + r",\quad \phi=" + latex(phi) + r",\quad c=" + latex(c),
        r"\textbf{Step 3: Move the step edge to the origin}",
        r"\text{Let }s=t-c.\text{ Then }u(t-c)=u(s)",
        r"e^{j(\omega_0 t+\phi)}=e^{j(\omega_0 s+\omega_0 c+\phi)}",
        r"\text{For this input, }\omega_0 c+\phi=" + latex(shifted_phase),
        r"\textbf{Step 4: Use the modulated unit-step transform}",
        r"\mathcal{F}\{e^{j\omega_0 t}u(t)\}=\pi\delta(\omega-\omega_0)-j\,\mathrm{PV}\!\left(\frac{1}{\omega-\omega_0}\right)",
        r"\text{PV appears because the modulated step is one-sided and not absolutely integrable.}",
        r"\textbf{Step 5: Apply the time-shift property}",
        r"\mathcal{F}\{y(t-c)\}=e^{-j\omega c}Y(\omega)",
    ]
    if coeff != 1:
        steps.append(r"\text{Apply the constant factor }" + latex(coeff) + r"\text{ by linearity.}")
    steps += _step_final_result(X)
    return ("distribution_form", True, X, steps, "", None)
    return None


def _rule_bspline2(f):
    if str(f).replace(" ","") in ["Heaviside(t)*Heaviside(t)", "Heaviside(t)**2", "Heaviside(t).Heaviside(t)"]:
        return _rule_shifted_poly_u_explicit(t*Heaviside(t))
    return None


def _rule_damped_oscillation(f):
    if isinstance(f,Mul) and f.has(Heaviside(t)):
        rest = simplify(f/Heaviside(t))
        a,b = symbols("a b", real=True, positive=True)
        m = rest.match(exp(-a*t)*sin(b*t))
        if m and a in m and b in m:
            aa,bb = m[a],m[b]
            X = bb/((aa+I*omega)**2 + bb**2)
            steps=[
                r"x(t)=e^{-a t}\sin(bt)u(t)",
                r"X(\omega)=\frac{b}{(a+j\omega)^2+b^2}",
                r"X(\omega)="+latex(X)
            ]
            return ("distribution_form",True,X,steps,r"\Re(a)>0",None)
    return None




def _polynomial_step_distribution_from_poly(g):
    g = expand(simplify(g))
    if not g.is_polynomial(t):
        return None

    try:
        poly = Poly(g, t)
    except Exception:
        return None

    def _basis(n):
        if n == 0:
            return pi*DiracDelta(omega) - I*(1/omega)
        return (
            pi*(I**n)*Derivative(DiracDelta(omega), (omega, n))
            + ((-1)**n)*(I**(n-1))*factorial(n)*PV(1/(omega**(n+1)))
        )

    pieces = []
    degree_values = []
    for (degree,), coeff in poly.terms():
        if degree < 0:
            return None
        degree_int = int(degree)
        degree_values.append(degree_int)
        pieces.append(simplify(coeff * _basis(degree_int)))
    if not pieces:
        return None

    return Add(*pieces, evaluate=False), sorted(degree_values)


def _rule_shifted_poly_times_step_distribution(f):
    """
    Distribution rule for p(t)u(t-c). Let s=t-c, derive p(s+c)u(s)
    with the polynomial-step basis, then apply time shift.
    """
    if not (isinstance(f, Mul) and f.has(Heaviside)):
        return None

    h_t = None
    c = None
    for h in f.atoms(Heaviside):
        if len(h.args) < 1:
            continue
        shift = _match_heaviside_shift(h)
        if shift is not None and simplify(shift) != 0:
            h_t = h
            c = simplify(shift)
            break
    if h_t is None:
        return None

    rest = simplify(f / h_t)
    shifted_poly = expand(simplify(rest.subs(t, t + c)))
    base = _polynomial_step_distribution_from_poly(shifted_poly)
    if base is None:
        return None
    Y, degree_values = base
    X = _simplify_spectrum(exp(-I*omega*c) * Y)

    steps = _step_start_definition(f)
    steps += [
        r"\textbf{Step 2: Identify a shifted polynomial multiplied by a shifted unit step}",
        r"x(t)=p(t)u(t-c),\quad c=" + latex(c),
        r"\textbf{Step 3: Move the step edge to the origin}",
        r"\text{Let }s=t-c.\text{ Then }u(t-c)=u(s)\text{ and }t=s+c",
        r"p(s+c)=" + _display_in_s(shifted_poly),
        r"\text{For this input, use }n\in\{" + ",".join(str(n) for n in degree_values) + r"\}\text{ term by term.}",
        r"\textbf{Step 4: Reuse the polynomial-step transform pair at the origin}",
        r"\mathcal{F}\{t^n u(t)\}=\pi j^n\delta^{(n)}(\omega)+(-1)^n j^{\,n-1}n!\,\mathrm{PV}\!\left(\frac{1}{\omega^{n+1}}\right)",
        r"\textbf{Step 5: Apply the time-shift property}",
        r"\mathcal{F}\{y(t-c)\}=e^{-j\omega c}Y(\omega)",
    ]
    steps += _step_final_result(X)
    return ("distribution_form", True, X, steps, "", None)


def _rule_poly_times_step_distribution(f):
    """
    Explicit distribution for t^n * Heaviside(t):
      F{t^n u(t)} = pi*j^n*delta^{(n)}(omega) + (-1)^n*j^{n-1}*n!*PV(1/omega^{n+1})
    """
    if not (isinstance(f, Mul) and f.has(Heaviside)):
        return None

    h_t = None
    for h in f.atoms(Heaviside):
        if len(h.args) >= 1 and simplify(h.args[0] - t) == 0:
            h_t = h
            break
    if h_t is None:
        return None

    g = expand(simplify(f / h_t))
    base = _polynomial_step_distribution_from_poly(g)
    if base is None:
        return None
    X, degree_values = base
    steps = [
        r"\textbf{Method: Distribution rule (polynomial times step)}",
        r"x(t)=p(t)u(t),\;p(t)=" + latex(g),
        r"\mathcal{F}\{u(t)\}=\pi\delta(\omega)-j\,\mathrm{PV}\!\left(\frac{1}{\omega}\right)",
        r"\mathcal{F}\{t^n u(t)\}=\pi j^n\delta^{(n)}(\omega)+(-1)^n j^{\,n-1}n!\,\mathrm{PV}\!\left(\frac{1}{\omega^{n+1}}\right)",
        r"\text{Apply polynomial linearity term by term}",
        r"\Rightarrow X(\omega)=\pi j^n\delta^{(n)}(\omega)+(-1)^n j^{\,n-1}n!\,\mathrm{PV}\frac{1}{\omega^{n+1}}",
        r"X(\omega)=" + latex(X),
        ]
    steps = _step_start_definition(f)
    steps += [
        r"\textbf{Step 2: Identify a polynomial multiplied by the unit step}",
        r"x(t)=p(t)u(t),\quad p(t)=" + latex(g),
        r"\text{For this input, use }n\in\{" + ",".join(str(n) for n in degree_values) + r"\}\text{ term by term.}",
        r"\textbf{Step 3: Use the unit-step transform pair}",
        r"\mathcal{F}\{u(t)\}=\pi\delta(\omega)-j\,\mathrm{PV}\!\left(\frac{1}{\omega}\right)",
        r"\text{The PV term appears because }u(t)\text{ is one-sided and not absolutely integrable.}",
        r"\textbf{Step 4: Use frequency differentiation}",
        r"\mathcal{F}\{t^n u(t)\}=\pi j^n\delta^{(n)}(\omega)+(-1)^n j^{\,n-1}n!\,\mathrm{PV}\!\left(\frac{1}{\omega^{n+1}}\right)",
        r"\textbf{Step 5: Apply polynomial linearity term by term}",
    ]
    steps += _step_final_result(X)
    return ("distribution_form", True, X, steps, "", None)


def _rule_trig_times_step_distribution(f):
    """
    Distribution-first for sin(a t + b)u(t-c), cos(a t + b)u(t-c).
    Rewrite around s=t-c, then use Euler form + u(s) shifted in frequency.
    """
    if not (isinstance(f, Mul) and f.has(Heaviside)):
        return None

    # Accept any unit step of the form Heaviside(t-c), including c=0.
    h_t = None
    step_shift = None
    for h in f.atoms(Heaviside):
        if len(h.args) < 1:
            continue
        shift = _match_heaviside_shift(h)
        if shift is not None:
            h_t = h
            step_shift = simplify(shift)
            break
    if h_t is None:
        return None

    g = simplify(f / h_t)
    coeff, trig_part = g.as_independent(t, as_Add=False)

    # match sin(a*t+b) or cos(a*t+b) with linear phase
    a = Wild("a", exclude=[t])
    b = Wild("b", exclude=[t])
    mm_sin = None
    mm_cos = None

    if trig_part.func == sin:
        mm_sin = trig_part.args[0].match(a*t + b)
        trig = "sin"
    elif trig_part.func == cos:
        mm_cos = trig_part.args[0].match(a*t + b)
        trig = "cos"
    else:
        return None

    mm = mm_sin if mm_sin is not None else mm_cos
    if not (mm and a in mm and b in mm):
        return None

    aa = simplify(mm[a])
    bb = simplify(mm[b])
    cc = simplify(step_shift)
    shifted_phase = simplify(aa*cc + bb)

    # helper: F{e^{j*w0*t}u(t)} = pi*delta(omega-w0) - j*PV(1/(omega-w0))
    def _Ushift(w0):
        return pi*DiracDelta(omega - w0) - I*PV(1/(omega - w0))

    if trig == "sin":
        X = _simplify_spectrum((exp(I*shifted_phase)*_Ushift(aa) - exp(-I*shifted_phase)*_Ushift(-aa)) / (2*I))
        steps = [
            r"\textbf{Method: Distribution rule (trig times step)}",
            _linear_phase_display("sin", aa, bb) + r"u(t)",
            r"\sin(\theta)=\frac{e^{j\theta}-e^{-j\theta}}{2j}",
            r"\mathcal{F}\{e^{j\omega_0 t}u(t)\}=\pi\delta(\omega-\omega_0)-j\,\mathrm{PV}\!\left(\frac{1}{\omega-\omega_0}\right)",
            r"\text{Apply linearity and frequency shift } \omega_0=\pm " + latex(aa),
            r"X(\omega)=" + latex(X),
            ]
    else:
        X = _simplify_spectrum((exp(I*shifted_phase)*_Ushift(aa) + exp(-I*shifted_phase)*_Ushift(-aa)) / 2)
        steps = [
            r"\textbf{Method: Distribution rule (trig times step)}",
            _linear_phase_display("cos", aa, bb) + r"u(t)",
            r"\cos(\theta)=\frac{e^{j\theta}+e^{-j\theta}}{2}",
            r"\mathcal{F}\{e^{j\omega_0 t}u(t)\}=\pi\delta(\omega-\omega_0)-j\,\mathrm{PV}\!\left(\frac{1}{\omega-\omega_0}\right)",
            r"\text{Apply linearity and frequency shift } \omega_0=\pm " + latex(aa),
            r"X(\omega)=" + latex(X),
            ]

    X = _simplify_spectrum(coeff * exp(-I*omega*cc) * X)

    trig_display = r"\sin" if trig == "sin" else r"\cos"
    euler_identity = (
        r"\sin(\theta)=\frac{e^{j\theta}-e^{-j\theta}}{2j}"
        if trig == "sin"
        else r"\cos(\theta)=\frac{e^{j\theta}+e^{-j\theta}}{2}"
    )
    if coeff == 1:
        coeff_latex = ""
    elif coeff == -1:
        coeff_latex = "-"
    else:
        coeff_latex = latex(coeff)
    steps = _step_start_definition(f)
    steps += [
        r"\textbf{Step 2: Identify a sinusoid multiplied by a shifted unit step}",
        r"x(t)=" + coeff_latex + _linear_phase_display(trig, aa, bb) + r"u(t-c)",
        r"\text{Here }a=" + latex(aa) + r",\quad b=" + latex(bb) + r",\quad c=" + latex(cc),
        r"\textbf{Step 3: Move the step edge to the origin}",
        r"\text{Let }s=t-c.\text{ Then }u(t-c)=u(s)",
        trig_display + r"\!\left(a t+b\right)=" + trig_display + r"\!\left(a s+(a c+b)\right)",
        r"\text{For this input, }a c+b=" + latex(shifted_phase),
        r"\text{So the shifted sinusoid is }" + _linear_phase_display(trig, aa, shifted_phase, var="s"),
        r"\textbf{Step 4: Rewrite the sinusoid with Euler's identity}",
        euler_identity,
        r"\textbf{Step 5: Use the modulated unit-step transform}",
        r"\mathcal{F}\{e^{j\omega_0 t}u(t)\}=\pi\delta(\omega-\omega_0)-j\,\mathrm{PV}\!\left(\frac{1}{\omega-\omega_0}\right)",
        r"\text{PV appears because each modulated step remains one-sided and not absolutely integrable.}",
        r"\textbf{Step 6: Apply the time-shift property}",
        r"\mathcal{F}\{y(t-c)\}=e^{-j\omega c}Y(\omega)",
        r"\textbf{Step 7: Combine the }+\omega_0\text{ and }-\omega_0\text{ terms by linearity}",
    ]
    if coeff != 1:
        steps.append(r"\text{Apply the constant factor }" + latex(coeff) + r"\text{ by linearity.}")
    steps += _step_final_result(X)
    return ("distribution_form", True, X, steps, "", None)




def _rule_poly_distribution(f):
    """
    Distribution for pure polynomials t^n (no step):
      F{t^n} = 2*pi*j^n*delta^{(n)}(omega)
    """
    if f == t:
        n = 1
    elif f.is_Pow and f.base == t and f.exp.is_Integer and int(f.exp) >= 0:
        n = int(f.exp)
    else:
        return None

    X = _simplify_spectrum(2*pi*(I**n)*Derivative(DiracDelta(omega), (omega, n)))
    steps = _step_start_definition(f)
    steps += [
        r"\textbf{Step 2: Identify the polynomial degree}",
        r"x(t)=t^{" + str(n) + r"}",
        r"\text{Here }n=" + str(n),
        r"\textbf{Step 3: Use the constant transform pair}",
        r"\mathcal{F}\{1\}=2\pi\delta(\omega)",
        r"\textbf{Step 4: Use frequency differentiation}",
        r"\mathcal{F}\{t^n x(t)\}=j^n\frac{d^n}{d\omega^n}X(\omega)",
        r"\Rightarrow X(\omega)=2\pi j^n\delta^{(n)}(\omega)",
        ]
    steps += _step_final_result(X)
    return ("distribution_form", True, X, steps, "", None)


def _rule_pv_reciprocal(f):
    r"""
    PV distribution for 1/(a*t+b) (includes 1/(t+a)).

    Convention:
      \mathcal{F}\{\mathrm{PV}(1/t)\} = -i\pi\,\mathrm{sign}(\omega)

    Then:
      1/(a t+b) = (1/a) * 1/(t + b/a)
      => X(\omega) = -(i\pi/a) * e^{i\omega (b/a)} * sign(\omega)
    """
    try:
        num, den = f.as_numer_denom()
        if simplify(num - 1) != 0:
            return None

        # Keep the original neat display for 1/(t+a)
        a0 = _match_t_plus_a(den)
        if a0 is not None:
            X = exp(I*omega*a0) * (-I*pi*sign(omega))
            steps = [
                r"\textbf{Method: Distribution rule (principal value)}",
                r"\mathrm{PV}\!\int_{-\infty}^{\infty}\frac{e^{-i\omega t}}{t}\,dt=-i\pi\,\mathrm{sign}(\omega)",
                r"\Rightarrow\;\mathcal{F}\left\{\mathrm{PV}\frac{1}{t+a}\right\}=e^{i\omega a}\left(-i\pi\,\mathrm{sign}(\omega)\right)",
                r"X(\omega)=" + latex(X),
                ]
            X = _omega_real_cleanup(X)
            steps = _step_start_definition(f)
            steps += [
                r"\textbf{Step 2: Interpret the reciprocal as a principal-value distribution}",
                r"\mathrm{PV}\!\int_{-\infty}^{\infty}\frac{e^{-j\omega t}}{t}\,dt=-j\pi\,\mathrm{sign}(\omega)",
                r"\text{PV means the singularity at }t=0\text{ is handled by symmetric limiting.}",
                r"\textbf{Step 3: Apply the time-shift property}",
                r"\mathcal{F}\left\{\mathrm{PV}\frac{1}{t+a}\right\}=e^{j\omega a}\left(-j\pi\,\mathrm{sign}(\omega)\right)",
                r"\text{Substitute }a=" + latex(a0),
            ]
            steps += [
                r"\textbf{Final Result}",
                r"X(\omega)=" + (_format_pv_reciprocal_result_latex(f) or latex(X)),
            ]

            return ("distribution_form", True, X, steps, "", None)

        # General linear denominator: a*t + b
        lin = _as_linear_in_t(den)
        if lin is None:
            return None
        a, b = lin
        if simplify(a) == 0:
            return None

        shift = simplify(b / a)
        X = _simplify_spectrum((-I*pi/a) * exp(I*omega*shift) * sign(omega))

        steps = [
            r"\textbf{Method: Distribution rule (principal value)}",
            r"\mathrm{PV}\!\int_{-\infty}^{\infty}\frac{e^{-i\omega t}}{t}\,dt=-i\pi\,\mathrm{sign}(\omega)",
            r"\frac{1}{a t+b}=\frac{1}{a}\,\frac{1}{t+\frac{b}{a}}",
            r"\Rightarrow\;X(\omega)=-\frac{i\pi}{a}\,e^{i\omega\frac{b}{a}}\,\mathrm{sign}(\omega)",
            r"\text{Substitute }a=%s,\;b=%s\;\Rightarrow\;\frac{b}{a}=%s\;\Rightarrow\;X(\omega)=%s" % (latex(a), latex(b), latex(shift), (_format_pv_reciprocal_result_latex(f) or latex(_omega_real_cleanup(X)))),
            r"X(\omega)=" + latex(X),
            ]
        steps = _step_start_definition(f)
        steps += [
            r"\textbf{Step 2: Interpret the reciprocal as a principal-value distribution}",
            r"\mathrm{PV}\!\int_{-\infty}^{\infty}\frac{e^{-j\omega t}}{t}\,dt=-j\pi\,\mathrm{sign}(\omega)",
            r"\text{PV means the singularity is handled by symmetric limiting around the pole.}",
            r"\textbf{Step 3: Rewrite the denominator into shifted form}",
            r"\frac{1}{a t+b}=\frac{1}{a}\,\frac{1}{t+\frac{b}{a}}",
            r"\textbf{Step 4: Apply scaling, shift, and linearity}",
            r"X(\omega)=-\frac{j\pi}{a}\,e^{j\omega\frac{b}{a}}\,\mathrm{sign}(\omega)",
            r"\text{Substitute }a=%s,\;b=%s,\;\frac{b}{a}=%s" % (latex(a), latex(b), latex(shift)),
        ]
        steps += [
            r"\textbf{Final Result}",
            r"X(\omega)=" + (_format_pv_reciprocal_result_latex(f) or latex(_omega_real_cleanup(X))),
        ]
        return ("distribution_form", True, X, steps, "", None)
    except Exception:
        return None


def _format_pv_reciprocal_result_latex(f):
    """
    If f is 1/(a*t+b) (includes 1/(t+a)), return a stable LaTeX string
    without Piecewise.
    """
    try:
        num, den = f.as_numer_denom()
        if simplify(num - 1) != 0:
            return None

        a0 = _match_t_plus_a(den)
        if a0 is not None:
            if simplify(a0) == 0:
                return r"-i\pi\,\mathrm{sign}(\omega)"
            return _phase_factor_latex(a0) + r"\left(-j\pi\,\mathrm{sign}(\omega)\right)"

        lin = _as_linear_in_t(den)
        if lin is None:
            return None
        a, b = lin
        if simplify(a) == 0:
            return None

        shift = simplify(b / a)
        if simplify(shift) == 0:
            return r"-\frac{j\pi}{%s}\,\mathrm{sign}(\omega)" % latex(a)
        return r"-\frac{j\pi}{%s}\," % latex(a) + _phase_factor_latex(shift) + r"\,\mathrm{sign}(\omega)"
    except Exception:
        return None


def _rule_affine_step(f):
    pair = _extract_single_function_factor(f, Heaviside)
    if pair is None or not _is_heaviside(pair[1]): return None
    coeff, h = pair
    lin = _as_linear_in_t(h.args[0])
    if lin is None: return None
    a, b = lin
    conditions = _affine_real_conditions(a,b)
    if conditions is None: return None
    c = simplify(-b/a)
    X = coeff*exp(-I*omega*c)*(pi*DiracDelta(omega)-I*sign(a)*PV(1/omega))
    steps = [r'\textbf{Step 1: Identify the step edge and orientation}',
             r'c='+latex(c)+r',\quad a='+latex(a)+r',\quad C='+latex(coeff),
             r'\mathcal{F}\{u(a t)\}=\pi\delta(\omega)-j\,\mathrm{sign}(a)\,\mathrm{PV}\frac{1}{\omega}',
             r'\textbf{Step 2: Apply time shift and linearity}',
             r'\mathcal{F}\{C y(t-c)\}=C e^{-j\omega c}Y(\omega)'] + _step_final_result(X)
    _record_method('known_pair')
    return 'distribution_form',True,X,steps,conditions,None
