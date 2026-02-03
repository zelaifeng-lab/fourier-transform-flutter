from __future__ import annotations

import time
import contextvars
import uuid

from fastapi import FastAPI
from pydantic import BaseModel
from sympy import (
    symbols, I, exp, Integral, oo, latex, simplify, expand, diff, Derivative, re, im, sqrt, factorial,
    Add, Mul, pi, sin, cos, Piecewise, Abs, sign, factor_terms
, Function, together, fraction, Poly, div, apart, Wild, srepr)
from sympy.functions.special.delta_functions import Heaviside, DiracDelta
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application
)

# ============================================================
# Fourier Backend (Engineering Convention, ω real)
#   X(ω) = ∫_{-∞}^{∞} x(t) e^{-j ω t} dt , ω ∈ ℝ
#
# User conventions:
#   - Convolution uses '·' (U+00B7) ONLY
#   - Multiplication uses '*' (or implicit multiplication)
#
# Policy:
#   - Force ω real (avoid complex-ω arg(...) artifacts)
#   - For ANY expression containing trig, first rewrite to complex exponentials.
#     If it becomes a finite sum of pure tones C_k e^{j ω_k t}, return the
#     distribution result via the definition integral:
#         ∫ e^{-j(ω-ω0)t} dt = 2π δ(ω-ω0)
#   - Otherwise fall back to property rules + integral fallback.
# ============================================================

app = FastAPI(title="Fourier Backend (Engineering Convention, ω real)")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://zelaifeng-lab.github.io",   # 你的 GitHub Pages 域名
    ],
    allow_credentials=True,
    allow_methods=["*"],   # 允许 POST/OPTIONS 等
    allow_headers=["*"],   # 允许 Content-Type 等
)


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


BUILD_ID = "v2_fixed_20260121"
APART_FULL_DEFAULT = False  # always keep real-field partial fractions (avoid RootSum)

# ===== Omega-real cleanup (avoid Piecewise/arg/RootSum) =====
def _omega_real_cleanup(expr):
    """
    Post-process SymPy outputs assuming ω is real:
    - replace ω/|ω| with sign(ω)
    - drop Piecewise branches that only special-case ω=0 (removable after sign rewrite)
    """
    try:
        expr = expr.subs(omega/Abs(omega), sign(omega))
        expr = expr.subs(-omega/Abs(omega), -sign(omega))
        expr = expr.subs(Abs(omega)/omega, sign(omega))

        # Replace Abs(ω)*ω**(-1) patterns with sign(ω)
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
omega = symbols("omega", real=True)

# ---------- lightweight caches for expensive SymPy operations ----------
_TOGETHER_CACHE = {}
_APART_CACHE = {}

def _together_cached(expr):
    key = srepr(expr)
    v = _TOGETHER_CACHE.get(key)
    if v is not None:
        return v
    with _PerfTimer("sympy.together", info=str(expr)[:80]):
        v = together(expr)
    _TOGETHER_CACHE[key] = v
    return v

def _apart_cached(expr):
    key = srepr(expr)
    v = _APART_CACHE.get(key)
    if v is not None:
        return v
    with _PerfTimer("sympy.apart", info=str(expr)[:80]):
        v = apart(expr, t, full=APART_FULL_DEFAULT)
    _APART_CACHE[key] = v
    return v


# Rect(x): unit rectangular pulse, rect(x)=1 for |x|<=1/2 else 0
class Rect(Function):
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
                out.append(f"(({args[0]})/({args[1]}))")
            else:
                out.append("FRAC(" + ",".join(args) + ")")
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def _pre_normalize(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("（", "(").replace("）", ")")
    s = s.replace("π", "pi")
    s = s.replace("ω", "omega")
    s = s.replace("−", "-")

    # users often type '^' for power; sympy uses '**'
    s = s.replace("^", "**")

    # step/delta aliases
    s = s.replace("u(", "Heaviside(")
    s = s.replace("θ(", "Heaviside(")
    s = s.replace("heaviside(", "Heaviside(")

    s = s.replace("δ(", "DiracDelta(")
    s = s.replace("delta(", "DiracDelta(")

    # frac(a,b)
    s = s.replace("rect(", "Rect(")
    s = s.replace("frac(", "FRAC(")
    s = _convert_frac_calls(s)
    return s


def _parse_sympy(expr_str: str):
    expr_str = _pre_normalize(expr_str)
    local_dict = {
        "t": t,
        "omega": omega,
        "pi": pi,
        "Heaviside": Heaviside,
        "DiracDelta": DiracDelta,
        "I": I,
        "exp": exp,
        "sin": sin,
        "cos": cos,
        "Abs": Abs,
        "sign": sign,
        "Rect": Rect,
    }
    expr = parse_expr(expr_str, local_dict=local_dict, transformations=TRANSFORMS, evaluate=True)
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
    Convolution is triggered by '·' (U+00B7) or '•' (U+2022) at top level.
    Multiplication must be written as '*'.
    """
    s = _strip_outer_parens_once(s)
    depth = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if depth != 0:
            continue
        if ch in {"·", "•"}:
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
    try:
        X = X_def.doit()
        return True, simplify(X)
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



def _try_trig_as_exp_distribution(f):
    """
    Robust trig -> exp -> pure-tone detection.

    Also has a guaranteed path for sin(a*t+b) / cos(a*t+b) to avoid SymPy's arg(ω) Piecewise.

    NOTE: This function is "steps-only refactor": computation is unchanged, only steps text is made
    textbook-like.
    """
    # Guaranteed: sin(a*t+b), cos(a*t+b) -> delta-combination (distribution)
    if f.func in (sin, cos) and len(f.args) == 1:
        lin = _as_linear_in_t(f.args[0])
        if lin is not None:
            a, b = lin  # argument = a*t + b, with a,b independent of t
            if f.func == sin:
                X = simplify(pi / I * (exp(I*b) * DiracDelta(omega - a) - exp(-I*b) * DiracDelta(omega + a)))
                steps = [
                    r"\textbf{Step 1: Identify sinusoidal signal}",
                    r"x(t)=" + latex(f),
                    r"\textbf{Step 2: Use Euler identity}",
                    r"\sin(\theta)=\frac{e^{j\theta}-e^{-j\theta}}{2j},\;\;\theta=a t+b",
                    r"\textbf{Step 3: Use Fourier transform of complex exponentials}",
                    r"\mathcal{F}\{e^{j\omega_0 t}\}=2\pi\,\delta(\omega-\omega_0)",
                    r"\textbf{Step 4: Combine delta functions}",
                    r"X(\omega)=\frac{\pi}{j}\Big(e^{jb}\delta(\omega-a)-e^{-jb}\delta(\omega+a)\Big)",
                    r"\textbf{Final Result}",
                    r"X(\omega)=" + latex(X),
                    ]
                return ("distribution_form", True, X, steps, "", None)
            else:
                X = simplify(pi * (exp(I*b) * DiracDelta(omega - a) + exp(-I*b) * DiracDelta(omega + a)))
                steps = [
                    r"\textbf{Step 1: Identify sinusoidal signal}",
                    r"x(t)=" + latex(f),
                    r"\textbf{Step 2: Use Euler identity}",
                    r"\cos(\theta)=\frac{e^{j\theta}+e^{-j\theta}}{2},\;\;\theta=a t+b",
                    r"\textbf{Step 3: Use Fourier transform of complex exponentials}",
                    r"\mathcal{F}\{e^{j\omega_0 t}\}=2\pi\,\delta(\omega-\omega_0)",
                    r"\textbf{Step 4: Combine delta functions}",
                    r"X(\omega)=\pi\Big(e^{jb}\delta(\omega-a)+e^{-jb}\delta(\omega+a)\Big)",
                    r"\textbf{Final Result}",
                    r"X(\omega)=" + latex(X),
                    ]
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
            # constant term -> would transform to delta(ω); but keep this function for pure tones only
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

    # Build X(ω)=∑ 2π Ck δ(ω-wk)
    X = 0
    for Ck, wk in pairs:
        X += simplify(2*pi*Ck*DiracDelta(omega - wk))
    X = _omega_real_cleanup(X)

    # Textbook-like steps (no internal logs)
    steps = [
        r"\textbf{Step 1: Identify sinusoidal signal}",
        r"x(t)=" + latex(f),
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
      π e^{-√c |ω|} ( b/√c - i a sign(ω) )
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

        steps = [
            r"\text{Known pair: }\frac{1}{t^2+c}\;\xleftrightarrow{\mathcal{F}}\;\frac{\pi}{\sqrt{c}}e^{-\sqrt{c}|\omega|},\;c>0",
            r"\text{And }\frac{t}{t^2+c}\;\xleftrightarrow{\mathcal{F}}\;-j\pi\,\mathrm{sign}(\omega)e^{-\sqrt{c}|\omega|}",
            r"\text{Decompose: }\frac{a t + b}{t^2+c}=a\frac{t}{t^2+c}+b\frac{1}{t^2+c}",
            r"c=" + latex(c),
            r"X(\omega)=" + latex(X),
            ]
        X = _omega_real_cleanup(X)
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
            # If denominator factors into two distinct linear factors in t, do not skip apart.
            facs = Pd0.factor_list()[1]  # [(factor_poly, exp), ...]
            degs = [fp.degree() for (fp, _e) in facs]
            if not (len(facs) == 2 and degs == [1, 1]):
                return None
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
        if pf == f_rem:
            return None

        # transform each additive term (computation unchanged)

        terms = pf.as_ordered_terms() if pf.is_Add else [pf]

        X_terms = []


        # Build term-wise derivations for display (textbook style)

        termwise_steps = []

        for k, term in enumerate(terms, start=1):

            # Compute transform of this term using the same engine (does not change math rules)

            form_k, ok_k, Xk, steps_k, cond_k, err_k = _derive_with_properties(term)


            X_terms.append(Xk)


            termwise_steps.append(r"\textbf{Term %d}:\quad x_{%d}(t)=%s" % (k, k, latex(term)))


            # Include the term's own derivation steps, but avoid repeating global headers/markers.

            for s in (steps_k or []):

                if not s:

                    continue

                ss = str(s).strip()

                if not ss:

                    continue

                if ("=\\mathcal" in ss and "X(" in ss):

                    continue

                if ss.startswith("Method:"):

                    continue

                if "Final Result" in ss:

                    continue

                termwise_steps.append(ss)


            termwise_steps.append(r"\Rightarrow\; X_{%d}(ω)=%s" % (k, latex(Xk)))


        X = _omega_real_cleanup(Add(*X_terms, evaluate=False))

        # Textbook-like steps
        steps = [
            r"\textbf{Step 1: Identify a rational time-domain function}",
            r"x(t)=" + latex(f),
            r"\textbf{Step 2: Apply partial fraction decomposition}",
            ]
        if div_line is not None:
            steps.append(div_line)
        steps.append(latex(f_rem) + "=" + latex(pf))
        steps.append(r"\textbf{Step 3: Use known Fourier transform pairs}")
        steps.extend(termwise_steps)
        steps.append(r"\textbf{Step 4: Combine by linearity}")
        steps.append(r"X(\omega)=\sum_k \mathcal{F}\{t_k\}")
        steps.append(r"\textbf{Final Result}")
        steps.append(r"X(\omega)=" + latex(X))

        return ("distribution_form", True, X, steps, "", None)
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




# ---------- main derivation ----------

def _derive_with_properties(f):
    """
    Returns: (form, ok, X_expr, steps_latex, conditions_latex, error_or_None)
    form ∈ {"closed_form","integral_form","distribution_form","divergent"}
    """

    # 0) Trig-first policy (user request)

    poly_step_res = _rule_poly_times_step_distribution(f)

    pv_res = _rule_pv_reciprocal(f)
    if pv_res is not None:
        return pv_res

    poly_res = _rule_poly_distribution(f)
    if poly_res is not None:
        return poly_res

    if poly_step_res is not None:
        return poly_step_res

    trig_step_res = _rule_trig_times_step_distribution(f)
    if trig_step_res is not None:
        return trig_step_res

    trig_res = _try_trig_as_exp_distribution(f)
    if trig_res is not None:
        return trig_res

    # --- Known pair: 1/(t^2 + c), c>0 (force ω real; avoid half-branch Piecewise) ---
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
                        X = simplify(pi/alpha * exp(-alpha*Abs(omega)))
                        # Use Unicode ω in steps to avoid flutter_math_fork \omega parser issues.
                        steps = [
                            r"X(ω)=\int_{-\infty}^{\infty}x(t)\,e^{-iω t}\,dt",
                            rf"x(t)=\frac{{1}}{{t^{{2}}+{latex(c)}}}",
                            rf"\Rightarrow\;X(ω)=\int_{{-\infty}}^{{\infty}}\frac{{e^{{-iω t}}}}{{t^{{2}}+{latex(c)}}}\,dt\;\;({latex(c)}>0)",
                            rf"X(ω)=\frac{{\pi}}{{{latex(alpha)}}}e^{{-{latex(alpha)}|ω|}}",
                        ]
                        X = _omega_real_cleanup(X)
                        return "distribution_form", True, X, steps, "", None
    except Exception:
        pass

    quadlin_res = _rule_linear_over_t2_plus_c(f)
    if quadlin_res is not None:
        return quadlin_res

    rat_res = _rule_rational_apart_linearity(f)
    if rat_res is not None:
        return rat_res



    # 0.5) Constant (distribution)
    # x(t)=C  -> X(ω)=2π C δ(ω)
    if f.free_symbols.isdisjoint({t}):
        X = simplify(2*pi*f*DiracDelta(omega))
        steps = [
            r"x(t)=" + latex(f),
            r"X(\omega)=\int_{-\infty}^{\infty}C\,e^{-j\omega t}dt=2\pi C\,\delta(\omega)\quad(\text{distribution})",
            r"X(\omega)=" + latex(X),
            ]
        return "distribution_form", True, X, steps, "", None

    # 0.6) DiracDelta(t-a) shift
    if f.func == DiracDelta and len(f.args)==1:
        arg = f.args[0]
        lin = _as_linear_in_t(arg)
        if lin is not None:
            a, b = lin  # arg = a*t + b
            # Only handle a = 1, -1 cleanly for now
            if a == 1:
                t0 = -b
                X = simplify(exp(-I*omega*t0))
                steps = [
                    r"x(t)=\delta(t-t_0)",
                    r"X(\omega)=\int \delta(t-t_0)e^{-j\omega t}dt=e^{-j\omega t_0}",
                    r"t_0=" + latex(t0),
                    r"X(\omega)=" + latex(X),
                    ]
                return "distribution_form", True, X, steps, "", None
            if a == -1:
                # δ(-t + b) = δ(t-b)
                t0 = b
                X = simplify(exp(-I*omega*t0))
                steps = [
                    r"x(t)=\delta(-t+t_0)=\delta(t-t_0)",
                    r"X(\omega)=e^{-j\omega t_0}",
                    r"t_0=" + latex(t0),
                    r"X(\omega)=" + latex(X),
                    ]
                return "distribution_form", True, X, steps, "", None

    # 0.7) Heaviside(t) and Heaviside(t-a) (distribution)
    if f == Heaviside(t) or (f.func==Heaviside and len(f.args)==1 and _as_linear_in_t(f.args[0]) is not None):
        if f == Heaviside(t):
            a_shift = 0
        else:
            lin = _as_linear_in_t(f.args[0])
            a1, b1 = lin  # a1*t + b1
            if a1 != 1:
                a_shift = None
            else:
                a_shift = -b1
        if a_shift is not None:
            base = simplify(pi*DiracDelta(omega) - I*(1/omega))
            # Use PV via sign rule? We'll keep PV as 1/omega with distribution note.
            # More standard: πδ(ω) - j PV(1/ω). Here we show as πδ(ω) - i*PV(1/ω).
            X = simplify(pi*DiracDelta(omega) - I*sign(omega)*0)  # placeholder to keep simplify stable
            X = pi*DiracDelta(omega) - I* (1/omega)
            steps = [
                r"x(t)=u(t)=\mathrm{Heaviside}(t)",
                r"\int_0^{\infty}e^{-j\omega t}dt=\lim_{\varepsilon\to0^+}\frac{1}{\varepsilon+j\omega}=\pi\delta(\omega)-j\,\mathrm{PV}\frac{1}{\omega}",
            ]
            if a_shift != 0:
                X = simplify(exp(-I*omega*a_shift) * (pi*DiracDelta(omega) - I*(1/omega)))
                steps.append(r"u(t-a)\Rightarrow e^{-j\omega a}U(\omega)")
            else:
                X = pi*DiracDelta(omega) - I*(1/omega)
            steps.append(r"X(\omega)=" + latex(X))
            return "distribution_form", True, X, steps, "", None

    # 0.8) exp(a*t)*Heaviside(t) (Laplace-type)
    if isinstance(f, Mul) and any((getattr(arg,'func',None)==Heaviside) for arg in f.args):
        h = next(arg for arg in f.args if getattr(arg,'func',None)==Heaviside)
        g = simplify(f / h)
        if g.func == exp and len(g.args)==1:
            lin = _as_linear_in_t(g.args[0])
            if lin is not None:
                acoef, b = lin  # exponent = acoef*t + b
                # x(t)=e^{a t + b}u(t) -> e^{b} /(jω - a)
                X = simplify(exp(b) / (I*omega - acoef))
                cond = r"\Re(a)<0" if acoef.free_symbols else ("" )
                steps = [
                    r"x(t)=e^{a t}u(t)",
                    r"X(\omega)=\int_0^{\infty}e^{(a-j\omega)t}dt=\frac{1}{j\omega-a}\quad(\Re(a)<0)",
                    r"X(\omega)=" + latex(X),
                    ]
                return "closed_form", True, X, steps, (r"\text{Requires }\Re(a)<0" if cond else ""), None

    # 0.9) exp(-a*Abs(t)) (common integrable)
    if f.func == exp and len(f.args)==1 and f.args[0].has(Abs(t)):
        arg = f.args[0]
        # match -a*Abs(t)
        a_sym = symbols('a_sym', real=True, positive=True)
        m = arg.match(-a_sym*Abs(t))
        if m and a_sym in m:
            a_val = m[a_sym]
            X = simplify(2*a_val/(a_val**2 + omega**2))
            steps = [
                r"x(t)=e^{-a|t|}\ (a>0)",
                r"X(\omega)=\int_{-\infty}^{\infty}e^{-a|t|}e^{-j\omega t}dt=\frac{2a}{a^2+\omega^2}",
                r"X(\omega)=" + latex(X),
                ]
            return "closed_form", True, X, steps, r"a>0", None

    # 0.10) Pure tone exp(I*ω0*t + I*φ)
    if f.func == exp and len(f.args)==1:
        inside = simplify(f.args[0]/I)
        if not inside.has(I):
            lin = _as_linear_in_t(inside)
            if lin is not None:
                w0, phi = lin
                X = simplify(2*pi*exp(I*phi)*DiracDelta(omega - w0))
                steps = [
                    r"x(t)=e^{j(\omega_0 t+\phi)}",
                    r"X(\omega)=\int e^{-j(\omega-\omega_0)t}dt\,e^{j\phi}=2\pi e^{j\phi}\delta(\omega-\omega_0)",
                    r"X(\omega)=" + latex(X),
                    ]
                return "distribution_form", True, X, steps, "", None

    # 0.11) t^n * exp(I*ω0*t) (frequency differentiation)
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
            X = simplify(coeff * 2*pi * (I**tpow) * diff(DiracDelta(omega-exp_factor), omega, tpow))
            steps = [
                r"x(t)=t^n e^{j\omega_0 t}",
                r"\mathcal{F}\{e^{j\omega_0 t}\}=2\pi\delta(\omega-\omega_0)",
                r"\mathcal{F}\{t\,x(t)\}=j\,\frac{d}{d\omega}X(\omega)\Rightarrow\mathcal{F}\{t^n x(t)\}=j^n\frac{d^n}{d\omega^n}X(\omega)",
                r"\Rightarrow\;X(\omega)=2\pi j^n\,\delta^{(n)}(\omega-\omega_0)",
                r"X(\omega)=" + latex(X),
                ]
            return "distribution_form", True, X, steps, "", None


    # 1) DiracDelta basics
    if f == DiracDelta(t):
        X = 1
        steps = [
            r"x(t)=\delta(t)",
            r"X(\omega)=\int_{-\infty}^{\infty}\delta(t)e^{-j\omega t}dt=e^{-j\omega\cdot 0}=1",
            r"X(\omega)=1",
        ]
        return "distribution_form", True, X, steps, "", None


    # 2) PV rational distributions: 1/(t+a), 1/(t+a)^2
    # Allow an overall constant factor c: F{c*g(t)} = c*G(ω)
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
        X = simplify(c0 * (-I*pi*sign(omega) * exp(I*omega*a1)))
        steps = [
            r"\text{(Distribution)}\;\mathcal{F}\{\mathrm{PV}\tfrac{1}{t}\}=-i\pi\,\mathrm{sign}(\omega)",
            r"\text{Time shift: }\mathcal{F}\{g(t-t_0)\}=e^{-i\omega t_0}G(\omega)",
        ]
        if c0 != 1:
            steps.append(r"\text{Linearity: }\mathcal{F}\{c\,g(t)\}=c\,G(\omega),\; c=" + latex(c0))
        steps += [
            r"\Rightarrow\;X(\omega)=c\,e^{i\omega a}\left(-i\pi\,\mathrm{sign}(\omega)\right)",
            r"X(\omega)=" + latex(X),
            ]
        return "distribution_form", True, X, steps, "", None

    a2 = _match_shifted_power(f0, 2)
    if a2 is not None:
        X = simplify(c0 * (-pi*omega*sign(omega) * exp(I*omega*a2)))
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
        X_sum = simplify(X_sum)

        if win is not None:
            a, b = win
            steps = [
                r"\textbf{Step 1: Identify a finite-duration signal}",
                r"x(t)=" + latex(f),
                r"\textbf{Step 2: Determine the nonzero interval}",
                r"x(t)=1\;\;\text{for }t\in[" + latex(a) + "," + latex(b) + r"],\;\;0\text{ otherwise}",
                r"\textbf{Step 3: Write the Fourier transform integral}",
                r"X(ω)=\int_{" + latex(a) + r"}^{" + latex(b) + r"} e^{-iω t}\,dt",
                r"\textbf{Step 4: Evaluate the integral}",
                r"X(ω)=\frac{e^{-iω " + latex(a) + r"}-e^{-iω " + latex(b) + r"}}{iω}\quad(\text{with distributional interpretation at }ω=0)",
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
            formG, okG, G, G_steps, condG, errG = _derive_with_properties(rest)
            X = simplify(coeff * G, doit=False)
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


@app.post("/fourier", response_model=FourierResponse)
def fourier(req: FourierRequest):
    raw = (req.expression or "").strip()
    if not raw:
        return FourierResponse(
            build_id=BUILD_ID,
            ok=False,
            input_latex="",
            result_latex="",
            steps_latex=[r"\text{Empty input.}"],
            error="Empty input",
            form="error",
            conditions_latex="",
            method="unknown",

        )

    # Convolution: only '·'
    conv = _split_convolution_top_level(raw)
    if conv is not None:
        left_s, right_s = conv
        try:
            f = _parse_sympy(left_s)
            g = _parse_sympy(right_s)
        except Exception as e:
            msg = str(e).replace("\\", "/")
            return FourierResponse(
                build_id=BUILD_ID,
                ok=False,
                input_latex="",
                result_latex="",
                steps_latex=[r"\text{Parser error: " + msg + r"}"],
                error=str(e),
                form="error",
                conditions_latex="",
                method="unknown",

            )

        # Convolution theorem (engineering)
        # IMPORTANT: compute each factor using the SAME rule-first pipeline
        # as the non-convolution path. Otherwise, sin/cos/pure-tones etc.
        # fall back to the raw definition integral and SymPy may return 0.
        F_form, okF, F, F_steps, _, _ = _derive_with_properties(f)
        G_form, okG, G, G_steps, _, _ = _derive_with_properties(g)

        steps = []
        steps.append(r"x(t)=(f\star g)(t)=\int_{-\infty}^{\infty}f(\tau)g(t-\tau)\,d\tau")
        steps.append(r"\Rightarrow\;X(\omega)=F(\omega)\,G(\omega)")
        steps.append(r"\text{Compute }F(\omega)\text{ (rule-first):}")
        steps.extend(F_steps)
        steps.append(r"F(\omega)=" + latex(F))
        steps.append(r"\text{Compute }G(\omega)\text{ (rule-first):}")
        steps.extend(G_steps)
        steps.append(r"G(\omega)=" + latex(G))
        X = simplify(F * G) if (okF and okG) else (F * G)
        steps.append(r"\Rightarrow\;X(\omega)=(" + latex(F) + r")(" + latex(G) + r")")

        return FourierResponse(
            build_id=BUILD_ID,
            ok=True,
            input_latex=latex(f) + r"\cdot " + latex(g),
            result_latex=(_format_pv_reciprocal_result_latex(f) or latex(X)),
            steps_latex=steps,
            error=None,
            form="closed_form" if (okF and okG) else "integral_form",
            conditions_latex="",
            method="unknown",

        )

    # Normal case
    try:
        f = _parse_sympy(raw)
    except Exception as e:
        msg = str(e).replace("\\", "/")
        return FourierResponse(
            build_id=BUILD_ID,
            ok=False,
            input_latex="",
            result_latex="",
            steps_latex=[r"\text{Parser error: " + msg + r"}"],
            error=str(e),
            form="error",
            conditions_latex="",
        )

    form, ok, X, steps, conditions, err = _derive_with_properties(f)
    # --- method probe (no change to math logic) ---
    method = "unknown"
    if form == "distribution_form":
        method = "distribution"
    elif form == "convolution":
        method = "convolution"
    elif form == "closed_form":
        method = "closed_form"
    elif form == "fallback":
        method = "fallback"

    if steps:
        s0 = steps[0]
        if "Convolution" in s0 or "convolution" in s0:
            method = "convolution_rule"
        elif "Distribution rule" in s0 or "(Distribution)" in s0:
            method = "distribution_rule"
        elif "Direct integral" in s0 or "integral" in s0:
            method = "direct_integral"


    legacy_ok = (form in {"closed_form", "distribution_form"})


    return FourierResponse(
        build_id=BUILD_ID,
        ok=legacy_ok,
        input_latex=latex(f),
        result_latex=(_format_pv_reciprocal_result_latex(f) or latex(X)),
        steps_latex=steps,
        error=err,
        form=form,
        conditions_latex=conditions,
        method="unknown",

    )



# ===== Extra rules: modulated step, B-spline, damped oscillation =====

def _rule_modulated_step(f):
    if isinstance(f, Mul) and f.has(Heaviside):
        for h in f.atoms(Heaviside):
            if len(h.args)==1:
                a = symbols("a", real=True)
                m = h.args[0].match(t-a)
                if m and a in m:
                    a0 = m[a]
                    rest = simplify(f/h)
                    w0 = symbols("w0", real=True)
                    mm = rest.match(exp(I*w0*t))
                    if mm and w0 in mm:
                        ww = mm[w0]
                        X = exp(-I*omega*a0)/(I*(omega-ww))
                        steps=[
                            r"x(t)=e^{j\omega_0 t}u(t-a)",
                            r"X(\omega)=e^{-j\omega a}\frac{1}{j(\omega-\omega_0)}",
                            r"X(\omega)="+latex(X)
                        ]
                        return ("distribution_form",True,X,steps,"",None)
    return None


def _rule_bspline2(f):
    if str(f).replace(" ","") in ["Heaviside(t)•Heaviside(t)","Heaviside(t).Heaviside(t)"]:
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



def _rule_poly_times_step_distribution(f):
    """
    Explicit distribution for t^n * Heaviside(t):
      F{t^n u(t)} = π j^n δ^{(n)}(ω) + (-1)^n j^{n-1} n! PV(1/ω^{n+1})
    """
    if not (isinstance(f, Mul) and f.has(Heaviside(t))):
        return None
    # exact factor Heaviside(t) must be present
    if Heaviside(t) not in f.args:
        return None
    g = simplify(f / Heaviside(t))
    if g == 1:
        n = 0
    elif g == t:
        n = 1
    elif g.is_Pow and g.base == t and g.exp.is_Integer and int(g.exp) >= 0:
        n = int(g.exp)
    else:
        return None

    X = simplify(
        pi*(I**n)*Derivative(DiracDelta(omega), (omega, n))
        + ((-1)**n)*(I**(n-1))*factorial(n)*PV(1/(omega**(n+1)))
    )
    steps = [
        r"\textbf{Method: Distribution rule (polynomial × step)}",
        r"x(t)=t^{" + str(n) + r"}u(t),\;u(t)=\mathrm{Heaviside}(t)",
        r"\mathcal{F}\{u(t)\}=\pi\delta(\omega)-j\,\mathrm{PV}\frac{1}{\omega}",
        r"\mathcal{F}\{t^n x(t)\}=j^n\frac{d^n}{d\omega^n}X(\omega)",
        r"\Rightarrow X(\omega)=\pi j^n\delta^{(n)}(\omega)+(-1)^n j^{\,n-1}n!\,\mathrm{PV}\frac{1}{\omega^{n+1}}",
        r"X(\omega)=" + latex(X),
        ]
    return ("distribution_form", True, X, steps, "", None)


def _rule_trig_times_step_distribution(f):
    """
    Distribution-first for sin(a t + b)u(t), cos(a t + b)u(t).
    Use Euler form + u(t) base pair shifted in frequency.
    """
    if not (isinstance(f, Mul) and f.has(Heaviside)):
        return None

    # Accept any Heaviside with first argument == t (ignore value-at-0 parameter)
    h_t = None
    for h in f.atoms(Heaviside):
        if len(h.args) >= 1 and simplify(h.args[0] - t) == 0:
            h_t = h
            break
    if h_t is None:
        return None

    g = simplify(f / h_t)

    # match sin(a*t+b) or cos(a*t+b) with linear phase
    a = Wild("a", exclude=[t])
    b = Wild("b", exclude=[t])
    mm_sin = None
    mm_cos = None

    if g.func == sin:
        mm_sin = g.args[0].match(a*t + b)
        trig = "sin"
    elif g.func == cos:
        mm_cos = g.args[0].match(a*t + b)
        trig = "cos"
    else:
        return None

    mm = mm_sin if mm_sin is not None else mm_cos
    if not (mm and a in mm and b in mm):
        return None

    aa = simplify(mm[a])
    bb = simplify(mm[b])

    # helper: F{e^{jω0 t}u(t)} = πδ(ω-ω0) - j PV(1/(ω-ω0))
    def _Ushift(w0):
        return pi*DiracDelta(omega - w0) - I*PV(1/(omega - w0))

    if trig == "sin":
        X = simplify((exp(I*bb)*_Ushift(aa) - exp(-I*bb)*_Ushift(-aa)) / (2*I))
        steps = [
            r"\textbf{Method: Distribution rule (trig × step)}",
            r"x(t)=\sin\!\left(" + latex(aa) + r"t+" + latex(bb) + r"\right)u(t)",
            r"\sin(\theta)=\frac{e^{j\theta}-e^{-j\theta}}{2j}",
            r"\mathcal{F}\{e^{j\omega_0 t}u(t)\}=\pi\delta(\omega-\omega_0)-j\,\mathrm{PV}\frac{1}{\omega-\omega_0}",
            r"\text{Apply linearity and frequency shift } \omega_0=\pm " + latex(aa),
            r"X(\omega)=" + latex(X),
            ]
    else:
        X = simplify((exp(I*bb)*_Ushift(aa) + exp(-I*bb)*_Ushift(-aa)) / 2)
        steps = [
            r"\textbf{Method: Distribution rule (trig × step)}",
            r"x(t)=\cos\!\left(" + latex(aa) + r"t+" + latex(bb) + r"\right)u(t)",
            r"\cos(\theta)=\frac{e^{j\theta}+e^{-j\theta}}{2}",
            r"\mathcal{F}\{e^{j\omega_0 t}u(t)\}=\pi\delta(\omega-\omega_0)-j\,\mathrm{PV}\frac{1}{\omega-\omega_0}",
            r"\text{Apply linearity and frequency shift } \omega_0=\pm " + latex(aa),
            r"X(\omega)=" + latex(X),
            ]

    return ("distribution_form", True, X, steps, "", None)




def _rule_poly_distribution(f):
    """
    Distribution for pure polynomials t^n (no step):
      F{t^n} = 2π i^n δ^{(n)}(ω)
    """
    if f == 1:
        n = 0
    elif f == t:
        n = 1
    elif f.is_Pow and f.base == t and f.exp.is_Integer and int(f.exp) >= 0:
        n = int(f.exp)
    else:
        return None

    X = simplify(2*pi*(I**n)*Derivative(DiracDelta(omega), (omega, n)))
    steps = [
        r"\textbf{Method: Distribution rule (polynomial)}",
        r"x(t)=t^{" + str(n) + r"}",
        r"\mathcal{F}\{1\}=2\pi\delta(\omega)",
        r"\mathcal{F}\{t^n x(t)\}=j^n\frac{d^n}{d\omega^n}X(\omega)",
        r"\Rightarrow X(\omega)=2\pi j^n\delta^{(n)}(\omega)",
        r"X(\omega)=" + latex(X),
        ]
    return ("distribution_form", True, X, steps, "", None)


def _rule_pv_reciprocal(f):
    """
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

            return ("distribution_form", True, X, steps, "", None)

        # General linear denominator: a*t + b
        lin = _as_linear_in_t(den)
        if lin is None:
            return None
        a, b = lin
        if simplify(a) == 0:
            return None

        shift = simplify(b / a)
        X = simplify((-I*pi/a) * exp(I*omega*shift) * sign(omega))

        steps = [
            r"\textbf{Method: Distribution rule (principal value)}",
            r"\mathrm{PV}\!\int_{-\infty}^{\infty}\frac{e^{-i\omega t}}{t}\,dt=-i\pi\,\mathrm{sign}(\omega)",
            r"\frac{1}{a t+b}=\frac{1}{a}\,\frac{1}{t+\frac{b}{a}}",
            r"\Rightarrow\;X(\omega)=-\frac{i\pi}{a}\,e^{i\omega\frac{b}{a}}\,\mathrm{sign}(\omega)",
            r"X(\omega)=" + latex(X),
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
            return r"e^{i\omega %s}\left(-i\pi\,\mathrm{sign}(\omega)\right)" % latex(a0)

        lin = _as_linear_in_t(den)
        if lin is None:
            return None
        a, b = lin
        if simplify(a) == 0:
            return None

        shift = simplify(b / a)
        if simplify(shift) == 0:
            return r"-\frac{i\pi}{%s}\,\mathrm{sign}(\omega)" % latex(a)
        return r"-\frac{i\pi}{%s}\,e^{i\omega %s}\,\mathrm{sign}(\omega)" % (latex(a), latex(shift))
    except Exception:
        return None
