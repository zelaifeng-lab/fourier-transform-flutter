from __future__ import annotations

import re as _regex

import pytest
from fastapi.testclient import TestClient

from backend import app
from cases import FORBIDDEN_TOKENS, FUTURE_CASES, IMPLEMENTED_CASES, KNOWN_GAP_CASES, FourierCase


client = TestClient(app)


def _post_fourier(expression: str) -> dict:
    response = client.post("/fourier", json={"expression": expression})
    assert response.status_code == 200
    return response.json()


def _joined_steps(payload: dict) -> str:
    return "\n".join(payload.get("steps_latex") or [])


def _assert_no_forbidden_tokens(payload: dict) -> None:
    text = "\n".join(
        [
            str(payload.get("result_latex") or ""),
            _joined_steps(payload),
        ]
    )
    for token in FORBIDDEN_TOKENS:
        assert token not in text


def _normalize_latex(text: str) -> str:
    return "".join(text.split())


@pytest.mark.parametrize("case", IMPLEMENTED_CASES, ids=lambda case: case.id)
def test_implemented_fourier_cases_return_teaching_results(case: FourierCase) -> None:
    _assert_teaching_result(case)


def test_known_gap_cases_are_empty() -> None:
    assert KNOWN_GAP_CASES == ()


TEACHING_STEP_SNAPSHOTS = (
    (
        "1",
        (
            r"\textbf{Step 1: Use the constant transform pair}",
            r"\mathcal{F}\{C\}=2\pi C\,\delta(\omega)",
            r"\textbf{Final Result}",
        ),
    ),
    (
        "delta(t-3)",
        (
            r"\textbf{Step 1: Identify the impulse location}",
            r"t_0=3",
            r"\textbf{Step 2: Use the sifting property of the delta function}",
            r"e^{-j\omega t_0}",
        ),
    ),
    (
        "u(t)",
        (
            r"\textbf{Step 1: Identify the signal as a shifted unit step}",
            r"\mathcal{F}\{u(t)\}=\pi\delta(\omega)-j\,\mathrm{PV}\!\left(\frac{1}{\omega}\right)",
            r"not absolutely integrable",
            r"Apply the time-shift property",
        ),
    ),
    (
        "t*u(t)",
        (
            r"\textbf{Step 1: Identify a polynomial multiplied by the unit step}",
            r"p(t)=t",
            r"n\in\{1\}",
            r"Use frequency differentiation",
            r"Apply polynomial linearity term by term",
        ),
    ),
    (
        "sin(t)*u(t)",
        (
            r"\textbf{Step 1: Identify a sinusoid multiplied by a shifted unit step}",
            r"a=1,\quad b=0,\quad c=0",
            r"Move the step edge to the origin",
            r"a c+b=0",
            r"Use the modulated unit-step transform",
            r"PV appears because each modulated step remains one-sided",
        ),
    ),
    (
        "t",
        (
            r"\textbf{Step 1: Identify the polynomial degree}",
            r"\text{Here }n=1",
            r"Use frequency differentiation",
        ),
    ),
    (
        "exp(-2*t)*u(t)",
        (
            r"\textbf{Step 1: Use the unit step to set the integration range}",
            r"a=-2,\quad b=0",
            r"\textbf{Step 2: Combine exponential terms}",
            r"\textbf{Step 3: Evaluate the convergent one-sided exponential integral}",
        ),
    ),
    (
        "sin(3*t+2)",
        (
            r"\textbf{Step 1: Identify the sinusoidal phase}",
            r"a=3,\quad b=2",
            r"\text{Here the two exponential frequencies are }\omega_0=a\text{ and }\omega_0=-a.",
            r"Combine delta functions",
        ),
    ),
    (
        "frac(1,3*t-2)",
        (
            r"\textbf{Step 1: Interpret the reciprocal as a principal-value distribution}",
            r"PV means the singularity is handled by symmetric limiting around the pole.",
            r"Rewrite the denominator into shifted form",
            r"a=3,\;b=-2",
        ),
    ),
    (
        "frac(1,t^2+1)",
        (
            r"\textbf{Step 1: Identify a standard rational transform pair}",
            r"Rewrite the denominator as }t^2+\alpha^2",
            r"\mathcal{F}\left\{\frac{1}{t^2+\alpha^2}\right\}",
            r"\frac{\pi}{\alpha}e^{-\alpha|\omega|}",
        ),
    ),
    (
        "u(t-2)-u(t-5)",
        (
            r"\textbf{Step 1: Identify a finite-duration signal}",
            r"x(t)=1\;\;\text{for }t\in[2,5]",
            r"\textbf{Step 3: Replace the full integral by the interval integral}",
            r"\textbf{Step 4: Evaluate the exponential integral}",
        ),
    ),
)


@pytest.mark.parametrize("expression, fragments", TEACHING_STEP_SNAPSHOTS)
def test_representative_teaching_step_snapshots(expression: str, fragments: tuple[str, ...]) -> None:
    payload = _post_fourier(expression)
    steps_latex = _joined_steps(payload)

    cursor = 0
    for fragment in fragments:
        found_at = steps_latex.find(fragment, cursor)
        assert found_at >= 0, f"{expression} steps should contain {fragment!r} after offset {cursor}:\n{steps_latex}"
        cursor = found_at + len(fragment)

    _assert_no_forbidden_tokens(payload)


@pytest.mark.parametrize(
    "expression",
    (
        "u(t)\u2022u(t)",
        "u(t)\u00b7u(t)",
        "u(t)\u2219u(t)",
        "u(t)\u22c6u(t)",
        "u(t)\u2217u(t)",
        r"u(t)\bullet u(t)",
        "delta(t-1)\u2022u(t)",
    ),
)
def test_convolution_operator_variants_use_convolution_rule(expression: str) -> None:
    payload = _post_fourier(expression)

    assert payload["ok"] is True
    assert payload["method"] == "convolution_rule"
    assert payload["result_latex"]

    steps_latex = _joined_steps(payload)
    assert r"(f\star g)(t)" in steps_latex
    assert r"X(\omega)=F(\omega)\,G(\omega)" in steps_latex
    assert r"F(\omega)=" in steps_latex
    assert r"G(\omega)=" in steps_latex

    _assert_no_forbidden_tokens(payload)


def _assert_teaching_result(case: FourierCase) -> None:
    payload = _post_fourier(case.expression)

    assert payload["ok"] is True
    assert payload["result_latex"]
    assert payload["steps_latex"], f"{case.id} should include teaching steps"

    result_latex = payload["result_latex"]
    if case.expected_result_latex is not None:
        assert _normalize_latex(result_latex) == _normalize_latex(case.expected_result_latex), (
            f"{case.id} result mismatch:\n"
            f"expected: {case.expected_result_latex}\n"
            f"actual:   {result_latex}"
        )

    for token in case.expected_result_contains:
        assert token in result_latex, f"{case.id} result should contain {token!r}: {result_latex}"

    steps_latex = _joined_steps(payload)
    assert "Start from the Fourier transform definition" not in steps_latex
    assert "Final Result" in steps_latex, f"{case.id} steps should include a final result section"
    assert r"\theta\left" not in steps_latex, f"{case.id} steps should display unit steps as u(t)"
    assert "e^{-i" not in steps_latex and "e^{i" not in steps_latex, f"{case.id} steps should use engineering j notation"
    assert not _regex.search(r"(?<![A-Za-z\\])i\s*\\omega", steps_latex), f"{case.id} steps should use engineering j notation"
    if "PV" in result_latex or "rational_pv" in case.category:
        assert (
            "PV" in steps_latex or "principal value" in steps_latex
        ), f"{case.id} steps should explain the principal-value part"

    assert r"\operatorname{PV}" not in result_latex
    assert r"\mathrm{PV}\frac" not in result_latex
    assert r"\operatorname{sign}" not in result_latex
    assert r"\left|{\omega}\right|" not in result_latex
    assert r"\delta^{\left(" not in result_latex
    assert r"\mathrm{PV}\frac" not in steps_latex
    assert "1t+" not in steps_latex
    assert "+-" not in steps_latex
    assert r"\pj" not in steps_latex
    assert not _regex.search(r"e\^\{[+-]j\\omega\s+[-0-9]", result_latex + steps_latex)
    assert not _regex.search(r"(?<![A-Za-z])i(?![A-Za-z])", result_latex), (
        f"{case.id} result should use engineering j notation: {result_latex}"
    )

    for token in case.expected_steps_contains:
        if token in steps_latex:
            continue
        step_match = _regex.match(r"\\textbf\{Step\s+\d+:\s*(.*?)\}$", token)
        if step_match and step_match.group(1) in steps_latex:
            continue
        assert False, f"{case.id} steps should contain {token!r}: {steps_latex}"

    _assert_no_forbidden_tokens(payload)


@pytest.mark.parametrize("case", FUTURE_CASES, ids=lambda case: case.id)
def test_future_cases_are_documented_but_not_active_regressions(case: FourierCase) -> None:
    assert case.implemented is False
