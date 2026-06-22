from __future__ import annotations

from dataclasses import dataclass


FORBIDDEN_TOKENS = (
    "Piecewise",
    "RootSum",
    "arg",
    "polar_lift",
    "meijerg",
    "\\begin{cases}",
    "Method:",
    "_rule_",
    "matcher",
    "debug",
    "srepr",
)


@dataclass(frozen=True)
class FourierCase:
    id: str
    expression: str
    category: str
    expected_result_contains: tuple[str, ...] = ()
    expected_steps_contains: tuple[str, ...] = ()
    implemented: bool = True
    strict_ok: bool = True
    expected_result_latex: str | None = None


IMPLEMENTED_CASES: tuple[FourierCase, ...] = (
    FourierCase("B01", "1", "basic", ("2", "pi", "delta"), expected_result_latex=r"2 \pi \delta\left(\omega\right)"),
    FourierCase("B02", "5", "basic", ("10", "pi", "delta"), expected_result_latex=r"10 \pi \delta\left(\omega\right)"),
    FourierCase("B03", "delta(t)", "basic", ("1",), expected_result_latex=r"1"),
    FourierCase("B04", "delta(t-3)", "basic", ("e", "3", "omega"), expected_result_latex=r"e^{- 3 j \omega}"),
    FourierCase("B05", "delta(t+2)", "basic", ("e", "2", "omega"), expected_result_latex=r"e^{2 j \omega}"),
    FourierCase("B06", "u(t)", "basic", ("pi", "delta", "omega"), expected_result_latex=r"\pi \delta\left(\omega\right) - \frac{j}{\omega}"),
    FourierCase("B07", "u(t-2)", "basic", ("e", "2", "omega", "delta"), expected_result_latex=r"\left(\pi \delta\left(\omega\right) - \frac{j}{\omega}\right) e^{- 2 j \omega}"),
    FourierCase("B08", "u(t+1)", "basic", ("e", "omega", "delta"), expected_result_latex=r"\left(\pi \delta\left(\omega\right) - \frac{j}{\omega}\right) e^{j \omega}"),
    FourierCase("B09", "sign(t)", "sign_distribution", ("PV", "omega"), expected_result_latex=r"- 2 j \mathrm{PV}\frac{1}{\omega}"),
    FourierCase("B10", "sign(t-2)", "sign_distribution", ("PV", "omega", "e"), expected_result_latex=r"- 2 j e^{- 2 j \omega} \mathrm{PV}\frac{1}{\omega}"),
    FourierCase("B11", "sign(-3*(t+1))", "sign_distribution", ("PV", "omega", "e"), expected_result_latex=r"2 j e^{j \omega} \mathrm{PV}\frac{1}{\omega}"),
    FourierCase("P01", "t", "polynomial", ("delta", "omega"), expected_result_latex=r"2 j \pi \delta^{(1)}(\omega)"),
    FourierCase("P02", "t^2", "polynomial", ("delta", "omega"), expected_result_latex=r"- 2 \pi \delta^{(2)}(\omega)"),
    FourierCase("P03", "t^3", "polynomial", ("delta", "omega"), expected_result_latex=r"- 2 j \pi \delta^{(3)}(\omega)"),
    FourierCase("P04", "3*t", "polynomial", ("delta", "omega"), expected_result_latex=r"6 j \pi \delta^{(1)}(\omega)"),
    FourierCase("P05", "2*t^2+3*t+1", "polynomial", ("delta", "omega"), expected_result_latex=r"2 \pi \delta\left(\omega\right) + 6 j \pi \delta^{(1)}(\omega) - 4 \pi \delta^{(2)}(\omega)"),
    FourierCase("P06", "t^4", "polynomial", ("delta", "omega"), expected_result_latex=r"2 \pi \delta^{(4)}(\omega)"),
    FourierCase("PU01", "t*u(t)", "polynomial_step", ("PV", "omega"), expected_result_latex=r"j \pi \delta^{(1)}(\omega) - \mathrm{PV}\frac{1}{\omega^{2}}"),
    FourierCase("PU02", "t^2*u(t)", "polynomial_step", ("PV", "omega"), expected_result_latex=r"- \pi \delta^{(2)}(\omega) + 2 j \mathrm{PV}\frac{1}{\omega^{3}}"),
    FourierCase("PU03", "t^3*u(t)", "polynomial_step", ("PV", "omega"), expected_result_latex=r"- j \pi \delta^{(3)}(\omega) + 6 \mathrm{PV}\frac{1}{\omega^{4}}"),
    FourierCase("PU04", "2*t*u(t)", "polynomial_step", ("PV", "omega"), expected_result_latex=r"2 j \pi \delta^{(1)}(\omega) - 2 \mathrm{PV}\frac{1}{\omega^{2}}"),
    FourierCase("PU05", "(t^2+2*t+1)*u(t)", "polynomial_step", ("PV", "omega"), expected_result_latex=r"\left(\pi \delta\left(\omega\right) - \frac{j}{\omega}\right) - \left(\pi \delta^{(2)}(\omega) - 2 j \mathrm{PV}\frac{1}{\omega^{3}}\right) + \left(2 j \pi \delta^{(1)}(\omega) - 2 \mathrm{PV}\frac{1}{\omega^{2}}\right)"),
    FourierCase(
        "PU06",
        "(t-2)*u(t-2)",
        "shifted_polynomial_step",
        ("PV", "delta", "omega", "e"),
        expected_steps_contains=(
            r"\textbf{Step 2: Identify a shifted polynomial multiplied by a shifted unit step}",
            r"c=2",
            r"p(t)\rightarrow p(s+c)=t",
            r"\mathcal{F}\{y(t-c)\}=e^{-j\omega c}Y(\omega)",
        ),
    ),
    FourierCase(
        "PU07",
        "t*u(t-2)",
        "shifted_polynomial_step",
        ("PV", "delta", "omega", "e"),
        expected_steps_contains=(
            r"\textbf{Step 2: Identify a shifted polynomial multiplied by a shifted unit step}",
            r"c=2",
            r"p(t)\rightarrow p(s+c)=t + 2",
            r"n\in\{0,1\}",
        ),
    ),
    FourierCase(
        "PU08",
        "(t+3)^2*u(t+3)",
        "shifted_polynomial_step",
        ("PV", "delta", "omega", "e"),
        expected_steps_contains=(
            r"\textbf{Step 2: Identify a shifted polynomial multiplied by a shifted unit step}",
            r"c=-3",
            r"p(t)\rightarrow p(s+c)=t^{2}",
            r"n\in\{2\}",
        ),
    ),
    FourierCase("E01", "exp(I*3*t)", "exponential", ("delta", "omega", "3"), expected_result_latex=r"2 \pi \delta\left(\omega - 3\right)"),
    FourierCase("E02", "exp(-I*2*t)", "exponential", ("delta", "omega", "2"), expected_result_latex=r"2 \pi \delta\left(\omega + 2\right)"),
    FourierCase("E03", "exp(I*(5*t+2))", "exponential", ("delta", "omega", "5"), expected_result_latex=r"2 \pi e^{2 j} \delta\left(\omega - 5\right)"),
    FourierCase("E04", "exp(-t)*u(t)", "exponential", ("omega",), expected_result_latex=r"\frac{1}{j \omega + 1}"),
    FourierCase("E05", "exp(-2*t)*u(t)", "exponential", ("omega", "2"), expected_result_latex=r"\frac{1}{j \omega + 2}"),
    FourierCase("E06", "exp(-3*abs(t))", "exponential", ("omega",), expected_result_latex=r"\frac{6}{\omega^{2} + 9}"),
    FourierCase(
        "E09",
        "exp(I*2*t)*u(t-3)",
        "exponential_modulated_step",
        ("PV", "delta", "omega - 2"),
        expected_steps_contains=(
            r"\textbf{Step 2: Identify a modulated unit-step signal}",
            r"\omega_0=2,\quad \phi=0,\quad c=3",
            r"\omega_0 c+\phi=6",
            r"\mathcal{F}\{y(t-c)\}=e^{-j\omega c}Y(\omega)",
        ),
    ),
    FourierCase(
        "E10",
        "exp(I*(2*t+1))*u(t+3)",
        "exponential_modulated_step",
        ("PV", "delta", "omega - 2"),
        expected_steps_contains=(
            r"\textbf{Step 2: Identify a modulated unit-step signal}",
            r"\omega_0=2,\quad \phi=1,\quad c=-3",
            r"\omega_0 c+\phi=-5",
            r"\mathcal{F}\{y(t-c)\}=e^{-j\omega c}Y(\omega)",
        ),
    ),
    FourierCase(
        "E11",
        "3*exp(-I*t)*u(t-2)",
        "exponential_modulated_step",
        ("PV", "delta", "omega + 1"),
        expected_steps_contains=(
            r"\textbf{Step 2: Identify a modulated unit-step signal}",
            r"\omega_0=-1,\quad \phi=0,\quad c=2",
            r"\omega_0 c+\phi=-2",
            r"\text{Apply the constant factor }3\text{ by linearity.}",
        ),
    ),
    FourierCase("T01", "sin(t)", "trigonometric", ("delta", "omega"), expected_result_latex=r"j \pi \left(- \delta\left(\omega - 1\right) + \delta\left(\omega + 1\right)\right)"),
    FourierCase("T02", "cos(t)", "trigonometric", ("delta", "omega"), expected_result_latex=r"\pi \left(\delta\left(\omega - 1\right) + \delta\left(\omega + 1\right)\right)"),
    FourierCase("T03", "sin(3*t)", "trigonometric", ("delta", "omega", "3"), expected_result_latex=r"j \pi \left(- \delta\left(\omega - 3\right) + \delta\left(\omega + 3\right)\right)"),
    FourierCase("T04", "cos(4*t)", "trigonometric", ("delta", "omega", "4"), expected_result_latex=r"\pi \left(\delta\left(\omega - 4\right) + \delta\left(\omega + 4\right)\right)"),
    FourierCase("T05", "sin(3*t+2)", "trigonometric", ("delta", "omega", "3"), expected_result_latex=r"j \pi \left(- e^{4 j} \delta\left(\omega - 3\right) + \delta\left(\omega + 3\right)\right) e^{- 2 j}"),
    FourierCase("T06", "cos(4*t-1)", "trigonometric", ("delta", "omega", "4"), expected_result_latex=r"\pi \left(\delta\left(\omega - 4\right) + e^{2 j} \delta\left(\omega + 4\right)\right) e^{- j}"),
    FourierCase("T07", "sin(t)+cos(2*t)", "trigonometric", ("delta", "omega"), expected_result_latex=r"\pi \left(- j \left(\delta\left(\omega - 1\right) - \delta\left(\omega + 1\right)\right) + \delta\left(\omega - 2\right) + \delta\left(\omega + 2\right)\right)"),
    FourierCase("T08", "sin(t)*u(t)", "trigonometric_step", ("PV", "omega"), expected_result_latex=r"- \frac{j \pi \delta\left(\omega - 1\right)}{2} + \frac{j \pi \delta\left(\omega + 1\right)}{2} - \frac{\mathrm{PV}\frac{1}{\omega - 1}}{2} + \frac{\mathrm{PV}\frac{1}{\omega + 1}}{2}"),
    FourierCase("T09", "cos(t)*u(t)", "trigonometric_step", ("PV", "omega"), expected_result_latex=r"\frac{\pi \delta\left(\omega - 1\right)}{2} + \frac{\pi \delta\left(\omega + 1\right)}{2} - \frac{j \mathrm{PV}\frac{1}{\omega - 1}}{2} - \frac{j \mathrm{PV}\frac{1}{\omega + 1}}{2}"),
    FourierCase(
        "T10",
        "sin(t-1)*u(t+3)",
        "trigonometric_shifted_step",
        ("PV", "delta", "omega"),
        expected_steps_contains=(
            r"\textbf{Step 3: Move the step edge to the origin}",
            r"c=-3",
            r"a c+b=-4",
            r"\mathcal{F}\{y(t-c)\}=e^{-j\omega c}Y(\omega)",
        ),
    ),
    FourierCase(
        "T11",
        "cos(2*t+1)*u(t-3)",
        "trigonometric_shifted_step",
        ("PV", "delta", "omega"),
        expected_steps_contains=(
            r"\textbf{Step 3: Move the step edge to the origin}",
            r"c=3",
            r"a c+b=7",
            r"\mathcal{F}\{y(t-c)\}=e^{-j\omega c}Y(\omega)",
        ),
    ),
    FourierCase(
        "D01",
        "(exp(I*2*t)+exp(-I*t))*u(t-3)",
        "distributive_linearity",
        ("PV", "delta", "omega - 2", "omega + 1"),
        expected_steps_contains=(
            r"\textbf{Step 2: Distribute multiplication over addition}",
            r"\textbf{Step 3: Apply linearity term by term}",
            r"e^{2 j t} u(t - 3) + e^{- j t} u(t - 3)",
        ),
    ),
    FourierCase(
        "D02",
        "2*(t*u(t)+sin(t)*u(t))",
        "distributive_linearity",
        ("PV", "delta", "omega"),
        expected_steps_contains=(
            r"\textbf{Step 2: Distribute multiplication over addition}",
            r"2 t u(t) + 2 \sin{\left(t \right)} u(t)",
            r"\textbf{Step 3: Apply linearity term by term}",
        ),
    ),
    FourierCase(
        "W08",
        "t*(u(t)-u(t-1))",
        "polynomial_window",
        ("omega", "e"),
        expected_steps_contains=(
            r"\textbf{Step 2: Identify a polynomial finite-window signal}",
            r"p(t)=t",
            r"\text{Here }a=0,\quad b=1",
            r"\textbf{Step 3: Replace the full integral by the finite interval}",
        ),
    ),
    FourierCase("R01", "frac(1,t)", "rational_pv", ("sign", "omega"), expected_result_latex=r"-j\pi\,\mathrm{sign}(\omega)"),
    FourierCase("R02", "frac(1,t+3)", "rational_pv", ("sign", "omega", "3"), expected_result_latex=r"e^{j\omega 3}\left(-j\pi\,\mathrm{sign}(\omega)\right)"),
    FourierCase("R03", "frac(1,3*t-2)", "rational_pv", ("sign", "omega"), expected_result_latex=r"-\frac{j\pi}{3}\,e^{j\omega - \frac{2}{3}}\,\mathrm{sign}(\omega)"),
    FourierCase("R04", "frac(1,t^2+1)", "rational", ("pi", "omega"), expected_result_latex=r"\pi e^{- |\omega|}"),
    FourierCase("R05", "frac(1,t^2+4)", "rational", ("pi", "omega"), expected_result_latex=r"\frac{\pi e^{- 2 |\omega|}}{2}"),
    FourierCase("R06", "frac(t,t^2+1)", "rational", ("sign", "omega"), expected_result_latex=r"- j \pi e^{- |\omega|} \mathrm{sign}(\omega)"),
    FourierCase("R07", "frac(2*t+3,t^2+6)", "rational", ("omega",), expected_result_latex=r"\pi \left(- 2 j \mathrm{sign}(\omega) + \frac{\sqrt{6}}{2}\right) e^{- \sqrt{6} |\omega|}"),
    FourierCase("R08", "frac(2*t,3*t^2+4*t-1)", "rational", ("omega",), expected_result_latex=r"- \frac{j \pi \left(7 - 2 \sqrt{7}\right) e^{\frac{j \omega \left(2 - \sqrt{7}\right)}{3}} \mathrm{sign}(\omega)}{21} - \frac{j \pi \left(2 \sqrt{7} + 7\right) e^{\frac{j \omega \left(2 + \sqrt{7}\right)}{3}} \mathrm{sign}(\omega)}{21}"),
    FourierCase("R09", "frac(t^2,t(t+1))", "rational", ("sign", "delta", "omega"), expected_result_latex=r"j \pi e^{j \omega} \mathrm{sign}(\omega) + 2 \pi \delta\left(\omega\right)"),
    FourierCase("R10", "frac(t^2(t+1),t^3(t+1))", "rational_pv", ("sign", "omega"), expected_result_latex=r"-j\pi\,\mathrm{sign}(\omega)"),
    FourierCase("R11", "frac(t^2（t+1）,t^3(t+1))", "rational_pv", ("sign", "omega"), expected_result_latex=r"-j\pi\,\mathrm{sign}(\omega)"),
    FourierCase("W01", "u(t-2)-u(t-5)", "window", ("omega",), expected_result_latex=r"- \frac{j \left(e^{- 2 j \omega} - e^{- 5 j \omega}\right)}{\omega}"),
    FourierCase("W02", "u(t+1)-u(t-1)", "window", ("omega",), expected_result_latex=r"- \frac{j \left(e^{j \omega} - e^{- j \omega}\right)}{\omega}"),
    FourierCase("W03", "rect((t-2)/3)", "rect_window", ("omega", "sin", "e"), expected_result_latex=r"\frac{2 e^{- 2 j \omega} \sin{\left(\frac{3 \omega}{2} \right)}}{\omega}"),
    FourierCase("W04", "4*rect((t+1)/2)", "rect_window", ("omega", "sin", "e"), expected_result_latex=r"\frac{8 e^{j \omega} \sin{\left(\omega \right)}}{\omega}"),
    FourierCase("W09", "tri((t-1)/2)", "tri_window", ("omega", "sin", "e"), expected_result_latex=r"\frac{2 e^{- j \omega} \sin^{2}{\left(\omega \right)}}{\omega^{2}}"),
    FourierCase("W10", "3*tri((t+2)/4)", "tri_window", ("omega", "sin", "e"), expected_result_latex=r"\frac{3 e^{2 j \omega} \sin^{2}{\left(2 \omega \right)}}{\omega^{2}}"),
)


KNOWN_GAP_CASES: tuple[FourierCase, ...] = ()


FUTURE_CASES: tuple[FourierCase, ...] = (
    FourierCase("E07", "exp(-a*t)*u(t)", "future_parameter", implemented=False),
    FourierCase("E08", "exp(-a*abs(t))", "future_parameter", implemented=False),
    FourierCase("R12", "frac(1,t^2)", "future_distribution", implemented=False),
    FourierCase("R13", "frac(1,(t+2)^2)", "future_distribution", implemented=False),
    FourierCase("W05", "frac(sin(t),pi*t)", "future_sinc", implemented=False),
    FourierCase("W06", "frac(sin(3*t),pi*t)", "future_sinc", implemented=False),
    FourierCase("W07", "frac(sin(t),t)", "future_sinc", implemented=False),
    FourierCase("F01", "exp(-t^2)", "future_gaussian", implemented=False),
    FourierCase("F02", "exp(-2*t^2)", "future_gaussian", implemented=False),
)

