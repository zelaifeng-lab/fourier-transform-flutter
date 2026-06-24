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
    FourierCase("B06", "u(t)", "basic", ("pi", "delta", "omega"), expected_result_latex=r"\pi \delta\left(\omega\right) - j \mathrm{PV}\!\left(\frac{1}{\omega}\right)"),
    FourierCase("B07", "u(t-2)", "basic", ("e", "2", "omega", "delta"), expected_result_latex=r"\left(\pi \delta\left(\omega\right) - j \mathrm{PV}\!\left(\frac{1}{\omega}\right)\right) e^{- 2 j \omega}"),
    FourierCase("B08", "u(t+1)", "basic", ("e", "omega", "delta"), expected_result_latex=r"\left(\pi \delta\left(\omega\right) - j \mathrm{PV}\!\left(\frac{1}{\omega}\right)\right) e^{j \omega}"),
    FourierCase("B09", "sign(t)", "sign_distribution", ("PV", "omega"), expected_result_latex=r"- 2 j \mathrm{PV}\!\left(\frac{1}{\omega}\right)"),
    FourierCase("B10", "sign(t-2)", "sign_distribution", ("PV", "omega", "e"), expected_result_latex=r"- 2 j e^{- 2 j \omega} \mathrm{PV}\!\left(\frac{1}{\omega}\right)"),
    FourierCase("B11", "sign(-3*(t+1))", "sign_distribution", ("PV", "omega", "e"), expected_result_latex=r"2 j e^{j \omega} \mathrm{PV}\!\left(\frac{1}{\omega}\right)"),
    FourierCase("P01", "t", "polynomial", ("delta", "omega"), expected_result_latex=r"2 j \pi \delta^{(1)}(\omega)"),
    FourierCase("P02", "t^2", "polynomial", ("delta", "omega"), expected_result_latex=r"- 2 \pi \delta^{(2)}(\omega)"),
    FourierCase("P03", "t^3", "polynomial", ("delta", "omega"), expected_result_latex=r"- 2 j \pi \delta^{(3)}(\omega)"),
    FourierCase("P04", "3*t", "polynomial", ("delta", "omega"), expected_result_latex=r"6 j \pi \delta^{(1)}(\omega)"),
    FourierCase("P05", "2*t^2+3*t+1", "polynomial", ("delta", "omega"), expected_result_latex=r"2 \pi \delta\left(\omega\right) + 6 j \pi \delta^{(1)}(\omega) - 4 \pi \delta^{(2)}(\omega)"),
    FourierCase("P06", "t^4", "polynomial", ("delta", "omega"), expected_result_latex=r"2 \pi \delta^{(4)}(\omega)"),
    FourierCase("PU01", "t*u(t)", "polynomial_step", ("PV", "omega"), expected_result_latex=r"j \pi \delta^{(1)}(\omega) - \mathrm{PV}\!\left(\frac{1}{\omega^{2}}\right)"),
    FourierCase("PU02", "t^2*u(t)", "polynomial_step", ("PV", "omega"), expected_result_latex=r"- \pi \delta^{(2)}(\omega) + 2 j \mathrm{PV}\!\left(\frac{1}{\omega^{3}}\right)"),
    FourierCase("PU03", "t^3*u(t)", "polynomial_step", ("PV", "omega"), expected_result_latex=r"- j \pi \delta^{(3)}(\omega) + 6 \mathrm{PV}\!\left(\frac{1}{\omega^{4}}\right)"),
    FourierCase("PU04", "2*t*u(t)", "polynomial_step", ("PV", "omega"), expected_result_latex=r"2 j \pi \delta^{(1)}(\omega) - 2 \mathrm{PV}\!\left(\frac{1}{\omega^{2}}\right)"),
    FourierCase("PU05", "(t^2+2*t+1)*u(t)", "polynomial_step", ("PV", "omega"), expected_result_latex=r"\left(\pi \delta\left(\omega\right) - j \mathrm{PV}\!\left(\frac{1}{\omega}\right)\right) - \left(\pi \delta^{(2)}(\omega) - 2 j \mathrm{PV}\!\left(\frac{1}{\omega^{3}}\right)\right) + \left(2 j \pi \delta^{(1)}(\omega) - 2 \mathrm{PV}\!\left(\frac{1}{\omega^{2}}\right)\right)"),
    FourierCase(
        "PU06",
        "(t-2)*u(t-2)",
        "shifted_polynomial_step",
        ("PV", "delta", "omega", "e"),
        expected_steps_contains=(
            r"\textbf{Step 2: Identify a shifted polynomial multiplied by a shifted unit step}",
            r"c=2",
            r"p(s+c)=s",
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
            r"p(s+c)=s + 2",
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
            r"p(s+c)=s^{2}",
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
    FourierCase("T08", "sin(t)*u(t)", "trigonometric_step", ("PV", "omega"), expected_result_latex=r"- \frac{j \pi \delta\left(\omega - 1\right)}{2} + \frac{j \pi \delta\left(\omega + 1\right)}{2} - \frac{\mathrm{PV}\!\left(\frac{1}{\omega - 1}\right)}{2} + \frac{\mathrm{PV}\!\left(\frac{1}{\omega + 1}\right)}{2}"),
    FourierCase("T09", "cos(t)*u(t)", "trigonometric_step", ("PV", "omega"), expected_result_latex=r"\frac{\pi \delta\left(\omega - 1\right)}{2} + \frac{\pi \delta\left(\omega + 1\right)}{2} - \frac{j \mathrm{PV}\!\left(\frac{1}{\omega - 1}\right)}{2} - \frac{j \mathrm{PV}\!\left(\frac{1}{\omega + 1}\right)}{2}"),
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
    FourierCase("R02", "frac(1,t+3)", "rational_pv", ("sign", "omega", "3"), expected_result_latex=r"e^{3 j \omega}\left(-j\pi\,\mathrm{sign}(\omega)\right)"),
    FourierCase("R03", "frac(1,3*t-2)", "rational_pv", ("sign", "omega"), expected_result_latex=r"-\frac{j\pi}{3}\,e^{- \frac{2 j \omega}{3}}\,\mathrm{sign}(\omega)"),
    FourierCase("R04", "frac(1,t^2+1)", "rational", ("pi", "omega"), expected_result_latex=r"\pi e^{- |\omega|}"),
    FourierCase("R05", "frac(1,t^2+4)", "rational", ("pi", "omega"), expected_result_latex=r"\frac{\pi e^{- 2 |\omega|}}{2}"),
    FourierCase("R06", "frac(t,t^2+1)", "rational", ("sign", "omega"), expected_result_latex=r"- j \pi e^{- |\omega|} \mathrm{sign}(\omega)"),
    FourierCase("R07", "frac(2*t+3,t^2+6)", "rational", ("omega",), expected_result_latex=r"\pi \left(- 2 j \mathrm{sign}(\omega) + \frac{\sqrt{6}}{2}\right) e^{- \sqrt{6} |\omega|}"),
    FourierCase("R08", "frac(2*t,3*t^2+4*t-1)", "rational", ("omega",), expected_result_latex=r"- \frac{j \pi \left(2 \sqrt{7} + 7\right) e^{- j \omega \left(- \frac{\sqrt{7}}{3} - \frac{2}{3}\right)} \mathrm{sign}(\omega)}{21} + \frac{j \pi \left(-7 + 2 \sqrt{7}\right) e^{- j \omega \left(- \frac{2}{3} + \frac{\sqrt{7}}{3}\right)} \mathrm{sign}(\omega)}{21}"),
    FourierCase(
        "R15",
        "frac(t^5+t^4+t^3,(t+1)(t^2+1)(t+6)(t^2+6))",
        "rational_composite",
        ("sign", "omega", "sqrt", "e"),
        expected_result_latex=r"\pi \left(- \frac{6 j \mathrm{sign}(\omega)}{35} - \frac{\sqrt{6}}{7}\right) e^{- \sqrt{6} |\omega|} + \pi \left(\frac{7 j \mathrm{sign}(\omega)}{370} + \frac{1}{74}\right) e^{- |\omega|} - \frac{1116 j \pi e^{6 j \omega} \mathrm{sign}(\omega)}{1295} + \frac{j \pi e^{j \omega} \mathrm{sign}(\omega)}{70}",
        expected_steps_contains=(
            r"\textbf{Step 2: Apply partial fraction decomposition}",
            r"\frac{6 \left(t - 5\right)}{35 \left(t^{2} + 6\right)}",
            r"\frac{1116}{1295 \left(t + 6\right)}",
            r"\textbf{Step 4: Combine by linearity}",
        ),
    ),
    FourierCase("R09", "frac(t^2,t(t+1))", "rational", ("sign", "delta", "omega"), expected_result_latex=r"j \pi e^{j \omega} \mathrm{sign}(\omega) + 2 \pi \delta\left(\omega\right)"),
    FourierCase("R10", "frac(t^2(t+1),t^3(t+1))", "rational_pv", ("sign", "omega"), expected_result_latex=r"-j\pi\,\mathrm{sign}(\omega)"),
    FourierCase("R11", "frac(t^2（t+1）,t^3(t+1))", "rational_pv", ("sign", "omega"), expected_result_latex=r"-j\pi\,\mathrm{sign}(\omega)"),
    FourierCase("W01", "u(t-2)-u(t-5)", "window", ("omega",), expected_result_latex=r"- \frac{j \left(e^{- 2 j \omega} - e^{- 5 j \omega}\right)}{\omega}"),
    FourierCase("W02", "u(t+1)-u(t-1)", "window", ("omega",), expected_result_latex=r"- \frac{j \left(e^{j \omega} - e^{- j \omega}\right)}{\omega}"),
    FourierCase("W03", "rect((t-2)/3)", "rect_window", ("omega", "sin", "e"), expected_result_latex=r"\frac{2 e^{- 2 j \omega} \sin{\left(\frac{3 \omega}{2} \right)}}{\omega}"),
    FourierCase("W04", "4*rect((t+1)/2)", "rect_window", ("omega", "sin", "e"), expected_result_latex=r"\frac{8 e^{j \omega} \sin{\left(\omega \right)}}{\omega}"),
    FourierCase("W09", "tri((t-1)/2)", "tri_window", ("omega", "sin", "e"), expected_result_latex=r"\frac{2 e^{- j \omega} \sin^{2}{\left(\omega \right)}}{\omega^{2}}"),
    FourierCase("W10", "3*tri((t+2)/4)", "tri_window", ("omega", "sin", "e"), expected_result_latex=r"\frac{3 e^{2 j \omega} \sin^{2}{\left(2 \omega \right)}}{\omega^{2}}"),
    FourierCase("W05", "frac(sin(t),pi*t)", "sinc", ("Rect", "omega"), expected_result_latex=r"\operatorname{Rect}{\left(\frac{\omega}{2} \right)}"),
    FourierCase("W06", "frac(sin(a*t),pi*t)", "sinc", ("Rect", "omega", "a"), expected_result_latex=r"\operatorname{Rect}{\left(\frac{\omega}{2 a} \right)}"),
    FourierCase("W07", "frac(sin(t),t)", "sinc", ("Rect", "omega", "pi"), expected_result_latex=r"\pi \operatorname{Rect}{\left(\frac{\omega}{2} \right)}"),
    FourierCase("W11", "frac(sin(a*t),t)", "sinc", ("Rect", "omega", "a"), expected_result_latex=r"\pi \operatorname{Rect}{\left(\frac{\omega}{2 a} \right)}"),
    FourierCase("F01", "exp(-t^2)", "gaussian", ("omega",), expected_result_latex=r"\sqrt{\pi} e^{- \frac{\omega^{2}}{4}}"),
    FourierCase("F02", "exp(-2*t^2)", "gaussian", ("omega",), expected_result_latex=r"\frac{\sqrt{2} \sqrt{\pi} e^{- \frac{\omega^{2}}{8}}}{2}"),
    FourierCase("F08", "exp(-a*t^2)", "gaussian", ("omega", "a"), expected_result_latex=r"\sqrt{\pi} \sqrt{\frac{1}{a}} e^{- \frac{\omega^{2}}{4 a}}"),
    FourierCase(
        "PR01",
        "exp(-(t+1)^2)",
        "property_time_shift",
        ("sqrt", "pi", "e", "omega"),
        expected_steps_contains=(
            r"\textbf{Step 2: Identify a time shift}",
            r"x(t)=g(t-c),\quad c=-1",
            r"\mathcal{F}\{g(t-c)\}=e^{-j\omega c}G(\omega)",
        ),
    ),
    FourierCase(
        "PR02",
        "exp(-abs(t-2))",
        "property_time_shift",
        ("omega", r"e^{- 2 j \omega}"),
        expected_steps_contains=(
            r"\textbf{Step 2: Identify a time shift}",
            r"x(t)=g(t-c),\quad c=2",
            r"G(\omega)=\frac{2}{\omega^{2} + 1}",
        ),
    ),
    FourierCase(
        "PR03",
        "exp(I*3*t)*exp(-t^2)",
        "property_modulation",
        (r"\omega - 3", "sqrt", "pi"),
        expected_steps_contains=(
            r"\textbf{Step 2: Identify a modulation factor}",
            r"\omega_0=3,\quad \phi=0",
            r"\mathcal{F}\{e^{j\omega_0 t}g(t)\}=G(\omega-\omega_0)",
        ),
    ),
    FourierCase(
        "PR04",
        "exp(I*(2*t+1))*exp(-abs(t))",
        "property_modulation",
        (r"\omega - 2", r"e^{j}"),
        expected_steps_contains=(
            r"\textbf{Step 2: Identify a modulation factor}",
            r"\omega_0=2,\quad \phi=1",
            r"\text{The constant phase }e^{j\phi}\text{ is kept as a multiplier.}",
        ),
    ),
    FourierCase(
        "PR05",
        "exp(I*3*t)*exp(-(t+1)^2)",
        "property_modulation_time_shift",
        (r"\omega - 3", "sqrt", "pi", r"e^{j \left(\omega - 3\right)}"),
        expected_steps_contains=(
            r"\textbf{Step 2: Identify a modulation factor}",
            r"g(t)=e^{- \left(t + 1\right)^{2}}",
            r"G(\omega)=\sqrt{\pi} e^{- \frac{\omega^{2}}{4}} e^{j \omega}",
        ),
    ),
    FourierCase(
        "PR06",
        "t*exp(-t^2)",
        "property_time_multiplication",
        ("j", "sqrt", "pi", "omega"),
        expected_steps_contains=(
            r"\textbf{Step 2: Identify multiplication by }t",
            r"\mathcal{F}\{t g(t)\}=j\frac{d}{d\omega}G(\omega)",
        ),
    ),
    FourierCase(
        "PR07",
        "t*exp(-abs(t))",
        "property_time_multiplication",
        ("j", "omega", r"\left(\omega^{2} + 1\right)^{2}"),
        expected_steps_contains=(
            r"\textbf{Step 2: Identify multiplication by }t",
            r"G(\omega)=\frac{2}{\omega^{2} + 1}",
            r"\mathcal{F}\{t g(t)\}=j\frac{d}{d\omega}G(\omega)",
        ),
    ),
)


KNOWN_GAP_CASES: tuple[FourierCase, ...] = ()


FUTURE_CASES: tuple[FourierCase, ...] = ()

