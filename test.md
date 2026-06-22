# Fourier Transform Backend Test Data

This file defines recommended backend test cases for the Fourier Transform Engine.
The convention is:

```latex
X(\omega)=\int_{-\infty}^{\infty}x(t)e^{-j\omega t}\,dt
```

Assumptions:

- `t` and `omega` are real variables.
- Outputs should prefer engineering/distribution forms.
- Regression tests should reject unwanted symbolic artifacts such as `Piecewise`, `RootSum`, `arg`, `polar_lift`, and `meijerg`.
- Cases marked `implemented: false` are useful future-rule test data and can be converted to active tests once the corresponding rules are added.

## 1. Basic Distributions And Constants

| ID | Expression | Expected result | Implemented | Notes |
|---|---|---|---|---|
| B01 | `1` | `2*pi*delta(omega)` | true | Constant signal |
| B02 | `5` | `10*pi*delta(omega)` | true | Constant multiplier |
| B03 | `delta(t)` | `1` | true | Unit impulse |
| B04 | `delta(t-3)` | `exp(-I*3*omega)` | true | Shifted impulse |
| B05 | `delta(t+2)` | `exp(I*2*omega)` | true | Shifted impulse |
| B06 | `u(t)` | `pi*delta(omega) - I*PV(1/omega)` | true | Heaviside step |
| B07 | `u(t-2)` | `exp(-I*2*omega)*(pi*delta(omega)-I*PV(1/omega))` | true | Shifted step |
| B08 | `u(t+1)` | `exp(I*omega)*(pi*delta(omega)-I*PV(1/omega))` | true | Shifted step |
| B09 | `sign(t)` | `-2*I*PV(1/omega)` | true | Sign distribution pair |
| B10 | `sign(t-2)` | `exp(-2*I*omega)*(-2*I*PV(1/omega))` | true | Shifted sign distribution |
| B11 | `sign(-3*(t+1))` | `exp(I*omega)*(2*I*PV(1/omega))` | true | Negative scale plus time shift |

## 2. Polynomial Test Data

Pure polynomial rule:

```latex
\mathcal{F}\{t^n\}=2\pi j^n\delta^{(n)}(\omega)
```

| ID | Expression | Expected result | Implemented | Notes |
|---|---|---|---|---|
| P01 | `t` | `2*pi*I*delta'(omega)` | true | First derivative of delta |
| P02 | `t^2` | `-2*pi*delta''(omega)` | true | Second derivative of delta |
| P03 | `t^3` | `-2*pi*I*delta'''(omega)` | true | Third derivative of delta |
| P04 | `3*t` | `6*pi*I*delta'(omega)` | true | Constant multiplier |
| P05 | `2*t^2+3*t+1` | `-4*pi*delta''(omega)+6*pi*I*delta'(omega)+2*pi*delta(omega)` | true | Polynomial linearity |
| P06 | `t^4` | `2*pi*delta''''(omega)` | true | Higher-order polynomial |

Polynomial times step rule:

```latex
\mathcal{F}\{t^n u(t)\}
=\pi j^n\delta^{(n)}(\omega)
+(-1)^n j^{n-1}n!\,\mathrm{PV}(1/\omega^{n+1})
```

| ID | Expression | Expected result | Implemented | Notes |
|---|---|---|---|---|
| PU01 | `t*u(t)` | `I*pi*delta'(omega) - PV(1/omega^2)` | true | Polynomial step |
| PU02 | `t^2*u(t)` | `-pi*delta''(omega) - 2*I*PV(1/omega^3)` | true | Polynomial step |
| PU03 | `t^3*u(t)` | `-I*pi*delta'''(omega) + 6*PV(1/omega^4)` | true | Polynomial step |
| PU04 | `2*t*u(t)` | `2*I*pi*delta'(omega) - 2*PV(1/omega^2)` | true | Constant multiplier |
| PU05 | `(t^2+2*t+1)*u(t)` | Expanded linear combination of `t^2*u(t)`, `2*t*u(t)`, and `u(t)` | true | Polynomial expansion |
| PU06 | `(t-2)*u(t-2)` | Shifted `t*u(t)` result: multiply the origin polynomial-step transform by `exp(-2*I*omega)` | true | Reuse shifted step helper; no direct integral |
| PU07 | `t*u(t-2)` | Rewrite with `s=t-2`, so `t=s+2`; combine `s*u(s)` and `u(s)` then time shift | true | Shifted polynomial-step expansion |
| PU08 | `(t+3)^2*u(t+3)` | Shifted `t^2*u(t)` result with factor `exp(3*I*omega)` | true | Shifted polynomial-step rule |

## 3. Exponential Signals

| ID | Expression | Expected result | Implemented | Notes |
|---|---|---|---|---|
| E01 | `exp(I*3*t)` | `2*pi*delta(omega-3)` | true | Pure tone |
| E02 | `exp(-I*2*t)` | `2*pi*delta(omega+2)` | true | Pure tone |
| E03 | `exp(I*(5*t+2))` | `2*pi*exp(2*I)*delta(omega-5)` | true | Pure tone with phase |
| E04 | `exp(-t)*u(t)` | `1/(1+I*omega)` | true | One-sided exponential |
| E05 | `exp(-2*t)*u(t)` | `1/(2+I*omega)` | true | One-sided exponential |
| E06 | `exp(-3*abs(t))` | `6/(9+omega^2)` | true | Two-sided decaying exponential |
| E07 | `exp(-a*t)*u(t)` | `1/(a+I*omega), a>0` | false | Future parameter rule |
| E08 | `exp(-a*abs(t))` | `2*a/(a^2+omega^2), a>0` | false | Future parameter rule |
| E09 | `exp(I*2*t)*u(t-3)` | Shifted modulated step: `exp(-I*(omega-2)*3)*(pi*delta(omega-2)-I*PV(1/(omega-2)))` | true | Distribution form; no direct integral |
| E10 | `exp(I*(2*t+1))*u(t+3)` | Shifted modulated step with phase: `exp(I)*exp(I*(omega-2)*3)*(pi*delta(omega-2)-I*PV(1/(omega-2)))` | true | Distribution form; no direct integral |
| E11 | `3*exp(-I*t)*u(t-2)` | Scaled shifted modulated step around `omega=-1` | true | Distribution form; no direct integral |

## 4. Trigonometric Signals

| ID | Expression | Expected result | Implemented | Notes |
|---|---|---|---|---|
| T01 | `sin(t)` | `pi/I*(delta(omega-1)-delta(omega+1))` | true | Pure sinusoid |
| T02 | `cos(t)` | `pi*(delta(omega-1)+delta(omega+1))` | true | Pure sinusoid |
| T03 | `sin(3*t)` | `pi/I*(delta(omega-3)-delta(omega+3))` | true | Scaled sinusoid |
| T04 | `cos(4*t)` | `pi*(delta(omega-4)+delta(omega+4))` | true | Scaled sinusoid |
| T05 | `sin(3*t+2)` | `pi/I*(exp(2I)*delta(omega-3)-exp(-2I)*delta(omega+3))` | true | Phase shift |
| T06 | `cos(4*t-1)` | `pi*(exp(-I)*delta(omega-4)+exp(I)*delta(omega+4))` | true | Phase shift |
| T07 | `sin(t)+cos(2*t)` | Linear combination of T01 and `cos(2*t)` | true | Linearity |
| T08 | `sin(t)*u(t)` | Principal-value rational terms plus delta terms | true | Assert equivalent distribution form |
| T09 | `cos(t)*u(t)` | Principal-value rational terms plus delta terms | true | Assert equivalent distribution form |
| T10 | `sin(t-1)*u(t+3)` | Shifted-step sinusoid: rewrite with `s=t+3`, then use modulated unit-step PV/delta terms and time shift | true | Assert no direct-integral `Piecewise` result |
| T11 | `cos(2*t+1)*u(t-3)` | Shifted-step cosine: rewrite with `s=t-3`, then use modulated unit-step PV/delta terms and time shift | true | Assert no direct-integral `Piecewise` result |

## 4.1 Distributive And Window Linearity

| ID | Expression | Expected result | Implemented | Notes |
|---|---|---|---|---|
| D01 | `(exp(I*2*t)+exp(-I*t))*u(t-3)` | Distribute into two shifted modulated-step terms and sum their PV/delta results | true | Assert no direct-integral `Piecewise` result |
| D02 | `2*(t*u(t)+sin(t)*u(t))` | Distribute into polynomial-step and trig-step terms, then apply linearity | true | Assert no direct-integral `Piecewise` result |
| W08 | `t*(u(t)-u(t-1))` | Finite polynomial-window integral over `[0,1]` | true | Assert no direct-integral `Piecewise` result |

## 5. Rational Functions And Principal Values

| ID | Expression | Expected result | Implemented | Notes |
|---|---|---|---|---|
| R01 | `frac(1,t)` | `-I*pi*sign(omega)` | true | PV reciprocal |
| R02 | `frac(1,t+3)` | `exp(I*3*omega)*(-I*pi*sign(omega))` | true | Shifted PV reciprocal |
| R03 | `frac(1,3*t-2)` | `-(I*pi/3)*exp(-I*2*omega/3)*sign(omega)` | true | Linear denominator |
| R04 | `frac(1,t^2+1)` | `pi*exp(-abs(omega))` | true | Standard pair |
| R05 | `frac(1,t^2+4)` | `(pi/2)*exp(-2*abs(omega))` | true | Standard pair |
| R06 | `frac(t,t^2+1)` | `-I*pi*sign(omega)*exp(-abs(omega))` | true | Odd rational pair |
| R07 | `frac(2*t+3,t^2+6)` | `2*F{t/(t^2+6)} + 3*F{1/(t^2+6)}` | true | Rational linearity |
| R08 | `frac(2*t,3*t^2+4*t-1)` | Real partial fractions, then termwise PV results | true | More complex rational function |
| R09 | `frac(1,t^2)` | `-pi*abs(omega)` | false | Future distribution rule |
| R10 | `frac(1,(t+2)^2)` | `exp(I*2*omega)*(-pi*abs(omega))` | false | Future shifted distribution rule |

## 6. Window, Rect, And Sinc Family

| ID | Expression | Expected result | Implemented | Notes |
|---|---|---|---|---|
| W01 | `u(t-2)-u(t-5)` | `(exp(-I*2*omega)-exp(-I*5*omega))/(I*omega)` | true | Finite window |
| W02 | `u(t+1)-u(t-1)` | `2*sin(omega)/omega` | true | Symmetric rectangular window |
| W03 | `rect((t-2)/3)` | `exp(-2*I*omega)*2*sin(3*omega/2)/omega` | true | Shifted and scaled rectangular pulse |
| W04 | `4*rect((t+1)/2)` | `4*exp(I*omega)*2*sin(omega)/omega` | true | Amplitude, shift, and width scaling |
| W09 | `tri((t-1)/2)` | `2*exp(-I*omega)*(sin(omega)/omega)^2` | true | Shifted and scaled triangular pulse |
| W10 | `3*tri((t+2)/4)` | `3*exp(2*I*omega)*sin(2*omega)^2/omega^2` | true | Amplitude, shift, and width scaling |
| W05 | `frac(sin(t),pi*t)` | `1 for abs(omega)<1` | false | Future sinc rule |
| W06 | `frac(sin(3*t),pi*t)` | `1 for abs(omega)<3` | false | Future sinc rule |
| W07 | `frac(sin(t),t)` | `pi for abs(omega)<1` | false | Future sinc rule |

## 7. Common Future Rules

| ID | Expression | Expected result | Implemented | Notes |
|---|---|---|---|---|
| F01 | `exp(-t^2)` | `sqrt(pi)*exp(-omega^2/4)` | false | Gaussian |
| F02 | `exp(-2*t^2)` | `sqrt(pi/2)*exp(-omega^2/8)` | false | Gaussian |
| F03 | `x(t-a)` | `exp(-I*omega*a)*X(omega)` | false | Abstract time shift |
| F04 | `exp(I*w0*t)*x(t)` | `X(omega-w0)` | false | Abstract modulation |
| F05 | `x(a*t)` | `1/abs(a)*X(omega/a)` | false | Abstract scaling |
| F06 | `diff(x(t),t)` | `I*omega*X(omega)` | false | Time differentiation |
| F07 | `t*x(t)` | `I*dX/domega` | false | Frequency differentiation |

## 8. Convolution Cases

Use the convolution symbol that the backend parser actually supports. If frontend and backend symbols differ, keep separate compatibility tests for both.

| ID | Expression | Expected result | Implemented | Notes |
|---|---|---|---|---|
| C01 | `delta(t)·sin(t)` | `F{sin(t)}` | true | Identity convolution |
| C02 | `delta(t-1)·sin(t)` | `exp(-I*omega)*F{sin(t)}` | true | Shifted impulse convolution |
| C03 | `u(t)·exp(-t)*u(t)` | `F{u(t)} * F{exp(-t)*u(t)}` | true | Convolution theorem |
| C04 | `cos(t)·cos(t)` | `F{cos(t)} * F{cos(t)}` | true | Convolution theorem |
| C05 | `frac(1,t^2+1)·frac(1,t^2+1)` | `(pi*exp(-abs(omega)))^2` | true | Convolution theorem |

## 9. Suggested Test Schema

Recommended Python parameter record:

```python
{
    "id": "R04",
    "expression": "frac(1,t^2+1)",
    "expected_contains": ["pi", "exp", "omega"],
    "expected_math": "pi*exp(-Abs(omega))",
    "category": "rational_known_pair",
    "implemented": True,
}
```

For future rules:

```python
{
    "id": "F01",
    "expression": "exp(-t^2)",
    "expected_math": "sqrt(pi)*exp(-omega**2/4)",
    "category": "future_gaussian",
    "implemented": False,
}
```

Recommended generic assertions for implemented cases:

- HTTP/API call returns success for active tests.
- `ok == true` when the rule is expected to be implemented.
- `result_latex` contains the expected stable mathematical components.
- `steps_latex` is non-empty and readable.
- `result_latex` and `steps_latex` do not contain `Piecewise`, `RootSum`, `arg`, `polar_lift`, `meijerg`, `_rule_`, `matcher`, `debug`, or `srepr`.

## 10. Backend Test Requirements

When converting this data into automated tests, follow these requirements:

1. Active implemented cases must call the backend `/fourier` endpoint rather than only testing helper functions directly.
2. Each implemented case should assert `ok == true`, unless the case is explicitly testing parser failure or unsupported input behavior.
3. Each implemented case should verify that `result_latex` contains stable mathematical components rather than relying only on exact string equality.
4. Each implemented case should verify that `steps_latex` is non-empty and follows a teaching-style derivation flow.
5. Every active test should reject internal/debug artifacts in both `result_latex` and `steps_latex`: `Piecewise`, `RootSum`, `arg`, `polar_lift`, `meijerg`, `_rule_`, `matcher`, `debug`, and `srepr`.
6. Future cases with `implemented: false` should be kept as pending or skipped tests until the corresponding backend rule is implemented.
7. When a new rule is added, move its related cases from future/pending to active regression tests.
8. For expressions with mathematically equivalent output forms, prefer component-based assertions or SymPy-equivalence checks over brittle exact string matching.
9. Test IDs in code should match the IDs in this file, so failures can be traced back to the documented transform pair.
