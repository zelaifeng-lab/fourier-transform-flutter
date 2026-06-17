# Flutter Example List Guide

This document explains the Dart preset expression list used by the Flutter app.

## Purpose

The examples are preset expressions. They save users from typing common Fourier-transform inputs manually and demonstrate the supported transform range of the app.

Users can rotate through the list with:

- `Previous`
- `Next`

The list rolls over at both ends.

## Location

The example list is defined in:

```text
flutter_app/lib/main.dart
```

The data model is:

```dart
class FourierExample {
  final String expression;
  final String title;
  final String category;
  final String inputLatex;
  final String transformLatex;
  final String description;
}
```

## Fields

| Field | Meaning |
|---|---|
| `expression` | The actual expression sent through the local frontend transform flow and then to the backend result page. |
| `title` | Short human-readable name of the example. |
| `category` | Transform family or property demonstrated by the example. |
| `inputLatex` | LaTeX display for the original function `x(t)`. |
| `transformLatex` | LaTeX display for the transform object, known pair, or property used. |
| `description` | Short explanation for the user. |

## Current Example Categories

The current list includes:

- Basic pairs
- Time shift
- Scale by coefficient
- Frequency scale
- Phase shift
- Combination
- Polynomial distribution
- Polynomial times step
- Trigonometric pairs
- PV distributions
- Modulation
- One-sided exponentials
- Rational pairs
- Generic rational forms
- Finite windows

## Examples of Supported Presets

```text
u(t)
u(t-2)
delta(t-1)
t^2
(t^2+2*t+1)*u(t)
sin(3*t+2)
sin(t)+cos(2*t)
exp(-3*abs(t))
frac(1,3*t-2)
frac(2*t,3*t^2+4*t-1)
u(t-2)-u(t-5)
```

## How To Add An Example

1. Confirm the backend can process the expression.

   Preferred backend check:

   ```powershell
   cd backend
   .\.venv\Scripts\python.exe -m pytest tests -q
   ```

2. Add a new `FourierExample` entry in `flutter_app/lib/main.dart`.

   Template:

   ```dart
   FourierExample(
     expression: 'your_expression',
     title: 'Short title',
     category: 'Transform category',
     inputLatex: r'your x(t) latex',
     transformLatex: r'your X(\omega) or property latex',
     description: 'One short explanation.',
   ),
   ```

3. Keep `expression` in the same syntax accepted by the app and backend.

   Common syntax:

   ```text
   u(t)
   delta(t)
   frac(1,t)
   exp(I*3*t)
   exp(-3*abs(t))
   sin(t)+cos(2*t)
   ```

4. Prefer examples that demonstrate a capability, not only a trivial case.

   Good example types:

   - shifted function
   - scaled function
   - combination of known pairs
   - PV/distribution case
   - finite window
   - rational expression matched to a symbolic family

5. Run Flutter tests.

   ```powershell
   cd flutter_app
   flutter test
   ```

## Test Coverage

Flutter tests are in:

```text
flutter_app/test/home_page_test.dart
flutter_app/test/frontend_backend_e2e_test.dart
```

Current frontend tests verify:

- the home page renders
- the example metadata appears
- `Previous` and `Next` rotate examples
- general Fourier formulas are visible
- PV explanation is visible
- common screen sizes do not throw layout exceptions
- optional frontend-backend e2e is available when explicitly enabled

## Frontend-Backend E2E Test

The e2e test is intentionally skipped by default so GitHub Actions does not depend on a live backend.

To enable it locally:

```powershell
cd flutter_app
flutter test test/frontend_backend_e2e_test.dart `
  --dart-define=RUN_BACKEND_E2E=true `
  --dart-define=API_BASE_URL=http://127.0.0.1:8000
```

Run the backend separately before enabling the e2e test:

```powershell
cd backend
.\.venv\Scripts\python.exe -m uvicorn backend:app --host 127.0.0.1 --port 8000
```

## Completion Status

- [x] Complex examples: shift / scale / combination
- [x] Original function display for the current example
- [x] Transform object / property display for the current example
- [x] LaTeX rendering with `Math.tex`
- [x] General Fourier formulas
- [x] PV explanation
- [x] Enhanced Flutter tests
- [x] Optional frontend-backend e2e test
- [x] Documentation for adding examples
