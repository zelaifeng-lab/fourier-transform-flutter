# Brief Q&A for Email Reply

This is a short version for email communication. The detailed Q&A file is `docs/QA.md`.

| Question / Concern | Status | Brief Reply |
|---|---|---|
| What is PV? | Solved | PV is explained as Cauchy principal value, used for distribution-style Fourier transforms such as `1/t` and `u(t)`. |
| Can the app handle general user input? | Solved within scope | The app supports general user input within a defined engineering rule range, including impulses, steps, exponentials, trigonometric functions, polynomial-step forms, PV reciprocals, rational forms, windows, and combinations. |
| Can the app find more generic Fourier transforms within a certain range? | Solved within scope | The backend is rule-first and handles transform families by matching general forms and substituting parameters, rather than only using fixed numeric examples. |
| Are there preset examples for users? | Solved | The Flutter app includes categorized examples with previous/next navigation to help users enter suitable expressions. |
| Are examples complex enough? | Solved | The examples now include shift, scale, combination, PV, rational, polynomial-step, and finite-window cases. |
| Does the result show the original function and transform? | Solved | The result page displays both `x(t)` and `X(\omega)` using LaTeX. |
| Are formulas displayed using LaTeX? | Solved | The frontend uses `Math.tex` for formula preview, result display, derivation steps, and general Fourier formulas. |
| Are derivation steps suitable for teaching? | Solved | Backend steps now start from the Fourier definition and explain the transform pair/property used, parameter substitution, and final result. |
| Are the formulas consistent with engineering notation? | Solved | User-facing output now uses `j`, `u(t)`, `\mathrm{PV}`, `\mathrm{sign}(\omega)`, compact `|\omega|`, and cleaner delta derivative notation. |
| Does the frontend handle long formulas/notes on small screens? | Solved | Long LaTeX formulas and raw inputs now scroll horizontally, while long example notes use bounded local vertical scrolling. |
| Has the project been tested? | Solved | Backend regression tests pass with `68 passed`; Flutter tests pass with `6 passed, 1 skipped`. |
| Is there a Q&A file instead of a long email? | Solved | A detailed Q&A file and this brief email version are now available. |
| Is the dissertation-format report completed here? | Deferred | The dissertation-format report will be adjusted separately using the provided template. |
