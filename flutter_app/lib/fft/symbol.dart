import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter_math_fork/flutter_math.dart';
import 'dart:convert';
import 'package:http/http.dart' as http;


class SymbolicResult {
  final bool ok;
  final String inputLatex; // RHS only, displayed as x(t)=...
  final String resultLatex; // RHS only, displayed as X(ω)=...
  final List<String> stepsLatex;

  // Optional for charts
  final List<double> omegaAxis;
  final List<double> spectrumData;

  const SymbolicResult._({
    required this.ok,
    required this.inputLatex,
    required this.resultLatex,
    required this.stepsLatex,
    required this.omegaAxis,
    required this.spectrumData,
  });

  factory SymbolicResult.ok({
    required String inputLatex,
    required String resultLatex,
    required List<String> stepsLatex,
    required List<double> omegaAxis,
    required List<double> spectrumData,
  }) {
    return SymbolicResult._(
      ok: true,
      inputLatex: inputLatex,
      resultLatex: resultLatex,
      stepsLatex: stepsLatex,
      omegaAxis: omegaAxis,
      spectrumData: spectrumData,
    );
  }

  factory SymbolicResult.fail({
    required String inputLatex,
    String messageLatex = r'\text{Unable to compute}',
  }) {
    return SymbolicResult._(
      ok: false,
      inputLatex: inputLatex,
      resultLatex: messageLatex,
      stepsLatex: const [],
      omegaAxis: const [],
      spectrumData: const [],
    );
  }
}
Future<SymbolicResult> computeByBackendOnly(String expression) async {
// Backend selection:
// - Web (GitHub Pages/Flutter Web): use your public Render URL (HTTPS)
// - Android emulator: use 10.0.2.2 to reach host machine
// - You can override for any build with:
//   flutter run/build ... --dart-define=API_BASE_URL=https://your-backend
  const String envBase = String.fromEnvironment('API_BASE_URL', defaultValue: '');
  const String renderBase = 'https://fourier-transform-flutter.onrender.com';
  const String androidEmulatorBase = 'http://10.0.2.2:8000';

  final String base = envBase.isNotEmpty ? envBase : (kIsWeb ? renderBase : androidEmulatorBase);
  final Uri uri = Uri.parse(base.endsWith('/') ? '${base}fourier' : '${base}/fourier');


  try {
    final res = await http.post(
      uri,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'expression': expression}),
    );

    if (res.statusCode != 200) {
      return SymbolicResult.fail(
        inputLatex: r'\text{(backend error)}',
        messageLatex: r'\text{HTTP }' + res.statusCode.toString(),
      );
    }

    final j = jsonDecode(res.body) as Map<String, dynamic>;
    final ok = (j['ok'] == true);

    final inputLatex = (j['input_latex'] ?? r'\text{(parse failed)}').toString();
    final resultLatex = (j['result_latex'] ?? r'\text{Unable to compute}').toString();
    final steps = (j['steps_latex'] as List<dynamic>? ?? const [])
        .map((e) => e.toString())
        .toList();

    return SymbolicResult._(
      ok: ok,
      inputLatex: inputLatex,
      resultLatex: resultLatex,
      stepsLatex: steps,
      omegaAxis: const [],
      spectrumData: const [],
    );
  } catch (_) {
    return SymbolicResult.fail(
      inputLatex: r'\text{(network error)}',
      messageLatex: r'\text{Unable to reach backend}',
    );
  }
}


// UI page

class SymbolPage extends StatefulWidget {
  final String expression;
  const SymbolPage({super.key, required this.expression});

  @override
  State<SymbolPage> createState() => _SymbolPageState();
}

class _SymbolPageState extends State<SymbolPage> {
  late Future<SymbolicResult> _future;

  @override
  void initState() {
    super.initState();
    _future = computeByBackendOnly(widget.expression);
  }

  @override
  void didUpdateWidget(covariant SymbolPage oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.expression != widget.expression) {
      _future = computeByBackendOnly(widget.expression);
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    Widget texLine(String latex, {TextStyle? style}) {
      return SingleChildScrollView(
        scrollDirection: Axis.horizontal,
        child: Math.tex(
          latex,
          textStyle: style ?? theme.textTheme.bodyLarge,
        ),
      );
    }

    Widget statusLine(String text) {
      return Text(
        text,
        style: theme.textTheme.bodyMedium?.copyWith(
          color: theme.colorScheme.onSurface.withValues(alpha: 0.7),
        ),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('Symbolic Fourier (Backend)'),
      ),
      body: FutureBuilder<SymbolicResult>(
        future: _future,
        builder: (context, snap) {
          if (snap.connectionState != ConnectionState.done) {
            return const Center(child: CircularProgressIndicator());
          }

          final res = snap.data ??
              SymbolicResult.fail(
                inputLatex: r'\text{(no data)}',
                messageLatex: r'\text{Unable to compute}',
              );

          return ListView(
            padding: const EdgeInsets.all(16),
            children: [
              _Card(
                title: 'Input',
                child: texLine(
                  r'\displaystyle x(t)=' + res.inputLatex,
                  style: theme.textTheme.titleLarge,
                ),
              ),
              const SizedBox(height: 12),
              _Card(
                title: 'Result',
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    texLine(
                      r'\displaystyle X(\omega)=' + res.resultLatex,
                      style: theme.textTheme.titleLarge,
                    ),
                    const SizedBox(height: 8),
                    if (!res.ok) statusLine('Closed-form not found; showing integral / symbolic form.'),
                  ],
                ),
              ),
              const SizedBox(height: 12),
              _Card(
                title: 'Derivation',
                child: res.stepsLatex.isEmpty
                    ? statusLine('No steps.')
                    : Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    for (int i = 0; i < res.stepsLatex.length; i++) ...[
                      Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          SizedBox(
                            width: 28,
                            child: Text(
                              '${i + 1}.',
                              style: theme.textTheme.bodyMedium?.copyWith(
                                color: theme.colorScheme.onSurface.withValues(alpha: 0.65),
                              ),
                            ),
                          ),
                          Expanded(
                            child: texLine(
                              r'\displaystyle ' + res.stepsLatex[i],
                              style: theme.textTheme.bodyLarge,
                            ),
                          ),
                        ],
                      ),
                      if (i != res.stepsLatex.length - 1) ...[
                        const SizedBox(height: 10),
                        Divider(color: theme.colorScheme.outlineVariant.withValues(alpha: 0.35)),
                        const SizedBox(height: 10),
                      ],
                    ],
                  ],
                ),
              ),
            ],
          );
        },
      ),
    );
  }
}

class _Card extends StatelessWidget {
  final String title;
  final Widget child;
  const _Card({required this.title, required this.child});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: theme.colorScheme.outlineVariant.withValues(alpha: 0.35),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(title, style: theme.textTheme.titleSmall),
          const SizedBox(height: 10),
          child,
        ],
      ),
    );
  }
}


/// Symbolic

class SymbolicFourier {
  static const int _samples = 2000;
  static final double _wMin = -10 * math.pi;
  static final double _wMax = 10 * math.pi;

  static SymbolicResult tryCompute(String raw) {
    final expr0 = _normalize(raw);
    var expr = _rewrite(expr0);
    debugPrint('RAW = <$raw>');
    //final expr = _normalize(raw);
    debugPrint('NORM = <$expr>');
    if (expr.startsWith('+')) expr = expr.substring(1);
    // Unary minus at head: treat as constant multiplier -1
    if (expr.startsWith('-')) {
      final rest = expr.substring(1);
      if (rest.isEmpty) {
        return SymbolicResult.fail(inputLatex: r'-', messageLatex: r'\text{Unable to compute}');
      }
      final inner = tryCompute(rest);
      if (!inner.ok) return inner;
      return SymbolicResult.ok(
        inputLatex: r'-\left(' + inner.inputLatex + r'\right)',
        resultLatex: r'-\left(' + inner.resultLatex + r'\right)',
        stepsLatex: _aligned([
          r'X(\omega)=\int_{-\infty}^{\infty}x(t)e^{-j\omega t}\,dt',
          r'x(t)=-g(t)',
          r'\text{Constant multiplier property：}\;\mathcal{F}\{-g(t)\}=-G(\omega)',
          ...inner.stepsLatex,
        ]),
        omegaAxis: inner.omegaAxis,
        spectrumData: inner.spectrumData.map((v) => -v).toList(),
      );
    }
    // Constant: x(t)=C  =>  X(ω)=2π C δ(ω)
    final cst = _matchConstant(expr);
    if (cst != null) {
      final omegaAxis = _linspace(_wMin, _wMax, _samples);
      final cLatex = cst.coeffLatex;
      final resultLatex = cst.isZero
          ? r'0'
          : (cLatex == r'1'
          ? r'2\pi\,\delta(\omega)'
          : r'2\pi\,' + cLatex + r'\,\delta(\omega)');
      return SymbolicResult.ok(
        inputLatex: cst.inputLatex,
        resultLatex: resultLatex,
        stepsLatex: _aligned([
          r'X(\omega)=\int_{-\infty}^{\infty}x(t)e^{-j\omega t}\,dt',
          r'x(t)=C',
          r'X(\omega)=C\int_{-\infty}^{\infty}e^{-j\omega t}\,dt',
          r'\int_{-\infty}^{\infty}e^{-j\omega t}\,dt=2\pi\,\delta(\omega)',
          r'\Rightarrow X(\omega)=2\pi C\,\delta(\omega)',
        ]),
        omegaAxis: omegaAxis,
        spectrumData: List<double>.filled(omegaAxis.length, 0.0),
      );
    }

    // Basic pair shortcut: 1/(t^2 + a^2) (symbolic a allowed) BEFORE PF
    final invT2A2 = _matchInvT2PlusA2(expr);
    if (invT2A2 != null) {
      final omegaAxis = _linspace(_wMin, _wMax, _samples);
      if (invT2A2.aNumeric != null) {
        final a = invT2A2.aNumeric!;
        if (a <= 0) {
          return SymbolicResult.fail(inputLatex: invT2A2.inputLatex, messageLatex: r'\text{Unable to compute（requires }a>0\text{）}');
        }
        final aL = invT2A2.aLatex ?? _numToLatex(a);
        return SymbolicResult.ok(
          inputLatex: invT2A2.inputLatex,
          resultLatex: r'\frac{\pi}{' + aL + r'}e^{-(' + aL + r')|\omega|}',
          stepsLatex: _aligned([
            r'X(\omega)=\int_{-\infty}^{\infty}\frac{1}{t^{2}+a^{2}}e^{-j\omega t}\,dt',
            r'\text{基本对：}\;\mathcal{F}\left\{\frac{1}{t^{2}+a^{2}}\right\}=\frac{\pi}{a}e^{-a|\omega|}\;(a>0)',
            r'\Rightarrow X(\omega)=\frac{\pi}{' + aL + r'}e^{-(' + aL + r')|\omega|}',
          ]),
          omegaAxis: omegaAxis,
          spectrumData: omegaAxis.map((w) => (math.pi / a) * math.exp(-a * w.abs())).toList(),
        );
      }
      if (invT2A2.aSymbol != null) {
        final aS = invT2A2.aSymbol!;
        return SymbolicResult.ok(
          inputLatex: invT2A2.inputLatex,
          resultLatex: r'\frac{\pi}{' + aS + r'}e^{-(' + aS + r')|\omega|}\quad(a>0)',
          stepsLatex: _aligned([
            r'X(\omega)=\int_{-\infty}^{\infty}\frac{1}{t^{2}+a^{2}}e^{-j\omega t}\,dt',
            r'\text{基本对：}\;\mathcal{F}\left\{\frac{1}{t^{2}+a^{2}}\right\}=\frac{\pi}{a}e^{-a|\omega|}\;(a>0)',
            r'\Rightarrow X(\omega)=\frac{\pi}{' + aS + r'}e^{-(' + aS + r')|\omega|}\;(a>0)',
          ]),
          omegaAxis: omegaAxis,
          spectrumData: List<double>.filled(omegaAxis.length, 0.0),
        );
      }
    }

    // Early handling: 1/(t±a)^n (n=1..3). Avoid PF numeric-root noise for repeated roots like (t+3)^2.
    final invShiftEarly = _matchInvTShiftPowSigned(expr);
    if (invShiftEarly != null) {
      final sign = invShiftEarly.sign;
      final a = invShiftEarly.a;
      final n = invShiftEarly.n;

      final omegaAxisLocal = _linspace(_wMin, _wMax, _samples);
      final baseRes = _computeInvTPowN(n, omegaAxisLocal);
      if (baseRes == null) return SymbolicResult.fail(inputLatex: _toLatexFallback(expr));

      final phase = (sign == '-')
          ? (r'e^{-j\omega ' + a + r'}')
          : (r'e^{j\omega ' + a + r'}');

      final inputLatex = r'\frac{1}{(t' + sign + a + r')^{' + n.toString() + r'}}';

      return SymbolicResult.ok(
        inputLatex: inputLatex,
        resultLatex: phase + r'\left(' + baseRes.resultLatex + r'\right)',
        stepsLatex: _aligned([
          r'X(\omega)=\int_{-\infty}^{\infty}\frac{1}{(t' + sign + a + r')^{' + n.toString() + r'}}e^{-j\omega t}\,dt',
          r'\text{Let }g(t)=\frac{1}{t^{' + n.toString() + r'}}\Rightarrow G(\omega)=' + baseRes.resultLatex,
          r'x(t)=g(t' + (sign == '-' ? '+' : '-') + a + r') \;\Rightarrow\; X(\omega)=e' +
              (sign == '-' ? r'^{-j\omega ' : r'^{j\omega ') +
              a +
              r'}\,G(\omega)',
        ]),
        omegaAxis: omegaAxisLocal,
        spectrumData: baseRes.spectrumData,
      );
    }

// PF: rational polynomial P(t)/Q(t) -> partial fractions (real-linear only).
    // Only triggers when both numerator and denominator are polynomials (implicit multiplication allowed),
    // and Q(t) can be fully factorized into real linear factors (including repeated roots).
    // Unexpand perfect-square quadratic: 1/(t^2 + b*t + c) where discriminant==0 -> 1/(t+s)^2 (computed directly)
    final sqShift = _matchInvPerfectSquareQuadraticShift(expr);
    if (sqShift != null) {
      final s = sqShift; // s=b/2, so t^2+2*s*t+s^2=(t+s)^2
      // x(t)=1/(t+s)^2 = f(t-(-s)), f(t)=1/t^2 => X(w)=e^{+j w s} F{1/t^2}
      final omegaAxisLocal = _linspace(_wMin, _wMax, _samples);
      final baseRes = _computeInvTPowN(2, omegaAxisLocal);
      final sLatex = _numToLatex(s);
      final phase = r'e^{j\omega(' + sLatex + r')}';
      return SymbolicResult.ok(
        inputLatex: r'\frac{1}{t^{2}+' + _numToLatex(2*s) + r't+' + _numToLatex(s*s) + r'}',
        resultLatex: phase + r'\left(-\pi|\omega|\right)',
        stepsLatex: _aligned([
          r'X(\omega)=\int_{-\infty}^{\infty}\frac{1}{t^{2}+' + _numToLatex(2*s) + r't+' + _numToLatex(s*s) + r'}e^{-j\omega t}\,dt',
          r't^{2}+' + _numToLatex(2*s) + r't+' + _numToLatex(s*s) + r'=(t+' + sLatex + r')^{2}',
          r'\Rightarrow\;x(t)=\frac{1}{(t+' + sLatex + r')^{2}}',
          r'\text{Let }f(t)=\frac{1}{t^{2}},\;x(t)=f\bigl(t-(-' + sLatex + r')\bigr)',
          r'\text{Time-shift property: }\mathcal{F}\{f(t-a)\}=e^{-j\omega a}F(\omega)',
          r'\Rightarrow\;X(\omega)=e^{j\omega(' + sLatex + r')}\,\mathcal{F}\left\{\frac{1}{t^{2}}\right\}',
          r'\mathcal{F}\left\{\frac{1}{t^{2}}\right\}=-\pi|\omega|',
        ]),
        omegaAxis: omegaAxisLocal,
        spectrumData: baseRes?.spectrumData ?? const <double>[],
      );
    }

    final pf = _tryPartialFractionRealOnly(expr);
    if (pf != null) {
      if (!pf.ok) {
        return SymbolicResult.fail(
          inputLatex: pf.inputLatex,
          messageLatex: pf.messageLatex ?? r'\text{Unsupported in current version}',
        );
      }
      // Avoid infinite recursion if decomposition returns the same expression.
      if (pf.expandedExpr != null && pf.expandedExpr != expr) {
        final r = tryCompute(pf.expandedExpr!);
        if (!r.ok) return SymbolicResult.fail(inputLatex: pf.inputLatex);
        return SymbolicResult.ok(
          inputLatex: pf.inputLatex,
          resultLatex: r.resultLatex,
          stepsLatex: _aligned([
            r'X(\omega)=\int_{-\infty}^{\infty}x(t)e^{-j\omega t}\,dt',
            r'\text{perform partial fraction decomposition on the rational expression：}',
            pf.decomposeLatex ?? r'\text{(decompose)}',
            r'\text{Calculate the integral of each item according to the definition and then add them together.。}',
            ...r.stepsLatex,
          ]),
          omegaAxis: r.omegaAxis,
          spectrumData: r.spectrumData,
        );
      }
    }

    final omegaAxis = _linspace(_wMin, _wMax, _samples);

// Perfect-square quadratic (expanded): 1/(t^2 + b*t + c) where b^2-4c = 0
    final psq = _matchInvPerfectSquareExpanded(expr);
    if (psq != null) {
      final s = psq.shift; // t^2 + b t + c = (t + s)^2
      final baseRes = _computeInvTPowN(2, omegaAxis);
      if (baseRes == null) return SymbolicResult.fail(inputLatex: _toLatexFallback(expr));

      final sLatex = _numToLatex(s.abs());
      final phase = s >= 0 ? (r'e^{j\omega ' + sLatex + r'}') : (r'e^{-j\omega ' + sLatex + r'}');

      return SymbolicResult.ok(
        inputLatex: r'\frac{1}{(t' + (s >= 0 ? '+' : '-') + sLatex + r')^{2}}',
        resultLatex: phase + r'\left(' + baseRes.resultLatex + r'\right)',
        stepsLatex: _aligned([
          r'X(\omega)=\int_{-\infty}^{\infty}\frac{1}{t^{2}' +
              (psq.b >= 0 ? '+' : '-') +
              _numToLatex(psq.b.abs()) +
              r't+' +
              _numToLatex(psq.c) +
              r'}e^{-j\omega t}\,dt',
          r't^{2}' +
              (psq.b >= 0 ? '+' : '-') +
              _numToLatex(psq.b.abs()) +
              r't+' +
              _numToLatex(psq.c) +
              r'=(t' +
              (s >= 0 ? '+' : '-') +
              sLatex +
              r')^{2}',
          r'\Rightarrow\;x(t)=\frac{1}{(t' +
              (s >= 0 ? '+' : '-') +
              sLatex +
              r')^{2}}=f(t' +
              (s >= 0 ? '+' : '-') +
              sLatex +
              r'),\;f(t)=\frac{1}{t^{2}}',
          s >= 0
              ? r'f(t+a)\Longleftrightarrow e^{j\omega a}F(\omega)'
              : r'f(t-a)\Longleftrightarrow e^{-j\omega a}F(\omega)',
          r'F(\omega)=\mathcal{F}\left\{\frac{1}{t^{2}}\right\}=-\pi|\omega|',
        ]),
        omegaAxis: omegaAxis,
        spectrumData: baseRes.spectrumData,
      );
    }


    // 0) Expand polynomial * u(t): (P(t))*u(t) or u(t)*(P(t))
    final polyU = _matchPolynomialTimesU(expr);
    if (polyU != null) {
      // Only expand if it really is a sum polynomial (avoid loops)
      if (polyU.terms.length >= 2) {
        final expanded = polyU.terms.join('+');
        final r = tryCompute(expanded);
        if (!r.ok) return SymbolicResult.fail(inputLatex: _toLatexFallback(expr));

        return SymbolicResult.ok(
          inputLatex: _toLatexFallback(expr),
          resultLatex: r.resultLatex,
          stepsLatex: [
            r'X(\omega)=\int_{-\infty}^{\infty}x(t)e^{-j\omega t}\,dt',
            r'\text{对 }(P(t))u(t)\text{ Expand using the distributive property of multiplication：}',
            r'x(t)=(' + _toLatexFallback(polyU.poly) + r')u(t)=' +
                polyU.terms.map(_toLatexFallback).join('+'),
            r'\text{再用Linearity：}\;\mathcal{F}\{\sum x_i\}=\sum \mathcal{F}\{x_i\}.',
            ...r.stepsLatex,
          ],
          omegaAxis: r.omegaAxis,
          spectrumData: r.spectrumData,
        );
      }
    }

    // 1) Polynomial / sum splitting: top-level +/-
    final sumParts = _splitTopLevelSum(expr);
    if (sumParts != null) {
      final sub = <SymbolicResult>[];
      for (final p in sumParts) {
        final part = p.trim().startsWith('+') ? p.trim().substring(1) : p.trim();
        final r = tryCompute(part);
        if (!r.ok) return SymbolicResult.fail(inputLatex: _toLatexFallback(expr));
        sub.add(r);
      }

      final combined = List<double>.filled(omegaAxis.length, 0.0);
      bool canCombine = true;
      for (final r in sub) {
        if (r.spectrumData.length != omegaAxis.length) {
          canCombine = false;
          break;
        }
      }
      if (canCombine) {
        for (final r in sub) {
          for (int i = 0; i < omegaAxis.length; i++) {
            combined[i] += r.spectrumData[i];
          }
        }
      }

      final steps = <String>[
        r'X(\omega)=\int_{-\infty}^{\infty}x(t)e^{-j\omega t}\,dt',
        r'\text{Linearity：}\;\mathcal{F}\{x_1+x_2+\cdots\}=\mathcal{F}\{x_1\}+\mathcal{F}\{x_2\}+\cdots',
      ];
      for (int i = 0; i < sub.length; i++) {
        steps.add(r'X_' + (i + 1).toString() + r'(\omega)=' + sub[i].resultLatex);
      }

      return SymbolicResult.ok(
        inputLatex: _toLatexFallback(expr),
        resultLatex: sub.map((r) => '(' + r.resultLatex + ')').join('+'),
        stepsLatex: steps,
        omegaAxis: omegaAxis,
        spectrumData: canCombine ? combined : List<double>.filled(omegaAxis.length, 0.0),
      );
    }

    // 2) Convolution (black dot) – derivation only (numeric plot is placeholder)
    if (expr.contains('•') || expr.contains(r'\bullet') || expr.contains('∙') || expr.contains('⋆')) {
      final conv = _splitConvolutionTopLevel(expr);
      if (conv != null) {
        final fExpr = conv.$1.trim();
        final gExpr = conv.$2.trim();

        final fRes = tryCompute(fExpr);
        if (!fRes.ok) {
          return SymbolicResult.fail(inputLatex: _toLatexFallback(expr));
        }
        final gRes = tryCompute(gExpr);
        if (!gRes.ok) {
          return SymbolicResult.fail(inputLatex: _toLatexFallback(expr));
        }

        // Numeric spectrum: only multiply if both sides provide compatible numeric spectra on the same omega grid.
        List<double> spec = const <double>[];
        if (fRes.spectrumData.isNotEmpty &&
            gRes.spectrumData.isNotEmpty &&
            fRes.spectrumData.length == gRes.spectrumData.length &&
            fRes.omegaAxis.isNotEmpty &&
            gRes.omegaAxis.isNotEmpty &&
            fRes.omegaAxis.length == gRes.omegaAxis.length) {
          bool axisCompatible = true;
          final n = fRes.omegaAxis.length;
          // compare a few anchor points to avoid heavy loops
          for (final k in <int>[0, n ~/ 2, n - 1]) {
            if ((fRes.omegaAxis[k] - gRes.omegaAxis[k]).abs() > 1e-9) {
              axisCompatible = false;
              break;
            }
          }
          if (axisCompatible) {
            spec = List<double>.generate(
              fRes.spectrumData.length,
                  (i) => fRes.spectrumData[i] * gRes.spectrumData[i],
            );
          }
        }

        return SymbolicResult.ok(
          inputLatex: r'(' + fRes.inputLatex + r')\bullet(' + gRes.inputLatex + r')',
          resultLatex: r'(' + fRes.resultLatex + r')\cdot(' + gRes.resultLatex + r')',
          stepsLatex: _aligned([
            r'X(\omega)=\int_{-\infty}^{\infty}x(t)e^{-j\omega t}\,dt',
            r'x(t)=(f\bullet g)(t)=\int_{-\infty}^{\infty}f(\tau)\,g(t-\tau)\,d\tau',
            r'X(\omega)=\int_{-\infty}^{\infty}\left[\int_{-\infty}^{\infty}f(\tau)\,g(t-\tau)\,d\tau\right]e^{-j\omega t}\,dt',
            r'\text{Swap integrals and let }u=t-\tau\Rightarrow t=u+\tau,\;dt=du',
            r'=\left[\int_{-\infty}^{\infty}f(\tau)e^{-j\omega \tau}\,d\tau\right]\left[\int_{-\infty}^{\infty}g(u)e^{-j\omega u}\,du\right]',
            r'=F(\omega)\,G(\omega)',
            r'F(\omega)=' + fRes.resultLatex,
            r'G(\omega)=' + gRes.resultLatex,
          ]),
          omegaAxis: (spec.isNotEmpty && fRes.omegaAxis.isNotEmpty) ? fRes.omegaAxis : omegaAxis,
          spectrumData: (spec.isNotEmpty) ? spec : (omegaAxis.map((w) => w.abs()).toList()),
        );
      }
    }

    // 3) Dirac delta δ(t), δ(t-a)
    if (_isDelta0(expr)) {
      return SymbolicResult.ok(
        inputLatex: r'\delta(t)',
        resultLatex: r'1',
        stepsLatex: _aligned([
          r'X(\omega)=\int_{-\infty}^{\infty}\delta(t)e^{-j\omega t}\,dt',
          r'=1',
        ]),
        omegaAxis: omegaAxis,
        spectrumData: List<double>.filled(omegaAxis.length, 0.0),
      );
    }
    final dShift = _matchDeltaShift(expr);
    if (dShift != null) {
      final a = dShift;
      return SymbolicResult.ok(
        inputLatex: r'\delta(t-' + a + r')',
        resultLatex: r'e^{-j\omega ' + a + r'}',
        stepsLatex: _aligned([
          r'X(\omega)=\int_{-\infty}^{\infty}\delta(t-' + a + r')e^{-j\omega t}\,dt',
          r'=e^{-j\omega ' + a + r'}',
        ]),
        omegaAxis: omegaAxis,
        spectrumData: List<double>.filled(omegaAxis.length, 0.0),
      );
    }

    // 4) Unit step u(t), u(t±a)
    if (_isU0(expr)) {
      return SymbolicResult.ok(
        inputLatex: r'u(t)',
        resultLatex: r'\pi\delta(\omega)+\frac{1}{j\omega}',
        stepsLatex: _aligned([
          r'X(\omega)=\int_{-\infty}^{\infty}u(t)e^{-j\omega t}\,dt=\int_{0}^{\infty}e^{-j\omega t}\,dt',
          r'=\frac{1}{j\omega}+\pi\delta(\omega)',
        ]),
        omegaAxis: omegaAxis,
        spectrumData: omegaAxis.map((w) => w.abs()).toList(),
      );
    }

    final uSigned = _matchUTShiftSigned(expr);
    if (uSigned != null) {
      // u(t-a) => lower=a, factor e^{-jωa}
      // u(t+a) => u(t-(-a)) => lower=-a, factor e^{+jωa}
      final sign = uSigned.sign;
      final a = uSigned.a;

      final phase = (sign == '-')
          ? (r'e^{-j\omega ' + a + r'}')
          : (r'e^{j\omega ' + a + r'}');

      final inputLatex = (sign == '-')
          ? (r'u(t-' + a + r')')
          : (r'u(t+' + a + r')');

      final lower = (sign == '-') ? a : ('-' + a);

      return SymbolicResult.ok(
        inputLatex: inputLatex,
        resultLatex: phase + r'\left(\pi\delta(\omega)+\frac{1}{j\omega}\right)',
        stepsLatex: _aligned([
          r'X(\omega)=\int_{-\infty}^{\infty}u(t' +
              (sign == '-' ? '-' : '+') +
              a +
              r')e^{-j\omega t}\,dt',
          r'=\int_{' + lower + r'}^{\infty}e^{-j\omega t}\,dt',
          r'\text{Let }\tau=t-(' + lower + r')\Rightarrow t=\tau+(' + lower + r')',
          r'=e^{-j\omega(' + lower + r')}\int_{0}^{\infty}e^{-j\omega\tau}\,d\tau',
          r'=e^{-j\omega(' +
              lower +
              r')}\left(\frac{1}{j\omega}+\pi\delta(\omega)\right)',
          r'\Rightarrow X(\omega)=' + phase + r'\left(\pi\delta(\omega)+\frac{1}{j\omega}\right)',
        ]),
        omegaAxis: omegaAxis,
        spectrumData: omegaAxis.map((w) => w.abs()).toList(),
      );
    }

    // 5) exp(a t) u(t)  (a can be numeric or symbolic)
    final expU = _matchExpU(expr);
    if (expU != null) {
      final aStr = expU.a; // could be "-6" or "a"
      final aNum = double.tryParse(aStr);

      if (aNum != null) {
        final aLatex = _numToLatex(aNum);
        if (aNum >= 0) {
          return SymbolicResult.fail(
            inputLatex: r'e^{' + aLatex + r't}u(t)',
            messageLatex: r'\text{Unable to compute（requires }\mathrm{Re}(a)<0\text{）}',
          );
        }
        return SymbolicResult.ok(
          inputLatex: r'e^{' + aLatex + r't}u(t)',
          resultLatex: r'\frac{1}{-(' + aLatex + r')+j\omega}',
          stepsLatex: _aligned([
            r'X(\omega)=\int_{-\infty}^{\infty}x(t)e^{-j\omega t}\,dt',
            r'=\int_{0}^{\infty}e^{' + aLatex + r't}\cdot e^{-j\omega t}\,dt',
            r'=\int_{0}^{\infty}e^{(' + aLatex + r'-j\omega)t}\,dt',
            r'=\left[\frac{1}{-(' + aLatex + r')+j\omega}e^{(' + aLatex + r'-j\omega)t}\right]_{0}^{\infty}',
            r'=\frac{1}{-(' + aLatex + r')+j\omega}\quad(\mathrm{Re}(a)<0)',
          ]),
          omegaAxis: omegaAxis,
          spectrumData: omegaAxis
              .map((w) => 1.0 / math.sqrt((-aNum) * (-aNum) + w * w))
              .toList(),
        );
      } else {
        // symbolic a
        return SymbolicResult.ok(
          inputLatex: r'e^{' + aStr + r't}u(t)',
          resultLatex: r'\frac{1}{-(' + aStr + r')+j\omega}\quad(\mathrm{Re}(a)<0)',
          stepsLatex: _aligned([
            r'X(\omega)=\int_{0}^{\infty}e^{(' + aStr + r'-j\omega)t}\,dt',
            r'=\frac{1}{-(' + aStr + r')+j\omega}\quad(\mathrm{Re}(a)<0)',
          ]),
          omegaAxis: omegaAxis,
          spectrumData: List<double>.filled(omegaAxis.length, 0.0),
        );
      }
    }

    // 6) 1/(t±a)^n  (n=1/2/3) via time shift from 1/t^n
    final invShift = _matchInvTShiftPowSigned(expr);
    if (invShift != null) {
      final sign = invShift.sign;
      final a = invShift.a;
      final n = invShift.n;

      final baseRes = _computeInvTPowN(n, omegaAxis);
      if (baseRes == null) return SymbolicResult.fail(inputLatex: _toLatexFallback(expr));

      final phase = (sign == '-')
          ? (r'e^{-j\omega ' + a + r'}')
          : (r'e^{j\omega ' + a + r'}');

      final inputLatex = r'\frac{1}{(t' + sign + a + r')^{' + n.toString() + r'}}';

      return SymbolicResult.ok(
        inputLatex: inputLatex,
        resultLatex: phase + r'\left(' + baseRes.resultLatex + r'\right)',
        stepsLatex: _aligned([
          r'X(\omega)=\int_{-\infty}^{\infty}\frac{1}{(t' +
              sign +
              a +
              r')^{' +
              n.toString() +
              r'}}e^{-j\omega t}\,dt',
          r'\text{令 }f(t)=\frac{1}{t^{' + n.toString() + r'}}\Rightarrow x(t)=f(t' + sign + a + r')',
          (sign == '-')
              ? r'f(t-a)\Longleftrightarrow e^{-j\omega a}F(\omega)'
              : r'f(t+a)=f(t-(-a))\Longleftrightarrow e^{j\omega a}F(\omega)',
          r'F(\omega)=\mathcal{F}\left\{\frac{1}{t^{' + n.toString() + r'}}\right\}=' + baseRes.resultLatex,
          r'\Rightarrow X(\omega)=' + phase + r'F(\omega)',
        ]),
        omegaAxis: omegaAxis,
        // amplitude plot placeholder (phase ignored), reuse base magnitude
        spectrumData: baseRes.spectrumData,
      );
    }

    // 7) 1/t^n (n=1/2/3)
    final invN = _matchInvTPowN(expr);
    if (invN != null) {
      final baseRes = _computeInvTPowN(invN, omegaAxis);
      if (baseRes == null) return SymbolicResult.fail(inputLatex: _toLatexFallback(expr));
      return baseRes;
    }

    // 8) c*t^n*u(t) (n>=0)
    final coeffPowU = _matchCoeffTPowNU(expr);
    if (coeffPowU != null) {
      final c = coeffPowU.c;
      final n = coeffPowU.n;

      final base = tryCompute('t^$n*u(t)');
      if (!base.ok) return SymbolicResult.fail(inputLatex: _toLatexFallback(expr));

      final cLatex = _numToLatex(c);

      return SymbolicResult.ok(
        inputLatex: cLatex + r'\,' + base.inputLatex,
        resultLatex: cLatex + r'\left(' + base.resultLatex + r'\right)',
        stepsLatex: [
          r'X(\omega)=\int_{-\infty}^{\infty}x(t)e^{-j\omega t}\,dt',
          r'x(t)=' + cLatex + r'\,t^{' + n.toString() + r'}u(t)',
          r'\text{Constant multiplier property：}\;\mathcal{F}\{c\,x(t)\}=c\,X(\omega)',
          ...base.stepsLatex,
        ],
        omegaAxis: base.omegaAxis,
        spectrumData: base.spectrumData.map((v) => v * c).toList(),
      );
    }

    // 9) t^n u(t) (formal)
    final nPowU = _matchTPowNU(expr);
    if (nPowU != null) {
      final nStr = nPowU.toString();
      return SymbolicResult.ok(
        inputLatex: r't^{' + nStr + r'}u(t)',
        resultLatex: r'j^n\pi\delta^{(n)}(\omega)+\frac{n!}{(j\omega)^{n+1}}',
        stepsLatex: _aligned([
          r'X(\omega)=\int_{-\infty}^{\infty}t^n u(t)e^{-j\omega t}\,dt=\int_{0}^{\infty}t^n e^{-j\omega t}\,dt',
          r'U(\omega)=\int_{0}^{\infty}e^{-j\omega t}\,dt=\frac{1}{j\omega}+\pi\delta(\omega)',
          r'\int_{0}^{\infty}t^n e^{-j\omega t}\,dt=j^n\frac{d^n}{d\omega^n}U(\omega)',
          r'\Rightarrow X(\omega)=j^n\pi\delta^{(n)}(\omega)+\frac{n!}{(j\omega)^{n+1}}',
        ]),
        omegaAxis: omegaAxis,
        spectrumData: omegaAxis.map((w) => w.abs()).toList(),
      );
    }

    // 10) sin(at+phi), cos(at+phi)
    final trig = _matchTrig(expr);
    if (trig != null) {
      final kind = trig.kind;
      final a = trig.a;
      final phi = trig.phi;

      if (kind == 'cos') {
        return SymbolicResult.ok(
          inputLatex: r'\cos((' + a + r')t' + _fmtPhase(phi) + r')',
          resultLatex:
          r'\pi\left(e^{j(' +
              _phiNoSign(phi) +
              r')}\delta(\omega-(' +
              a +
              r'))+e^{-j(' +
              _phiNoSign(phi) +
              r')}\delta(\omega+(' +
              a +
              r'))\right)',
          stepsLatex: _aligned([
            r'X(\omega)=\int_{-\infty}^{\infty}\cos((' + a + r')t' + _fmtPhase(phi) + r')e^{-j\omega t}\,dt',
            r'\cos(\theta)=\frac{e^{j\theta}+e^{-j\theta}}{2}',
            r'=\frac{e^{j\phi}}{2}\int e^{-j(\omega-a)t}\,dt+\frac{e^{-j\phi}}{2}\int e^{-j(\omega+a)t}\,dt',
            r'\int_{-\infty}^{\infty}e^{-j\beta t}\,dt=2\pi\delta(\beta)',
            r'\Rightarrow X(\omega)=\pi\left(e^{j\phi}\delta(\omega-a)+e^{-j\phi}\delta(\omega+a)\right)',
          ]),
          omegaAxis: omegaAxis,
          spectrumData: List<double>.filled(omegaAxis.length, 0.0),
        );
      } else {
        return SymbolicResult.ok(
          inputLatex: r'\sin((' + a + r')t' + _fmtPhase(phi) + r')',
          resultLatex:
          r'\frac{\pi}{j}\left(e^{j(' +
              _phiNoSign(phi) +
              r')}\delta(\omega-(' +
              a +
              r'))-e^{-j(' +
              _phiNoSign(phi) +
              r')}\delta(\omega+(' +
              a +
              r'))\right)',
          stepsLatex: _aligned([
            r'X(\omega)=\int_{-\infty}^{\infty}\sin((' + a + r')t' + _fmtPhase(phi) + r')e^{-j\omega t}\,dt',
            r'\sin(\theta)=\frac{e^{j\theta}-e^{-j\theta}}{2j}',
            r'=\frac{e^{j\phi}}{2j}\int e^{-j(\omega-a)t}\,dt-\frac{e^{-j\phi}}{2j}\int e^{-j(\omega+a)t}\,dt',
            r'\int_{-\infty}^{\infty}e^{-j\beta t}\,dt=2\pi\delta(\beta)',
            r'\Rightarrow X(\omega)=\frac{\pi}{j}\left(e^{j\phi}\delta(\omega-a)-e^{-j\phi}\delta(\omega+a)\right)',
          ]),
          omegaAxis: omegaAxis,
          spectrumData: List<double>.filled(omegaAxis.length, 0.0),
        );
      }
    }

    // 11) sin/cos(at+phi)*u(t)
    final trigU = _matchTrigTimesU(expr);
    if (trigU != null) {
      final kind = trigU.kind;
      final a = trigU.a;
      final phi = trigU.phi;

      final inputLatex = (kind == 'sin')
          ? (r'\sin((' + a + r')t' + _fmtPhase(phi) + r')u(t)')
          : (r'\cos((' + a + r')t' + _fmtPhase(phi) + r')u(t)');

      final steps = <String>[
        r'X(\omega)=\int_{-\infty}^{\infty}x(t)e^{-j\omega t}\,dt',
        r'=\int_{0}^{\infty}' +
            (kind == 'sin' ? r'\sin' : r'\cos') +
            r'((' +
            a +
            r')t' +
            _fmtPhase(phi) +
            r')\cdot e^{-j\omega t}\,dt',
        (kind == 'sin')
            ? r'\sin(\theta)=\frac{e^{j\theta}-e^{-j\theta}}{2j}'
            : r'\cos(\theta)=\frac{e^{j\theta}+e^{-j\theta}}{2}',
        r'\int_{0}^{\infty}e^{-j\beta t}\,dt=\frac{1}{j\beta}+\pi\delta(\beta)',
      ];

      final phiNo = _phiNoSign(phi);

      final resLatex = (kind == 'sin')
          ? (r'\frac{1}{2j}\left(e^{j(' +
          phiNo +
          r')}\left[\frac{1}{j(\omega-(' +
          a +
          r'))}+\pi\delta(\omega-(' +
          a +
          r'))\right]-e^{-j(' +
          phiNo +
          r')}\left[\frac{1}{j(\omega+(' +
          a +
          r'))}+\pi\delta(\omega+(' +
          a +
          r'))\right]\right)')
          : (r'\frac{1}{2}\left(e^{j(' +
          phiNo +
          r')}\left[\frac{1}{j(\omega-(' +
          a +
          r'))}+\pi\delta(\omega-(' +
          a +
          r'))\right]+e^{-j(' +
          phiNo +
          r')}\left[\frac{1}{j(\omega+(' +
          a +
          r'))}+\pi\delta(\omega+(' +
          a +
          r'))\right]\right)');

      return SymbolicResult.ok(
        inputLatex: inputLatex,
        resultLatex: resLatex,
        stepsLatex: steps,
        omegaAxis: omegaAxis,
        spectrumData: List<double>.filled(omegaAxis.length, 0.0),
      );
    }

    return SymbolicResult.fail(inputLatex: _toLatexFallback(expr));
  }


  // 1/t^n (n=1/2/3)

  static SymbolicResult? _computeInvTPowN(int n, List<double> omegaAxis) {
    if (n == 1) {
      return SymbolicResult.ok(
        inputLatex: r'\frac{1}{t}',
        resultLatex: r'-j\pi\,\mathrm{sgn}(\omega)',
        stepsLatex: _aligned([
          r'X(\omega)=\int_{-\infty}^{\infty}\frac{1}{t}\cdot e^{-j\omega t}\,dt',
          r'=\int_{-\infty}^{\infty}\frac{\cos(\omega t)}{t}\,dt-j\int_{-\infty}^{\infty}\frac{\sin(\omega t)}{t}\,dt',
          r'\int_{-\infty}^{\infty}\frac{\cos(\omega t)}{t}\,dt=0',
          r'\int_{-\infty}^{\infty}\frac{\sin(\omega t)}{t}\,dt=\pi\,\mathrm{sgn}(\omega)',
          r'\Rightarrow X(\omega)=-j\pi\,\mathrm{sgn}(\omega)',
        ]),
        omegaAxis: omegaAxis,
        spectrumData: omegaAxis.map((_) => math.pi).toList(),
      );
    }
    if (n == 2) {
      return SymbolicResult.ok(
        inputLatex: r'\frac{1}{t^{2}}',
        resultLatex: r'-\pi|\omega|',
        stepsLatex: _aligned([
          r'X(\omega)=\int_{-\infty}^{\infty}\frac{1}{t^{2}}e^{-j\omega t}\,dt',
          r'\text{Let }y(t)=\frac{1}{t}\Rightarrow y^{\prime}(t)=-\frac{1}{t^{2}}',
          r'\mathcal{F}\{y^{\prime}(t)\}=j\omega\,Y(\omega)\quad(\text{boundary term }=0)',
          r'\Rightarrow \mathcal{F}\left\{\frac{1}{t^{2}}\right\}=-j\omega\,\mathcal{F}\left\{\frac{1}{t}\right\}=-\pi|\omega|',
        ]),
        omegaAxis: omegaAxis,
        spectrumData: omegaAxis.map((w) => math.pi * w.abs()).toList(),
      );
    }
    if (n == 3) {
      return SymbolicResult.ok(
        inputLatex: r'\frac{1}{t^{3}}',
        resultLatex: r'\frac{j\pi}{2}\,\omega|\omega|',
        stepsLatex: _aligned([
          r'X(\omega)=\int_{-\infty}^{\infty}\frac{1}{t^{3}}e^{-j\omega t}\,dt',
          r'\text{Let }y(t)=\frac{1}{t^{2}}\Rightarrow y^{\prime}(t)=-\frac{2}{t^{3}}',
          r'\mathcal{F}\{y^{\prime}(t)\}=\int_{-\infty}^{\infty}y^{\prime}(t)e^{-j\omega t}\,dt=j\omega\,Y(\omega)',
          r'\Rightarrow \mathcal{F}\left\{\frac{1}{t^{3}}\right\}=-\frac{1}{2}j\omega\,\mathcal{F}\left\{\frac{1}{t^{2}}\right\}',
          r'=-\frac{1}{2}j\omega(-\pi|\omega|)=\frac{j\pi}{2}\omega|\omega|',
        ]),
        omegaAxis: omegaAxis,
        spectrumData: omegaAxis.map((w) => (math.pi / 2.0) * w.abs() * w.abs()).toList(),
      );
    }
    return null;
  }


  /// Normalization & rewriting

  static String _normalize(String expr) {
    var x = expr.trim();

    // whitespace / Chinese brackets
    x = x.replaceAll(' ', '');
    x = x.replaceAll('（', '(').replaceAll('）', ')');
    x = x.replaceAll('−', '-');
    x = x.replaceAll('，', ',');
    x = x.replaceAll('、', ',');

    // collapse repeated signs: ++ -> +, -- -> +, +-/-+ -> -
    while (true) {
      final before = x;
      x = x.replaceAll('++', '+');
      x = x.replaceAll('--', '+');
      x = x.replaceAll('+-', '-');
      x = x.replaceAll('-+', '-');
      if (x == before) break;
    }

    // LaTeX tokens -> plain
    x = x.replaceAll(r'\cdot', '*');
    x = x.replaceAll(r'\bullet', '•');
    x = x.replaceAll(r'\pi', 'pi');
    x = x.replaceAll('π', 'pi');
    x = x.replaceAll(r'\left', '');
    x = x.replaceAll(r'\right', '');

    // delta alias (optional)
    x = x.replaceAll('Δ', 'delta');

    // \frac{A}{B} -> (A)/(B) (nested)
    if (x.contains(r'\frac')) {
      x = _latexFracToPlain(x);
    }

    // frac(A,B) -> (A)/(B) (function-style, nested)
    if (x.contains('frac(')) {
      x = _funcFracToPlain(x);
    }

    // (1)/(...) -> 1/(...)
    x = x.replaceAll('(1)/', '1/');
    x = x.replaceAll('(+1)/', '1/');
    x = x.replaceAll('(-1)/', '-1/');

    // e^{...} / e^(...) -> exp(...)
    x = _latexEToExp(x);

    // implicit multiplications (best-effort)
    x = x.replaceAllMapped(RegExp(r'(\d)(t)'), (m) => '${m.group(1)}*t');
    x = x.replaceAllMapped(RegExp(r'(\d)\('), (m) => '${m.group(1)}*(');
    // implicit multiplication: t( -> t*( ; )( -> )*( ; )t -> )*t ; t t -> t*t
    x = x.replaceAll('t(', 't*(');
    x = x.replaceAll(')(', ')*(');
    x = x.replaceAll(')t', ')*t');
    x = x.replaceAll('tt', 't*t');

    return x;
  }

  static String _rewrite(String expr) => expr;


  /// LaTeX helpers

  static String _latexFracToPlain(String s) {
    String out = s;
    while (true) {
      final idx = out.indexOf(r'\frac');
      if (idx < 0) break;

      int i = idx + r'\frac'.length;
      if (i >= out.length || out[i] != '{') break;

      final a = _readLatexGroup(out, i);
      if (a == null) break;
      final aText = a.text;
      i = a.end;

      if (i >= out.length || out[i] != '{') break;
      final b = _readLatexGroup(out, i);
      if (b == null) break;
      final bText = b.text;
      i = b.end;

      final replaced = '($aText)/($bText)';
      out = out.substring(0, idx) + replaced + out.substring(i);
    }
    return out;
  }


  // frac(A,B) -> (A)/(B)  (supports nested parentheses)
  static String _funcFracToPlain(String s) {
    String out = s;
    while (true) {
      final idx = out.indexOf('frac(');
      if (idx < 0) break;

      int i = idx + 5; // after "frac("
      int depth = 1;
      int comma = -1;

      for (; i < out.length; i++) {
        final ch = out[i];
        if (ch == '(') depth++;
        if (ch == ')') depth--;
        if (depth == 1 && ch == ',' && comma < 0) comma = i;
        if (depth == 0) break;
      }
      if (comma < 0 || i >= out.length || out[i] != ')') break;

      final a = out.substring(idx + 5, comma).trim();
      final b = out.substring(comma + 1, i).trim();
      final repl = '($a)/($b)';
      out = out.substring(0, idx) + repl + out.substring(i + 1);
    }
    return out;
  }

  static _Group? _readLatexGroup(String s, int startBrace) {
    if (startBrace < 0 || startBrace >= s.length || s[startBrace] != '{') return null;
    int depth = 0;
    final buf = StringBuffer();
    for (int i = startBrace; i < s.length; i++) {
      final ch = s[i];
      if (ch == '{') {
        depth++;
        if (depth > 1) buf.write(ch);
      } else if (ch == '}') {
        depth--;
        if (depth == 0) {
          return _Group(buf.toString(), i + 1);
        }
        buf.write(ch);
      } else {
        buf.write(ch);
      }
    }
    return null;
  }

  static String _latexEToExp(String s) {
    var out = s;

    // e^{...} -> exp(...)
    while (true) {
      final idx = out.indexOf('e^{');
      if (idx < 0) break;
      final g = _readLatexGroup(out, idx + 2); // points to '{'
      if (g == null) break;
      final inside = g.text;
      out = out.substring(0, idx) + 'exp($inside)' + out.substring(g.end);
    }

    // e^(...) -> exp(...)
    out = out.replaceAllMapped(RegExp(r'e\^\(([^()]*)\)'), (m) => 'exp(${m.group(1)!})');

    return out;
  }

  /// =========================
  /// Top-level sum split
  /// =========================
  static List<String>? _splitTopLevelSum(String expr) {
    int depth = 0;
    final parts = <String>[];
    int start = 0;

    for (int i = 0; i < expr.length; i++) {
      final ch = expr[i];
      if (ch == '(') depth++;
      if (ch == ')') depth--;
      if (depth == 0 && (ch == '+' || ch == '-') && i != 0) {
        parts.add(expr.substring(start, i));
        start = i;
      }
    }
    if (start < expr.length) parts.add(expr.substring(start));
    if (parts.length < 2) return null;

    final cleaned = parts.map((p) => p.trim()).where((p) => p.isNotEmpty).toList();
    return cleaned.length >= 2 ? cleaned : null;
  }


  /// Matchers

  static bool _isU0(String expr) => expr == 'u(t)' || expr == 'u(t+0)' || expr == 'u(t-0)';
  static bool _isDelta0(String expr) => expr == 'δ(t)' || expr == 'delta(t)';

  static String? _matchDeltaShift(String expr) {
    final m = RegExp(r'^(?:δ|delta)\(t-([0-9]+(?:\.[0-9]+)?|[A-Za-z_]\w*)\)$')
        .firstMatch(expr);
    return m?.group(1);
  }
  // Match 1/(t^2 + a^2) (a numeric or symbolic) and 1/(t^2 + K) (K numeric).

  static _PerfectSquareExpanded? _matchInvPerfectSquareExpanded(String expr) {
    // Recognize expanded perfect square: 1/(t^2 + b*t + c) with numeric b,c and b^2-4c=0.
    final div = _splitTopLevelDivision(expr);
    if (div == null) return null;
    if (div.$1.trim() != '1') return null;

    var denom = div.$2.trim();
    denom = _stripOuterParen(denom).replaceAll(' ', '');

    // t^2 + b*t + c  (allow "*t" or implicit "t")
    final m = RegExp(r'^t\^2([+-][0-9]+(?:\.[0-9]+)?)\*?t([+-][0-9]+(?:\.[0-9]+)?)$').firstMatch(denom);
    if (m == null) return null;

    final b = double.tryParse(m.group(1)!);
    final c = double.tryParse(m.group(2)!);
    if (b == null || c == null) return null;

    final disc = b * b - 4.0 * c;
    if (disc.abs() > 1e-8) return null;

    final s = b / 2.0;
    if ((c - s * s).abs() > 1e-6) return null;

    return _PerfectSquareExpanded(shift: s, b: b, c: c);
  }

  static _InvT2A2? _matchInvT2PlusA2(String expr) {
    // Accept forms like: 1/(t^2+a^2), 1/(t^2+9), 1/(t^2+(a)^2)
    final m = RegExp(r'^1/\(t\^2\+(.+)\)$').firstMatch(expr);
    if (m == null) return null;
    final tail = m.group(1)!.trim();

    // numeric K: 1/(t^2+K) => a = sqrt(K)
    final k = double.tryParse(tail);
    if (k != null) {
      if (k <= 0) return _InvT2A2(inputLatex: r'\frac{1}{t^{2}+' + _numToLatex(k) + r'}');
      final kL = _numToLatex(k);
      return _InvT2A2(
        inputLatex: r'\frac{1}{t^{2}+' + kL + r'}',
        aNumeric: math.sqrt(k),
        aLatex: r'\sqrt{' + kL + r'}',
      );
    }

    // a^2 or 4^2
    final mPow2 = RegExp(r'^([A-Za-z_]\w*|[+-]?\d+(?:\.\d+)?)\^2$').firstMatch(tail);
    if (mPow2 != null) {
      final base = mPow2.group(1)!;
      final baseNum = double.tryParse(base);
      if (baseNum != null) {
        return _InvT2A2(
          inputLatex: r'\frac{1}{t^{2}+(' + _numToLatex(baseNum) + r')^{2}}',
          aNumeric: baseNum.abs(),
        );
      }
      return _InvT2A2(inputLatex: r'\frac{1}{t^{2}+' + base + r'^{2}}', aSymbol: base);
    }

    // (a)^2
    final mParen = RegExp(r'^\(([^()]+)\)\^2$').firstMatch(tail);
    if (mParen != null) {
      final base = mParen.group(1)!;
      final baseNum = double.tryParse(base);
      if (baseNum != null) {
        return _InvT2A2(
          inputLatex: r'\frac{1}{t^{2}+(' + _numToLatex(baseNum) + r')^{2}}',
          aNumeric: baseNum.abs(),
        );
      }
      if (RegExp(r'^[A-Za-z_]\w*$').hasMatch(base)) {
        return _InvT2A2(inputLatex: r'\frac{1}{t^{2}+' + base + r'^{2}}', aSymbol: base);
      }
    }

    // If tail is exactly like a2 (user shortcut), accept a=?? -> treat as a^2
    final mShort = RegExp(r'^([A-Za-z_]\w*)2$').firstMatch(tail);
    if (mShort != null) {
      final base = mShort.group(1)!;
      return _InvT2A2(inputLatex: r'\frac{1}{t^{2}+' + base + r'^{2}}', aSymbol: base);
    }
    return null;
  }

  static _SignedShift? _matchUTShiftSigned(String expr) {
    final m = RegExp(r'^u\(t([+-])([0-9]+(?:\.[0-9]+)?|[A-Za-z_]\w*)\)$').firstMatch(expr);
    if (m == null) return null;
    return _SignedShift(m.group(1)!, m.group(2)!);
  }

  // exp(a*t)*u(t) / u(t)*exp(a*t) / exp(at)*u(t) / exp(-6t)*u(t)
  static _ExpATU? _matchExpU(String expr) {
    final m1 = RegExp(r'^exp\((.+)\)\*u\(t\)$').firstMatch(expr);
    final m2 = RegExp(r'^u\(t\)\*exp\((.+)\)$').firstMatch(expr);

    String? inside;
    if (m1 != null) inside = m1.group(1);
    if (m2 != null) inside = m2.group(1);
    if (inside == null) return null;

    final a = _extractATCoeff(inside.trim());
    if (a == null) return null;

    return _ExpATU(a);
  }

  static String? _extractATCoeff(String inside) {
    var s = inside.replaceAll(' ', '');

    // "-6*t" or "a*t"
    final mStar = RegExp(r'^([+-]?(?:\d+(?:\.\d+)?|[A-Za-z_]\w*))\*t$').firstMatch(s);
    if (mStar != null) return mStar.group(1)!;

    // "-6t" or "at"
    final mNoStar = RegExp(r'^([+-]?(?:\d+(?:\.\d+)?|[A-Za-z_]\w*))t$').firstMatch(s);
    if (mNoStar != null) return mNoStar.group(1)!;

    // "t" / "-t"
    if (s == 't' || s == '+t') return '1';
    if (s == '-t') return '-1';

    return null;
  }

  // 1/(t±a)^n and 1/(t±a)
  static _InvShiftPow? _matchInvTShiftPowSigned(String expr) {
    // 1/(t+a)
    final m1 = RegExp(r'^1/\(t([+-])([0-9]+(?:\.[0-9]+)?|[A-Za-z_]\w*)\)$').firstMatch(expr);
    if (m1 != null) return _InvShiftPow(m1.group(1)!, m1.group(2)!, 1);

    // 1/(t+a)^n
    final m2 = RegExp(r'^1/\(t([+-])([0-9]+(?:\.[0-9]+)?|[A-Za-z_]\w*)\)\^(?:\(?\{?(\d+)\}?\)?)$')
        .firstMatch(expr);
    if (m2 != null) {
      final n = int.tryParse(m2.group(3)!);
      if (n == null || n < 1 || n > 3) return null;
      return _InvShiftPow(m2.group(1)!, m2.group(2)!, n);
    }

    // 1/((t+a)^n)
    final m3 = RegExp(r'^1/\(\(t([+-])([0-9]+(?:\.[0-9]+)?|[A-Za-z_]\w*)\)\^(?:\(?\{?(\d+)\}?\)?)\)$')
        .firstMatch(expr);
    if (m3 != null) {
      final n = int.tryParse(m3.group(3)!);
      if (n == null || n < 1 || n > 3) return null;
      return _InvShiftPow(m3.group(1)!, m3.group(2)!, n);
    }

    return null;
  }

  static int? _matchInvTPowN(String expr) {
    // allow: 1/t, 1/t^2, 1/t^(2), 1/(t^2), t^(-2)
    final m1 = RegExp(r'^1/\(?t\)?(?:\^\(?(\d+)\)?)?$').firstMatch(expr);
    if (m1 != null) return int.tryParse(m1.group(1) ?? '1');

    final m2 = RegExp(r'^1/\(t\^\(?(\d+)\)?\)$').firstMatch(expr);
    if (m2 != null) return int.tryParse(m2.group(1)!);

    final m3 = RegExp(r'^t\^\(-(\d+)\)$').firstMatch(expr);
    if (m3 != null) return int.tryParse(m3.group(1)!);

    return null;
  }

  // t^n*u(t), u(t)*t^n, t^n u(t)
  static int? _matchTPowNU(String expr) {
    final m1 = RegExp(r'^t\^\(?(\d+)\)?\*u\(t\)$').firstMatch(expr);
    if (m1 != null) return int.tryParse(m1.group(1)!);

    final m2 = RegExp(r'^u\(t\)\*t\^\(?(\d+)\)?$').firstMatch(expr);
    if (m2 != null) return int.tryParse(m2.group(1)!);

    final m3 = RegExp(r'^t\^\(?(\d+)\)?u\(t\)$').firstMatch(expr);
    if (m3 != null) return int.tryParse(m3.group(1)!);

    // t*u(t)
    final m4 = RegExp(r'^t\*u\(t\)$').firstMatch(expr);
    if (m4 != null) return 1;

    return null;
  }

  // c*t^n*u(t) / c*t*u(t) / ct*u(t)
  static _CoeffPowU? _matchCoeffTPowNU(String expr) {
    final m1 = RegExp(r'^([+-]?\d+(?:\.\d+)?)\*t\^\(?(\d+)\)?\*u\(t\)$').firstMatch(expr);
    if (m1 != null) return _CoeffPowU(double.parse(m1.group(1)!), int.parse(m1.group(2)!));

    final m2 = RegExp(r'^([+-]?\d+(?:\.\d+)?)\*t\*u\(t\)$').firstMatch(expr);
    if (m2 != null) return _CoeffPowU(double.parse(m2.group(1)!), 1);

    final m3 = RegExp(r'^([+-]?\d+(?:\.\d+)?)t\*u\(t\)$').firstMatch(expr);
    if (m3 != null) return _CoeffPowU(double.parse(m3.group(1)!), 1);

    return null;
  }

  // Polynomial * u(t) expansion
  static _PolyU? _matchPolynomialTimesU(String expr) {
    // (poly)*u(t) or poly*u(t)
    final m1 = RegExp(r'^\(?(.+)\)?\*u\(t\)$').firstMatch(expr);
    if (m1 != null) {
      final poly = m1.group(1)!.trim();
      final terms = _splitTopLevelSum(poly) ?? [poly];
      final expanded = terms.map((t) => _polyTermTimesU(t, rightU: true)).toList();
      return _PolyU(poly, expanded);
    }

    // u(t)*(poly)
    final m2 = RegExp(r'^u\(t\)\*\(?(.+)\)?$').firstMatch(expr);
    if (m2 != null) {
      final poly = m2.group(1)!.trim();
      final terms = _splitTopLevelSum(poly) ?? [poly];
      final expanded = terms.map((t) => _polyTermTimesU(t, rightU: false)).toList();
      return _PolyU(poly, expanded);
    }

    return null;
  }

  static String _polyTermTimesU(String termRaw, {required bool rightU}) {
    var term = termRaw.trim();
    if (term.isEmpty) return 'u(t)';
    if (term.startsWith('+')) term = term.substring(1);

    // pure number
    if (RegExp(r'^[+-]?\d+(\.\d+)?$').hasMatch(term)) {
      if (term == '1') return 'u(t)';
      if (term == '-1') return '-u(t)';
      return rightU ? '$term*u(t)' : 'u(t)*$term';
    }

    // t or -t
    if (term == 't') return rightU ? 't*u(t)' : 'u(t)*t';
    if (term == '-t') return rightU ? '-t*u(t)' : 'u(t)*(-t)';

    // 2t / -3t (no *)
    final mNoStar = RegExp(r'^([+-]?\d+(\.\d+)?)t$').firstMatch(term);
    if (mNoStar != null) {
      final c = mNoStar.group(1)!;
      return rightU ? '${c}*t*u(t)' : 'u(t)*${c}*t';
    }

    // keep as-is, just attach u(t)
    return rightU ? '$term*u(t)' : 'u(t)*$term';
  }

  // trig
  static _Trig? _matchTrig(String expr) {
    final m = RegExp(r'^(sin|cos)\((.+)\)$').firstMatch(expr);
    if (m == null) return null;
    final kind = m.group(1)!;
    final inside = m.group(2)!;
    final lin = _parseLinearForm(inside);
    if (lin == null) return null;
    return _Trig(kind, lin.a, lin.b);
  }

  static _Trig? _matchTrigTimesU(String expr) {
    final m1 = RegExp(r'^(sin|cos)\((.+)\)\*u\(t\)$').firstMatch(expr);
    if (m1 != null) {
      final lin = _parseLinearForm(m1.group(2)!);
      if (lin == null) return null;
      return _Trig(m1.group(1)!, lin.a, lin.b);
    }

    final m2 = RegExp(r'^u\(t\)\*(sin|cos)\((.+)\)$').firstMatch(expr);
    if (m2 != null) {
      final lin = _parseLinearForm(m2.group(2)!);
      if (lin == null) return null;
      return _Trig(m2.group(1)!, lin.a, lin.b);
    }
    return null;
  }

  // Parse inside trig as a*t + b
  //- sin(t) => a=1,b=0 ; sin(-t)=>a=-1,b=0
  // - sin(t+phi) => a=1,b=+phi
  // - sin(at) / sin(a*t) / sin(-6t) etc.
  static _LinForm? _parseLinearForm(String inside) {
    var s = inside.replaceAll(' ', '');

    if (s == 't' || s == '+t') return const _LinForm('1', '0');
    if (s == '-t') return const _LinForm('-1', '0');

    final mTpm = RegExp(r'^([+-]?)t([+-].+)$').firstMatch(s);
    if (mTpm != null) {
      final sign = mTpm.group(1);
      final a = (sign == '-') ? '-1' : '1';
      final b = mTpm.group(2)!;
      return _LinForm(a, b);
    }

    // normalize t*a -> a*t
    final mTA = RegExp(r'^t\*([A-Za-z_]\w*|\-?[0-9]+(?:\.[0-9]+)?)((?:[+-].+)?)$')
        .firstMatch(s);
    if (mTA != null) {
      s = '${mTA.group(1)}*t${mTA.group(2) ?? ''}';
    }

    final m = RegExp(r'^(.+)\*t([+-].+)?$').firstMatch(s);
    if (m != null) {
      final aRaw = m.group(1)!;
      final bRaw = (m.group(2) == null || m.group(2)!.isEmpty) ? '0' : m.group(2)!;
      final a = aRaw.isEmpty ? '1' : aRaw;
      return _LinForm(a, bRaw);
    }

    final mNoStar =
    RegExp(r'^([+-]?(?:[0-9]+(?:\.[0-9]+)?|[A-Za-z_]\w*))t([+-].+)?$').firstMatch(s);
    if (mNoStar != null) {
      final a = mNoStar.group(1)!;
      final b = (mNoStar.group(2) == null || mNoStar.group(2)!.isEmpty) ? '0' : mNoStar.group(2)!;
      return _LinForm(a, b);
    }

    return null;
  }


  //Formatting

  static List<String> _aligned(List<String> lines) => lines;
  static String _toLatexFallback(String expr) => expr;

  static String _fmtPhase(String b) {
    if (b == '0') return '';
    if (b.startsWith('+') || b.startsWith('-')) return b;
    return '+$b';
  }

  static String _phiNoSign(String b) {
    if (b == '0') return '0';
    if (b.startsWith('+')) return b.substring(1);
    return b;
  }

  static String _numToLatex(double x) {
    if ((x - x.roundToDouble()).abs() < 1e-12) return x.round().toInt().toString();
    var s = x.toString();
    if (s.contains('.')) {
      s = s.replaceAll(RegExp(r'0+$'), '');
      s = s.replaceAll(RegExp(r'\.$'), '');
    }
    return s;
  }


  // Detect Q(t) = c2*t^2 + c0 with c1≈0 and c0/c2 = a^2 > 0. Return k0=a^2.
  static double? _matchT2PlusA2Numeric(_Poly q) {
    if (q.deg != 2) return null;
    final c2 = q.coeff(2);
    if (c2.abs() < 1e-12) return null;
    final c1 = q.coeff(1) / c2;
    if (c1.abs() > 1e-8) return null;
    final k0 = q.coeff(0) / c2;
    if (k0 <= 0) return null;
    return k0;
  }

  static List<double> _linspace(double a, double b, int n) {
    if (n <= 1) return [a];
    final step = (b - a) / (n - 1);
    return List.generate(n, (i) => a + i * step);
  }
  //pf

  static _PFRealResult? _tryPartialFractionRealOnly(String expr) {
    final div = _splitTopLevelDivision(expr);
    if (div == null) return null;

    final pStr = div.$1;
    final qStr = div.$2;

    final p = _Poly.parse(pStr);
    final q = _Poly.parse(qStr);
    final inputLatex = r'\frac{' + (p?.toLatex() ?? _toLatexFallback(pStr)) + r'}{' + (q?.toLatex() ?? _toLatexFallback(qStr)) + r'}';

    if (q == null || q.isZero) {
      // Not a pure polynomial in t -> let other matchers handle.
      return null;
    }
    // Enable PF only for deg(Q) >= 2; linear denominators are handled by direct rules.
    if (q.deg < 2) return null;
    if (p == null) {
      // Not a pure polynomial in t -> let other matchers handle.
      return null;
    }

    //  needed for degree >= 2 denominators.

    if (q.deg < 2) return null;


    // Long division if deg(P) >= deg(Q)
    var quotient = _Poly.zero();
    var remainder = p;
    if (p.deg >= q.deg) {
      final dm = remainder.divmod(q);
      quotient = dm.$1;
      remainder = dm.$2;
    }

    // If remainder is zero -> polynomial
    if (remainder.isZero) {
      return _PFRealResult.ok(
        inputLatex: inputLatex,
        decomposeLatex: inputLatex + '=' + quotient.toLatex(),
        expandedExpr: quotient.toExprString(),
      );
    }

    // True fraction required
    if (remainder.deg >= q.deg) {
      return _PFRealResult.unsupported(
        inputLatex: inputLatex,
        messageLatex: r'\text{Unsupported in current version：无法得到严格真分式（请检查输入）。}',
      );
    }

    // Root finding over complex; reject if any non-real roots.
    final roots = _durandKernerRoots(q);
    if (roots == null || roots.isEmpty) {
      return _PFRealResult.unsupported(
        inputLatex: inputLatex,
        messageLatex: r'\text{Unsupported in current version：Failed to solve }Q(t)=0\text{ 的根。}',
      );
    }

    const imagEps = 1e-6;
    final realRoots = <double>[];
    for (final z in roots) {
      if (z.im.abs() < imagEps) realRoots.add(z.re);
    }

    if (realRoots.length != q.deg) {
      // Special-case: allow irreducible quadratic exactly of form t^2 + a^2 (numeric)
      final k0 = _matchT2PlusA2Numeric(q);
      if (k0 != null && remainder.deg == 0) {
        final c2 = q.coeff(2);
        final scale = remainder.coeff(0) / c2; // because q = c2*(t^2 + k0)
        final term = '${_numToPlain(scale)}/(t^2+${_numToPlain(k0)})';
        final expanded = (quotient.isZero) ? term : (quotient.toExprString() + '+' + term);

        final inputLatex = r'\frac{' + p.toLatex() + r'}{' + q.toLatex() + r'}';
        final decompLatex = quotient.isZero
            ? (inputLatex + '=' + r'\frac{' + _numToLatex(scale) + r'}{t^{2}+' + _numToLatex(k0) + r'}')
            : (inputLatex + '=' + quotient.toLatex() + '+' + r'\frac{' + _numToLatex(scale) + r'}{t^{2}+' + _numToLatex(k0) + r'}');

        return _PFRealResult.ok(
          inputLatex: inputLatex,
          decomposeLatex: decompLatex,
          expandedExpr: expanded,
        );
      }

      return _PFRealResult.unsupported(
        inputLatex: _toLatexFallback(expr),
        messageLatex: r'\text{Unsupported in current version：}Q(t)\text{ 含quadratic irreducible factor（complex conjugate roots），且不属于 }t^{2}+a^{2}\text{ 形式。}',
      );
    }

    // Cluster real roots into repeated roots
    realRoots.sort();
    const clusterEps = 1e-5;
    final factors = <_RealRootFactor>[];
    for (final r0 in realRoots) {
      if (factors.isEmpty) {
        factors.add(_RealRootFactor(r0, 1));
      } else {
        final last = factors.last;
        if ((r0 - last.a).abs() < clusterEps) {
          last.m += 1;
          last.a = (last.a * (last.m - 1) + r0) / last.m;
        } else {
          factors.add(_RealRootFactor(r0, 1));
        }
      }
    }

    // Verify Q(t) ≈ leading * Π(t-a)^m
    final qHat = _Poly.fromRealRootFactors(factors, leading: q.leading);
    final err = qHat.maxAbsDiff(q);
    if (err > 1e-3 * (1 + q.maxAbsCoeff)) {
      return _PFRealResult.unsupported(
        inputLatex: inputLatex,
        messageLatex: r'\text{Unsupported in current version：}Q(t)\text{ 未能稳定分解为实线性因子。}',
      );
    }

    // Build unknown list A_{i,k}
    final unknowns = <_UnknownAK>[];
    for (final f in factors) {
      for (int k = 1; k <= f.m; k++) {
        unknowns.add(_UnknownAK(f.a, k));
      }
    }
    final M = unknowns.length;
    final N = math.max(M + 4, M * 2);

    // Sample points for least squares (avoid poles)
    final samples = <double>[];
    double t = -math.max(6.0, factors.length * 3.0);
    while (samples.length < N) {
      bool nearPole = false;
      for (final f in factors) {
        if ((t - f.a).abs() < 1e-3) { nearPole = true; break; }
      }
      if (!nearPole) samples.add(t);
      t += 1.0;
      if (t > 5000) break;
    }
    if (samples.length < M) {
      return _PFRealResult.unsupported(
        inputLatex: inputLatex,
        messageLatex: r'\text{Unsupported in current version：Not enough sample points to solve for coefficients。}',
      );
    }

    // Build linear system A x = y where y = R(t)/Q(t)
    final A = List.generate(samples.length, (_) => List<double>.filled(M, 0.0));
    final y = List<double>.filled(samples.length, 0.0);

    for (int i = 0; i < samples.length; i++) {
      final tj = samples[i];
      final pj = remainder.eval(tj);
      final qj = q.eval(tj);
      if (qj.abs() < 1e-12) continue;
      y[i] = pj / qj;

      for (int u = 0; u < M; u++) {
        final a = unknowns[u].a;
        final k = unknowns[u].k;
        A[i][u] = 1.0 / math.pow(tj - a, k);
      }
    }

    final coeffs = _solveLeastSquares(A, y);
    if (coeffs == null) {
      return _PFRealResult.unsupported(
        inputLatex: inputLatex,
        messageLatex: r'\text{Unsupported in current version：Failed to solve for coefficients。}',
      );
    }

    // Build expanded expression and latex decomposition
    final termsExpr = <String>[];
    final termsLatex = <String>[];

    for (int i = 0; i < M; i++) {
      final c = coeffs[i];
      if (c.abs() < 1e-10) continue;

      final a = unknowns[i].a;
      final k = unknowns[i].k;

      final cStr = _numToPlain(c);
      final aStr = _numToPlain(a);

      // (t-a)
      final shift = (a >= 0) ? '-$aStr' : '+${_numToPlain(-a)}';
      final denom = (k == 1) ? 't$shift' : '(t$shift)^$k';
      termsExpr.add('$cStr/($denom)');

      final cL = _numToLatex(c);
      final aL = _numToLatex(a);
      final shiftL = (a >= 0) ? ('-' + aL) : ('+' + _numToLatex(-a));
      if (k == 1) {
        termsLatex.add(r'\frac{' + cL + r'}{t' + shiftL + r'}');
      } else {
        termsLatex.add(r'\frac{' + cL + r'}{(t' + shiftL + r')^{' + k.toString() + r'}}');
      }
    }

    if (termsExpr.isEmpty) {
      return _PFRealResult.unsupported(
        inputLatex: inputLatex,
        messageLatex: r'\text{Unsupported in current version：分解数值不稳定（得到空项）。}',
      );
    }

    final expandedProper = termsExpr.join('+');
    final properLatex = r'\frac{' + remainder.toLatex() + r'}{' + q.toLatex() + r'}=' + termsLatex.join('+');

    if (!quotient.isZero) {
      final expanded = quotient.toExprString() + '+' + expandedProper;
      final decomp = inputLatex + '=' + quotient.toLatex() + '+(' + properLatex + ')';
      return _PFRealResult.ok(inputLatex: inputLatex, decomposeLatex: decomp, expandedExpr: expanded);
    }

    return _PFRealResult.ok(inputLatex: inputLatex, decomposeLatex: properLatex, expandedExpr: expandedProper);
  }

  static (String, String)? _splitTopLevelDivision(String expr) {
    // Prefer explicit top-level '/'
    int depth = 0;
    for (int i = 0; i < expr.length; i++) {
      final ch = expr[i];
      if (ch == '(') depth++;
      if (ch == ')') depth--;
      if (depth == 0 && ch == '/') {
        final left = expr.substring(0, i).trim();
        final right = expr.substring(i + 1).trim();
        if (left.isEmpty || right.isEmpty) return null;
        return (_stripOuterParen(left), _stripOuterParen(right));
      }
    }

    // Also accept top-level frac(A,B) even if normalize missed it.
    if (expr.startsWith('frac(') && expr.endsWith(')')) {
      int i = 5; // after 'frac('
      depth = 1;
      int comma = -1;
      for (; i < expr.length; i++) {
        final ch = expr[i];
        if (ch == '(') depth++;
        if (ch == ')') depth--;
        if (depth == 1 && ch == ',' && comma < 0) comma = i;
        if (depth == 0) break;
      }
      if (comma > 0 && i == expr.length - 1) {
        final a = expr.substring(5, comma).trim();
        final b = expr.substring(comma + 1, i).trim();
        if (a.isNotEmpty && b.isNotEmpty) {
          return (_stripOuterParen(a), _stripOuterParen(b));
        }
      }
    }

    return null;
  }
  static String _stripOuterParen(String s) {
    var x = s.trim();
    while (x.startsWith('(') && x.endsWith(')')) {
      int depth = 0;
      bool ok = true;
      for (int i = 0; i < x.length; i++) {
        final ch = x[i];
        if (ch == '(') depth++;
        if (ch == ')') depth--;
        if (depth == 0 && i != x.length - 1) { ok = false; break; }
      }
      if (!ok) break;
      x = x.substring(1, x.length - 1).trim();
    }
    return x;
  }

  static List<double>? _solveLeastSquares(List<List<double>> A, List<double> y) {
    final m = A.length;
    final n = A[0].length;
    final ata = List.generate(n, (_) => List<double>.filled(n, 0.0));
    final aty = List<double>.filled(n, 0.0);

    for (int i = 0; i < m; i++) {
      final row = A[i];
      for (int j = 0; j < n; j++) {
        aty[j] += row[j] * y[i];
        for (int k = 0; k < n; k++) {
          ata[j][k] += row[j] * row[k];
        }
      }
    }
    return _gaussSolve(ata, aty);
  }

  static List<double>? _gaussSolve(List<List<double>> M, List<double> b) {
    final n = b.length;
    final a = List.generate(n, (i) => List<double>.from(M[i]));
    final x = List<double>.from(b);

    for (int col = 0; col < n; col++) {
      int pivot = col;
      double best = a[col][col].abs();
      for (int r = col + 1; r < n; r++) {
        final v = a[r][col].abs();
        if (v > best) { best = v; pivot = r; }
      }
      if (best < 1e-12) return null;

      if (pivot != col) {
        final tmp = a[pivot]; a[pivot] = a[col]; a[col] = tmp;
        final tb = x[pivot]; x[pivot] = x[col]; x[col] = tb;
      }

      final diag = a[col][col];
      for (int c = col; c < n; c++) a[col][c] /= diag;
      x[col] /= diag;

      for (int r = 0; r < n; r++) {
        if (r == col) continue;
        final f = a[r][col];
        if (f.abs() < 1e-12) continue;
        for (int c = col; c < n; c++) a[r][c] -= f * a[col][c];
        x[r] -= f * x[col];
      }
    }
    return x;
  }

  static String _numToPlain(double x) {
    if ((x - x.roundToDouble()).abs() < 1e-10) return x.round().toInt().toString();
    var s = x.toStringAsPrecision(12);
    s = s.replaceAll(RegExp(r'0+$'), '');
    s = s.replaceAll(RegExp(r'\.$'), '');
    return s;
  }

  static List<_C>? _durandKernerRoots(_Poly q) {
    final n = q.deg;
    if (n <= 0) return const [];
    if (n == 1) {
      // a1*t + a0 = 0
      final a1 = q.coeff(1);
      final a0 = q.coeff(0);
      if (a1.abs() < 1e-12) return null;
      return [_C(-a0 / a1, 0)];
    }

    // Normalize to monic for stability
    final lead = q.leading;
    if (lead.abs() < 1e-12) return null;
    final p = q.scale(1.0 / lead);

    // initial guesses on circle
    final roots = List<_C>.generate(n, (k) {
      final theta = 2 * math.pi * k / n;
      final r = 0.4 + 0.9; // radius
      return _C(r * math.cos(theta), r * math.sin(theta));
    });

    const int iters = 200;
    const double eps = 1e-12;

    for (int iter = 0; iter < iters; iter++) {
      bool allSmall = true;
      for (int i = 0; i < n; i++) {
        var denom = _C(1, 0);
        for (int j = 0; j < n; j++) {
          if (j == i) continue;
          denom = denom * (roots[i] - roots[j]);
        }
        final f = p.evalC(roots[i]);
        final delta = f / denom;
        roots[i] = roots[i] - delta;
        if (delta.abs() > eps) allSmall = false;
      }
      if (allSmall) break;
    }

    return roots;
  }
}


// Helper structs

// PF + Polynomial utilities

class _PFRealResult {
  final bool ok;
  final String inputLatex;
  final String? decomposeLatex;
  final String? expandedExpr;
  final String? messageLatex;

  const _PFRealResult._({
    required this.ok,
    required this.inputLatex,
    this.decomposeLatex,
    this.expandedExpr,
    this.messageLatex,
  });

  factory _PFRealResult.ok({
    required String inputLatex,
    required String decomposeLatex,
    required String expandedExpr,
  }) =>
      _PFRealResult._(
        ok: true,
        inputLatex: inputLatex,
        decomposeLatex: decomposeLatex,
        expandedExpr: expandedExpr,
      );

  factory _PFRealResult.unsupported({
    required String inputLatex,
    required String messageLatex,
  }) =>
      _PFRealResult._(
        ok: false,
        inputLatex: inputLatex,
        messageLatex: messageLatex,
      );
}

class _RealRootFactor {
  double a;
  int m;
  _RealRootFactor(this.a, this.m);
}

class _UnknownAK {
  final double a;
  final int k;
  const _UnknownAK(this.a, this.k);
}

// Minimal complex number for Durand–Kerner.
class _C {
  final double re;
  final double im;
  const _C(this.re, this.im);

  _C operator +(Object other) {
    final o = other as _C;
    return _C(re + o.re, im + o.im);
  }

  _C operator -(Object other) {
    final o = other as _C;
    return _C(re - o.re, im - o.im);
  }

  _C operator *(Object other) {
    final o = other as _C;
    return _C(re * o.re - im * o.im, re * o.im + im * o.re);
  }

  _C operator /(Object other) {
    final o = other as _C;
    final d = o.re * o.re + o.im * o.im;
    if (d == 0) return const _C(double.nan, double.nan);
    return _C((re * o.re + im * o.im) / d, (im * o.re - re * o.im) / d);
  }

  double abs() => math.sqrt(re * re + im * im);
}

//Polynomial in variable t with real coefficients.
// coeffs[i] = coefficient of t^i
class _Poly {
  final List<double> coeffs;

  const _Poly(this.coeffs);

  static _Poly zero() => const _Poly([0.0]);

  bool get isZero => coeffs.every((c) => c.abs() < 1e-12);

  int get deg {
    for (int i = coeffs.length - 1; i >= 0; i--) {
      if (coeffs[i].abs() > 1e-12) return i;
    }
    return 0;
  }

  double get leading => coeff(deg);

  double get maxAbsCoeff => coeffs.fold(0.0, (m, v) => math.max(m, v.abs()));

  double coeff(int i) => (i >= 0 && i < coeffs.length) ? coeffs[i] : 0.0;

  _Poly _trim() {
    int d = coeffs.length - 1;
    while (d > 0 && coeffs[d].abs() < 1e-12) d--;
    return _Poly(coeffs.sublist(0, d + 1));
  }

  _Poly add(_Poly other) {
    final n = math.max(coeffs.length, other.coeffs.length);
    final out = List<double>.filled(n, 0.0);
    for (int i = 0; i < n; i++) out[i] = coeff(i) + other.coeff(i);
    return _Poly(out)._trim();
  }

  _Poly sub(_Poly other) {
    final n = math.max(coeffs.length, other.coeffs.length);
    final out = List<double>.filled(n, 0.0);
    for (int i = 0; i < n; i++) out[i] = coeff(i) - other.coeff(i);
    return _Poly(out)._trim();
  }

  _Poly mul(_Poly other) {
    final out = List<double>.filled(coeffs.length + other.coeffs.length - 1, 0.0);
    for (int i = 0; i < coeffs.length; i++) {
      for (int j = 0; j < other.coeffs.length; j++) {
        out[i + j] += coeffs[i] * other.coeffs[j];
      }
    }
    return _Poly(out)._trim();
  }

  _Poly scale(double s) => _Poly(coeffs.map((c) => c * s).toList())._trim();

  _Poly powInt(int n) {
    if (n < 0) throw ArgumentError('negative power not allowed');
    var r = _Poly(const [1.0]);
    var a = this;
    int k = n;
    while (k > 0) {
      if ((k & 1) == 1) r = r.mul(a);
      k >>= 1;
      if (k > 0) a = a.mul(a);
    }
    return r._trim();
  }

  ( _Poly, _Poly ) divmod(_Poly divisor) {
    if (divisor.isZero) throw ArgumentError('divide by zero poly');
    var a = List<double>.from(coeffs);
    final b = divisor._trim().coeffs;
    int da = _Poly(a).deg;
    final db = divisor.deg;
    final leadB = divisor.leading;
    if (da < db) return (_Poly.zero(), _Poly(a)._trim());

    final q = List<double>.filled(da - db + 1, 0.0);
    while (da >= db) {
      final coef = a[da] / leadB;
      final shift = da - db;
      q[shift] = coef;
      for (int i = 0; i <= db; i++) {
        a[i + shift] -= coef * b[i];
      }
      a[da] = 0.0;
      da = _Poly(a).deg;
      if (da == 0 && a[0].abs() < 1e-12) break;
    }
    return (_Poly(q)._trim(), _Poly(a)._trim());
  }

  double eval(double t) {
    double s = 0.0;
    for (int i = coeffs.length - 1; i >= 0; i--) {
      s = s * t + coeffs[i];
    }
    return s;
  }

  _C evalC(_C z) {
    var s = const _C(0, 0);
    for (int i = coeffs.length - 1; i >= 0; i--) {
      s = s * z + _C(coeffs[i], 0);
    }
    return s;
  }

  double maxAbsDiff(_Poly other) {
    final n = math.max(coeffs.length, other.coeffs.length);
    double m = 0.0;
    for (int i = 0; i < n; i++) {
      m = math.max(m, (coeff(i) - other.coeff(i)).abs());
    }
    return m;
  }

  static _Poly fromRealRootFactors(List<_RealRootFactor> factors, {required double leading}) {
    var p = _Poly(const [1.0]);
    for (final f in factors) {
      // (t - a)^m
      final base = _Poly([ -f.a, 1.0 ]);
      p = p.mul(base.powInt(f.m));
    }
    return p.scale(leading);
  }

  String toLatex() {
    final d = deg;
    if (isZero) return '0';
    final parts = <String>[];
    for (int i = d; i >= 0; i--) {
      final c = coeff(i);
      if (c.abs() < 1e-12) continue;
      final sign = (c < 0) ? '-' : '+';
      final absC = c.abs();
      String term;
      if (i == 0) {
        term = _latexNum(absC);
      } else if (i == 1) {
        if ((absC - 1.0).abs() < 1e-12) term = 't';
        else term = _latexNum(absC) + 't';
      } else {
        if ((absC - 1.0).abs() < 1e-12) term = 't^{${i}}';
        else term = _latexNum(absC) + 't^{${i}}';
      }
      parts.add((parts.isEmpty && sign == '+') ? term : sign + term);
    }
    return parts.join('');
  }

  String toExprString() {
    final d = deg;
    if (isZero) return '0';
    final parts = <String>[];
    for (int i = d; i >= 0; i--) {
      final c = coeff(i);
      if (c.abs() < 1e-12) continue;
      final sign = (c < 0) ? '-' : '+';
      final absC = c.abs();
      String term;
      if (i == 0) {
        term = _plainNum(absC);
      } else if (i == 1) {
        if ((absC - 1.0).abs() < 1e-12) term = 't';
        else term = '${_plainNum(absC)}*t';
      } else {
        if ((absC - 1.0).abs() < 1e-12) term = 't^$i';
        else term = '${_plainNum(absC)}*t^$i';
      }
      parts.add((parts.isEmpty && sign == '+') ? term : sign + term);
    }
    return parts.join('');
  }

  static String _plainNum(double x) {
    if ((x - x.roundToDouble()).abs() < 1e-10) return x.round().toInt().toString();
    var s = x.toStringAsPrecision(12);
    s = s.replaceAll(RegExp(r'0+$'), '');
    s = s.replaceAll(RegExp(r'\.$'), '');
    return s;
  }

  static String _latexNum(double x) => _plainNum(x);


  // Implicit multiplication supported: 2t, t(t+1), (t+1)(t-1), 2(t+1)
  static _Poly? parse(String expr) {
    try {
      final lx = _PolyLexer(expr);
      final toks = lx.tokenize();
      final p = _PolyParser(toks);
      final poly = p.parseExpr();
      if (!p.isAtEnd) return null;
      return poly._trim();
    } catch (_) {
      return null;
    }
  }
}

enum _TokType { num, t, plus, minus, mul, pow, lpar, rpar, end }

class _Tok {
  final _TokType type;
  final String text;
  const _Tok(this.type, this.text);
}

class _PolyLexer {
  final String s;
  int i = 0;
  _PolyLexer(this.s);

  bool get done => i >= s.length;

  List<_Tok> tokenize() {
    final out = <_Tok>[];
    while (!done) {
      final ch = s[i];
      if (ch == ' ' || ch == '\n' || ch == '\t') { i++; continue; }
      if (ch == '+') { out.add(const _Tok(_TokType.plus, '+')); i++; continue; }
      if (ch == '-') { out.add(const _Tok(_TokType.minus, '-')); i++; continue; }
      if (ch == '*') { out.add(const _Tok(_TokType.mul, '*')); i++; continue; }
      if (ch == '^') { out.add(const _Tok(_TokType.pow, '^')); i++; continue; }
      if (ch == '(') { out.add(const _Tok(_TokType.lpar, '(')); i++; continue; }
      if (ch == ')') { out.add(const _Tok(_TokType.rpar, ')')); i++; continue; }
      if (ch == 't') { out.add(const _Tok(_TokType.t, 't')); i++; continue; }

      // number (supports decimal)
      if (RegExp(r'[0-9\.]').hasMatch(ch)) {
        final start = i;
        bool dot = false;
        while (!done) {
          final c = s[i];
          if (c == '.') {
            if (dot) break;
            dot = true;
            i++;
            continue;
          }
          if (!RegExp(r'[0-9]').hasMatch(c)) break;
          i++;
        }
        final txt = s.substring(start, i);
        if (txt == '.' || txt.isEmpty) throw FormatException('bad number');
        out.add(_Tok(_TokType.num, txt));
        continue;
      }

      // Unknown symbol => not a polynomial.
      throw FormatException('unknown token: $ch');
    }
    out.add(const _Tok(_TokType.end, ''));
    return _insertImplicitMul(out);
  }

  // Insert '*' when two factors are adjacent: num t, num (, t (, ) t, ) (, etc.
  List<_Tok> _insertImplicitMul(List<_Tok> toks) {
    bool isFactorEnd(_Tok t) =>
        t.type == _TokType.num || t.type == _TokType.t || t.type == _TokType.rpar;
    bool isFactorStart(_Tok t) =>
        t.type == _TokType.num || t.type == _TokType.t || t.type == _TokType.lpar;

    final out = <_Tok>[];
    for (int k = 0; k < toks.length; k++) {
      final cur = toks[k];
      if (out.isNotEmpty) {
        final prev = out.last;
        if (isFactorEnd(prev) && isFactorStart(cur)) {
          // don't insert before end token
          if (cur.type != _TokType.end) out.add(const _Tok(_TokType.mul, '*'));
        }
      }
      out.add(cur);
    }
    return out;
  }
}

class _PolyParser {
  final List<_Tok> toks;
  int p = 0;
  _PolyParser(this.toks);

  bool get isAtEnd => toks[p].type == _TokType.end;

  _Tok get cur => toks[p];

  _Tok eat(_TokType t) {
    if (cur.type != t) throw FormatException('expected $t got ${cur.type}');
    return toks[p++];
  }

  bool match(_TokType t) {
    if (cur.type == t) { p++; return true; }
    return false;
  }

  _Poly parseExpr() {
    var x = parseTerm();
    while (true) {
      if (match(_TokType.plus)) {
        x = x.add(parseTerm());
      } else if (match(_TokType.minus)) {
        x = x.sub(parseTerm());
      } else {
        break;
      }
    }
    return x;
  }

  _Poly parseTerm() {
    var x = parsePow();
    while (true) {
      if (match(_TokType.mul)) {
        x = x.mul(parsePow());
      } else {
        break;
      }
    }
    return x;
  }

  _Poly parsePow() {
    var x = parseUnary();
    if (match(_TokType.pow)) {
      // exponent must be integer number
      final expTok = cur;
      if (expTok.type != _TokType.num) throw FormatException('power exponent must be number');
      eat(_TokType.num);
      final n = int.tryParse(expTok.text);
      if (n == null || n < 0) throw FormatException('power exponent must be nonnegative int');
      x = x.powInt(n);
    }
    return x;
  }

  _Poly parseUnary() {
    if (match(_TokType.plus)) return parseUnary();
    if (match(_TokType.minus)) return _Poly(const [-1.0]).mul(parseUnary());
    return parsePrimary();
  }

  _Poly parsePrimary() {
    if (match(_TokType.num)) {
      final v = double.parse(toks[p - 1].text);
      return _Poly([v]);
    }
    if (match(_TokType.t)) {
      return _Poly(const [0.0, 1.0]);
    }
    if (match(_TokType.lpar)) {
      final x = parseExpr();
      eat(_TokType.rpar);
      return x;
    }
    throw FormatException('bad primary');
  }
}
class _InvT2A2 {
  final String inputLatex;
  final double? aNumeric;
  final String? aSymbol;
  final String? aLatex; // optional display override (e.g. \sqrt{K})

  const _InvT2A2({
    required this.inputLatex,
    this.aNumeric,
    this.aSymbol,
    this.aLatex,
  });
}

// Constant matcher: supports numeric like 3, -2.5 and 'pi'
double? _matchInvPerfectSquareQuadraticShift(String expr) {
  // Matches: 1/(t^2 + b*t + c) where b,c are numeric, and allows optional '*' between b and t.
  final s = expr.replaceAll(' ', '');
  final m = RegExp(r'^1/\(t\^2([+-]\d+(?:\.\d+)?)\*?t([+-]\d+(?:\.\d+)?)\)$').firstMatch(s);
  if (m == null) return null;
  final b = double.tryParse(m.group(1)!);
  final c = double.tryParse(m.group(2)!);
  if (b == null || c == null) return null;
  final disc = b * b - 4.0 * c;
  if (disc.abs() > 1e-8) return null;
  return b / 2.0;
}


_ConstMatch? _matchConstant(String expr) {
  final s = expr.trim();
  if (s.isEmpty) return null;

  if (s == 'pi') {
    return const _ConstMatch(inputLatex: r'\pi', coeffLatex: r'\pi', isZero: false);
  }

  final v = double.tryParse(s);
  if (v != null) {
    final cLatex = SymbolicFourier._numToLatex(v);
    return _ConstMatch(
      inputLatex: cLatex,
      coeffLatex: (v == 1.0) ? r'1' : cLatex,
      isZero: v == 0.0,
    );
  }
  return null;
}



class _ConstMatch {
  final String inputLatex;
  final String coeffLatex;
  final bool isZero;
  const _ConstMatch({required this.inputLatex, required this.coeffLatex, required this.isZero});
}

class _Group {
  final String text;
  final int end;
  const _Group(this.text, this.end);
}

class _SignedShift {
  final String sign; // '+' or '-'
  final String a;
  const _SignedShift(this.sign, this.a);
}


class _PerfectSquareExpanded {
  final double shift;
  final double b;
  final double c;
  const _PerfectSquareExpanded({required this.shift, required this.b, required this.c});
}

class _InvShiftPow {
  final String sign; // '+' or '-'
  final String a;
  final int n;
  const _InvShiftPow(this.sign, this.a, this.n);
}

class _ExpATU {
  final String a;
  const _ExpATU(this.a);
}

class _CoeffPowU {
  final double c;
  final int n;
  const _CoeffPowU(this.c, this.n);
}

class _PolyU {
  final String poly;
  final List<String> terms;
  const _PolyU(this.poly, this.terms);
}

class _LinForm {
  final String a;
  final String b;
  const _LinForm(this.a, this.b);
}

class _Trig {
  final String kind; // sin/cos
  final String a;
  final String phi;
  const _Trig(this.kind, this.a, this.phi);
}

// -------- v45x convolution helper (top-level) --------
// Splits f•g at TOP-LEVEL only (respects parentheses). Supports • ∙ ⋆ and \bullet.
(String, String)? _splitConvolutionTopLevel(String expr) {
  var s = expr.trim();

  // Strip one outer pair of parentheses if it wraps the whole expression.
  String stripOuterOnce(String x) {
    x = x.trim();
    if (!(x.startsWith('(') && x.endsWith(')'))) return x;
    int depth = 0;
    for (int i = 0; i < x.length; i++) {
      final ch = x[i];
      if (ch == '(') depth++;
      if (ch == ')') depth--;
      if (depth == 0 && i != x.length - 1) return x; // outer parens do not wrap all
    }
    return x.substring(1, x.length - 1).trim();
  }

  s = stripOuterOnce(s);

  const ops = ['•', '∙', '⋆'];
  int depth = 0;
  for (int i = 0; i < s.length; i++) {
    final ch = s[i];
    if (ch == '(') depth++;
    if (ch == ')') depth--;
    if (depth != 0) continue;

    if (ops.contains(ch)) {
      final left = s.substring(0, i).trim();
      final right = s.substring(i + 1).trim();
      if (left.isEmpty || right.isEmpty) return null;
      return (left, right);
    }
  }

  // LaTeX bullet: f\bullet g
  final idx = s.indexOf(r'\bullet');
  if (idx != -1) {
    // ensure \bullet is at top level by checking depth up to idx
    depth = 0;
    for (int i = 0; i < idx; i++) {
      final ch = s[i];
      if (ch == '(') depth++;
      if (ch == ')') depth--;
    }
    if (depth == 0) {
      final left = s.substring(0, idx).trim();
      final right = s.substring(idx + r'\bullet'.length).trim();
      if (left.isEmpty || right.isEmpty) return null;
      return (left, right);
    }
  }

  return null;
}
