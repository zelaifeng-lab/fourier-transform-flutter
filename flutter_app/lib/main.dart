import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

import 'fft/fourier_transform_bloc.dart';
import 'fft/fourier_transform_event.dart';
import 'fft/fourier_transform_state.dart';
import 'fft/charts.dart';
import 'responsive.dart';
import 'scrollable_content.dart';

void main() {
  runApp(const AppRoot());
}

class AppRoot extends StatelessWidget {
  const AppRoot({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Fourier Transform',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.indigo),
        useMaterial3: true,
      ),
      home: BlocProvider(
        create: (_) => FourierTransformBloc(),
        child: const HomePage(),
      ),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});
  @override
  State<HomePage> createState() => _HomePageState();
}

class FourierExample {
  final String expression;
  final String title;
  final String category;
  final String inputLatex;
  final String description;

  const FourierExample({
    required this.expression,
    required this.title,
    required this.category,
    required this.inputLatex,
    required this.description,
  });
}

class _HomePageState extends State<HomePage> {
  String _expr = 'sign(t-2)';
  int _cursor = 0;

  int _pow2 = 9;
  static const double _defaultA = 1.0;

  /// Example inputs focus on medium and advanced patterns that are useful
  /// as editable templates for user input.
  static const List<FourierExample> _examples = <FourierExample>[
    FourierExample(
      expression: 'sign(t-2)',
      title: 'Shifted sign',
      category: 'PV distribution',
      inputLatex: r'\mathrm{sign}(t-2)',
      description: 'Shifted sign distribution with a principal-value spectrum.',
    ),
    FourierExample(
      expression: 'rect((t-2)/3)',
      title: 'Shifted rectangular pulse',
      category: 'Window scale and shift',
      inputLatex: r'\mathrm{rect}\left(\frac{t-2}{3}\right)',
      description: 'A finite pulse with non-unit width and center shift.',
    ),
    FourierExample(
      expression: 'tri((t-1)/2)',
      title: 'Shifted triangular pulse',
      category: 'Window scale and shift',
      inputLatex: r'\mathrm{tri}\left(\frac{t-1}{2}\right)',
      description: 'Triangular pulse using both scaling and time shift.',
    ),
    FourierExample(
      expression: 'u(t-2)-u(t-5)',
      title: 'Finite shifted window',
      category: 'Step combination',
      inputLatex: r'u(t-2)-u(t-5)',
      description: 'A rectangular interval written as two shifted steps.',
    ),
    FourierExample(
      expression: 't*(u(t)-u(t-1))',
      title: 'Polynomial finite window',
      category: 'Windowed polynomial',
      inputLatex: r't\,[u(t)-u(t-1)]',
      description: 'Combines polynomial input with a finite step window.',
    ),
    FourierExample(
      expression: '(t^2+2*t+1)*u(t)',
      title: 'Polynomial-step combination',
      category: 'Polynomial times step',
      inputLatex: r'(t^2+2t+1)u(t)',
      description: 'Linearity across several polynomial-step terms.',
    ),
    FourierExample(
      expression: '(t-2)*u(t-2)',
      title: 'Shifted ramp step',
      category: 'Shifted polynomial step',
      inputLatex: r'(t-2)u(t-2)',
      description: 'Moves a ramp edge away from the origin.',
    ),
    FourierExample(
      expression: 't*u(t-2)',
      title: 'Uncentered shifted ramp',
      category: 'Shifted polynomial step',
      inputLatex: r'tu(t-2)',
      description: 'Shows how the polynomial changes after shifting the step edge.',
    ),
    FourierExample(
      expression: 'sin(3*t+2)',
      title: 'Phase-shifted sine',
      category: 'Trig phase',
      inputLatex: r'\sin(3t+2)',
      description: 'Demonstrates phase factors in a sinusoid transform.',
    ),
    FourierExample(
      expression: 'cos(4*t-1)',
      title: 'Phase-shifted cosine',
      category: 'Trig phase',
      inputLatex: r'\cos(4t-1)',
      description: 'Cosine with nonzero phase and shifted frequency impulses.',
    ),
    FourierExample(
      expression: 'sin(t)+cos(2*t)',
      title: 'Trig combination',
      category: 'Linearity',
      inputLatex: r'\sin(t)+\cos(2t)',
      description: 'A compact example of linearity across trigonometric terms.',
    ),
    FourierExample(
      expression: 'sin(t-1)*u(t+3)',
      title: 'Shifted one-sided sine',
      category: 'Trig times shifted step',
      inputLatex: r'\sin(t-1)u(t+3)',
      description: 'Combines sinusoidal phase with a shifted step boundary.',
    ),
    FourierExample(
      expression: 'cos(2*t+1)*u(t-3)',
      title: 'Shifted one-sided cosine',
      category: 'Trig times shifted step',
      inputLatex: r'\cos(2t+1)u(t-3)',
      description: 'A one-sided trigonometric signal with both phase and shift.',
    ),
    FourierExample(
      expression: 'exp(I*(5*t+2))',
      title: 'Complex tone with phase',
      category: 'Modulation',
      inputLatex: r'e^{j(5t+2)}',
      description: 'Pure tone with frequency shift and constant phase.',
    ),
    FourierExample(
      expression: 'exp(I*2*t)*u(t-3)',
      title: 'Modulated shifted step',
      category: 'Modulation plus shift',
      inputLatex: r'e^{j2t}u(t-3)',
      description: 'Combines modulation with a delayed unit step.',
    ),
    FourierExample(
      expression: 'exp(-a*t)*u(t)',
      title: 'Parameterized causal exponential',
      category: 'One-sided exponential',
      inputLatex: r'e^{-at}u(t)',
      description: 'A reusable template for stable one-sided exponentials.',
    ),
    FourierExample(
      expression: 'exp(-a*abs(t))',
      title: 'Parameterized two-sided exponential',
      category: 'Even integrable signal',
      inputLatex: r'e^{-a|t|}',
      description: 'A two-sided decaying exponential with a symbolic parameter.',
    ),
    FourierExample(
      expression: 'exp(I*3*t)*exp(-(t+1)^2)',
      title: 'Modulated shifted Gaussian',
      category: 'Gaussian plus modulation',
      inputLatex: r'e^{j3t}e^{-(t+1)^2}',
      description: 'Uses modulation and time shift on a Gaussian transform pair.',
    ),
    FourierExample(
      expression: 't*exp(-t^2)',
      title: 'Gaussian frequency differentiation',
      category: 'Frequency differentiation',
      inputLatex: r'te^{-t^2}',
      description: 'Shows how multiplication by t differentiates the spectrum.',
    ),
    FourierExample(
      expression: 'frac(1,3*t-2)',
      title: 'Scaled shifted reciprocal',
      category: 'PV rational',
      inputLatex: r'\frac{1}{3t-2}',
      description: 'Principal-value reciprocal with scale and shift.',
    ),
    FourierExample(
      expression: 'frac(1,(3*t-2)^2)',
      title: 'Second-order PV reciprocal',
      category: 'PV rational',
      inputLatex: r'\frac{1}{(3t-2)^2}',
      description: 'A higher-order singular rational distribution.',
    ),
    FourierExample(
      expression: 'frac(2*t+3,t^2+6)',
      title: 'Linear over quadratic',
      category: 'Rational template',
      inputLatex: r'\frac{2t+3}{t^2+6}',
      description: 'Matches a reusable rational transform template.',
    ),
    FourierExample(
      expression: 'frac(2*t,3*t^2+4*t-1)',
      title: 'General quadratic rational',
      category: 'Partial fractions',
      inputLatex: r'\frac{2t}{3t^2+4t-1}',
      description: 'Uses real-root partial fractions and known PV pairs.',
    ),
    FourierExample(
      expression: 'frac(t^5+t^4+t^3,(t+1)(t^2+1)(t+6)(t^2+6))',
      title: 'High-order rational combination',
      category: 'Partial fractions',
      inputLatex: r'\frac{t^5+t^4+t^3}{(t+1)(t^2+1)(t+6)(t^2+6)}',
      description: 'A larger rational example that combines several rule families.',
    ),
    FourierExample(
      expression: 'frac(sin(a*t),pi*t)',
      title: 'Parameterized sinc window',
      category: 'Sinc to rect',
      inputLatex: r'\frac{\sin(a\,t)}{\pi\,t}',
      description: 'A parameterized sinc transform into a frequency window.',
    ),
  ];

  int _exampleIndex = 0;

  @override
  void initState() {
    super.initState();
    //
    final idx = _examples.indexWhere((example) => example.expression == _expr);
    if (idx >= 0) {
      _exampleIndex = idx;
    } else {
      _exampleIndex = 0;
      _expr = _examples[0].expression;
    }
    _cursor = _expr.indexOf('□');
    if (_cursor < 0) _cursor = _expr.length;
  }

  void _setExpr(String next, {int? cursor}) {
    setState(() {
      _expr = next;
      _cursor = (cursor ?? _cursor).clamp(0, _expr.length);
    });
  }

  void _applyExample(int idx) {
    if (_examples.isEmpty) return;
    final i = (idx % _examples.length + _examples.length) % _examples.length;
    setState(() {
      _exampleIndex = i;
      _expr = _examples[i].expression;
      final c = _expr.indexOf('□');
      _cursor = (c >= 0) ? c : _expr.length;
    });
  }

  void _nextExample() => _applyExample(_exampleIndex + 1);

  void _prevExample() => _applyExample(_exampleIndex - 1);

  void _insert(String s) {
    if (_cursor < _expr.length && _expr[_cursor] == '□') {
      final before = _expr.substring(0, _cursor);
      final after = _expr.substring(_cursor + 1);
      _setExpr(before + s + after, cursor: _cursor + s.length);
      return;
    }
    final before = _expr.substring(0, _cursor);
    final after = _expr.substring(_cursor);
    _setExpr(before + s + after, cursor: _cursor + s.length);
  }

  void _backspace() {
    if (_cursor <= 0 || _expr.isEmpty) return;
    final before = _expr.substring(0, _cursor);
    final after = _expr.substring(_cursor);
    final nb = before.substring(0, before.length - 1);
    _setExpr(nb + after, cursor: _cursor - 1);
  }

  void _clear() => _setExpr('', cursor: 0);
  void _left() => _setExpr(_expr, cursor: _cursor - 1);
  void _right() => _setExpr(_expr, cursor: _cursor + 1);

  void _insertFrac() {
    final before = _expr.substring(0, _cursor);
    final after = _expr.substring(_cursor);
    const insert = 'frac(□,□)';
    final next = before + insert + after;
    final numPos = (before + 'frac(').length;
    _setExpr(next, cursor: numPos);
  }

  void _insertAbs() {
    final before = _expr.substring(0, _cursor);
    final after = _expr.substring(_cursor);
    const insert = 'abs(□)';
    final next = before + insert + after;
    final argPos = (before + 'abs(').length;
    _setExpr(next, cursor: argPos);
  }

  (int start, int comma, int end)? _nearestFrac() {
    final left = _expr.substring(0, _cursor);
    final idx = left.lastIndexOf('frac(');
    if (idx < 0) return null;

    int depth = 0;
    int comma = -1;
    for (int i = idx; i < _expr.length; i++) {
      final c = _expr[i];
      if (c == '(') depth++;
      if (c == ')') depth--;
      if (c == ',' && depth == 1 && comma < 0) comma = i;
      if (depth == 0) {
        if (comma < 0) return null;
        return (idx, comma, i);
      }
    }
    return null;
  }

  void _jumpToNum() {
    final frac = _nearestFrac();
    if (frac == null) return;
    _setExpr(_expr, cursor: frac.$1 + 'frac('.length);
  }

  void _jumpToDen() {
    final frac = _nearestFrac();
    if (frac == null) return;
    _setExpr(_expr, cursor: frac.$2 + 1);
  }

  @override
  Widget build(BuildContext context) {
    final currentExample = _examples[_exampleIndex];
    return BlocListener<FourierTransformBloc, FourierTransformState>(
      listenWhen: (p, c) => p.status != c.status,
      listener: (context, state) {
        if (state.status == FourierStatus.success) {
          Navigator.of(context).push(
            MaterialPageRoute(
              builder: (_) => ResultsPage(
                expression: state.expression,
                t: state.t,
                signal: state.signal,
                omega: state.omega,
                dt: state.dt,
                magnitude: state.magnitude,
              ),
            ),
          );
        }
        if (state.status == FourierStatus.failure && state.error != null) {
          ScaffoldMessenger.of(
            context,
          ).showSnackBar(SnackBar(content: Text(state.error!)));
        }
      },
      child: Scaffold(
        appBar: AppBar(title: const Text('Fourier Transform')),
        body: ResponsiveScrollView(
          children: [
            _Section(
              title: 'Expression (preview is the display)',
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Expanded(
                        child: _FormulaPreview(
                          expression: _expr,
                          cursor: _cursor,
                        ),
                      ),
                      const SizedBox(width: 8),
                      Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          SizedBox(
                            width: 112,
                            child: OutlinedButton(
                              onPressed: _prevExample,
                              child: const Text('Previous'),
                            ),
                          ),
                          const SizedBox(height: 8),
                          SizedBox(
                            width: 112,
                            child: OutlinedButton(
                              onPressed: _nextExample,
                              child: const Text('Next'),
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                  const SizedBox(height: 10),
                  Text(
                    'Example ${_exampleIndex + 1}/${_examples.length}',
                    style: Theme.of(context).textTheme.labelLarge,
                  ),
                  const SizedBox(height: 6),
                  ScrollableMathLine(
                    semanticsLabel:
                        'current-example:${currentExample.expression}',
                    latex: r'\displaystyle x(t)=' + currentExample.inputLatex,
                    textStyle: Theme.of(context).textTheme.titleMedium,
                  ),
                  const SizedBox(height: 12),
                  _Keypad(
                    onInsert: _insert,
                    onBackspace: _backspace,
                    onClear: _clear,
                    onLeft: _left,
                    onRight: _right,
                    onInsertFrac: _insertFrac,
                    onInsertAbs: _insertAbs,
                    onJumpNum: _jumpToNum,
                    onJumpDen: _jumpToDen,
                    currentExpr: _expr,
                    cursor: _cursor,
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),
            _Section(
              title: 'FFT size',
              child: Row(
                children: [
                  Expanded(
                    child: Slider(
                      min: 6,
                      max: 13,
                      divisions: 7,
                      value: _pow2.toDouble(),
                      label: 'N=${1 << _pow2}',
                      onChanged: (v) => setState(() => _pow2 = v.round()),
                    ),
                  ),
                  SizedBox(
                    width: 96,
                    child: Text('N=${1 << _pow2}', textAlign: TextAlign.end),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),
            BlocBuilder<FourierTransformBloc, FourierTransformState>(
              builder: (context, state) {
                final running = state.status == FourierStatus.running;
                return FilledButton.icon(
                  onPressed: running
                      ? null
                      : () {
                          context.read<FourierTransformBloc>().add(
                            TransformExpressionRequested(
                              expression: _expr,
                              pow2: _pow2,
                              a: _defaultA,
                            ),
                          );
                        },
                  icon: running
                      ? const SizedBox.square(
                          dimension: 18,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Icon(Icons.play_arrow),
                  label: Text(running ? 'Running…' : 'Run transform'),
                );
              },
            ),
            const SizedBox(height: 8),
            Text(
              '— inserts a fraction skeleton with visible boxes. Active field is highlighted like a formula editor. '
              'Singular functions like 1/t^n are stabilized by clamping near t=0 using eps=dt/2.',
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ],
        ),
      ),
    );
  }
}

class _FormulaPreview extends StatelessWidget {
  final String expression;
  final int cursor;

  const _FormulaPreview({required this.expression, required this.cursor});

  String _latexify(String expr) {
    final c = cursor.clamp(0, expr.length);

    final left = expr.substring(0, c);
    final fracStart = left.lastIndexOf('frac(');
    int activeComma = -1;
    int activeEnd = -1;
    bool inFrac = false;
    bool activeIsNum = true;

    if (fracStart >= 0) {
      int depth = 0;
      int comma = -1;
      for (int i = fracStart; i < expr.length; i++) {
        final ch = expr[i];
        if (ch == '(') depth++;
        if (ch == ')') depth--;
        if (ch == ',' && depth == 1 && comma < 0) comma = i;
        if (depth == 0) {
          activeComma = comma;
          activeEnd = i;
          inFrac = (comma >= 0 && c >= fracStart && c <= i);
          break;
        }
      }
      if (inFrac && activeComma >= 0) {
        activeIsNum = c <= activeComma;
      }
    }

    String convert(String s) {
      String convFrac(String input) {
        int i = 0;
        final buf = StringBuffer();
        while (i < input.length) {
          if (input.startsWith('exp(', i)) {
            i += 4; // after 'exp('
            int depth = 1;
            final argStart = i;
            while (i < input.length) {
              final ch = input[i];
              if (ch == '(') depth++;
              if (ch == ')') depth--;
              if (depth == 0) break;
              i++;
            }

            final complete = i < input.length && input[i] == ')';
            var argRaw = input.substring(argStart, complete ? i : input.length);
            final active = argRaw.contains('|');
            argRaw = argRaw.replaceAll('|', '').trim();

            String expLatex;
            if (argRaw.isEmpty) {
              expLatex = active ? r'\boxed{\phantom{0}}' : r'\square';
            } else {
              final inner = convert(argRaw);
              expLatex = active ? (r'\boxed{' + inner + '}') : inner;
            }

            buf.write('e^{');
            buf.write(expLatex);
            buf.write('}');
            if (complete) i++; // skip ')'
            continue;
          }

          if (input.startsWith('frac(', i)) {
            final localStart = i;
            i += 5;
            int depth = 1;
            final start = i;
            int comma = -1;
            while (i < input.length) {
              final ch = input[i];
              if (ch == '(') depth++;
              if (ch == ')') depth--;
              if (ch == ',' && depth == 1 && comma < 0) comma = i;
              if (depth == 0) break;
              i++;
            }
            if (i >= input.length || input[i] != ')' || comma < 0) {
              buf.write(r'\mathrm{frac}(');
              buf.write(convert(input.substring(start, i)));
              buf.write(')');
              if (i < input.length && input[i] == ')') i++;
              continue;
            }

            final numRaw = input.substring(start, comma).trim();
            final denRaw = input.substring(comma + 1, i).trim();

            String boxify(String raw, {required bool active}) {
              final cleaned = raw.trim();
              if (cleaned.isEmpty || cleaned == '□') {
                return active ? r'\boxed{\phantom{0}}' : r'\square';
              }
              final inner = convert(cleaned);
              return active ? (r'\boxed{' + inner + '}') : inner;
            }

            final isThisActive =
                inFrac &&
                localStart == fracStart &&
                i == activeEnd &&
                comma == activeComma;
            final num = boxify(numRaw, active: isThisActive && activeIsNum);
            final den = boxify(denRaw, active: isThisActive && !activeIsNum);

            buf.write(r'\frac{' + num + '}{' + den + '}');
            i++;
            continue;
          }

          if (input.startsWith('t^(', i)) {
            // Parse t^(...) as a power with exponent box/highlight.
            i += 3; // after 't^('
            int depth = 1;
            final expStart = i;
            while (i < input.length) {
              final ch = input[i];
              if (ch == '(') depth++;
              if (ch == ')') depth--;
              if (depth == 0) break;
              i++;
            }
            if (i >= input.length || input[i] != ')') {
              // malformed, fallback
              buf.write('t^(');
              buf.write(convert(input.substring(expStart, i)));
              if (i < input.length && input[i] == ')') buf.write(')');
              continue;
            }

            var expRaw = input.substring(expStart, i);
            final active = expRaw.contains('|');
            expRaw = expRaw.replaceAll('|', '').trim();

            String expLatex;
            if (expRaw.isEmpty || expRaw == '□') {
              expLatex = active ? r'\boxed{\phantom{0}}' : r'\square';
            } else {
              final inner = convert(expRaw);
              expLatex = active ? (r'\boxed{' + inner + '}') : inner;
            }

            buf.write('t^{');
            buf.write(expLatex);
            buf.write('}');
            i++; // skip ')'
            continue;
          }

          if (input.startsWith('abs(', i)) {
            i += 4; // after 'abs('
            int depth = 1;
            final argStart = i;
            while (i < input.length) {
              final ch = input[i];
              if (ch == '(') depth++;
              if (ch == ')') depth--;
              if (depth == 0) break;
              i++;
            }
            if (i >= input.length || input[i] != ')') {
              buf.write(r'\left|');
              buf.write(convert(input.substring(argStart, i)));
              buf.write(r'\right|');
              if (i < input.length && input[i] == ')') i++;
              continue;
            }

            var argRaw = input.substring(argStart, i);
            final active = argRaw.contains('|');
            argRaw = argRaw.replaceAll('|', '').trim();
            String inner;
            if (argRaw.isEmpty || argRaw == '□') {
              inner = active ? r'\boxed{\phantom{0}}' : r'\square';
            } else {
              final converted = convert(argRaw);
              inner = active ? (r'\boxed{' + converted + '}') : converted;
            }

            buf.write(r'\left|');
            buf.write(inner);
            buf.write(r'\right|');
            i++; // skip ')'
            continue;
          }

          if (input[i] == '|') {
            buf.write(r'\vert ');
            i++;
            continue;
          }
          if (input[i] == '□') {
            buf.write(r'\square');
            i++;
            continue;
          }
          buf.write(input[i]);
          i++;
        }
        return buf.toString();
      }

      var out = convFrac(s);

      out = out.replaceAllMapped(RegExp(r'\bpi\b'), (_) => r'\pi');
      out = out.replaceAllMapped(RegExp(r'\bsin\b'), (_) => r'\sin');
      out = out.replaceAllMapped(RegExp(r'\bcos\b'), (_) => r'\cos');
      out = out.replaceAllMapped(RegExp(r'\bsign\b'), (_) => r'\mathrm{sign}');
      out = out.replaceAllMapped(RegExp(r'\brect\b'), (_) => r'\mathrm{rect}');
      out = out.replaceAllMapped(RegExp(r'\btri\b'), (_) => r'\mathrm{tri}');
      out = out.replaceAllMapped(RegExp(r'\bexp\b'), (_) => r'\mathrm{exp}');
      out = out.replaceAllMapped(RegExp(r'\bdelta\b'), (_) => r'\delta');
      // Imaginary unit
      out = out.replaceAllMapped(RegExp(r'\bI\b'), (_) => r'i');

      // Recursive conversion can revisit already converted LaTeX commands inside frac(...).
      // Collapse doubled command prefixes before handing the string to flutter_math_fork.
      for (final command in <String>[
        'pi',
        'sin',
        'cos',
        'delta',
        'mathrm',
        'frac',
        'sqrt',
        'left',
        'right',
      ]) {
        out = out.replaceAll('\\\\$command', '\\$command');
      }

      // NOTE: constant 'a' substitution removed (UI no longer sets a).

      out = out.replaceAll('*', r'\,');
      out = out.replaceAll('•', r'\bullet ');
      return out;
    }

    final withCursor = expr.substring(0, c) + '|' + expr.substring(c);
    final trimmed = withCursor.trim();
    return trimmed.isEmpty ? r'\vert' : convert(trimmed);
  }

  @override
  Widget build(BuildContext context) {
    final latex = _latexify(expression);

    return Container(
      padding: const EdgeInsets.all(3),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [Colors.indigo, Colors.purple, Colors.blueAccent],
        ),
        borderRadius: BorderRadius.circular(18),
      ),
      child: Container(
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: Theme.of(context).colorScheme.surface,
          borderRadius: BorderRadius.circular(16),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Text('Preview', style: Theme.of(context).textTheme.titleMedium),
                const Spacer(),
              ],
            ),
            const SizedBox(height: 8),
            ScrollableMathLine(
              latex: latex,
              textStyle: Theme.of(context).textTheme.titleLarge,
            ),
            const SizedBox(height: 8),
            ScrollableTextLine(
              text: expression.isEmpty ? '(empty)' : expression,
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ],
        ),
      ),
    );
  }
}

class _Section extends StatelessWidget {
  final String title;
  final Widget child;
  const _Section({required this.title, required this.child});

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(18),
        side: BorderSide(color: Theme.of(context).dividerColor),
      ),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(title, style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 10),
            child,
          ],
        ),
      ),
    );
  }
}

class _Keypad extends StatelessWidget {
  final void Function(String) onInsert;
  final VoidCallback onBackspace;
  final VoidCallback onClear;
  final VoidCallback onLeft;
  final VoidCallback onRight;

  final VoidCallback onInsertFrac;
  final VoidCallback onInsertAbs;
  final VoidCallback onJumpNum;
  final VoidCallback onJumpDen;

  final String currentExpr;
  final int cursor;

  const _Keypad({
    required this.onInsert,
    required this.onBackspace,
    required this.onClear,
    required this.onLeft,
    required this.onRight,
    required this.onInsertFrac,
    required this.onInsertAbs,
    required this.onJumpNum,
    required this.onJumpDen,
    required this.currentExpr,
    required this.cursor,
  });

  void _dot() {
    final left = currentExpr.substring(0, cursor);
    int i = left.length - 1;
    while (i >= 0) {
      final c = left[i];
      final isDigit = c.codeUnitAt(0) >= 48 && c.codeUnitAt(0) <= 57;
      if (isDigit || c == '.') {
        i--;
      } else {
        break;
      }
    }
    final seg = left.substring(i + 1);
    if (seg.contains('.')) return;
    onInsert(seg.isEmpty ? '0.' : '.');
  }

  Widget _btn(
    String label, {
    required VoidCallback onTap,
    required double width,
    bool wide = false,
  }) {
    return SizedBox(
      width: wide ? (width * 2 + 8) : width,
      height: 54,
      child: OutlinedButton(
        onPressed: onTap,
        child: Text(
          label,
          style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w600),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        const gap = 8.0;
        final available = constraints.maxWidth.isFinite
            ? constraints.maxWidth
            : MediaQuery.sizeOf(context).width;
        final buttonWidth = ((available - gap * 3) / 4).clamp(64.0, 96.0);

        Widget row(List<Widget> children) {
          return Padding(
            padding: const EdgeInsets.only(bottom: gap),
            child: Wrap(spacing: gap, runSpacing: gap, children: children),
          );
        }

        Widget btn(
          String label, {
          required VoidCallback onTap,
          bool wide = false,
        }) {
          return _btn(label, onTap: onTap, width: buttonWidth, wide: wide);
        }

        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            row([
              btn('⟵', onTap: onLeft),
              btn('⟶', onTap: onRight),
              btn('⌫', onTap: onBackspace),
              btn('AC', onTap: onClear),
            ]),
            row([
              btn('— (frac)', onTap: onInsertFrac, wide: true),
              btn('•', onTap: () => onInsert('•')),
              btn('i', onTap: () => onInsert('I')),
            ]),
            row([
              btn('sin', onTap: () => onInsert('sin(')),
              btn('cos', onTap: () => onInsert('cos(')),
              btn('exp', onTap: () => onInsert('exp(')),
              btn('abs', onTap: onInsertAbs),
            ]),
            row([
              btn('sign', onTap: () => onInsert('sign(')),
              btn('rect', onTap: () => onInsert('rect(')),
              btn('tri', onTap: () => onInsert('tri(')),
            ]),
            row([
              btn('u(t)', onTap: () => onInsert('u(t)')),
              btn('δ(t)', onTap: () => onInsert('delta(t)')),
              btn('π', onTap: () => onInsert('pi')),
              btn('t', onTap: () => onInsert('t')),
            ]),
            row([
              btn('7', onTap: () => onInsert('7')),
              btn('8', onTap: () => onInsert('8')),
              btn('9', onTap: () => onInsert('9')),
              btn('÷', onTap: () => onInsert('/')),
            ]),
            row([
              btn('4', onTap: () => onInsert('4')),
              btn('5', onTap: () => onInsert('5')),
              btn('6', onTap: () => onInsert('6')),
              btn('×', onTap: () => onInsert('*')),
            ]),
            row([
              btn('1', onTap: () => onInsert('1')),
              btn('2', onTap: () => onInsert('2')),
              btn('3', onTap: () => onInsert('3')),
              btn('+', onTap: () => onInsert('+')),
            ]),
            row([
              btn('0', onTap: () => onInsert('0')),
              btn('.', onTap: _dot),
              btn('-', onTap: () => onInsert('-')),
              btn('^', onTap: () => onInsert('^')),
            ]),
            Wrap(
              spacing: gap,
              runSpacing: gap,
              children: [
                btn('(', onTap: () => onInsert('(')),
                btn(')', onTap: () => onInsert(')')),
                btn(',', onTap: () => onInsert(',')),
              ],
            ),
          ],
        );
      },
    );
  }
}
