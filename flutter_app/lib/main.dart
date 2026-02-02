import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:flutter_math_fork/flutter_math.dart';

import 'fft/fourier_transform_bloc.dart';
import 'fft/fourier_transform_event.dart';
import 'fft/fourier_transform_state.dart';
import 'fft/charts.dart';

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

class _HomePageState extends State<HomePage> {
  String _expr = 'u(t)';
  int _cursor = 0;

  int _pow2 = 9;
  double _a = 1.0;

  @override
  void initState() {
    super.initState();
    _cursor = _expr.indexOf('□');
    if (_cursor < 0) _cursor = _expr.length;
  }

  void _setExpr(String next, {int? cursor}) {
    setState(() {
      _expr = next;
      _cursor = (cursor ?? _cursor).clamp(0, _expr.length);
    });
  }

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
  void _insertTPow() {
    final before = _expr.substring(0, _cursor);
    final after = _expr.substring(_cursor);
    const insert = 't^(□)';
    final next = before + insert + after;
    final expPos = (before + 't^(').length; // place cursor at exponent box
    _setExpr(next, cursor: expPos);
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
          ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(state.error!)));
        }
      },
      child: Scaffold(
        appBar: AppBar(
          title: const Text('Fourier Transform'),
          actions: [
            TextButton(
              onPressed: () async {
                final v = await showDialog<double>(
                  context: context,
                  builder: (_) => _AInputDialog(initial: _a),
                );
                if (v != null) setState(() => _a = v);
              },
              child: Text('a = ${_a.toStringAsPrecision(4)}'),
            ),
          ],
        ),
        body: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            _Section(
              title: 'Expression (preview is the display)',
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _FormulaPreview(expression: _expr, cursor: _cursor, a: _a),
                  const SizedBox(height: 12),
                  _Keypad(
                    onInsert: _insert,
                    onBackspace: _backspace,
                    onClear: _clear,
                    onLeft: _left,
                    onRight: _right,
                    onInsertFrac: _insertFrac,
                    onInsertTPow: _insertAbs,
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
                  SizedBox(width: 96, child: Text('N=${1 << _pow2}', textAlign: TextAlign.end)),
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
                        a: _a,
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
  final double a;

  const _FormulaPreview({
    required this.expression,
    required this.cursor,
    required this.a,
  });

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

            final isThisActive = inFrac && localStart == fracStart && i == activeEnd && comma == activeComma;
            final num = boxify(numRaw, active: isThisActive && activeIsNum);
            final den = boxify(denRaw, active: isThisActive && !activeIsNum);

            buf.write(r'\frac{' + num + '}{' + den + '}');
            i++;
            continue;
          }

          if (input.startsWith('t^(', i)) {
            // Parse t^(...) as a power with exponent box/highlight.
            final start = i;
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
            buf.write(r'\mid ');
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
      out = out.replaceAllMapped(RegExp(r'\bexp\b'), (_) => r'\mathrm{exp}');
      out = out.replaceAllMapped(RegExp(r'\bdelta\b'), (_) => r'\delta');
      // Imaginary unit
      out = out.replaceAllMapped(RegExp(r'\bI\b'), (_) => r'i');

      out = out.replaceAllMapped(RegExp(r'\ba\b'), (_) => a.toStringAsPrecision(4));

      out = out.replaceAll('*', r'\cdot ');
      out = out.replaceAll('•', r'\bullet ');
      return out;
    }

    final withCursor = expr.substring(0, c) + '|' + expr.substring(c);
    final trimmed = withCursor.trim();
    return trimmed.isEmpty ? r'\mid' : convert(trimmed);
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
                Text('a=$a', style: Theme.of(context).textTheme.bodySmall),
              ],
            ),
            const SizedBox(height: 8),
            SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              child: Math.tex(
                latex,
                textStyle: Theme.of(context).textTheme.titleLarge,
              ),
            ),
            const SizedBox(height: 8),
            Text(expression.isEmpty ? '(empty)' : expression, style: Theme.of(context).textTheme.bodySmall),
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
  final VoidCallback onInsertTPow;
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
    required this.onInsertTPow,
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

  Widget _btn(String label, {required VoidCallback onTap, bool wide = false}) {
    return SizedBox(
      width: wide ? 138 : 78,
      height: 54,
      child: OutlinedButton(
        onPressed: onTap,
        child: Text(label, style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w600)),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: [
            _btn('⟵', onTap: onLeft),
            _btn('⟶', onTap: onRight),
            _btn('⌫', onTap: onBackspace),
            _btn('AC', onTap: onClear),
          ],
        ),
        const SizedBox(height: 8),
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: [
            _btn('— (frac)', onTap: onInsertFrac, wide: true),
            //  _btn('↑', onTap: onJumpNum),
            //  _btn('↓', onTap: onJumpDen),
            _btn('•', onTap: () => onInsert('•')),
            // Imaginary unit (SymPy uses capital I)
            _btn('i', onTap: () => onInsert('I')),

          ],
        ),
        const SizedBox(height: 8),
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: [
            _btn('sin', onTap: () => onInsert('sin(')),
            _btn('cos', onTap: () => onInsert('cos(')),
            _btn('exp', onTap: () => onInsert('exp(')),
            _btn('abs', onTap: onInsertTPow),],
        ),
        const SizedBox(height: 8),
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: [
            _btn('u(t)', onTap: () => onInsert('u(t)')),
            _btn('δ(t)', onTap: () => onInsert('delta(t)')),
            _btn('π', onTap: () => onInsert('pi')),
            _btn('t', onTap: () => onInsert('t')),

          ],
        ),
        const SizedBox(height: 8),
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: [
            _btn('7', onTap: () => onInsert('7')),
            _btn('8', onTap: () => onInsert('8')),
            _btn('9', onTap: () => onInsert('9')),
            _btn('÷', onTap: () => onInsert('/')),
          ],
        ),
        const SizedBox(height: 8),
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: [
            _btn('4', onTap: () => onInsert('4')),
            _btn('5', onTap: () => onInsert('5')),
            _btn('6', onTap: () => onInsert('6')),
            _btn('×', onTap: () => onInsert('*')),
          ],
        ),
        const SizedBox(height: 8),
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: [
            _btn('1', onTap: () => onInsert('1')),
            _btn('2', onTap: () => onInsert('2')),
            _btn('3', onTap: () => onInsert('3')),
            _btn('+', onTap: () => onInsert('+')),
          ],
        ),
        const SizedBox(height: 8),
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: [
            _btn('0', onTap: () => onInsert('0')),
            _btn('.', onTap: _dot),
            _btn('-', onTap: () => onInsert('-')),
            _btn('^', onTap: () => onInsert('^')),
          ],
        ),
        const SizedBox(height: 8),
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: [
            _btn('a', onTap: () => onInsert('a')),
            _btn('(', onTap: () => onInsert('(')),
            _btn(')', onTap: () => onInsert(')')),
            _btn(',', onTap: () => onInsert(',')),
          ],
        ),
      ],
    );
  }
}

class _AInputDialog extends StatefulWidget {
  final double initial;
  const _AInputDialog({required this.initial});

  @override
  State<_AInputDialog> createState() => _AInputDialogState();
}

class _AInputDialogState extends State<_AInputDialog> {
  late String _text;

  @override
  void initState() {
    super.initState();
    _text = widget.initial.toString();
  }

  void _append(String s) => setState(() => _text += s);
  void _backspace() => setState(() => _text.isEmpty ? '' : _text.substring(0, _text.length - 1));
  void _clear() => setState(() => _text = '');
  void _dot() {
    if (_text.contains('.')) return;
    setState(() => _text = _text.isEmpty ? '0.' : '$_text.');
  }

  void _neg() {
    setState(() {
      if (_text.startsWith('-')) {
        _text = _text.substring(1);
      } else {
        _text = '-$_text';
      }
    });
  }

  double? _parse() => double.tryParse(_text.trim());

  Widget _k(String label, VoidCallback onTap, {bool wide = false}) => SizedBox(
    width: wide ? 140 : 80,
    height: 54,
    child: OutlinedButton(onPressed: onTap, child: Text(label, style: const TextStyle(fontSize: 18))),
  );

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: const Text('Set parameter a'),
      content: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              border: Border.all(color: Theme.of(context).dividerColor),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Text(_text.isEmpty ? '0' : _text, style: Theme.of(context).textTheme.titleLarge),
          ),
          const SizedBox(height: 12),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              _k('7', () => _append('7')),
              _k('8', () => _append('8')),
              _k('9', () => _append('9')),
              _k('⌫', _backspace),
              _k('4', () => _append('4')),
              _k('5', () => _append('5')),
              _k('6', () => _append('6')),
              _k('AC', _clear),
              _k('1', () => _append('1')),
              _k('2', () => _append('2')),
              _k('3', () => _append('3')),
              _k('±', _neg),
              _k('0', () => _append('0')),
              _k('.', _dot),
              _k('OK', () {
                final v = _parse();
                if (v == null) return;
                Navigator.of(context).pop(v);
              }, wide: true),
            ],
          ),
        ],
      ),
    );
  }
}