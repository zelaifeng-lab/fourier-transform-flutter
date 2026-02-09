import 'dart:math' as math;
import 'package:flutter_bloc/flutter_bloc.dart';
import 'fourier_transform_event.dart';
import 'fourier_transform_state.dart';

// Dart's `dart:math` does not provide cosh/sinh. Implement them here.
double _cosh(double x) => (math.exp(x) + math.exp(-x)) / 2.0;
double _sinh(double x) => (math.exp(x) - math.exp(-x)) / 2.0;

// Insert implicit multiplication markers.
// Examples:
//   5t -> 5*t
//   2sin(t) -> 2*sin(t)
//   sin(5t) -> sin(5*t)
//   (t+1)(t+2) -> (t+1)*(t+2)
// Note: we avoid inserting '*' between a known function name and '(' (e.g., sin(...)).
String _insertImplicitMul(String s) {
  s = s.replaceAll(' ', '');
  bool isDigit(int c) => c >= 48 && c <= 57;
  bool isAlpha(int c) =>
      (c >= 65 && c <= 90) || (c >= 97 && c <= 122) || c == 95; // A-Z a-z _
  bool isOp(String ch) => '+-*/^,•'.contains(ch);

  final out = StringBuffer();

  String prevType = 'none'; // none|num|id|rpar
  String prevId = '';

  bool prevIsFuncName() => prevType == 'id' && _isFuncName(prevId);

  int i = 0;
  while (i < s.length) {
    final ch = s[i];
    final cu = ch.codeUnitAt(0);

    // number: 12, 12.3, .5
    if (isDigit(cu) || (ch == '.' && i + 1 < s.length && isDigit(s.codeUnitAt(i + 1)))) {
      int j = i;
      bool seenDot = false;
      if (s[j] == '.') { seenDot = true; j++; }
      while (j < s.length) {
        final cj = s[j];
        final u = cj.codeUnitAt(0);
        if (isDigit(u)) { j++; continue; }
        if (cj == '.' && !seenDot) { seenDot = true; j++; continue; }
        break;
      }

      if (prevType == 'num' || prevType == 'id' || prevType == 'rpar') {
        out.write('*');
      }

      out.write(s.substring(i, j));
      prevType = 'num';
      prevId = '';
      i = j;
      continue;
    }

    // identifier: sin, cos, t, u, abs, exp, pi, I, etc.
    if (isAlpha(cu)) {
      int j = i + 1;
      while (j < s.length) {
        final uj = s.codeUnitAt(j);
        if (isAlpha(uj) || isDigit(uj)) { j++; continue; }
        break;
      }
      final id = s.substring(i, j);

      if (prevType == 'num' || prevType == 'id' || prevType == 'rpar') {
        out.write('*');
      }

      out.write(id);
      prevType = 'id';
      prevId = id;
      i = j;
      continue;
    }

    // parentheses
    if (ch == '(') {
      if (!prevIsFuncName() && (prevType == 'num' || prevType == 'id' || prevType == 'rpar')) {
        out.write('*');
      }
      out.write('(');
      prevType = 'none';
      prevId = '';
      i++;
      continue;
    }
    if (ch == ')') {
      out.write(')');
      prevType = 'rpar';
      prevId = '';
      i++;
      continue;
    }

    // operators / others
    if (isOp(ch)) {
      out.write(ch);
      prevType = 'none';
      prevId = '';
      i++;
      continue;
    }

    // passthrough
    out.write(ch);
    prevType = 'none';
    prevId = '';
    i++;
  }

  return out.toString();
}

bool _isFuncName(String id) {
  // Must match the function names supported by the local parser.
  // Add more names here if you add more _Func1/_Func2 implementations.
  const funcs = <String>{
    'sin', 'cos', 'tan',
    'sinh', 'cosh',
    'exp', 'abs',
    'u', 'heaviside',
    'delta',
    'frac',
  };
  return funcs.contains(id);
}

class FourierTransformBloc extends Bloc<FourierTransformEvent, FourierTransformState> {
  FourierTransformBloc() : super(FourierTransformState.initial()) {
    on<TransformExpressionRequested>(_onTransform);
    on<ClearRequested>(_onClear);
  }

  void _onClear(ClearRequested event, Emitter<FourierTransformState> emit) {
    emit(FourierTransformState.initial());
  }

  Future<void> _onTransform(
      TransformExpressionRequested event,
      Emitter<FourierTransformState> emit,
      ) async {
    final expr = _insertImplicitMul(event.expression.trim());
    final n = event.n;

    if (expr.isEmpty) {
      emit(state.copyWith(status: FourierStatus.failure, error: '表达式为空'));
      return;
    }

    if (expr.contains('□')) {
      emit(state.copyWith(status: FourierStatus.failure, error: '还有未填写的空框（□），请先补全分子/分母'));
      return;
    }

    emit(state.copyWith(
      status: FourierStatus.running,
      expression: expr,
      pow2: event.pow2,
      n: n,
      a: event.a,
      error: null,
    ));

    try {
      // Sampling window: [-π, π)
      final tMin = -math.pi;
      final tMax = math.pi;
      final dt = (tMax - tMin) / n;
      final t = List<double>.generate(n, (i) => tMin + i * dt);

      // Parse -> AST -> array evaluation (supports convolution + frac + imaginary unit I)
      final ast = _Parser(expr).parse();
      final ctx = _EvalContext(t: t, dt: dt, a: event.a, eps: dt / 2);

      // Complex-valued signal (allows expressions like I*sin(t), exp(I*t), etc.)
      final signalC = ast.evalComplexArray(ctx);
      // Keep a real-part trace for the time-domain chart (existing UI is real-only)
      final signal = List<double>.generate(n, (i) => signalC[i].re);

      // Apply Hann window to reduce spectral leakage.
      // w[n] = 0.5 - 0.5 cos(2π n/(N-1)),  n=0..N-1
      // We also apply a simple amplitude correction by dividing by mean(w),
      // so that a constant signal keeps approximately the same DC level.
      final win = (n <= 1)
          ? List<double>.filled(n, 1.0)
          : List<double>.generate(
        n,
            (i) => 0.5 - 0.5 * math.cos(2.0 * math.pi * i / (n - 1)),
      );
      final meanW = win.isEmpty ? 1.0 : (win.reduce((a, b) => a + b) / win.length);
      final windowedC = List<_C>.generate(
        n,
            (i) => signalC[i].scale(win[i]),
      );

      // Continuous FT approximation using FFT:

      // X(ω_k) ≈ dt * Σ x[n] e^{-j ω_k t_n}
      final X = _fft(windowedC, inverse: false);
      for (int i = 0; i < X.length; i++) {
        X[i] = X[i].scale(dt / (meanW == 0 ? 1.0 : meanW));
      }

      // Full spectrum (fftshift): ω from negative to positive
      final Xs = _fftShift(X);
      final omega = List<double>.generate(n, (i) => 2.0 * math.pi * (i - n / 2) / (n * dt));
      final magnitude = List<double>.generate(n, (i) => Xs[i].abs());

      emit(state.copyWith(
        status: FourierStatus.success,
        dt: dt,
        t: t,
        signal: signal,
        omega: omega,
        magnitude: magnitude,
        error: null,
      ));
    } catch (e) {
      emit(state.copyWith(status: FourierStatus.failure, error: '解析/计算失败：$e'));
    }
  }
}

/// =======================
/// Complex + FFT
/// =======================

class _C {
  final double re;
  final double im;
  const _C(this.re, this.im);

  _C operator +(_C o) => _C(re + o.re, im + o.im);
  _C operator -(_C o) => _C(re - o.re, im - o.im);
  _C operator *(_C o) => _C(re * o.re - im * o.im, re * o.im + im * o.re);
  _C scale(double s) => _C(re * s, im * s);
  double abs() => math.sqrt(re * re + im * im);
}

_C _cDiv(_C a, _C b) {
  // a / b
  final denom = b.re * b.re + b.im * b.im;
  if (denom == 0) {
    // Avoid hard crash on singularities like 1/t at t=0.
    // Callers should have regularized the denominator, but keep a safe guard.
    const tiny = 1e-12;
    return _C(a.re / tiny, a.im / tiny);
  }
  return _C((a.re * b.re + a.im * b.im) / denom, (a.im * b.re - a.re * b.im) / denom);
}

_C _cExp(_C z) {
  // exp(x + jy) = exp(x)(cos y + j sin y)
  final ex = math.exp(z.re);
  return _C(ex * math.cos(z.im), ex * math.sin(z.im));
}

_C _cSin(_C z) {
  // sin(x + jy) = sin x cosh y + j cos x sinh y
  final x = z.re;
  final y = z.im;
  return _C(math.sin(x) * _cosh(y), math.cos(x) * _sinh(y));
}

_C _cCos(_C z) {
  // cos(x + jy) = cos x cosh y - j sin x sinh y
  final x = z.re;
  final y = z.im;
  return _C(math.cos(x) * _cosh(y), -math.sin(x) * _sinh(y));
}

List<_C> _fftReal(List<double> x) => _fft(x.map((v) => _C(v, 0)).toList(), inverse: false);
List<_C> _ifft(List<_C> X) => _fft(X, inverse: true);

List<_C> _fft(List<_C> a, {required bool inverse}) {
  final n = a.length;
  final out = List<_C>.from(a);

  int j = 0;
  for (int i = 1; i < n; i++) {
    int bit = n >> 1;
    while (j & bit != 0) {
      j ^= bit;
      bit >>= 1;
    }
    j ^= bit;
    if (i < j) {
      final tmp = out[i];
      out[i] = out[j];
      out[j] = tmp;
    }
  }

  for (int len = 2; len <= n; len <<= 1) {
    final ang = 2.0 * math.pi / len * (inverse ? 1 : -1);
    final wlen = _C(math.cos(ang), math.sin(ang));
    for (int i = 0; i < n; i += len) {
      var w = const _C(1, 0);
      for (int j0 = 0; j0 < len ~/ 2; j0++) {
        final u = out[i + j0];
        final v = out[i + j0 + len ~/ 2] * w;
        out[i + j0] = u + v;
        out[i + j0 + len ~/ 2] = u - v;
        w = w * wlen;
      }
    }
  }

  if (inverse) {
    final invN = 1.0 / n;
    for (int i = 0; i < n; i++) {
      out[i] = out[i].scale(invN);
    }
  }
  return out;
}

List<_C> _fftShift(List<_C> x) {
  final n = x.length;
  final half = n ~/ 2;
  final out = List<_C>.filled(n, const _C(0, 0));
  for (int i = 0; i < n; i++) {
    out[i] = x[(i + half) % n];
  }
  return out;
}

/// Circular convolution (periodic boundary):
/// (f • g)[k] ≈ dt * Σ f[n] g[(k-n) mod N]
List<_C> _circularConvolutionC(List<_C> f, List<_C> g, double dt) {
  final n = f.length;
  final F = _fft(f, inverse: false);
  final G = _fft(g, inverse: false);
  final H = List<_C>.generate(n, (i) => F[i] * G[i]);
  final h = _ifft(H);
  return List<_C>.generate(n, (i) => h[i].scale(dt));
}

/// =======================
/// Parser -> AST -> array evaluation
/// Supports: + - * / ^ ( ) , sin cos exp u delta
/// Convolution operator: '•'
/// Fraction function: frac(x,y)  (pointwise division)
/// Variables: t, a; constants: pi, e
/// =======================

class _EvalContext {
  final List<double> t;
  final double dt;
  final double a;
  final double eps;
  const _EvalContext({required this.t, required this.dt, required this.a, required this.eps});
}

abstract class _Node {
  /// Evaluate as a complex-valued array.
  List<_C> evalComplexArray(_EvalContext ctx);

  /// Convenience: real part (for existing charts/UI that are real-only).
  List<double> evalArray(_EvalContext ctx) {
    final z = evalComplexArray(ctx);
    return List<double>.generate(z.length, (i) => z[i].re);
  }
}

class _Num extends _Node {
  final double v;
  _Num(this.v);
  @override
  List<_C> evalComplexArray(_EvalContext ctx) => List<_C>.filled(ctx.t.length, _C(v, 0));
}

class _VarT extends _Node {
  @override
  List<_C> evalComplexArray(_EvalContext ctx) => ctx.t.map((x) => _C(x, 0)).toList();
}

class _VarA extends _Node {
  @override
  List<_C> evalComplexArray(_EvalContext ctx) => List<_C>.filled(ctx.t.length, _C(ctx.a, 0));
}

class _ConstPi extends _Node {
  @override
  List<_C> evalComplexArray(_EvalContext ctx) => List<_C>.filled(ctx.t.length, _C(math.pi, 0));
}

class _ConstE extends _Node {
  @override
  List<_C> evalComplexArray(_EvalContext ctx) => List<_C>.filled(ctx.t.length, _C(math.e, 0));
}

/// Imaginary unit: I = sqrt(-1)
class _ConstI extends _Node {
  @override
  List<_C> evalComplexArray(_EvalContext ctx) => List<_C>.filled(ctx.t.length, const _C(0, 1));
}

class _Unary extends _Node {
  final String op;
  final _Node child;
  _Unary(this.op, this.child);

  @override
  List<_C> evalComplexArray(_EvalContext ctx) {
    final a = child.evalComplexArray(ctx);
    if (op == '+') return a;
    return a.map((z) => _C(-z.re, -z.im)).toList();
  }
}

class _Binary extends _Node {
  final String op; // + - * / ^ •
  final _Node left;
  final _Node right;
  _Binary(this.op, this.left, this.right);

  @override
  List<_C> evalComplexArray(_EvalContext ctx) {
    if (op == '•') {
      final f = left.evalComplexArray(ctx);
      final g = right.evalComplexArray(ctx);
      return _circularConvolutionC(f, g, ctx.dt);
    }

    final a = left.evalComplexArray(ctx);
    final b = right.evalComplexArray(ctx);
    final n = a.length;
    final out = List<_C>.filled(n, const _C(0, 0));

    for (int i = 0; i < n; i++) {
      switch (op) {
        case '+':
          out[i] = a[i] + b[i];
          break;
        case '-':
          out[i] = a[i] - b[i];
          break;
        case '*':
          out[i] = a[i] * b[i];
          break;
        case '/':
          final den = b[i];
          final denAbs = den.abs();
          // Regularize division near 0 to avoid division-by-zero for 1/t, etc.
          final safeDen = (denAbs < ctx.eps)
              ? (denAbs == 0 ? _C(ctx.eps, 0) : den.scale(ctx.eps / denAbs))
              : den;
          out[i] = _cDiv(a[i], safeDen);
          break;
        case '^':
        // Power is only supported for real base and real exponent in this numeric engine.
          final base = a[i];
          final exp = b[i];
          if (base.im.abs() > 1e-12 || exp.im.abs() > 1e-12) {
            throw 'power on complex values is not supported in numeric FFT mode';
          }
          final br = base.re;
          final er = exp.re;
          if (er < 0 && br.abs() < ctx.eps) {
            final s = br.isNegative ? -1.0 : 1.0;
            out[i] = _C(math.pow(s * ctx.eps, er).toDouble(), 0);
          } else {
            out[i] = _C(math.pow(br, er).toDouble(), 0);
          }
          break;
        default:
          throw 'unknown op $op';
      }
    }
    return out;
  }
}

class _Func1 extends _Node {
  final String name;
  final _Node arg;
  _Func1(this.name, this.arg);

  @override
  List<_C> evalComplexArray(_EvalContext ctx) {
    final x = arg.evalComplexArray(ctx);
    final n = x.length;
    final out = List<_C>.filled(n, const _C(0, 0));

    switch (name) {
      case 'sin':
        for (int i = 0; i < n; i++) out[i] = _cSin(x[i]);
        break;
      case 'cos':
        for (int i = 0; i < n; i++) out[i] = _cCos(x[i]);
        break;
      case 'exp':
        for (int i = 0; i < n; i++) out[i] = _cExp(x[i]);
        break;
      case 'abs':
      // abs(z): magnitude for complex, |x| for real.
        for (int i = 0; i < n; i++) out[i] = _C(x[i].abs(), 0);
        break;
      case 'u':
        for (int i = 0; i < n; i++) out[i] = x[i].re >= 0 ? const _C(1, 0) : const _C(0, 0);
        break;
      case 'delta':
        for (int i = 0; i < n; i++) {
          out[i] = (x[i].re.abs() <= ctx.dt / 2) ? _C(1.0 / ctx.dt, 0) : const _C(0, 0);
        }
        break;
      default:
        throw '不支持函数：$name';
    }
    return out;
  }
}

class _Func2 extends _Node {
  final String name;
  final _Node a1;
  final _Node a2;
  _Func2(this.name, this.a1, this.a2);

  @override
  List<_C> evalComplexArray(_EvalContext ctx) {
    final x = a1.evalComplexArray(ctx);
    final y = a2.evalComplexArray(ctx);
    final n = x.length;
    final out = List<_C>.filled(n, const _C(0, 0));

    switch (name) {
      case 'frac':
        for (int i = 0; i < n; i++) {
          final den = y[i];
          final denAbs = den.abs();
          final safeDen = (denAbs < ctx.eps)
              ? (denAbs == 0 ? _C(ctx.eps, 0) : den.scale(ctx.eps / denAbs))
              : den;
          out[i] = _cDiv(x[i], safeDen);
        }
        break;
      default:
        throw '不支持函数：$name';
    }
    return out;
  }
}

enum _TokType { number, ident, plus, minus, star, slash, caret, dot, comma, lparen, rparen, eof }

class _Tok {
  final _TokType type;
  final String lexeme;
  final double? num;
  const _Tok(this.type, this.lexeme, [this.num]);
}

class _Lexer {
  final String s;
  int i = 0;
  _Lexer(this.s);

  List<_Tok> lex() {
    final out = <_Tok>[];
    while (!_end()) {
      _ws();
      if (_end()) break;
      final c = s[i];

      if (_digit(c) || c == '.') {
        out.add(_number());
        continue;
      }
      if (_alpha(c)) {
        out.add(_ident());
        continue;
      }

      switch (c) {
        case '+':
          out.add(const _Tok(_TokType.plus, '+'));
          i++;
          break;
        case '-':
          out.add(const _Tok(_TokType.minus, '-'));
          i++;
          break;
        case '*':
          out.add(const _Tok(_TokType.star, '*'));
          i++;
          break;
        case '/':
          out.add(const _Tok(_TokType.slash, '/'));
          i++;
          break;
        case '^':
          out.add(const _Tok(_TokType.caret, '^'));
          i++;
          break;
        case '(':
          out.add(const _Tok(_TokType.lparen, '('));
          i++;
          break;
        case ')':
          out.add(const _Tok(_TokType.rparen, ')'));
          i++;
          break;
        case '•':
          out.add(const _Tok(_TokType.dot, '•'));
          i++;
          break;
        case ',':
          out.add(const _Tok(_TokType.comma, ','));
          i++;
          break;
        default:
          throw '无法识别字符：$c';
      }
    }
    out.add(const _Tok(_TokType.eof, ''));
    return out;
  }

  _Tok _number() {
    final start = i;
    bool seenDot = false;
    if (s[i] == '.') {
      seenDot = true;
      i++;
    }
    while (!_end()) {
      final c = s[i];
      if (_digit(c)) {
        i++;
        continue;
      }
      if (c == '.' && !seenDot) {
        seenDot = true;
        i++;
        continue;
      }
      break;
    }
    final text = s.substring(start, i);
    if (text == '.') throw '不合法小数：$text';
    return _Tok(_TokType.number, text, double.parse(text));
  }

  _Tok _ident() {
    final start = i;
    while (!_end()) {
      final c = s[i];
      if (_alpha(c) || _digit(c) || c == '_') {
        i++;
      } else {
        break;
      }
    }
    final text = s.substring(start, i);
    return _Tok(_TokType.ident, text);
  }

  void _ws() {
    while (!_end() && s.codeUnitAt(i) <= 32) i++;
  }

  bool _end() => i >= s.length;
  bool _digit(String c) => c.codeUnitAt(0) >= 48 && c.codeUnitAt(0) <= 57;
  bool _alpha(String c) {
    final u = c.codeUnitAt(0);
    return (u >= 65 && u <= 90) || (u >= 97 && u <= 122);
  }
}

class _Parser {
  final List<_Tok> toks;
  int p = 0;

  _Parser(String input) : toks = _Lexer(input).lex();

  _Node parse() {
    final n = _expr();
    _expect(_TokType.eof);
    return n;
  }

  _Node _expr() {
    var n = _conv();
    while (_match(_TokType.plus) || _match(_TokType.minus)) {
      final op = _prev().lexeme;
      final r = _conv();
      n = _Binary(op, n, r);
    }
    return n;
  }

  _Node _conv() {
    var n = _term();
    while (_match(_TokType.dot)) {
      final r = _term();
      n = _Binary('•', n, r);
    }
    return n;
  }

  _Node _term() {
    var n = _power();
    while (_match(_TokType.star) || _match(_TokType.slash)) {
      final op = _prev().lexeme;
      final r = _power();
      n = _Binary(op, n, r);
    }
    return n;
  }

  _Node _power() {
    var n = _unary();
    if (_match(_TokType.caret)) {
      final r = _power();
      n = _Binary('^', n, r);
    }
    return n;
  }

  _Node _unary() {
    if (_match(_TokType.plus)) return _Unary('+', _unary());
    if (_match(_TokType.minus)) return _Unary('-', _unary());
    return _primary();
  }

  _Node _primary() {
    if (_match(_TokType.number)) return _Num(_prev().num!);

    if (_match(_TokType.ident)) {
      final name = _prev().lexeme;

      if (name == 't') return _VarT();
      if (name == 'a') return _VarA();
      if (name == 'pi') return _ConstPi();
      if (name == 'e') return _ConstE();
      if (name == 'I' || name == 'i') return _ConstI();

      if (_match(_TokType.lparen)) {
        final first = _expr();
        if (_match(_TokType.comma)) {
          final second = _expr();
          _expect(_TokType.rparen);
          return _Func2(name, first, second);
        }
        _expect(_TokType.rparen);
        return _Func1(name, first);
      }

      throw '未知标识符：$name';
    }

    if (_match(_TokType.lparen)) {
      final n = _expr();
      _expect(_TokType.rparen);
      return n;
    }

    throw 'error：${_peek().lexeme}';
  }

  bool _match(_TokType t) {
    if (_peek().type == t) {
      p++;
      return true;
    }
    return false;
  }

  void _expect(_TokType t) {
    if (!_match(t)) throw '期望 $t 但得到 ${_peek().type}';
  }

  _Tok _peek() => toks[p];
  _Tok _prev() => toks[p - 1];
}