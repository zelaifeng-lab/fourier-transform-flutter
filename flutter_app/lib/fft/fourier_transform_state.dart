import 'package:equatable/equatable.dart';

enum FourierStatus { idle, running, success, failure }

class FourierTransformState extends Equatable {
  final FourierStatus status;
  final String expression;
  final int pow2;
  final int n;

  final double a;
  final double dt;

  final List<double> t;
  final List<double> signal;

  final List<double> omega;
  final List<double> magnitude;

  final String? error;

  const FourierTransformState({
    required this.status,
    required this.expression,
    required this.pow2,
    required this.n,
    required this.a,
    required this.dt,
    required this.t,
    required this.signal,
    required this.omega,
    required this.magnitude,
    this.error,
  });

  factory FourierTransformState.initial() => const FourierTransformState(
    status: FourierStatus.idle,
    expression: 'e^(a*t) • u(t)',
    pow2: 9,
    n: 512,
    a: 1.0,
    dt: 0.0,
    t: [],
    signal: [],
    omega: [],
    magnitude: [],
    error: null,
  );

  FourierTransformState copyWith({
    FourierStatus? status,
    String? expression,
    int? pow2,
    int? n,
    double? a,
    double? dt,
    List<double>? t,
    List<double>? signal,
    List<double>? omega,
    List<double>? magnitude,
    String? error,
  }) {
    return FourierTransformState(
      status: status ?? this.status,
      expression: expression ?? this.expression,
      pow2: pow2 ?? this.pow2,
      n: n ?? this.n,
      a: a ?? this.a,
      dt: dt ?? this.dt,
      t: t ?? this.t,
      signal: signal ?? this.signal,
      omega: omega ?? this.omega,
      magnitude: magnitude ?? this.magnitude,
      error: error,
    );
  }

  @override
  List<Object?> get props => [status, expression, pow2, n, a, dt, t, signal, omega, magnitude, error];
}