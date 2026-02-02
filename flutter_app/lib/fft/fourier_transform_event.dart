import 'package:equatable/equatable.dart';

abstract class FourierTransformEvent extends Equatable {
  const FourierTransformEvent();
  @override
  List<Object?> get props => [];
}

class TransformExpressionRequested extends FourierTransformEvent {
  final String expression;
  final int pow2;
  final double a;

  const TransformExpressionRequested({
    required this.expression,
    required this.pow2,
    required this.a,
  });

  int get n => 1 << pow2;

  @override
  List<Object?> get props => [expression, pow2, a];
}

class ClearRequested extends FourierTransformEvent {
  const ClearRequested();
}