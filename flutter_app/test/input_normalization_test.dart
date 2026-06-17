import 'package:flutter_test/flutter_test.dart';
import 'package:fourier_transform/fft/fourier_transform_bloc.dart';

void main() {
  test('normalizes implicit multiplication around polynomial parentheses', () {
    expect(normalizeInputExpression('frac(t^2,t(t+1))'), 'frac(t^2,t*(t+1))');
    expect(
      normalizeInputExpression('frac(t^2(t+1),t^3(t+1))'),
      'frac(t^2*(t+1),t^3*(t+1))',
    );
    expect(
      normalizeInputExpression('frac(t^2\uFF08t+1\uFF09,t^3(t+1))'),
      'frac(t^2*(t+1),t^3*(t+1))',
    );
  });

  test('normalizes convolution operator variants to bullet', () {
    expect(normalizeInputExpression('u(t)\u2022u(t)'), 'u(t)\u2022u(t)');
    expect(normalizeInputExpression('u(t)\u00B7u(t)'), 'u(t)\u2022u(t)');
    expect(normalizeInputExpression('u(t)\u2219u(t)'), 'u(t)\u2022u(t)');
    expect(normalizeInputExpression('u(t)\u22C6u(t)'), 'u(t)\u2022u(t)');
    expect(normalizeInputExpression('u(t)\u2217u(t)'), 'u(t)\u2022u(t)');
    expect(normalizeInputExpression(r'u(t)\bullet u(t)'), 'u(t)\u2022u(t)');
  });
}
