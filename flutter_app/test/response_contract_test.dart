import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:fourier_transform/fft/symbol.dart';
import 'package:fourier_transform/page/Formula.dart';
import 'package:fourier_transform/scrollable_content.dart';

void main() {
  Map<String, dynamic> payload({bool ok = true}) => {
    'ok': ok,
    'input_latex': 'u(t)',
    'result_latex': r'\frac{1}{a+j\omega}',
    'steps_latex': <String>[],
    'conditions_latex': 'a>0',
    'form': ok ? 'closed_form' : 'integral_form',
    'method': 'known_pair',
    'error': ok ? null : 'Closed form not established.',
  };
  test('response model preserves all contract fields', () {
    final result = SymbolicResult.fromJson(payload(ok: false));
    expect(result.conditionsLatex, 'a>0');
    expect(result.error, 'Closed form not established.');
    expect(result.form, 'integral_form');
    expect(result.method, 'known_pair');
    expect(result.ok, isFalse);
  });
  test('legacy response with optional fields absent remains supported', () {
    final result = SymbolicResult.fromJson({'ok': true, 'result_latex': '1'});
    expect(result.ok, isTrue);
    expect(result.conditionsLatex, isEmpty);
    expect(result.error, isNull);
  });
  test('request serialisation and default or override endpoint', () async {
    final client = MockClient((request) async {
      const override = String.fromEnvironment('API_BASE_URL');
      final base = override.isEmpty
          ? 'https://fourier-transform-flutter.onrender.com'
          : override;
      expect(
        request.url.toString(),
        '${base.replaceFirst(RegExp(r'/$'), '')}/fourier',
      );
      expect(jsonDecode(request.body), {'expression': 'exp(-a*t)*u(t)'});
      return http.Response(jsonEncode(payload()), 200);
    });
    final result = await computeByBackendOnly('exp(-a*t)*u(t)', client: client);
    expect(result.conditionsLatex, 'a>0');
    expect(result.method, 'known_pair');
    client.close();
  });
  test('HTTP error is readable plain text', () async {
    final client = MockClient((_) async => http.Response('failure', 503));
    final result = await computeByBackendOnly('t', client: client);
    expect(result.ok, isFalse);
    expect(result.error, contains('HTTP 503'));
    expect(result.resultLatex, r'\text{HTTP }503');
    client.close();
  });
  test('network exception is distinguished from computation failure', () async {
    final client = MockClient(
      (_) async => throw http.ClientException('offline'),
    );
    final result = await computeByBackendOnly('t', client: client);
    expect(result.error, contains('Unable to reach'));
    client.close();
  });
  for (final body in ['not-json', '[]', '{"ok":"true"}']) {
    test(
      'invalid response is not labelled as a network failure: $body',
      () async {
        final client = MockClient((_) async => http.Response(body, 200));
        final result = await computeByBackendOnly('t', client: client);
        expect(result.error, contains('invalid response'));
        client.close();
      },
    );
  }
  testWidgets(
    'conditions render as math and errors as text without internal method',
    (tester) async {
      final result = SymbolicResult.fromJson(payload(ok: false));
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(body: ResultNotices(result: result)),
        ),
      );
      expect(find.text('Conditions'), findsOneWidget);
      expect(
        tester
            .widget<ScrollableMathLine>(
              find.byKey(const Key('result-conditions')),
            )
            .latex,
        'a>0',
      );
      expect(find.text('Closed form not established.'), findsOneWidget);
      expect(find.text('known_pair'), findsNothing);
      expect(tester.takeException(), isNull);
    },
  );
  testWidgets(
    'failed SymbolPage shows readable error without false integral status',
    (tester) async {
      await tester.pumpWidget(
        MaterialApp(
          home: SymbolPage(
            expression: 'bad',
            compute: (_) async => SymbolicResult.fromJson({
              'ok': false,
              'form': 'error',
              'error': 'Parser error: check parentheses.',
            }),
          ),
        ),
      );
      await tester.pumpAndSettle();
      expect(find.text('Parser error: check parentheses.'), findsOneWidget);
      expect(
        find.text(
          'Integral representation only; a closed form has not been established.',
        ),
        findsNothing,
      );
      expect(tester.takeException(), isNull);
    },
  );
  testWidgets('HTTP adapter retains paper format without displaying a spectrum', (tester) async {
    final client = MockClient((_) async => http.Response('failure', 503));
    final result = await computeByBackendOnly('t', client: client);
    await tester.pumpWidget(MaterialApp(home: SymbolPage(
      expression: 't', compute: (_) async => result,
    )));
    await tester.pumpAndSettle();
    expect(find.text('Backend request failed (HTTP 503).'), findsOneWidget);
    expect(find.bySemanticsLabel('symbolic-result:${result.resultLatex}'), findsNothing);
    expect(tester.takeException(), isNull);
    client.close();
  });
  test('input page is exported by Formula.dart', () {
    expect(const HomePage(), isA<StatefulWidget>());
  });
}
