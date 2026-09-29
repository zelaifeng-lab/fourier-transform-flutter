import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:fourier_transform/fft/symbol.dart';
import 'package:fourier_transform/main.dart';
import 'package:fourier_transform/scrollable_content.dart';

const bool _runBackendE2e = bool.fromEnvironment('RUN_BACKEND_E2E');
const String _backendBaseUrl = String.fromEnvironment('API_BASE_URL');

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  Future<void> pumpUntilFound(
    WidgetTester tester,
    Finder finder, {
    Duration timeout = const Duration(seconds: 20),
  }) async {
    final end = DateTime.now().add(timeout);
    while (DateTime.now().isBefore(end)) {
      // Real sockets need real time; advancing the fake clock triggers HTTP
      // timeouts before the operating system can complete the request.
      await tester.runAsync(
        () => Future<void>.delayed(const Duration(milliseconds: 100)),
      );
      await tester.pump();
      if (finder.evaluate().isNotEmpty) {
        return;
      }
    }
    throw TestFailure(
      'Timed out waiting for $finder. Visible text: '
      '${tester.widgetList<Text>(find.byType(Text)).map((w) => w.data).toList()}',
    );
  }

  Future<void> tapVisibleText(WidgetTester tester, String text) async {
    final finder = find.text(text);
    for (var i = 0; i < 20; i++) {
      final visible = finder.hitTestable();
      if (visible.evaluate().isNotEmpty) {
        await tester.tap(visible.first);
        await tester.pump();
        return;
      }
      final scrollables = find.byType(Scrollable).hitTestable();
      if (scrollables.evaluate().isEmpty) {
        break;
      }
      await tester.drag(scrollables.last, const Offset(0, -120));
      await tester.pump();
    }
    throw TestFailure('Could not tap visible text: $text');
  }

  testWidgets(
    'keypad input is sent to backend and backend result is rendered',
    (tester) async {
      final previousHttpOverrides = HttpOverrides.current;
      HttpOverrides.global = null;
      addTearDown(() => HttpOverrides.global = previousHttpOverrides);

      tester.view.devicePixelRatio = 1;
      tester.view.physicalSize = const Size(900, 900);
      addTearDown(tester.view.resetPhysicalSize);
      addTearDown(tester.view.resetDevicePixelRatio);

      await tester.pumpWidget(const AppRoot());
      await tester.pump();

      await tapVisibleText(tester, 'AC');
      await tapVisibleText(tester, 'sin');
      await tapVisibleText(tester, 't');
      await tapVisibleText(tester, ')');

      expect(find.text('sin(t)'), findsOneWidget);

      await tapVisibleText(tester, 'Run transform');

      await pumpUntilFound(tester, find.text('Results'));
      await pumpUntilFound(
        tester,
        find.byWidgetPredicate(
          (widget) =>
              widget is ScrollableMathLine &&
              widget.latex.startsWith(r'\displaystyle X(\omega)=') &&
              widget.latex.contains(r'\delta'),
        ),
      );

      final backendResult = (await tester.runAsync(
        () => computeByBackendOnly('sin(t)'),
      ))!;
      expect(backendResult.ok, isTrue);
      expect(backendResult.resultLatex, contains(r'\delta'));
      expect(backendResult.resultLatex, contains(r'\omega'));
      expect(tester.takeException(), isNull);
    },
    skip: !_runBackendE2e || _backendBaseUrl.isEmpty,
  );

  testWidgets(
    'live backend conditions and parser errors reach the displayed page',
    (tester) async {
      final previous = HttpOverrides.current;
      HttpOverrides.global = null;
      addTearDown(() => HttpOverrides.global = previous);
      await tester.pumpWidget(
        const MaterialApp(home: SymbolPage(expression: 'exp(-a*t)*u(t)')),
      );
      await pumpUntilFound(tester, find.byKey(const Key('result-conditions')));
      expect(find.text('Conditions'), findsOneWidget);
      expect(tester.takeException(), isNull);
      await tester.pumpWidget(
        const MaterialApp(home: SymbolPage(expression: '__bad__')),
      );
      await pumpUntilFound(tester, find.byKey(const Key('result-error')));
      expect(find.textContaining('Parser error:'), findsOneWidget);
      expect(tester.takeException(), isNull);
    },
    skip: !_runBackendE2e || _backendBaseUrl.isEmpty,
  );
}
