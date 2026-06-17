import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:fourier_transform/fft/symbol.dart';
import 'package:fourier_transform/main.dart';

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
      await tester.pump(const Duration(milliseconds: 100));
      if (finder.evaluate().isNotEmpty) {
        return;
      }
    }
    throw TestFailure('Timed out waiting for $finder.');
  }

  Future<void> tapVisibleText(WidgetTester tester, String text) async {
    final finder = find.text(text);
    await tester.scrollUntilVisible(finder, 80);
    await tester.tap(finder);
    await tester.pump();
  }

  testWidgets(
    'keypad input is sent to backend and backend result is rendered',
    (tester) async {
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
      await pumpUntilFound(tester, find.text('Result'));
      await pumpUntilFound(tester, find.bySemanticsLabel(RegExp(r'symbolic-result:.*delta')));

      final backendResult = await computeByBackendOnly('sin(t)');
      expect(backendResult.ok, isTrue);
      expect(backendResult.resultLatex, contains(r'\delta'));
      expect(backendResult.resultLatex, contains(r'\omega'));
      expect(tester.takeException(), isNull);
    },
    skip: !_runBackendE2e || _backendBaseUrl.isEmpty,
  );
}
