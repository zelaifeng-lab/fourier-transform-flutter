import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_math_fork/flutter_math.dart';
import 'package:fourier_transform/fft/step.dart';
import 'package:fourier_transform/fft/symbol.dart';
import 'package:fourier_transform/main.dart';
import 'package:fourier_transform/responsive.dart';
import 'package:fourier_transform/scrollable_content.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  void setSurfaceSize(WidgetTester tester, Size size) {
    tester.view.devicePixelRatio = 1;
    tester.view.physicalSize = size;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
  }

  test('responsive breakpoints return expected layout values', () {
    expect(AppBreakpoints.compact(359), isTrue);
    expect(AppBreakpoints.compact(600), isFalse);
    expect(AppBreakpoints.medium(600), isTrue);
    expect(AppBreakpoints.medium(1024), isFalse);

    expect(AppBreakpoints.maxContentWidth(390), double.infinity);
    expect(AppBreakpoints.maxContentWidth(800), 840);
    expect(AppBreakpoints.maxContentWidth(1280), 1040);

    expect(AppBreakpoints.chartHeight(390), 260);
    expect(AppBreakpoints.chartHeight(800), 310);
    expect(AppBreakpoints.chartHeight(1280), 360);
  });

  test('principal value notation is detected in result or steps', () {
    expect(
      containsPrincipalValueNotation(
        resultLatex: r'(\mathrm{PV})\frac{1}{\omega}',
        stepsLatex: const [],
      ),
      isTrue,
    );
    expect(
      containsPrincipalValueNotation(
        resultLatex: r'\pi\delta(\omega)',
        stepsLatex: const [r'PV appears because the signal is distributional.'],
      ),
      isTrue,
    );
    expect(
      containsPrincipalValueNotation(
        resultLatex: r'\pi\delta(\omega)',
        stepsLatex: const [r'ordinary absolutely convergent integral'],
      ),
      isFalse,
    );
  });

  testWidgets('principal value notice appears only when needed', (
    tester,
  ) async {
    await tester.pumpWidget(
      const MaterialApp(
        home: Scaffold(
          body: PrincipalValueNotice(visible: true),
        ),
      ),
    );
    expect(find.byKey(const Key('principal-value-notice')), findsOneWidget);
    expect(find.textContaining('PV means Cauchy principal value'), findsOneWidget);

    await tester.pumpWidget(
      const MaterialApp(
        home: Scaffold(
          body: PrincipalValueNotice(visible: false),
        ),
      ),
    );
    expect(find.byKey(const Key('principal-value-notice')), findsNothing);
  });

  test('chart sanitizer splits singular-looking traces', () {
    expect(
      chartSegmentLengthsForTest(
        const [-2, -1, 0, 1, 2],
        const [-2, -1, 0, 1, 2],
      ),
      [5],
    );

    expect(
      chartSegmentLengthsForTest(
        const [-2, -1, -0.1, 0, 0.1, 1, 2],
        const [-0.5, -1, -10, 20, 10, 1, 0.5],
      ),
      [2, 2],
    );
  });

  testWidgets('home page renders core controls', (tester) async {
    setSurfaceSize(tester, const Size(900, 900));

    await tester.pumpWidget(const AppRoot());
    await tester.pump();

    expect(find.text('Fourier Transform'), findsOneWidget);
    expect(find.text('Preview'), findsOneWidget);
    expect(find.text('Previous'), findsOneWidget);
    expect(find.text('Next'), findsOneWidget);
    expect(find.text('FFT size'), findsOneWidget);
    expect(find.text('Run transform'), findsOneWidget);
    expect(find.byType(Slider), findsOneWidget);
    expect(find.textContaining('Example'), findsOneWidget);
    expect(find.text('Unit step'), findsNothing);
    expect(find.text('Advanced impulse'), findsNothing);
    expect(find.text('General Fourier formulas'), findsNothing);
    expect(
      find.textContaining('PV means Cauchy principal value'),
      findsNothing,
    );
    expect(_currentExampleFormula('sign(t-2)'), findsOneWidget);
    expect(find.byType(ScrollableMathLine), findsNWidgets(2));
    expect(find.byType(Math), findsAtLeastNWidgets(2));
    expect(tester.takeException(), isNull);
  });

  testWidgets('exp input previews as e power notation', (tester) async {
    setSurfaceSize(tester, const Size(900, 900));

    await tester.pumpWidget(const AppRoot());
    await tester.pump();

    await tester.tap(find.text('exp'));
    await tester.pump();

    expect(
      find.byWidgetPredicate(
        (widget) =>
            widget is ScrollableMathLine &&
            widget.latex.contains('e^{') &&
            !widget.latex.contains(r'\mathrm{exp}'),
      ),
      findsOneWidget,
    );
    expect(tester.takeException(), isNull);
  });

  testWidgets('next and previous buttons rotate preset examples', (
    tester,
  ) async {
    setSurfaceSize(tester, const Size(900, 900));

    await tester.pumpWidget(const AppRoot());
    await tester.pump();

    expect(find.textContaining('Example 1/25'), findsOneWidget);
    expect(_currentExampleFormula('sign(t-2)'), findsOneWidget);

    await tester.tap(find.text('Next'));
    await tester.pump();
    expect(find.textContaining('Example 2/25'), findsOneWidget);
    expect(_currentExampleFormula('rect((t-2)/3)'), findsOneWidget);

    await tester.tap(find.text('Previous'));
    await tester.pump();
    expect(find.textContaining('Example 1/25'), findsOneWidget);
    expect(_currentExampleFormula('sign(t-2)'), findsOneWidget);
  });


  testWidgets('all preset examples render without math parser exceptions', (
    tester,
  ) async {
    setSurfaceSize(tester, const Size(900, 900));

    await tester.pumpWidget(const AppRoot());
    await tester.pump();
    expect(tester.takeException(), isNull);

    for (var i = 0; i < 24; i++) {
      await tester.tap(find.text('Next'));
      await tester.pump();
      expect(
        tester.takeException(),
        isNull,
        reason: 'Example ${i + 2}/25 should render without thrown exceptions.',
      );
      expect(
        find.textContaining('Parser Error'),
        findsNothing,
        reason: 'Example ${i + 2}/25 should not show a math parser error.',
      );
      expect(
        find.textContaining('Build Exception'),
        findsNothing,
        reason: 'Example ${i + 2}/25 should not show a math build error.',
      );
    }
  });

  testWidgets(
    'home page adapts to common screen sizes without layout exceptions',
    (tester) async {
      const sizes = <Size>[Size(360, 780), Size(768, 1024), Size(1280, 900)];

      for (final size in sizes) {
        setSurfaceSize(tester, size);

        await tester.pumpWidget(const AppRoot());
        await tester.pump();

        expect(find.text('Run transform'), findsOneWidget);
        expect(find.byType(Slider), findsOneWidget);
        expect(
          tester.takeException(),
          isNull,
          reason:
              'Expected no Flutter layout exception at ${size.width}x${size.height}.',
        );
      }
    },
  );

  testWidgets(
    'home page uses local scroll containers for narrow long content',
    (tester) async {
      setSurfaceSize(tester, const Size(360, 780));

      await tester.pumpWidget(const AppRoot());
      await tester.pump();

      expect(find.byType(ScrollableMathLine), findsNWidgets(2));
      expect(find.byType(ScrollableTextLine), findsOneWidget);
      expect(find.byType(BoundedScrollableText), findsNothing);
      expect(find.byType(Scrollbar), findsAtLeastNWidgets(3));
      expect(tester.takeException(), isNull);
    },
  );

  testWidgets(
    'ft steps page keeps long formulas scrollable on narrow screens',
    (tester) async {
      setSurfaceSize(tester, const Size(360, 780));

      final t = List<double>.generate(96, (i) => -2.4 + i * 0.05);
      final x = t.map((v) => v == 0 ? 1.0 : math.sin(3 * v) / v).toList();

      await tester.pumpWidget(
        MaterialApp(
          home: StepPage(t: t, x: x, dt: 0.05),
        ),
      );
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));

      expect(find.byType(ScrollableMathLine), findsAtLeastNWidgets(8));
      expect(find.byType(Scrollbar), findsAtLeastNWidgets(8));
      expect(tester.takeException(), isNull);
    },
  );
}

Finder _currentExampleFormula(String expression) {
  return find.byWidgetPredicate(
    (widget) =>
        widget is ScrollableMathLine &&
        widget.semanticsLabel == 'current-example:$expression',
  );
}
