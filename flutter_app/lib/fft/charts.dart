import 'package:flutter/material.dart';

import 'step.dart';
import './symbol.dart';

/// Results page with two tabs:
/// - Backend (left): symbolic Fourier transform via backend (steps + final result)
/// - Chart (right): numerical-definition/step visualization (plots)
///
/// Note: We keep the same constructor signature used by the FFT pipeline,
/// even though the Backend tab only needs [expression].
class ResultsPage extends StatelessWidget {
  final String expression;

  // Time-domain samples (for StepPage / charts)
  final List<double> t;
  final List<double> signal;
  final double dt;

  // FFT output (currently unused by this tab layout, but kept for compatibility)
  final List<double> omega;
  final List<double> magnitude;

  const ResultsPage({
    super.key,
    required this.expression,
    required this.t,
    required this.signal,
    required this.dt,
    required this.omega,
    required this.magnitude,
  });

  @override
  Widget build(BuildContext context) {
    return DefaultTabController(
      length: 2,
      child: Scaffold(
        appBar: AppBar(
          title: const Text('Results'),
          bottom: const TabBar(
            tabs: [
              Tab(text: 'Step'),
              Tab(text: 'Chart'),
            ],
          ),
        ),
        body: TabBarView(
          // Order must match TabBar order:
          // left = Backend, right = Chart
          children: [
            SymbolPage(expression: expression),
            StepPage(t: t, x: signal, dt: dt),
          ],
        ),
      ),
    );
  }
}
