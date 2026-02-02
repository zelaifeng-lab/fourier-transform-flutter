import 'package:flutter/material.dart';

import 'step.dart';
import 'symbol.dart';

class ResultsPage extends StatelessWidget {
  final String expression;

  final List<double> t;
  final List<double> signal;
  final double dt;

  // Kept for compatibility with bloc output (not used by Symbol tab)
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
              Tab(text: 'Symbol'),
              Tab(text: 'Steps'),
            ],
          ),
        ),
        body: TabBarView(
          children: [
            StepPage(t: t, x: signal, dt: dt),
            SymbolPage(expression: expression),
          ],
        ),
      ),
    );
  }
}