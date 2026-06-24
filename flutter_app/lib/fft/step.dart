import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart' show visibleForTesting;
import 'package:syncfusion_flutter_charts/charts.dart';

import '../responsive.dart';
import '../scrollable_content.dart';

class StepPage extends StatefulWidget {
  final List<double> t;
  final List<double> x;
  final double dt;

  const StepPage({
    super.key,
    required this.t,
    required this.x,
    required this.dt,
  });

  @override
  State<StepPage> createState() => _StepPageState();
}

class _StepPageState extends State<StepPage> {
  late double _omega;
  late final TextEditingController _omegaCtrl;

  // ω-sweep (integral result as a function of ω)
  bool _sweepLoading = true;
  List<double> _sweepOmega = const [];
  List<double> _sweepRe = const [];
  List<double> _sweepIm = const [];
  List<double> _sweepMag = const [];

  // Controls for sweep density (kept modest for performance)
  final int _sweepPoints = 201; // odd -> includes ω=0
  int _timeStride = 1; // downsample in time for sweep

  @override
  void initState() {
    super.initState();
    _omega = 0.0;
    _omegaCtrl = TextEditingController(text: '0');

    // Precompute ω-sweep once when entering this page
    WidgetsBinding.instance.addPostFrameCallback((_) => _computeSweep());
  }

  @override
  void dispose() {
    _omegaCtrl.dispose();
    super.dispose();
  }

  double get _omegaMax => math.pi / widget.dt; // Nyquist (rad/s)

  void _setOmega(double v) {
    setState(() {
      _omega = v;
      _omegaCtrl.text = v.toStringAsPrecision(8);
    });
  }

  void _applyOmegaFromText() {
    final v = double.tryParse(_omegaCtrl.text.trim());
    if (v == null) return;
    final clamped = v.clamp(-_omegaMax, _omegaMax);
    _setOmega(clamped);
  }

  Future<void> _computeSweep() async {
    setState(() => _sweepLoading = true);

    // Time downsample for performance on large N
    final n = widget.t.length;
    final stride = math.max(1, n ~/ 1200); // ~1200 samples max
    _timeStride = stride;

    final m = _sweepPoints;
    final wMin = -_omegaMax;
    final wMax = _omegaMax;

    final omega = List<double>.generate(
      m,
      (i) => wMin + (wMax - wMin) * i / (m - 1),
    );
    final re = List<double>.filled(m, 0);
    final im = List<double>.filled(m, 0);
    final mag = List<double>.filled(m, 0);

    // Compute X(ω) by Riemann sum for each ω (using downsampled time grid)
    for (int k = 0; k < m; k++) {
      final w = omega[k];
      double reSum = 0;
      double imSum = 0;

      for (int i = 0; i < n; i += stride) {
        final wt = w * widget.t[i];
        final c = math.cos(wt);
        final ns = -math.sin(wt);
        final xi = widget.x[i];

        reSum += xi * c;
        imSum += xi * ns;
      }

      // scale by dt * stride because we skipped points
      final scale = widget.dt * stride;
      re[k] = reSum * scale;
      im[k] = imSum * scale;
      mag[k] = math.sqrt(re[k] * re[k] + im[k] * im[k]);
    }

    if (!mounted) return;
    setState(() {
      _sweepOmega = omega;
      _sweepRe = re;
      _sweepIm = im;
      _sweepMag = mag;
      _sweepLoading = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    final t = widget.t;
    final x = widget.x;
    final n = t.length;

    final cosW = List<double>.filled(n, 0);
    final negSinW = List<double>.filled(n, 0);

    final reIntegrand = List<double>.filled(n, 0);
    final imIntegrand = List<double>.filled(n, 0);

    // running integral of the complex integrand
    final reCum = List<double>.filled(n, 0);
    final imCum = List<double>.filled(n, 0);
    final magCum = List<double>.filled(n, 0);

    double reSum = 0;
    double imSum = 0;

    for (int i = 0; i < n; i++) {
      final wt = _omega * t[i];
      final c = math.cos(wt);
      final ns = -math.sin(wt);

      cosW[i] = c;
      negSinW[i] = ns;

      final re = x[i] * c;
      final im = x[i] * ns;

      reIntegrand[i] = re;
      imIntegrand[i] = im;

      // Running integral (Riemann sum)
      reSum += re * widget.dt;
      imSum += im * widget.dt;

      reCum[i] = reSum;
      imCum[i] = imSum;
      magCum[i] = math.sqrt(reSum * reSum + imSum * imSum);
    }

    final Xre = reCum.isEmpty ? 0.0 : reCum.last;
    final Xim = imCum.isEmpty ? 0.0 : imCum.last;
    final Xmag = magCum.isEmpty ? 0.0 : magCum.last;

    return Scaffold(
      appBar: AppBar(
        title: const Text('FT Steps'),
        actions: [
          IconButton(
            tooltip: 'Recompute ω-sweep',
            onPressed: _computeSweep,
            icon: const Icon(Icons.refresh),
          ),
        ],
      ),
      body: ResponsiveScrollView(
        children: [
          _Card(
            title: ScrollableMathLine(
              latex: r'\text{Set }\omega\ \text{(rad/s)}',
              textStyle: Theme.of(context).textTheme.titleMedium,
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Expanded(
                      child: TextField(
                        controller: _omegaCtrl,
                        keyboardType: const TextInputType.numberWithOptions(
                          decimal: true,
                          signed: true,
                        ),
                        decoration: const InputDecoration(
                          labelText: 'ω',
                          hintText: 'Enter ω (rad/s)',
                          border: OutlineInputBorder(),
                          isDense: true,
                        ),
                        onSubmitted: (_) => _applyOmegaFromText(),
                      ),
                    ),
                    const SizedBox(width: 10),
                    FilledButton(
                      onPressed: _applyOmegaFromText,
                      child: const Text('Apply'),
                    ),
                  ],
                ),
                const SizedBox(height: 10),
                ScrollableMathLine(
                  latex: r'\omega = ' + _omega.toStringAsPrecision(8),
                ),
                Slider(
                  min: -_omegaMax,
                  max: _omegaMax,
                  value: _omega.clamp(-_omegaMax, _omegaMax),
                  onChanged: (v) => _setOmega(v),
                ),
                ScrollableMathLine(
                  latex:
                      r'\text{Nyquist range: }\omega\in['
                      '${(-_omegaMax).toStringAsPrecision(4)}, '
                      '${_omegaMax.toStringAsPrecision(4)}]',
                  textStyle: Theme.of(context).textTheme.bodySmall,
                ),
                const SizedBox(height: 10),
                ScrollableMathLine(
                  latex:
                      r'X(\omega)\approx ' +
                      Xre.toStringAsPrecision(6) +
                      r' + j\,' +
                      Xim.toStringAsPrecision(6) +
                      r',\quad '
                          r'\mathrm{Re}\{X\}=' +
                      Xre.toStringAsPrecision(6) +
                      r',\ '
                          r'\mathrm{Im}\{X\}=' +
                      Xim.toStringAsPrecision(6) +
                      r',\ |X|=' +
                      Xmag.toStringAsPrecision(6),
                  textStyle: Theme.of(context).textTheme.bodyMedium,
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),
          _ChartCard(
            title: const ScrollableMathLine(
              latex: r'1)\ \mathrm{Original\ signal}\ \; x(t)',
            ),
            xTitle: 't',
            yTitle: 'x(t)',
            series: [_SeriesData(name: 'x(t)', x: t, y: x)],
          ),
          const SizedBox(height: 16),
          _ChartCard(
            title: const ScrollableMathLine(
              latex:
                  r'2)\ \mathrm{Kernel}\ \; e^{-j\omega t}=\cos(\omega t)-j\sin(\omega t)',
            ),
            xTitle: 't',
            yTitle: 'kernel',
            series: [
              _SeriesData(name: 'cos(ωt)', x: t, y: cosW),
              _SeriesData(name: '-sin(ωt)', x: t, y: negSinW),
            ],
          ),
          const SizedBox(height: 16),
          _ChartCard(
            title: const ScrollableMathLine(
              latex: r'3)\ \mathrm{Integrand}\ \; x(t)\cdot e^{-j\omega t}',
            ),
            xTitle: 't',
            yTitle: 'integrand',
            series: [
              _SeriesData(name: 'Re: x(t)cos(ωt)', x: t, y: reIntegrand),
              _SeriesData(name: 'Im: -x(t)sin(ωt)', x: t, y: imIntegrand),
            ],
          ),
          const SizedBox(height: 16),
          _ChartCard(
            title: const ScrollableMathLine(
              latex:
                  r'4)\ \mathrm{Running\ integral\ (cumulative)}\ \;'
                  r'\int x(t)e^{-j\omega t}\,dt',
            ),
            xTitle: 't',
            yTitle: 'X(ω) up to t',
            series: [
              _SeriesData(name: 'Re cumulative', x: t, y: reCum),
              _SeriesData(name: 'Im cumulative', x: t, y: imCum),
              _SeriesData(name: '|X| cumulative', x: t, y: magCum),
            ],
          ),
          const SizedBox(height: 16),
          _Card(
            title: ScrollableMathLine(
              latex:
                  r'5)\ \text{Integral result vs }\omega\ \text{(}\omega\text{-axis curve)}',
              textStyle: Theme.of(context).textTheme.titleMedium,
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  _sweepLoading
                      ? 'Computing…'
                      : 'Computed with ${_sweepOmega.length} ω points, time stride=$_timeStride (downsampled for speed).',
                  style: Theme.of(context).textTheme.bodySmall,
                ),
                const SizedBox(height: 8),
                SizedBox(
                  height: AppBreakpoints.chartHeight(
                    MediaQuery.sizeOf(context).width,
                  ),
                  child: _sweepLoading
                      ? const Center(child: CircularProgressIndicator())
                      : _OmegaChart(
                          omega: _sweepOmega,
                          re: _sweepRe,
                          im: _sweepIm,
                          mag: _sweepMag,
                        ),
                ),
                const SizedBox(height: 8),
                // A LaTeX-rendered legend hint (Syncfusion legend itself is plain text).
                ScrollableMathLine(
                  latex:
                      r'\mathrm{Legend:}\ \mathrm{Re}\{X(\omega)\},\ \mathrm{Im}\{X(\omega)\},\ |X(\omega)|',
                  textStyle: Theme.of(context).textTheme.bodySmall,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

/// ---------- helpers UI ----------

class _OmegaChart extends StatelessWidget {
  final List<double> omega;
  final List<double> re;
  final List<double> im;
  final List<double> mag;

  const _OmegaChart({
    required this.omega,
    required this.re,
    required this.im,
    required this.mag,
  });

  @override
  Widget build(BuildContext context) {
    final series = <_SeriesData>[
      _SeriesData(name: 'Re{X(omega)}', x: omega, y: re),
      _SeriesData(name: 'Im{X(omega)}', x: omega, y: im),
      _SeriesData(name: '|X(omega)|', x: omega, y: mag),
    ];

    return SfCartesianChart(
      legend: const Legend(isVisible: true, position: LegendPosition.bottom),
      primaryXAxis: const NumericAxis(title: AxisTitle(text: 'omega (rad/s)')),
      primaryYAxis: const NumericAxis(title: AxisTitle(text: 'X(omega)')),
      series: _buildSegmentedLineSeries(series),
    );
  }
}

class _Card extends StatelessWidget {
  final Widget title;
  final Widget child;
  const _Card({required this.title, required this.child});

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
        side: BorderSide(color: Theme.of(context).dividerColor),
      ),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            DefaultTextStyle.merge(
              style: Theme.of(context).textTheme.titleMedium,
              child: title,
            ),
            const SizedBox(height: 8),
            child,
          ],
        ),
      ),
    );
  }
}

class _SeriesData {
  final String name;
  final List<double> x;
  final List<double> y;
  _SeriesData({required this.name, required this.x, required this.y});
}

class _ChartCard extends StatelessWidget {
  final Widget title;
  final String xTitle;
  final String yTitle;
  final List<_SeriesData> series;

  const _ChartCard({
    required this.title,
    required this.xTitle,
    required this.yTitle,
    required this.series,
  });

  @override
  Widget build(BuildContext context) {
    final seriesList = _buildSegmentedLineSeries(series);

    return _Card(
      title: title,
      child: SizedBox(
        height: AppBreakpoints.chartHeight(MediaQuery.sizeOf(context).width),
        child: SfCartesianChart(
          legend: const Legend(
            isVisible: true,
            position: LegendPosition.bottom,
          ),
          primaryXAxis: NumericAxis(title: AxisTitle(text: xTitle)),
          primaryYAxis: NumericAxis(title: AxisTitle(text: yTitle)),
          series: seriesList,
        ),
      ),
    );
  }
}

const List<Color> _chartSegmentColors = <Color>[
  Color(0xFF2563EB),
  Color(0xFFDC2626),
  Color(0xFF16A34A),
  Color(0xFF9333EA),
  Color(0xFFEA580C),
  Color(0xFF0891B2),
];

List<LineSeries<_Pt, double>> _buildSegmentedLineSeries(
  List<_SeriesData> series,
) {
  final lines = <LineSeries<_Pt, double>>[];
  for (var i = 0; i < series.length; i++) {
    final color = _chartSegmentColors[i % _chartSegmentColors.length];
    final segments = _splitRenderableSegments(series[i].x, series[i].y);
    for (var j = 0; j < segments.length; j++) {
      lines.add(
        LineSeries<_Pt, double>(
          name: series[i].name,
          dataSource: segments[j],
          color: color,
          isVisibleInLegend: j == 0,
          xValueMapper: (p, _) => p.x,
          yValueMapper: (p, _) => p.y,
        ),
      );
    }
  }
  return lines;
}

List<List<_Pt>> _splitRenderableSegments(List<double> x, List<double> y) {
  final n = math.min(x.length, y.length);
  if (n < 2) return const <List<_Pt>>[];

  final finiteAbs = <double>[];
  for (var i = 0; i < n; i++) {
    if (x[i].isFinite && y[i].isFinite) {
      final v = y[i].abs();
      if (v > 1e-12) finiteAbs.add(v);
    }
  }
  if (finiteAbs.isEmpty) return const <List<_Pt>>[];

  finiteAbs.sort();
  final scale = finiteAbs[finiteAbs.length ~/ 2].clamp(1e-9, 1e9).toDouble();
  final jumpLimit = math.max(scale * 8.0, 6.0);
  final highLimit = math.max(scale * 8.0, 8.0);
  const hardLimit = 1e9;

  final valid = List<bool>.filled(n, false);
  for (var i = 0; i < n; i++) {
    valid[i] = x[i].isFinite && y[i].isFinite && y[i].abs() <= hardLimit;
  }

  for (var i = 0; i < n; i++) {
    if (!valid[i] || y[i].abs() <= highLimit) continue;

    var hasSingularJump = false;
    if (i > 0 && valid[i - 1] && (y[i] - y[i - 1]).abs() > jumpLimit) {
      hasSingularJump = true;
    }
    if (i + 1 < n && valid[i + 1] && (y[i + 1] - y[i]).abs() > jumpLimit) {
      hasSingularJump = true;
    }
    if (hasSingularJump) valid[i] = false;
  }

  final segments = <List<_Pt>>[];
  var current = <_Pt>[];
  double? previousY;

  void closeSegment() {
    if (current.length >= 2) segments.add(current);
    current = <_Pt>[];
    previousY = null;
  }

  for (var i = 0; i < n; i++) {
    if (!valid[i]) {
      closeSegment();
      continue;
    }

    if (previousY != null) {
      final dy = (y[i] - previousY!).abs();
      final bothLarge = y[i].abs() > highLimit && previousY!.abs() > highLimit;
      final crossesSign = y[i].sign != previousY!.sign;
      if (dy > jumpLimit && (bothLarge || crossesSign)) {
        closeSegment();
      }
    }

    current.add(_Pt(x[i], y[i]));
    previousY = y[i];
  }
  closeSegment();
  return segments;
}

@visibleForTesting
List<int> chartSegmentLengthsForTest(List<double> x, List<double> y) {
  return _splitRenderableSegments(x, y).map((segment) => segment.length).toList();
}

class _Pt {
  final double x;
  final double y;
  _Pt(this.x, this.y);
}
