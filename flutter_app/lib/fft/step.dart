import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:syncfusion_flutter_charts/charts.dart';


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

    final omega = List<double>.generate(m, (i) => wMin + (wMax - wMin) * i / (m - 1));
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
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _Card(
            title: 'Set ω (rad/s)',
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Expanded(
                      child: TextField(
                        controller: _omegaCtrl,
                        keyboardType: const TextInputType.numberWithOptions(decimal: true, signed: true),
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
                Text('ω = ${_omega.toStringAsPrecision(8)}'),
                Slider(
                  min: -_omegaMax,
                  max: _omegaMax,
                  value: _omega.clamp(-_omegaMax, _omegaMax),
                  onChanged: (v) => _setOmega(v),
                ),
                Text(
                  'Nyquist range: [${(-_omegaMax).toStringAsPrecision(4)}, ${_omegaMax.toStringAsPrecision(4)}]',
                  style: Theme.of(context).textTheme.bodySmall,
                ),
                const SizedBox(height: 10),
                Text(
                  'X(ω) ≈ Re=${Xre.toStringAsPrecision(6)}, Im=${Xim.toStringAsPrecision(6)}, |X|=${Xmag.toStringAsPrecision(6)}',
                  style: Theme.of(context).textTheme.bodyMedium,
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),
          _ChartCard(
            title: '1) Original signal x(t)',
            xTitle: 't',
            yTitle: 'x(t)',
            series: [
              _SeriesData(name: 'x(t)', x: t, y: x),
            ],
          ),
          const SizedBox(height: 16),
          _ChartCard(
            title: '2) Kernel e^{-jωt} = cos(ωt) - j sin(ωt)',
            xTitle: 't',
            yTitle: 'kernel',
            series: [
              _SeriesData(name: 'cos(ωt)', x: t, y: cosW),
              _SeriesData(name: '-sin(ωt)', x: t, y: negSinW),
            ],
          ),
          const SizedBox(height: 16),
          _ChartCard(
            title: '3) Integrand x(t)·e^{-jωt}',
            xTitle: 't',
            yTitle: 'integrand',
            series: [
              _SeriesData(name: 'Re: x(t)cos(ωt)', x: t, y: reIntegrand),
              _SeriesData(name: 'Im: -x(t)sin(ωt)', x: t, y: imIntegrand),
            ],
          ),
          const SizedBox(height: 16),
          _ChartCard(
            title: '4) Running integral (cumulative)  ∫ x(t)e^{-jωt} dt',
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
            title: '5) Integral result vs ω  (ω-axis curve)',
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
                  height: 300,
                  child: _sweepLoading
                      ? const Center(child: CircularProgressIndicator())
                      : _OmegaChart(
                    omega: _sweepOmega,
                    re: _sweepRe,
                    im: _sweepIm,
                    mag: _sweepMag,
                  ),
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
    final reData = List<_Pt>.generate(omega.length, (i) => _Pt(omega[i], re[i]));
    final imData = List<_Pt>.generate(omega.length, (i) => _Pt(omega[i], im[i]));
    final magData = List<_Pt>.generate(omega.length, (i) => _Pt(omega[i], mag[i]));

    return SfCartesianChart(
      legend: const Legend(isVisible: true, position: LegendPosition.bottom),
      primaryXAxis: const NumericAxis(title: AxisTitle(text: 'ω (rad/s)')),
      primaryYAxis: const NumericAxis(title: AxisTitle(text: 'X(ω)')),
      series: <LineSeries<_Pt, double>>[
        LineSeries<_Pt, double>(
          name: 'Re{X(ω)}',
          dataSource: reData,
          xValueMapper: (p, _) => p.x,
          yValueMapper: (p, _) => p.y,
        ),
        LineSeries<_Pt, double>(
          name: 'Im{X(ω)}',
          dataSource: imData,
          xValueMapper: (p, _) => p.x,
          yValueMapper: (p, _) => p.y,
        ),
        LineSeries<_Pt, double>(
          name: '|X(ω)|',
          dataSource: magData,
          xValueMapper: (p, _) => p.x,
          yValueMapper: (p, _) => p.y,
        ),
      ],
    );
  }
}

class _Card extends StatelessWidget {
  final String title;
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
            Text(title, style: Theme.of(context).textTheme.titleMedium),
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
  final String title;
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
    final seriesList = series.map((s) {
      final data = List<_Pt>.generate(s.x.length, (i) => _Pt(s.x[i], s.y[i]));
      return LineSeries<_Pt, double>(
        name: s.name,
        dataSource: data,
        xValueMapper: (p, _) => p.x,
        yValueMapper: (p, _) => p.y,
      );
    }).toList();

    return _Card(
      title: title,
      child: SizedBox(
        height: 260,
        child: SfCartesianChart(
          legend: const Legend(isVisible: true, position: LegendPosition.bottom),
          primaryXAxis: NumericAxis(title: AxisTitle(text: xTitle)),
          primaryYAxis: NumericAxis(title: AxisTitle(text: yTitle)),
          series: seriesList,
        ),
      ),
    );
  }
}

class _Pt {
  final double x;
  final double y;
  _Pt(this.x, this.y);
}