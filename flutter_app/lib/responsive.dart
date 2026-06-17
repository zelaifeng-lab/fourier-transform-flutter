import 'package:flutter/material.dart';

class AppBreakpoints {
  static bool compact(double width) => width < 600;
  static bool medium(double width) => width >= 600 && width < 1024;

  static double maxContentWidth(double width) {
    if (compact(width)) return double.infinity;
    if (medium(width)) return 840;
    return 1040;
  }

  static EdgeInsets pagePadding(double width) {
    if (compact(width)) return const EdgeInsets.all(12);
    if (medium(width)) return const EdgeInsets.all(20);
    return const EdgeInsets.symmetric(horizontal: 28, vertical: 24);
  }

  static double chartHeight(double width) {
    if (compact(width)) return 260;
    if (medium(width)) return 310;
    return 360;
  }
}

class ResponsiveScrollView extends StatelessWidget {
  final List<Widget> children;
  final CrossAxisAlignment crossAxisAlignment;

  const ResponsiveScrollView({
    super.key,
    required this.children,
    this.crossAxisAlignment = CrossAxisAlignment.stretch,
  });

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final width = constraints.maxWidth;
        return SingleChildScrollView(
          padding: AppBreakpoints.pagePadding(width),
          child: Center(
            child: ConstrainedBox(
              constraints: BoxConstraints(
                maxWidth: AppBreakpoints.maxContentWidth(width),
              ),
              child: Column(
                crossAxisAlignment: crossAxisAlignment,
                children: children,
              ),
            ),
          ),
        );
      },
    );
  }
}
