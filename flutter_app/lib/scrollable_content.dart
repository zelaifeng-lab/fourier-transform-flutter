import 'package:flutter/material.dart';
import 'package:flutter_math_fork/flutter_math.dart';

class ScrollableMathLine extends StatefulWidget {
  final String latex;
  final TextStyle? textStyle;
  final String? semanticsLabel;

  const ScrollableMathLine({
    super.key,
    required this.latex,
    this.textStyle,
    this.semanticsLabel,
  });

  @override
  State<ScrollableMathLine> createState() => _ScrollableMathLineState();
}

class _ScrollableMathLineState extends State<ScrollableMathLine> {
  final ScrollController _controller = ScrollController();

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final line = Scrollbar(
      controller: _controller,
      thumbVisibility: true,
      child: SingleChildScrollView(
        controller: _controller,
        scrollDirection: Axis.horizontal,
        child: Padding(
          padding: const EdgeInsets.only(bottom: 8),
          child: Math.tex(
            widget.latex,
            textStyle:
                widget.textStyle ?? Theme.of(context).textTheme.bodyLarge,
          ),
        ),
      ),
    );

    if (widget.semanticsLabel == null) {
      return line;
    }

    return Semantics(label: widget.semanticsLabel, child: line);
  }
}

class ScrollableTextLine extends StatefulWidget {
  final String text;
  final TextStyle? style;
  final String? semanticsLabel;

  const ScrollableTextLine({
    super.key,
    required this.text,
    this.style,
    this.semanticsLabel,
  });

  @override
  State<ScrollableTextLine> createState() => _ScrollableTextLineState();
}

class _ScrollableTextLineState extends State<ScrollableTextLine> {
  final ScrollController _controller = ScrollController();

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final line = Scrollbar(
      controller: _controller,
      thumbVisibility: true,
      child: SingleChildScrollView(
        controller: _controller,
        scrollDirection: Axis.horizontal,
        child: Padding(
          padding: const EdgeInsets.only(bottom: 8),
          child: Text(
            widget.text,
            maxLines: 1,
            softWrap: false,
            style: widget.style,
          ),
        ),
      ),
    );

    if (widget.semanticsLabel == null) {
      return line;
    }

    return Semantics(label: widget.semanticsLabel, child: line);
  }
}

class BoundedScrollableText extends StatefulWidget {
  final String text;
  final TextStyle? style;
  final double maxHeight;

  const BoundedScrollableText({
    super.key,
    required this.text,
    this.style,
    this.maxHeight = 96,
  });

  @override
  State<BoundedScrollableText> createState() => _BoundedScrollableTextState();
}

class _BoundedScrollableTextState extends State<BoundedScrollableText> {
  final ScrollController _controller = ScrollController();

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return ConstrainedBox(
      constraints: BoxConstraints(maxHeight: widget.maxHeight),
      child: Scrollbar(
        controller: _controller,
        child: SingleChildScrollView(
          controller: _controller,
          child: Text(widget.text, style: widget.style),
        ),
      ),
    );
  }
}
