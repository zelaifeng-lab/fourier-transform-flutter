import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'fft/fourier_transform_bloc.dart';
import 'page/Formula.dart';

// Preserve source compatibility for existing consumers.
export 'page/Formula.dart' show HomePage, FourierExample;

void main() {
  runApp(const AppRoot());
}

class AppRoot extends StatelessWidget {
  const AppRoot({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Fourier Transform',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.indigo),
        useMaterial3: true,
      ),
      home: BlocProvider(
        create: (_) => FourierTransformBloc(),
        child: const HomePage(),
      ),
    );
  }
}
