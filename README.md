# Fourier Transform Engine (Engineering-Oriented)

A **rule-based symbolic Fourier Transform engine** designed for engineering applications, built with:

- 🧠 **FastAPI backend** for symbolic Fourier transform computation  
- 📱 **Flutter frontend** for LaTeX rendering and interactive visualization  

Unlike general CAS systems, this engine:

❌ Avoids RootSum, arg(ω), and complicated piecewise branches  
✅ Uses **distribution theory** (δ, PV, sign)  
✅ Follows **engineering transform tables**  
✅ Produces **step-by-step educational derivations**

---

## ✨ Key Features

✔ Rule-based Fourier transform (not `sympy.integrate`)  
✔ Supports generalized functions (Dirac δ, Heaviside u(t), PV integrals)  
✔ Step-by-step solution output for learning  
✔ Flutter UI renders math in LaTeX  
✔ Designed for signals & systems education

---

## 📂 Project Structure

```
backend/        → FastAPI symbolic FT engine  
flutter_app/    → Flutter frontend (UI + visualization)  
```

---

## 🧠 Backend (FastAPI)

### Install dependencies

```
cd backend
pip install -r requirements.txt
```

### Run server

```
uvicorn backend:app --reload
```

API docs will be available at:

👉 http://127.0.0.1:8000/docs

---

## 📱 Flutter Frontend

Make sure the backend server is running first.

```
cd flutter_app
flutter pub get
flutter run
```

---

## 📘 Supported Transform Types

### Elementary & Distribution Signals
- Dirac Delta δ(t)
- Heaviside step u(t)
- Sign function sgn(t)
- Principal Value integrals (PV)

### Exponentials & Trigonometric
- e^{iω₀t}
- sin(ω₀t), cos(ω₀t)
- Automatic trig → δ expansion

### Rational Functions
- (at+b)/(t²+c)
- 1/(t²+a²)
- 1/(t+a)
- Polynomial division + partial fractions

### Windowed Signals
- Finite interval signals via step functions  
  u(t−a) − u(t−b)

### Fourier Transform Properties
- Time shift  
- Frequency shift (modulation)  
- Scaling  
- Differentiation in frequency domain  
- Convolution

---

## 🎯 Project Goal

To create an **engineering-focused Fourier Transform learning tool** that shows derivations the way they appear in textbooks, rather than black-box symbolic outputs.

---

## 🚀 Future Improvements

- Online deployment of backend API  
- More transform pairs (Bessel, sinc, etc.)  
- Interactive spectrum visualization  
- Automatic function-type detection

---

## 📜 License

This project is for educational and research purposes.
