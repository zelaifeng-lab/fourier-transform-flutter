# 基于 Flutter 的傅里叶变换应用开发

## 副标题

符号推导、数值可视化与 Python 后端支持

## 作者信息

作者：Zelai Feng  
指导教师：待补充  
学院：School of Mechanical and Aerospace Engineering  
学校：Nanyang Technological University  
年份：2026

\newpage

# 原创性声明

本人声明，本论文所提交的工作为本人在指导教师指导下完成的项目成果。除文中已明确引用或说明的内容外，论文中的系统设计、程序实现、测试分析和文字表述均为本人完成。本论文未以相同或近似形式提交给其他课程、项目或学位申请。

\newpage

# 指导教师声明

本页用于放置指导教师声明。正式提交前应根据学院提供的模板补全签名、日期和声明文本。

\newpage

# 作者贡献说明

本项目由作者完成主要设计与实现工作。作者负责 Flutter 前端界面、FastAPI 后端接口、傅里叶变换规则引擎、测试用例设计、文档撰写和结果分析。若后续加入第三方库、参考代码或外部工具，应在本节中说明其用途和贡献边界。

\newpage

# 摘要

本项目开发了一套运行在智能设备和网页端的傅里叶变换应用。系统以 Flutter 作为前端框架，以 FastAPI 和 Python 作为符号计算后端。前端负责表达式输入、示例选择、公式渲染、推导步骤显示和图像展示；后端负责解析用户输入，识别常见信号结构，并返回工程风格的傅里叶变换结果。

项目的核心思想是采用规则优先的符号计算方式。系统优先使用已知傅里叶变换对、时移、调制、尺度变换、卷积和分布理论，而不是直接把所有表达式交给通用符号积分。这样可以避免 `Piecewise`、`RootSum` 和复杂条件表达式频繁出现在最终结果中。对于部分可控表达式，系统仍保留 SymPy 作为兜底路径，但其输出需要经过格式检查和教学步骤整理。

论文重点说明系统如何实现这些功能。报告不仅展示应用界面和功能效果，还给出每类变换规则对应的数学公式、关键代码片段、API 返回结构、前端调用方式和测试方法。测试部分覆盖后端回归测试、Flutter 前端测试、前后端联合测试和代表性数学结果校验。结果表明，该系统可以处理 Dirac delta、unit step、sign、rect、tri、三角函数、指数函数、高斯函数、sinc 类函数、多项式乘阶跃函数和部分高阶有理式等常见输入，并能输出较适合教学使用的步骤。

关键词：傅里叶变换；Flutter；FastAPI；符号计算；分布理论；规则引擎；数学可视化

\newpage

# 致谢

本项目在指导教师的建议下逐步完善。指导教师对报告结构、功能说明、代码展示方式和测试内容提出了重要意见，使项目从功能演示逐步转向更完整的工程型 dissertation 表达。作者也感谢开源社区提供的 Flutter、FastAPI、SymPy、pytest 和相关工具。这些工具为跨平台界面开发、后端服务、符号计算和自动化测试提供了基础支持。

\newpage

# 图目录

正式 Word 版本应使用自动图目录。本源文稿保留图目录位置。建议至少包含：

图 5.1 系统总体架构  
图 6.1 Flutter 输入界面  
图 6.2 示例按钮与输入区域布局  
图 13.1 实际应用输入界面截图  
图 13.2 实际应用步骤推导截图  
图 13.3 实际应用图像展示截图

\newpage

# 表目录

正式 Word 版本应使用自动表目录。本源文稿保留表目录位置。建议至少包含：

表 8.1 常见函数族与规则覆盖范围  
表 10.1 已支持函数类型概括  
表 14.1 代表性测试用例  
表 15.1 系统能力与限制总结

\newpage

# 符号与缩写表

| 符号或缩写 | 含义 |
|---|---|
| t | 时间变量 |
| ω | 角频率变量 |
| j | 虚数单位 |
| X(ω) | x(t) 的傅里叶变换 |
| δ(ω) | Dirac delta 分布 |
| PV | Cauchy principal value，柯西主值 |
| u(t) | Unit step / Heaviside step |
| API | Application Programming Interface |
| CAS | Computer Algebra System |
| UI | User Interface |

\newpage

# 1 引言

## 1.1 项目背景

傅里叶变换是信号分析、控制工程、通信系统和振动分析中的基础工具。学生在学习傅里叶变换时，常见困难并不只在计算结果本身，还包括如何识别信号类型、如何选择合适的变换性质，以及如何理解分布形式的结果。例如，阶跃函数、符号函数和 `1/t` 一类信号不能简单看成普通绝对可积函数，它们的傅里叶变换需要用 Dirac delta 或 Cauchy principal value 表达。

现有通用数学软件通常可以给出符号结果，但结果未必适合工程教学。某些输出会包含较复杂的分段形式、根和条件表达式。对于初学者来说，这类结果不一定能帮助理解傅里叶变换的结构。相比之下，工程课程更强调已知变换对和性质的组合使用，例如时移、调制、尺度变换和卷积。

本项目因此开发一套面向教学使用的傅里叶变换应用。系统把 Flutter 前端与 Python 后端结合起来。前端提供可交互输入和公式展示；后端通过规则优先的方式生成结果和步骤。项目目标不是取代完整 CAS，而是在一定范围内给出更清楚、更工程化的傅里叶变换输出。

## 1.2 项目目标

项目目标包括以下几项。

第一，开发可以在网页端和移动端运行的 Flutter 应用。用户可以输入时间域函数，也可以通过示例按钮快速选择常见信号。

第二，设计 FastAPI 后端。后端接收前端表达式，返回傅里叶变换结果、推导步骤、附加条件和错误信息。

第三，建立规则优先的符号傅里叶变换引擎。系统优先使用常见变换对和性质，而不是无条件调用 SymPy 积分。

第四，输出适合教学阅读的步骤。步骤应说明信号类型、使用的变换对、应用的性质和最终工程结果。

第五，建立测试数据和回归测试。测试不仅检查接口是否返回成功，还检查数学结果、LaTeX 输出和步骤文本是否满足预期。

## 1.3 项目范围

本项目主要关注一维连续时间傅里叶变换，采用工程中常见的角频率形式。系统默认时间变量为 `t`，频域变量为 `ω`。当前支持的函数族包括冲激、阶跃、符号函数、矩形脉冲、三角脉冲、三角函数、指数衰减函数、高斯函数、sinc 类函数、多项式乘阶跃函数和部分有理式。

项目不试图覆盖所有可能的符号表达式。对于超出规则范围的函数，系统会尝试受控 SymPy fallback。如果 fallback 输出不适合工程展示，系统应给出错误或提示，而不是把复杂内部表达式直接暴露给用户。

# 2 傅里叶变换理论基础

## 2.1 工程傅里叶变换约定

本项目采用工程中常见的傅里叶变换定义：

$$
X(\omega)=\int_{-\infty}^{\infty}x(t)e^{-j\omega t}\,dt
$$

对应的反变换为：

$$
x(t)=\frac{1}{2\pi}\int_{-\infty}^{\infty}X(\omega)e^{j\omega t}\,d\omega
$$

该约定决定了时移、调制和卷积公式中的相位符号。后端和前端显示均围绕这一约定组织。

## 2.2 分布意义下的信号

工程中常见的 `u(t)`、`sign(t)`、`1/t` 和常数信号并不总是普通绝对可积函数。它们的傅里叶变换需要在分布意义下理解。例如：

$$
\mathcal{F}\{u(t)\}=\pi\delta(\omega)+\mathrm{PV}\frac{1}{j\omega}
$$

$$
\mathcal{F}\{\operatorname{sign}(t)\}=\frac{2}{j\omega}
$$

其中 `PV` 表示 Cauchy principal value。系统在结果中使用 `(PV)` 的显示形式，以避免用户把 `PV` 误读为普通变量相乘。

## 2.3 常用傅里叶变换性质

系统大量使用傅里叶变换性质，而不是重复进行积分。若：

$$
x(t)\longleftrightarrow X(\omega)
$$

则时移性质为：

$$
x(t-t_0)\longleftrightarrow e^{-j\omega t_0}X(\omega)
$$

调制性质为：

$$
e^{j\omega_0t}x(t)\longleftrightarrow X(\omega-\omega_0)
$$

尺度性质为：

$$
x(at)\longleftrightarrow \frac{1}{|a|}X\left(\frac{\omega}{a}\right)
$$

这些性质也是后端 helper 设计的数学基础。

# 3 相关技术与设计依据

## 3.1 Flutter

Flutter 适合构建跨平台界面。项目使用 Flutter 负责输入框、示例按钮、结果显示、步骤显示和图像区域。它可以同时服务网页端和移动端，因此适合本项目“智能设备上的傅里叶变换应用”的目标。

## 3.2 FastAPI

FastAPI 用于构建后端接口。前端通过 HTTP 请求把表达式发给 `/fourier` 接口，后端返回 JSON 数据。FastAPI 的接口定义清晰，适合把符号计算服务和 Flutter 界面分离。

## 3.3 SymPy

SymPy 提供符号表达式解析、化简、积分和 LaTeX 输出能力。项目没有把 SymPy 作为唯一求解路径，而是把它放在 parser、表达式规范化和受控 fallback 中使用。这样既能利用 SymPy 的基础能力，又能避免最终结果被复杂中间表达式主导。

## 3.4 MathFork 与 LaTeX 渲染

前端使用 MathFork / `Math.tex` 渲染数学公式。用户输入、示例函数、变换结果和步骤公式都需要以 LaTeX 形式显示。为避免小屏幕上公式被裁切，前端对较长内容加入局部滚动区域。

# 4 系统总体设计

## 4.1 架构概述

系统采用前后端分离架构。Flutter 前端负责交互，FastAPI 后端负责计算。用户输入函数后，前端把表达式发送给后端。后端解析表达式，依次尝试规则匹配、性质推导和受控 fallback。计算完成后，后端返回 `result_latex`、`steps_latex`、`conditions_latex` 和错误信息。前端再把这些内容渲染给用户。

建议在正式 Word 版中加入系统架构图。图中应包含 Flutter UI、HTTP request、FastAPI API、parser、rule engine、SymPy fallback、cache 和 response renderer。

## 4.2 数据流

系统数据流如下。

1. 用户在 Flutter 输入表达式。
2. 前端生成请求体。
3. FastAPI 接收请求。
4. Parser 将输入转为 SymPy 表达式。
5. Rule engine 尝试识别函数结构。
6. Helper 提取时移、调制、尺度等参数。
7. 后端生成结果和步骤。
8. API 返回 JSON。
9. Flutter 使用 MathFork 渲染公式。
10. 图像区域根据后端或前端数据展示曲线。

## 4.3 API 返回结构

API 返回结构需要保持简洁。核心字段如下：

```python
{
    "ok": True,
    "result_latex": result_latex,
    "steps_latex": steps_latex,
    "conditions_latex": conditions_latex,
    "error": None,
}
```

该结构让前端不需要理解后端内部 rule 名称。前端只负责展示结果、步骤和错误提示。

# 5 Flutter 前端设计与实现

## 5.1 页面布局

前端页面围绕输入、示例、结果和图像四个区域展开。输入区域位于上方，方便用户先构造函数。示例按钮靠近输入框，帮助用户快速选择中等复杂度的表达式。结果和步骤区域放在输入下方，符合从问题到解答的阅读顺序。图像区域放在结果之后，用于辅助理解时域和频域变化。

页面需要适配不同屏幕大小。较小屏幕上，公式和注释可能过长，因此项目对局部内容采用滚动容器。这样不会破坏整个页面布局，也能避免长公式挤压其他组件。

## 5.2 输入与按钮设计

前端提供基础函数按钮，例如 `sin`、`cos`、`exp`、`u`、`delta`、`sign`、`rect` 和 `tri`。按钮的作用不是限制用户输入，而是降低输入成本。对于 `exp` 和幂函数，前端使用类似占位符的输入方式，使用户可以在指数位置继续输入，再通过方向键回到主表达式。

示例导航按钮被移动到输入框右侧，并采用上下排列。这种布局减少了按钮对输入区域下方空间的占用，也更适合移动端页面。

## 5.3 前端请求后端

前端通过环境变量或默认地址选择后端 API。网页端默认连接 Render 上的后端服务，Android emulator 则使用 `10.0.2.2` 访问本机后端。该设计使 GitHub Actions 构建前端网页时不需要同时构建后端，也不会破坏 Render 的部署方式。核心代码如代码 5.1 所示。

```dart
final String base = envBase.isNotEmpty
    ? envBase
    : (kIsWeb ? renderBase : androidEmulatorBase);

final Uri uri = Uri.parse(
  base.endsWith('/') ? '${base}fourier' : '${base}/fourier',
);
```

代码 5.1 的重点是把后端地址选择和请求路径拼接分开。用户如果需要测试其他后端地址，可以通过 `API_BASE_URL` 覆盖默认值。

请求发送后，前端只读取后端约定的 JSON 字段。这样前端不需要知道后端内部使用了哪一个 rule。

```dart
final res = await http.post(
  uri,
  headers: {'Content-Type': 'application/json'},
  body: jsonEncode({'expression': expression}),
);

final j = jsonDecode(res.body) as Map<String, dynamic>;
final resultLatex = (j['result_latex'] ?? r'\text{Unable to compute}').toString();
final steps = (j['steps_latex'] as List<dynamic>? ?? const [])
    .map((e) => e.toString())
    .toList();
```

## 5.4 LaTeX 渲染

公式显示使用 `Math.tex`。核心代码形式如下：

```dart
child: Math.tex(
  widget.latex,
  textStyle:
      widget.textStyle ?? Theme.of(context).textTheme.bodyLarge,
),
```

这里的 `??` 表示空值合并。如果 `widget.textStyle` 已经存在，就使用它；否则使用当前主题中的 `bodyLarge` 样式。这使公式组件可以在不同区域复用，同时保持默认显示风格。

## 5.5 局部滚动处理

当公式、步骤或图像注释过长时，前端使用局部滚动容器，而不是让整个页面布局被撑坏。该设计主要解决小屏幕上的显示异常。它不会改变原有功能，只改善用户阅读和操作体验。

局部滚动只包裹可能变长的公式行或文本行。代码 5.3 展示了公式行的处理方式。`SingleChildScrollView` 的滚动方向设为水平，因此长公式不会把页面整体撑宽。

```dart
child: SingleChildScrollView(
  controller: _controller,
  scrollDirection: Axis.horizontal,
  child: Padding(
    padding: const EdgeInsets.only(bottom: 8),
    child: Math.tex(widget.latex),
  ),
),
```

# 6 FastAPI 后端接口设计

## 6.1 接口职责

后端接口负责接收表达式、调用符号变换引擎并返回结果。它不把内部 rule 名称暴露给前端。这样前端和后端保持松耦合，后端规则扩展时不需要频繁修改前端显示逻辑。

## 6.2 输入解析

Parser 的作用不仅是读取字符串，也包括把用户输入规范成可计算的符号表达式。例如，前端或用户可能输入 `u(t)`、`frac(a,b)`、`exp(...)` 或带括号的有理式。Parser 需要把这些形式转换成后端规则可以识别的表达式。

规范化后的表达式进入 rule engine。若 parser 处理不充分，后端规则会更难匹配。因此 parser 是规则优先设计的第一层。

代码 6.1 展示了 parser 前的规范化处理。系统把工程写法 `u(t)`、`delta(t)`、`rect(t)` 和 `tri(t)` 映射为后端内部函数名，并把 `frac(a,b)` 转换为可由 SymPy 解析的除法结构。

```python
s = s.replace("^", "**")
s = s.replace("u(", "Heaviside(")
s = s.replace("delta(", "DiracDelta(")
s = s.replace("rect(", "Rect(")
s = s.replace("tri(", "Tri(")
s = s.replace("frac(", "FRAC(")
s = _convert_frac_calls(s)
```

规范化后，`parse_expr` 使用受控的 `local_dict`。这一步决定了哪些函数可以进入 rule-first 后端。

```python
local_dict = {
    "t": t, "omega": omega, "Heaviside": Heaviside,
    "DiracDelta": DiracDelta, "sign": sign,
    "Rect": Rect, "Tri": Tri,
}
expr = parse_expr(expr_str, local_dict=local_dict,
                  transformations=TRANSFORMS, evaluate=True)
```

## 6.3 错误处理

后端错误不应直接暴露 Python traceback。API 应返回可读错误信息。对于复杂表达式，系统可以说明该输入目前未被规则覆盖，或 fallback 结果不适合工程展示。

# 7 Rule-first 符号傅里叶变换引擎

## 7.1 设计思想

Rule-first engine 的核心思想是先识别信号类型，再应用已知变换对和性质。只有当规则无法覆盖且积分输出可控时，系统才使用 SymPy fallback。

这种设计更适合教学。学生看到的是“识别信号、选择性质、代入公式、整理结果”的过程，而不是复杂的 CAS 内部表达式。

## 7.2 Rule、helper 与 matcher 的关系

论文中需要清楚区分三个概念。

Rule 是具体的变换规则。例如 sign、rect、tri、Heaviside 和 Gaussian 都可以有对应 rule。Rule 决定某个表达式是否被该规则处理。

Helper 是供多个 rule 复用的工具函数。它可以提取仿射参数 `a*t+b`，也可以检查调制项、时移项或尺度项。通常是 rule 调用 helper，而不是 helper 调用 rule。

Matcher 是更宽泛的结构识别逻辑。它帮助系统判断表达式是否符合某类模式，例如乘积、加法、时移或调制。

代码 7.1 展示了 helper 如何提取 `a*t+b` 的中心和宽度。这个 helper 被 sign、rect、tri 等多个 rule 复用。它不直接决定最终变换结果，而是把参数提取出来交给具体 rule 使用。

```python
def _linear_center_and_width(arg):
    lin = _as_linear_in_t(arg)
    if lin is None:
        return None
    a, b = lin
    scale = _positive_linear_scale(a)
    if scale is None:
        return None
    width, sign_a = scale
    center = simplify(-b/a)
    return center, width, sign_a
```

## 7.3 通用处理流程

新增一个函数时，完整流程如下。

1. 明确数学变换对。
2. 判断结果属于普通函数还是分布形式。
3. 让 parser 能把用户输入转成规范表达式。
4. 在 rule 中识别函数结构。
5. 如函数内部含 `a*t+b`，调用 helper 提取参数。
6. 若存在时移、调制或尺度，应用对应性质。
7. 生成 `result_latex`。
8. 生成 `steps_latex`。
9. 通过 `/fourier` API 返回。
10. 增加 pytest 回归测试。

## 7.4 例：sign 函数

符号函数的基础变换对为：

$$
\operatorname{sign}(t)\longleftrightarrow \frac{2}{j\omega}
$$

对于 `sign(a*t+b)`，系统需要先识别内部仿射结构。若 `a*t+b = a(t-c)`，则可以转为带时移和尺度的形式。

代码 7.2 是实际 rule 的压缩片段。该 rule 先提取 `sign(...)`，再调用 helper 得到中心 `c` 和系数符号 `sign_a`。最终结果由基础变换对、尺度符号和时移相位共同构成。

```python
extracted = _extract_single_function_factor(f, sign)
if extracted is None:
    return None
coeff, core = extracted
params = _linear_center_and_width(core.args[0])
if params is None:
    return None
c, _width, sign_a = params
X = coeff * sign_a * exp(-I*omega*c) * (-2*I) * PV(1/omega)
```

这段代码说明新增函数时的基本方法：先识别函数结构，再提取参数，最后把数学变换对写成工程风格表达式。

## 7.5 例：rect 和 tri

矩形脉冲和三角脉冲都是有限支撑信号。它们的变换结果通常包含 sinc 类结构。系统先识别 `rect(a*t+b)` 或 `tri(a*t+b)` 的内部参数，再应用尺度和时移性质。

这类函数适合说明 helper 的复用。helper 不只服务 rect 和 tri，也可服务 sign、step、三角函数和部分高斯表达式。论文中不应把复用范围写得过窄。

rect 和 tri 的 rule 结构相似。不同之处在于变换对不同：rect 对应 sinc 型结果，tri 对应 sinc 平方型结果。代码 7.3 展示了这种复用方式。

```python
params = _linear_center_and_width(core.args[0])
c, width, _sign_a = params

X_rect = coeff * exp(-I*omega*c) * (2*sin(omega*width/2)/omega)

sinc_part = sin(omega*width/2)/(omega*width/2)
X_tri = coeff * exp(-I*omega*c) * width * sinc_part**2
```

因此，新增同类有限支撑函数时，通常不需要重写参数解析部分。开发者只需要给出新的基础变换对，并在 rule 中复用同一个 helper。

## 7.6 例：高阶有理式

部分高阶有理式可以通过有理式拆分进入已知规则。例如分母阶次高于分子阶次时，可以先做部分分式分解，再把每一项映射到指数衰减、正弦余弦或分布型结果。

这种情况不应描述成“直接积分得到结果”。更准确的说法是：系统先进行结构化拆分，再把拆分后的项交给已有规则处理。这样输出更可控，也更容易避免 `RootSum`。

# 8 数学公式与代码实现映射

## 8.1 Dirac delta

数学公式：

$$
\delta(t-t_0)\longleftrightarrow e^{-j\omega t_0}
$$

代码实现时，rule 识别 `DiracDelta`，再提取冲激位置 `t0`。结果直接构造指数相位项。

```python
if f.has(DiracDelta):
    t0 = _extract_delta_shift(f)
    result = exp(-I * omega * t0)
```

## 8.2 Unit step

数学公式：

$$
u(t)\longleftrightarrow \pi\delta(\omega)+(PV)\frac{1}{j\omega}
$$

阶跃函数需要分布理论。系统在结果中显示 `(PV)`，避免与变量相乘混淆。

```python
if _is_heaviside(f):
    result_latex = r"\pi\delta(\omega)+(PV)\frac{1}{j\omega}"
```

## 8.3 指数衰减乘阶跃

数学公式：

$$
e^{-at}u(t)\longleftrightarrow \frac{1}{a+j\omega},\quad a>0
$$

该规则在工程题中很常见。系统识别指数项和阶跃项的乘积，再提取衰减系数。

```python
if _has_exp_decay_and_step(f):
    a = _extract_decay_rate(f)
    result = 1 / (a + I * omega)
```

## 8.4 调制

数学公式：

$$
e^{j\omega_0t}x(t)\longleftrightarrow X(\omega-\omega_0)
$$

系统通过 matcher 检查乘积中是否含有调制因子。如果存在，则先计算基础信号的变换，再把频率变量替换为 `ω-ω0`。

```python
extracted = _extract_generic_modulation(f)
if extracted is None:
    return None
base, w0, phi = extracted
form, ok, G, _, conditions, error = _derive_with_properties(base)
shifted = G.subs(omega, omega - w0)
X = shifted if simplify(phi) == 0 else exp(I*phi) * shifted
```

这段代码体现了 rule-first 的递归思想。系统先求 `g(t)` 的变换 `G(ω)`，再应用调制性质，而不是重新对整个乘积积分。

## 8.5 时移

数学公式：

$$
x(t-t_0)\longleftrightarrow e^{-j\omega t_0}X(\omega)
$$

时移 helper 的作用是从表达式内部提取 `t-t0` 结构。对于 `u(t-c)`、`rect(t-c)` 和 `sin(t-c)` 等表达式，该 helper 可以减少重复代码。

```python
for c in _shift_candidates(f):
    base = simplify(f.subs(t, t + c))
    form, ok, G, _, conditions, error = _derive_with_properties(base)
    if not ok:
        continue
    X = exp(-I*omega*c) * G
    return form, True, X, steps, conditions, error
```

这里的 `base` 是移位前的基础信号。若基础信号能由已有 rule 处理，系统就把时移相位乘回结果。

## 8.6 尺度变换

数学公式：

$$
x(at)\longleftrightarrow \frac{1}{|a|}X\left(\frac{\omega}{a}\right)
$$

尺度变换对 `rect(a*t)`、`tri(a*t)`、`sign(a*t)` 和高斯函数都很重要。系统通过 helper 提取 `a`，再进行频域替换。

## 8.7 API 返回代码

后端完成推导后，把结果整理成简化 JSON：

```python
return {
    "ok": True,
    "result_latex": result_latex,
    "steps_latex": steps_latex,
    "conditions_latex": conditions_latex,
    "error": None,
}
```

该返回结构让前端无需知道 rule engine 的内部实现。

# 9 支持函数族与规则覆盖范围

## 9.1 分布型函数

系统支持 Dirac delta、Heaviside step、sign、`1/t` 和部分高阶 PV 有理函数。这些函数的共同特点是结果可能包含 `δ(ω)` 或 `(PV)` 项。系统优先使用分布形式，而不是给出普通积分结果。

## 9.2 有限支撑函数

系统支持 rect 和 tri。它们常用于信号处理教学，频域结果通常与 sinc 结构相关。对于带参数的 `rect(a*t+b)` 和 `tri(a*t+b)`，系统通过仿射参数 helper 处理尺度和时移。

## 9.3 振荡函数

系统支持 sin、cos 以及带相位或时移的形式。三角函数的结果通常由频域冲激组成。若表达式含调制项，系统再应用频移性质。

## 9.4 指数与高斯函数

系统支持 `exp(-a*t)u(t)`、`exp(-a*abs(t))`、`exp(-t^2)` 和 `exp(-a*t^2)` 等常见形式。指数衰减乘阶跃常用于一阶系统响应；高斯函数则展示了傅里叶变换中自相似结构。

## 9.5 多项式乘阶跃函数

系统支持 `t^n u(t)` 以及部分平移形式。该类函数的频域结果通常包含高阶 PV 项和 delta 导数项。系统需要把步骤写清楚，避免学生误以为这是普通函数相乘。

## 9.6 高阶有理式

系统支持部分分母阶次高于分子阶次的有理式。处理思路是先进行有理式拆分，再把拆分后的项交给已有变换规则。该方法可以处理一部分复杂输入，同时保持输出形式可读。

# 10 基础 SymPy 兜底能力

## 10.1 fallback 的位置

SymPy fallback 是补充路径，不是主路径。系统先尝试 rule-first 推导，包括已知变换对、时移、调制、尺度变换、卷积和有理式拆分。只有当这些规则无法覆盖，而且 SymPy 输出仍然可控时，系统才进入 fallback。

这种顺序有两个目的。第一，常见工程信号可以直接得到教材风格结果。第二，后端可以避免把复杂 CAS 中间形式暴露给用户。换句话说，fallback 不是系统能力不足的补丁，而是规则引擎外侧的一层受控扩展。

## 10.2 fallback 判断流程

fallback 的判断过程可以概括为三步。系统先解析表达式，再尝试规则匹配。如果规则没有给出有效结果，后端才考虑直接积分或 SymPy 化简。得到结果后，系统还会检查输出形式是否适合展示。

| 阶段 | 判断内容 | 通过条件 | 不通过时的处理 |
|---|---|---|---|
| 规则匹配 | 是否属于已知函数族 | 进入 rule-first 输出 | 尝试性质推导或 fallback |
| 直接积分 | SymPy 是否能得到结果 | 结果可整理成 LaTeX | 返回未覆盖或错误信息 |
| 输出过滤 | 是否包含复杂内部形式 | 不含禁用表达式 | 不直接展示给前端 |

这种流程保证了系统不会因为“能积分”就盲目接受结果。对于教学软件来说，可解释性和稳定格式与计算结果同样重要。

## 10.3 可接受的 fallback 类型

可接受的类型包括有限区间积分、部分指数衰减函数和某些可直接积分的表达式。例如，带有限支撑窗口的函数有时可以直接积分得到结果。只要输出能整理成清楚的 LaTeX，系统可以保留该结果。

比较适合 fallback 的情况包括：

* 有明确有限区间的普通函数；
* 衰减足够快且积分收敛的指数类函数；
* 不引入复杂根结构的简单有理函数；
* 可以经过简化变成已知变换对组合的表达式。

不适合 fallback 的情况包括：高度嵌套的复合函数、条件表达式非常复杂的分段函数，以及积分结果依赖难以解释的分支条件的表达式。这些情况即使 SymPy 能给出形式结果，也不一定适合作为学生看到的最终答案。

## 10.4 不适合直接展示的输出

若 SymPy 输出包含 `Piecewise`、`RootSum`、`arg`、`polar_lift` 或复杂条件表达式，系统不应直接把它们作为最终结果。这样的输出会降低教学可读性，也不符合工程风格。

本项目在测试中持续检查这些禁用形式。其目的不是否定通用 CAS 的能力，而是保证应用输出与工程课程中的习惯一致。例如，阶跃函数和 `1/t` 相关结果应优先写成 `δ` 和 `(PV)`，而不是显示难以解释的条件表达式。

# 11 高并发处理的相关问题

## 11.1 缓存设计

傅里叶变换应用存在重复查询。例如，用户可能反复点击同一个示例，或者在调试输入时多次提交相近表达式。后端缓存可以减少重复计算，尤其是复杂符号表达式的解析和推导。

缓存键可以由规范化后的输入表达式构成。缓存值包括结果 LaTeX、步骤 LaTeX 和附加条件。这样同一输入再次出现时，后端可以直接返回已有结果。

当前后端已经对部分昂贵的 SymPy 操作加入轻量缓存。代码 11.1 展示了 `together` 和 `apart` 的缓存结构。锁用于保护缓存字典，避免多个请求同时读写缓存。

```python
_CACHE_LOCK = RLock()
_CACHE_MAX_SIZE = 512
_TOGETHER_CACHE = {}
_APART_CACHE = {}

def _together_cached(expr):
    key = srepr(expr)
    with _CACHE_LOCK:
        v = _TOGETHER_CACHE.get(key)
        if v is not None:
            return v
    v = together(expr)
    return _cache_store(_TOGETHER_CACHE, key, v)
```

这部分缓存不是完整的工业级分布式缓存。它的目的更具体：减少同一进程内重复调用昂贵符号化简的次数。由于傅里叶变换示例、测试用例和课堂演示中经常重复出现同一表达式，这种缓存能带来实际收益。

## 11.2 锁的设计建议

对于复杂表达式，如果多个请求同时到达并计算同一内容，可能造成重复开销。可以为缓存 miss 的计算过程加入细粒度锁。一个合理设计是按表达式 key 建立锁：同一表达式只允许一个请求计算，其他请求等待结果；不同表达式则仍可并行处理。

该设计适合本项目规模。它不会把系统说成完整工业级高并发平台，但能说明后端已经考虑重复计算和并发保护问题。

后续若要把锁从“缓存字典保护”扩展到“按表达式保护计算过程”，可以采用 per-expression lock。该设计只锁住同一个表达式，不会阻塞不同表达式的请求。

```python
with lock_for(normalized_expression):
    cached = result_cache.get(normalized_expression)
    if cached is not None:
        return cached
    result = derive_fourier(normalized_expression)
    result_cache[normalized_expression] = result
    return result
```

这段设计目前作为扩展方案保留。它说明系统可以从单进程缓存自然扩展到请求级别的并发保护，而不需要改变前端 API。

## 11.3 部署场景下的影响

在 Render 这类轻量部署环境中，后端资源通常有限。符号计算又比普通字符串处理更耗时，因此缓存和锁的价值比普通 CRUD 服务更明显。缓存可以降低重复查询成本，锁可以避免多个相同复杂表达式同时触发重复计算。

本项目目前更接近教学和演示型应用。它没有实现复杂的分布式队列，也没有把高并发作为主要研究目标。因此报告中只把这部分作为后端稳定性和可扩展性的讨论，而不是夸大为完整工业高并发架构。

## 11.4 后续扩展

后续可以加入 TTL cache、最大缓存容量、后台任务队列和请求超时策略。这些设计可以进一步提高服务稳定性。

| 扩展方向 | 作用 | 适用场景 |
|---|---|---|
| TTL cache | 自动清理过期结果 | 长时间运行的后端服务 |
| LRU cache | 保留最近常用表达式 | 示例和课堂演示重复较多 |
| request queue | 控制同时计算数量 | 复杂表达式请求集中出现 |
| timeout policy | 避免单次请求占用过久 | fallback 积分或化简较慢 |

# 12 实际应用效果展示

本章用于展示应用的真实运行效果。截图由用户后续补充，因此本版报告只保留正式图位和说明文字，不使用占位截图。后续插入截图时，应保持图像清晰、大小统一，并在正文中引用图号。

## 12.1 输入界面

本节应插入输入界面截图。截图建议展示表达式输入框、函数按钮、示例导航和公式预览区域。正文可围绕以下内容展开：用户如何通过按钮输入 `sign`、`rect`、`tri`、`exp` 等函数；示例如何帮助学生快速尝试中等复杂度表达式；公式预览如何降低输入错误。

图 12.1 建议标题为“Flutter 应用的傅里叶变换输入界面”。插图后可补一段说明，指出该界面支持直接输入和示例选择两种工作流。

## 12.2 步骤推导界面

本节应插入步骤推导截图。截图需要展示 `steps_latex` 渲染后的教学步骤，包括信号识别、变换对选择、性质应用和最终结果。该图应体现本项目与普通计算器的区别：系统不仅返回答案，还展示推导路径。

图 12.2 建议标题为“傅里叶变换步骤推导显示界面”。插图后可说明步骤文本如何帮助学生理解 rule-first 的思路。

## 12.3 图像展示界面

本节应插入图像展示截图。截图建议展示时域曲线、频域曲线或数值积分曲线。对于包含奇点、冲激或定义上无穷的函数，图像区域应避免把不连续点错误连成普通曲线，并给出必要提示。

图 12.3 建议标题为“时域与频域图像展示界面”。插图后可说明图像在教学中的作用：它不能替代符号推导，但可以帮助学生观察信号变化和频谱趋势。

# 13 测试与评估

## 13.1 测试环境

测试环境分为后端、前端和前后端联合三类。后端测试主要在 Python 虚拟环境中运行，前端测试使用 Flutter test，联合测试通过前端触发请求并检查后端返回结果。由于前端部署在 GitHub Pages，后端部署在 Render，两者的测试需要避免互相覆盖构建流程。

| 测试层级 | 工具 | 主要目标 |
|---|---|---|
| 后端测试 | pytest + FastAPI TestClient | 验证 `/fourier` API 和数学输出 |
| 前端测试 | flutter test | 验证按钮、示例、渲染和局部滚动 |
| 联合测试 | Flutter widget/e2e + backend URL | 验证前端输入、请求发送和后端返回 |
| 文档检查 | LibreOffice/PDF render | 验证报告分页、目录和公式显示 |

## 13.2 后端回归测试

后端测试通过 FastAPI `TestClient` 调用 `/fourier`。测试不只检查 `ok == true`，还检查 `result_latex`、`steps_latex` 和禁止输出项。

典型检查包括：

* 结果包含预期变换形式。
* 步骤不暴露 rule 名称或 debug 信息。
* 输出中不包含 `Piecewise`、`RootSum`、`arg`。
* 常见函数族都有代表性输入。

代码 13.1 展示了实际测试中的核心检查。测试会比较期望 LaTeX，检查步骤是否包含最终结果，并禁止不适合工程展示的符号形式。

```python
assert payload["result_latex"]
assert payload["steps_latex"]

if case.expected_result_latex is not None:
    assert _normalize_latex(result_latex) == \
        _normalize_latex(case.expected_result_latex)

assert "Final Result" in steps_latex
assert r"\operatorname{PV}" not in result_latex
assert r"\mathrm{PV}\frac" not in result_latex
assert "e^{-i" not in steps_latex
```

这些检查说明测试并不是只看接口是否成功返回。它还验证结果表达、教学步骤和工程符号风格是否符合预期。

## 13.3 前端测试

前端测试应覆盖按钮输入、示例导航、公式预览、请求发送和结果渲染。特别需要检查长公式和长注释在小屏幕上是否能通过局部滚动正常显示。

前端测试的重点不是重新证明傅里叶变换数学结果，而是证明用户界面能正确组织输入和显示后端结果。例如，测试可以检查 `sign`、`rect`、`tri` 按钮是否插入正确函数名，指数和幂输入是否能保持光标位置，示例切换后公式预览是否仍能通过 MathFork 渲染。

| 前端行为 | 检查内容 | 风险 |
|---|---|---|
| 函数按钮 | 插入 `sign(`、`rect(`、`tri(` | 输入格式错误 |
| 示例导航 | Previous/Next 切换表达式 | 示例重复或缺失 |
| 公式预览 | MathFork 正常渲染 | Parser error 或 build exception |
| 局部滚动 | 长公式不撑坏页面 | 小屏幕显示异常 |

## 13.4 前后端联合测试

联合测试的目标是验证真实用户流程。测试从前端按钮输入开始，检查请求是否发送到后端，再验证后端返回结果是否被前端正确渲染。该测试可以发现单独后端测试无法覆盖的路径问题，例如 API 地址、JSON 字段名和渲染组件错误。

联合测试尤其适合验证部署边界。前端在 GitHub Actions 中构建，后端在 Render 上运行，两者通过 URL 连接。因此测试需要确认 `API_BASE_URL`、默认 Render 地址和 Android emulator 地址不会互相干扰。

## 13.5 代表性测试输入

代表性输入应覆盖以下类型：

| 类型 | 示例 | 主要验证点 |
|---|---|---|
| 冲激 | `delta(t-2)` | 时移相位 |
| 阶跃 | `u(t-1)` | δ 与 PV 分布 |
| 符号函数 | `sign(2*t-3)` | 仿射参数和 PV |
| 矩形脉冲 | `rect(2*t-1)` | 有限支撑与 sinc |
| 三角脉冲 | `tri(t/2+1)` | sinc 平方结构 |
| 指数衰减 | `exp(-3*t)*u(t)` | 一侧收敛积分 |
| 高斯 | `exp(-2*t^2)` | 高斯变换对 |
| sinc | `sin(3*t)/(pi*t)` | sinc 与 rect 对偶 |
| 多项式阶跃 | `(t-2)^2*u(t-2)` | 平移多项式分布 |
| 高阶有理式 | `frac(t^5+t^4+t^3,(t+1)(t^2+1)(t+6)(t^2+6))` | 有理式拆分 |
| 调制与时移 | `exp(j*3*t)*exp(-(t+1)^2)` | 调制和时移组合 |

## 13.6 测试结果解释方式

测试结果不应只写“全部通过”。更好的写法是说明通过意味着什么。例如，后端测试通过说明 rule-first 引擎在代表性输入上能返回预期 LaTeX；前端测试通过说明用户界面能正确输入和显示；联合测试通过说明前后端字段约定和部署地址设置是可用的。

如果某些输入仍然不能稳定处理，应放在 limitation 中说明，而不是从测试中消失。这样报告会更像真实工程评估，而不是简单功能展示。

# 14 结果分析与讨论

## 14.1 当前能力

系统已经可以覆盖工程课程和常见习题中的多类信号。对于规则覆盖的函数，结果形式较稳定，步骤也更接近教材推导。与直接符号积分相比，rule-first 方法更容易得到 `δ`、`PV` 和 sinc 等工程表达。

当前能力可以按函数族概括如下。

| 函数类型 | 当前支持情况 | 主要方法 | 输出特点 |
|---|---|---|---|
| Dirac delta | 支持平移形式 | 冲激变换对 | 指数相位 |
| Unit step | 支持平移形式 | 分布规则 + 时移 | δ 与 PV |
| sign | 支持仿射参数 | helper + 分布规则 | PV 结果 |
| rect / tri | 支持平移尺度 | helper + 有限支撑规则 | sinc / sinc² |
| sin / cos | 支持频率和相位 | Euler 分解或规则匹配 | 频域冲激 |
| 指数衰减 | 支持部分一侧/双侧形式 | 已知变换对 | 有理函数或指数频谱 |
| 高斯 | 支持参数形式 | 高斯变换对 | 高斯频谱 |
| sinc | 支持常见参数 | 对偶规则 | rect 频域窗口 |
| 多项式阶跃 | 支持若干阶数和平移 | 分布推导 | PV 与 δ 导数 |
| 高阶有理式 | 支持部分 proper rational | 有理式拆分 + 规则 | 组合结果 |

这张能力矩阵也回答了应用对 general user inputs 的范围问题：系统不是任意 CAS，但能覆盖工程课程中常见且可规则化的输入。

## 14.2 输出质量

输出质量主要体现在三方面。第一，结果采用工程符号，例如 `j`、`ω`、`δ(ω)` 和 `(PV)`。第二，步骤按教学顺序组织，而不是暴露 SymPy 内部过程。第三，前端对长公式和长文本使用局部滚动，减少小屏幕显示异常。

这种输出策略让系统更适合教学场景。学生看到的不是单一答案，而是从信号结构到变换性质的推导过程。

## 14.3 局限性

系统仍不是完整 CAS。任意复合函数、任意分段函数和复杂非线性组合不一定能得到结果。对于部分输入，SymPy 可能能积分，但输出不一定适合直接展示。系统需要在“能算”和“适合教学展示”之间做取舍。

当前局限主要包括：

* 对任意分段函数的自动推导仍有限；
* 对高阶复合函数的规则覆盖需要继续扩展；
* 数值图像不能完全表达冲激和 PV 这类分布对象；
* fallback 的结果仍需要过滤，不能盲目展示；
* 移动端小屏幕仍可能需要更细的公式排版优化。

这些限制并不否定系统价值。它们说明项目边界是“工程常见范围内的可解释傅里叶变换”，而不是完整通用符号数学平台。

## 14.4 与通用 CAS 的区别

通用 CAS 更关注数学表达式的广泛处理能力。本系统更关注一组工程常见信号的稳定输出和可解释步骤。它牺牲了一部分泛化范围，换取结果格式和教学表达的可控性。

在工程教学中，这种取舍是合理的。若学生输入 `u(t-1)`，系统应直接说明阶跃变换对和时移性质，而不是返回难以解释的条件表达式。若学生输入 `rect((t-2)/3)`，系统应展示有限支撑信号和 sinc 结果，而不是让学生从积分结果中再猜结构。

# 15 难点分析

## 15.1 分布结果的表达

分布型结果是本项目的主要难点之一。`PV`、`δ(ω)` 和 delta 导数项不容易用普通函数解释。系统必须在结果和步骤中保持一致格式，并避免把这些符号误显示为普通变量。

PV 的显示尤其容易产生误解。若直接写成 `PV 1/ω`，用户可能把 `PV` 看成两个变量 `P` 和 `V` 的乘积。因此前端和后端都倾向于使用 `(PV)` 或明确的 principal value 说明。该选择虽然只是显示层处理，但它直接影响数学结果的可读性。

## 15.2 规则覆盖与泛化之间的平衡

如果规则写得过窄，系统只能处理少量固定表达式。如果规则写得过宽，可能误匹配并返回错误结果。因此系统使用 helper 提取通用结构，同时在 rule 中保持数学条件检查。

例如，`sign(a*t+b)`、`rect(a*t+b)` 和 `tri(a*t+b)` 都可以复用仿射参数 helper。但它们的基础变换对不同，最终结果也不同。因此 helper 只负责提取 `center`、`width` 和 `sign(a)`，具体数学意义仍由 rule 决定。这种分层使系统更容易扩展，也降低了误匹配风险。

## 15.3 公式和图像显示

长公式在小屏幕上容易被裁切。图像中遇到奇点或无穷值时，也可能出现曲线连接错误。前端通过局部滚动和绘图提示缓解这些问题，但未来仍可以进一步改进图像采样和奇点处理。

图像显示还有一个数学层面的限制。冲激和 PV 不是普通函数，不能像普通连续曲线一样直接画出来。因此图像展示只能作为辅助，而不能替代符号结果。报告中需要明确这一点，避免把数值图像误解为完整数学表达。

## 15.4 计算速度与可解释性的矛盾

某些表达式可以通过直接积分得到结果，但速度较慢，而且结果可能是分段形式。系统如果完全追求“能算”，就可能牺牲响应速度和阅读体验。相反，如果完全拒绝 fallback，又会限制用户输入范围。

本项目采用折中方案：优先使用规则，必要时使用受控 fallback。这个设计使常见函数能快速返回，也保留了对部分泛化输入的处理能力。未来若加入更多规则，可以进一步减少 fallback 的使用频率。

## 15.5 文档表达的难点

本项目既包含数学推导，也包含 Flutter、FastAPI、SymPy、测试和部署。报告如果只列功能，会显得像产品介绍；如果粘贴大量代码，又会削弱论文结构。因此文档采用“公式 + 短代码 + 解释”的方式。每个代码片段只展示实现关键，不展示整套源码。

这种写法更符合老师的要求。读者可以看到某个傅里叶变换性质如何落到代码中，也能理解下一次新增函数时应修改 parser、rule、helper、steps 和 tests 哪几个位置。

# 16 结论与未来工作

本项目完成了一套 Flutter 与 FastAPI 结合的傅里叶变换应用。系统支持用户输入、示例选择、公式渲染、步骤推导和图像展示。后端采用 rule-first 方法，优先使用已知变换对和傅里叶变换性质，减少复杂 CAS 中间表达式对最终结果的影响。

测试结果表明，系统可以处理多类常见工程信号，并能输出较清楚的结果和步骤。项目的主要贡献在于把前端交互、符号规则、分布理论和教学步骤结合起来，形成一个适合学习傅里叶变换的应用原型。

未来工作包括扩大函数族覆盖范围、完善卷积和频移规则、增强前后端联合测试、改善图像奇点显示、增加用户自定义规则能力，并进一步优化缓存和并发保护。

# 参考文献

[1] A. V. Oppenheim, A. S. Willsky, and S. H. Nawab, *Signals and Systems*, 2nd ed. Prentice Hall, 1997.  
[2] R. N. Bracewell, *The Fourier Transform and Its Applications*, 3rd ed. McGraw-Hill, 2000.  
[3] E. Kreyszig, *Advanced Engineering Mathematics*, 10th ed. Wiley, 2011.  
[4] L. Schwartz, *Théorie des distributions*. Hermann, 1950.  
[5] SymPy Development Team, *SymPy Documentation*.  
[6] Flutter Team, *Flutter Documentation*.  
[7] FastAPI, *FastAPI Documentation*.

# 附录 A 代表性输入与输出范围

附录应列出所有函数类型的代表性输入。正式 Word 版本中可把本节整理成表格。

## A.1 基础函数

`delta(t)`、`delta(t-2)`、`u(t)`、`u(t-1)`、`sign(t)`、`sign(2*t-3)`

## A.2 有限支撑函数

`rect(t)`、`rect(2*t-1)`、`tri(t)`、`tri(t/2+1)`

## A.3 三角与调制函数

`sin(3*t)`、`cos(2*t+1)`、`exp(j*3*t)*u(t-1)`、`exp(j*3*t)*exp(-(t+1)^2)`

## A.4 指数与高斯函数

`exp(-a*t)*u(t)`、`exp(-a*abs(t))`、`exp(-t^2)`、`exp(-a*t^2)`

## A.5 sinc 与有理式

`sin(t)/(pi*t)`、`sin(a*t)/(pi*t)`、`sin(t)/t`、`1/t`、`1/(t+a)^2`

## A.6 多项式和高阶有理式

`t*u(t)`、`t^2*u(t)`、`(t-2)^2*u(t-2)`、`frac(t^5+t^4+t^3,(t+1)(t^2+1)(t+6)(t^2+6))`

