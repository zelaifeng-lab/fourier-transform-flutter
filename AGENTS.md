# Fourier Transform Backend Agent Guide

## 项目目标

这是一个面向工程风格的符号傅里叶变换系统。

系统目标：

* 输出工程风格频域表达式
* 使用分布理论（Distribution Theory）
* 使用规则优先（Rule-first）而不是纯符号积分
* 输出教材风格推导步骤
* 避免复杂 SymPy 中间表达式

## 核心数学哲学

优先使用：

* 已知傅里叶变换对
* 性质推导
* 分布理论
* 正则化方法
* `\delta` / `\mathrm{PV}` / `\operatorname{sign}` 形式

避免：

* 将 `SymPy.integrate` 作为主路径
* `RootSum`
* `Piecewise`
* `arg(\omega)`
* 复杂条件表达式

示例：

* `1/t -> \mathrm{PV}\frac{1}{j\omega}`
* `sign(t) -> \frac{2}{j\omega}`
* `Heaviside(t) -> \pi\delta(\omega)+\mathrm{PV}\frac{1}{j\omega}`

## 数学约定

默认使用工程傅里叶变换约定：

```latex
X(\omega)=\int_{-\infty}^{\infty}x(t)e^{-j\omega t}\,dt
```

默认假设：

* `t` 为时间变量
* `\omega` 为实角频率
* 频域结果优先整理为分布形式
* `Heaviside`、`DiracDelta`、`PV` 等分布规则优先于积分兜底

## Step 生成规范

`steps_latex` 必须符合教材风格。

必须：

1. 从定义开始：

```latex
X(\omega)=\int x(t)e^{-j\omega t}\,dt
```

2. 展开表达式
3. 使用已知变换对
4. 使用性质（时移、调制、卷积、尺度变换等）
5. 最终整理为工程风格结果

禁止：

* 暴露 matcher 内部逻辑
* 暴露 rule engine 细节
* 输出 SymPy 中间表达式
* 输出调试信息
* 在 `steps_latex` 中出现 `_rule_`、`matcher`、`debug`、`srepr`

## 工程输出规范

输出应符合工程习惯。

优先：

* `e^{-j\omega t_0}`
* `\delta(\omega)`
* `\mathrm{PV}\frac{1}{j\omega}`
* `\operatorname{sign}(\omega)`

避免：

* `Piecewise`
* `RootSum`
* `arg(\omega)`
* `polar_lift`
* `meijerg`
* complex conditionals

`result_latex` 和 `steps_latex` 不得包含：

* `Piecewise`
* `RootSum`
* `arg`
* `polar_lift`
* `meijerg`
* `_rule_`
* `matcher`
* `debug`

## 仓库结构约定

当前后端主要集中在：

* `fourier_backend/backend.py`

当前规则函数通常位于 `backend.py` 中，命名风格包括：

* `_rule_*`
* `_derive_with_properties`
* `_fourier_*_steps`
* `_tb_make_steps`
* `_format_*`

如果未来拆分模块，才优先修改：

* `fourier_backend/matcher/`
* `fourier_backend/steps/`
* `fourier_backend/tests/`

除非明确要求，否则不要修改：

* Flutter frontend：`lib/`
* Android/iOS/macOS/Linux/Windows 平台目录
* Render 部署配置
* CORS 配置
* GitHub Actions

## 搜索规范

搜索代码时优先使用 `rg`。

默认排除：

* `build/`
* `.dart_tool/`
* `fourier_backend/venv/`
* `__pycache__/`

推荐命令：

```powershell
rg -n "pattern" fourier_backend -g "!venv/**" -g "!__pycache__/**"
```

## Matcher 开发规范

新增规则时：

1. 先搜索已有 matcher / `_rule_*`
2. 尽量复用已有规则
3. 优先使用 transform pairs
4. 优先使用 properties
5. 避免直接 integrate
6. 只在规则无法覆盖且输出可控时使用积分兜底

支持的重要规则：

* `DiracDelta`
* `Heaviside`
* `1/t`
* `1/(t+a)`
* `sin(a t+b)`
* `cos(a t+b)`
* `t^n u(t)`
* convolution
* modulation
* scale
* shift

## 测试规范

所有规则修改必须增加 pytest 回归测试。

后端测试优先放在：

* `fourier_backend/tests/`

测试应通过 FastAPI `TestClient` 调用 `/fourier`，并覆盖：

* `ok == true`
* `result_latex` 正确
* `steps_latex` 正确
* 不出现 `Piecewise`
* 不出现 `RootSum`
* 不出现 `arg`
* 不出现内部调试/规则名
* 不超时

推荐测试命令：

```powershell
cd fourier_backend
.\venv\Scripts\python -m pytest tests
```

如果没有现成测试目录，新增最小回归测试目录和测试文件。

## 受保护行为

禁止破坏：

* `DiracDelta` 规则
* `Heaviside` 规则
* `PV` 分布规则
* convolution 规则
* 现有 LaTeX 风格
* 已有 API 响应字段：`ok`、`result_latex`、`steps_latex`、`conditions_latex`、`error`

## Agent Workflow

处理规则新增任务时：

1. 搜索已有 `_rule_*` 和相关 formatter
2. 分析可复用规则
3. 添加 matcher / rule
4. 添加 LaTeX formatter
5. 添加 step generator
6. 添加 regression tests
7. 运行后端 pytest
8. 输出 changelog

最终回复应包含：

* 修改了什么
* 新增或更新了哪些测试
* 实际运行的测试命令
* 是否存在未覆盖风险

## Backend Test Data Reference

后端傅里叶变换规则的测试数据统一维护在仓库根目录的 `test.md`。

使用规则：

* 新增或修改后端规则前，先查看 `test.md` 中是否已有对应用例。
* 已实现规则应优先转为 active regression tests。
* 未实现但常用的变换对保留为 future/pending cases，新增规则后再切换为 active。
* 测试应继续检查输出中不包含 `Piecewise`、`RootSum`、`arg`、`polar_lift`、`meijerg`、`_rule_`、`matcher`、`debug`、`srepr`。
* 自动化测试要求以 `test.md` 的 “Backend Test Requirements” 为准；新增后端规则时，应同步更新或激活对应测试用例。
