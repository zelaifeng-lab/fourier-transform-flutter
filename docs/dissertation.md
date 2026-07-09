# Dissertation Document Agent Guide

本文件用于约束 `docs/` 目录下 dissertation / project report 的后续修改。目标是让论文修改保持一致，不再把格式、分页、目录、代码展示和章节逻辑反复改乱。

## 1. 总体目标

论文应从普通工程报告改为 MSc dissertation 风格。写作重点不是只展示功能，而是说明功能如何被设计、实现、测试和验证。

报告需要证明：

* 项目实现了 Flutter 前端与 FastAPI / Python 后端结合的傅里叶变换应用。
* 系统采用 rule-first symbolic backend，而不是完全依赖 SymPy 积分。
* 前端支持公式输入、示例选择、MathFork / LaTeX 渲染、步骤展示和图像展示。
* 后端能处理常见傅里叶变换对、分布型结果、时移、调制、尺度、有理式拆分、多项式与部分 SymPy fallback 情况。
* 测试覆盖结果正确性、步骤可读性、格式输出和前后端联合流程。

## 2. 参考格式

格式应优先参考：

* `docs/FYP_AB_LYZ.pdf`

基本要求：

* A4 页面。
* 正文接近 Times New Roman 12 pt；中文可使用宋体或等价字体。
* 1.5 倍行距。
* 页边距接近 3 cm。
* 正文段落不要过密。
* 一级标题、二级标题应清楚分层。
* 图、表、代码片段必须编号，并在正文中引用。

修改 Word 后必须渲染为 PDF 或 PNG 检查格式，不应只依赖脚本生成成功。

## 3. 前置页分页规则

前几页必须参考 `FYP_AB_LYZ.pdf` 的正式论文结构。不要把多个前置部分压在同一页。

推荐分页：

1. Cover Page
2. Formal Title Page
3. Statement of Originality
4. Supervisor Declaration Statement
5. Authorship Attribution Statement
6. Table of Contents
7. Abstract
8. Acknowledgement
9. List of Figures
10. List of Tables
11. List of Symbols and Abbreviations
12. Main Text

正文的 `1 Introduction` / `1 引言` 必须从新页开始。

## 4. 目录规则

目录必须使用 Word 自带目录字段，而不是手写目录文本。

要求：

* 目录项应能点击跳转到对应章节。
* 目录应基于 Word heading styles 自动生成。
* 目录层级建议覆盖 Heading 1 到 Heading 3。
* 修改标题后，应重新更新目录页码。

可使用的 TOC 字段形式：

```text
TOC \o "1-3" \h \z \u
```

## 5. 公式规则

正文中的展示公式必须优先使用 Word 自带“插入公式”的显示方式，即 Word equation / OMML 公式对象。不要把正文展示公式长期保留为普通文本、截图或原始 LaTeX 字符串。

要求：

* 展示公式单独一行。
* 展示公式居中。
* 公式编号放在同一行右侧或公式后方，保持学术论文习惯。
* 正文中的 omega 应尽量显示为 `ω`，代码片段中的变量名不强制替换。
* 展示公式应转换为 Word equation / OMML；只有在转换工具暂时不可用时，才允许临时使用居中的普通数学文本，并必须在交付说明中标记为待升级。
* 公式不要整体斜体化成普通文本。
* 不要让长公式被裁切或挤出页面。
* 代码块和表格中的公式不强制转换为 Word equation，以免破坏代码可读性。

## 6. 写作风格

英文报告：

* 使用 third person，少用 `I implemented`。
* 使用清楚、直接、学术但不过度复杂的句子。
* 不要写成 sales pitch。
* 不要堆砌功能列表，应解释设计原因和实现方法。

中文报告：

* 保持学术表达，但句子不宜过长。
* 每句话尽量有明确主语。
* 避免明显 AI 模板句，例如过多的“首先、其次、此外、综上所述”。
* 保持自然的人类写作节奏。

中英文内容应保持一致，但不需要逐字直译。

## 7. 推荐章节结构

推荐中文结构：

1. 摘要
2. 引言
3. 傅里叶变换理论基础
4. 相关技术与设计依据
5. 系统总体设计
6. Flutter 前端设计与实现
7. FastAPI 后端接口设计
8. Rule-first 符号傅里叶变换引擎
9. 数学公式与代码实现映射
10. 支持函数族与规则覆盖范围
11. 基础 SymPy 兜底能力
12. 高并发处理的相关问题
13. 实际应用效果展示
14. 测试与评估
15. 结果分析与讨论
16. 难点分析
17. 结论与未来工作
18. 参考文献
19. Appendix / 附录

其中“实际应用效果展示”应包含三类截图：

* 输入界面
* 步骤推导界面
* 图像展示界面

截图排版应保持美观，不要过大或过小。每张图需要图题，并在正文中引用。

## 8. 代码展示原则

老师明确要求：不要粘贴整套代码，而要给出与每个功能直接相关的短代码片段。

代码展示应说明：

* 使用了哪条数学公式或变换性质。
* 代码中哪几行用于识别该函数。
* 哪几行用于构造结果。
* 哪几行用于生成步骤或 API 返回。
* 如果下次添加类似函数，应如何复用这个模式。

代码片段应短小，通常 5 到 15 行为宜。每段代码需要配简短注释或正文解释。

重点代码类型：

* parser 如何规范输入。
* rule 如何识别函数类型。
* helper 如何提取仿射参数，例如 `a*t+b`。
* rule 如何引用 helper。
* result_latex 如何生成。
* steps_latex 如何生成。
* API 如何返回简化结构。
* Flutter 如何发送请求。
* Flutter 如何用 MathFork / Math.tex 渲染公式。
* Flutter 如何处理 example buttons 和页面布局。

## 9. Rule-first 实现说明要求

论文中解释新增函数时，建议使用以下逻辑：

1. 确定数学信号和傅里叶变换对。
2. 判断结果是普通函数还是分布形式。
3. 在 parser 层把输入规范成后端可识别的 SymPy 表达式。
4. 在 rule 层识别函数结构。
5. 如有时移、调制、尺度变换，调用通用 helper。
6. 生成工程风格 `result_latex`。
7. 生成教学风格 `steps_latex`。
8. 通过 FastAPI 返回前端。
9. 在 Flutter 中渲染公式、步骤和图像。
10. 添加回归测试。

不要把 matcher / helper 的概念写混。通常是 rule 调用 helper，而不是 helper 调用 rule。

## 10. 必须覆盖的后端能力

报告中应概括说明下列能力：

* Dirac delta
* Heaviside step
* sign
* rect
* tri
* sin / cos
* exponential with step
* Gaussian
* sinc-type functions
* polynomial times step
* high-order rational functions where denominator degree is higher than numerator degree
* distributional rational forms such as PV terms
* time shift
* modulation
* scale transform
* convolution-related rules where implemented
* controlled SymPy fallback

高阶多项式除式需要说明：它通过有理式拆分、部分分式或结构化规则匹配进入已知变换规则，不应描述成纯粹盲积分。

## 11. SymPy fallback 说明

报告需要说明：系统不是完全拒绝 SymPy，而是把 SymPy 放在受控 fallback 位置。

可以说明的 fallback 类型：

* 有限区间积分可控的函数。
* 一些指数衰减函数。
* 一些可直接积分且输出不含复杂中间结构的表达式。

需要避免把以下输出作为最终工程结果：

* `Piecewise`
* `RootSum`
* `arg`
* `polar_lift`
* `meijerg`
* 复杂条件表达式

## 12. 高并发处理章节

后端虽然不是大型高并发系统，但报告中可以合理说明已做和建议做的处理。

可以写入：

* 结果缓存用于减少重复计算。
* 缓存适合傅里叶变换这种重复查询较多的应用。
* 锁可以避免多个请求同时计算同一复杂表达式。
* API 返回结构保持简洁，降低前后端通信复杂度。
* 后续可以扩展为 TTL cache、request queue 或 background task。

不要夸大成已经完成完整工业级高并发架构。

## 13. 测试章节要求

测试与评估需要覆盖：

* 后端 regression tests。
* FastAPI `/fourier` 测试。
* Flutter widget 或 smoke tests。
* 前后端联合测试。
* 代表性数学结果校验。
* 输出格式检查。
* 步骤输出检查。
* 图像显示异常处理。

测试应说明不仅检查 `ok == true`，还要检查：

* `result_latex` 是否数学等价或符合预期形式。
* `steps_latex` 是否教学上可读。
* 输出中不含 `Piecewise`、`RootSum`、`arg` 等不希望暴露的形式。

## 14. 附录规则

目前不保留单独的 Appendix A: Question & Answer Summary。

附录应主要保留：

* 所有函数类型的代表性输入。
* 部署或运行命令。
* 必要的补充代码片段。

附录不要成为整套源码粘贴区。

## 15. 文件保留规则

`docs/` 目录下不要长期保留大量临时 Word / PDF 文件。

建议保留：

* 中文旧版最多 2 个，作为历史参考。
* 英文旧版最多 2 个，作为历史参考。
* 中文新版最多 1 个。
* 英文新版最多 1 个。
* 必要的最终 PDF。
* `FYP_AB_LYZ.pdf` 作为格式参考。

多余的旧版 Word / PDF 文件应在新版确认后删除，避免 `docs/` 目录堆积过多相似文件。临时渲染目录、LibreOffice profile、截图检查目录应在确认后清理。

## 16. 修改工作流

每次正式修改 dissertation 文件时：

1. 先确认使用哪个 Word 文件作为源文件。
2. 不要在没有检查的情况下覆盖唯一版本。
3. 修改内容和结构。
4. 渲染为 PDF。
5. 检查前置页分页。
6. 检查目录是否为可跳转目录。
7. 检查公式是否居中、单独成行且不被裁切。
8. 检查是否存在空白页或异常分页。
9. 检查页数是否合理。
10. 给用户说明修改文件、验证结果和剩余风险。

如果工具无法进行可视化检查，应明确告诉用户，不要假装已经目视确认。



