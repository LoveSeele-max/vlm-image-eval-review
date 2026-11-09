# VLM 辅助生图评测与人工复核工具

基于历史 FLUX 生成图，按可见主体特征、结构完整性和选定提示词条目进行评测。Python 标准库提供本机 HTTP 服务与接口调用，原生 HTML/CSS/JavaScript 提供页面，SQLite 保存状态与历史，Pillow 处理图片。实际实现没有使用 Streamlit，也不是自主 Agent。

已完成 40 张去重样本、720 项人工初标、开发集 V1、冻结保留集 V1/V2 和 18 项作者分歧复核，并整理 5 个真实案例。各项交付进度见 [任务记录表](./生图评测与人工复核工具_两天任务记录表.md)。

## 结果入口

| 保留集总体，冻结初标 | V1 | V2 |
| --- | --- | --- |
| 工程成功图片 | 29/30 | 27/30 |
| 决策覆盖率 | 453/517 = 87.62% | 407/517 = 78.72% |
| 条件一致率 | 436/453 = 96.25% | 392/407 = 96.31% |
| 共同可判断 388 项的一致率 | 374/388 = 96.39% | 374/388 = 96.39% |

共同条目的两版二元判断完全相同，未见 V2 明确质量提升，V2 覆盖率下降。高条件一致率不能替代覆盖率或错误类别分布。工程成功表示响应通过本地校验，不表示图片质量通过。

- [H01 结果报告](./H01_结果报告.md)：数据、标签分布、全部维度分母、失败及限制。
- [G02 完整主要指标](./G02_保留集主要指标.md)：主表、共同子集、定义和本机 CSV 位置。
- [G04 五个真实案例](./G04_真实复核案例.md)：图片、规则、初标、两版证据和作者结论。
- [G03 复核记录](./G03_保留集三方复核页.md)：18 项冲突/16 张图，7 项维持、11 项修订；其他队列未全量复核。
- [H03 交付与三分钟演示](./H03_交付与三分钟演示.md)：检查清单、可照读脚本及隔离回放命令；双击 [打开项目演示.cmd](./打开项目演示.cmd) 查看本机汇总页。

## 环境与安装

已验证：Windows、Python 3.13.7（本机 `D:/PYTHON313/python.exe`）、Pillow 12.2.0、Tk 8.6、Edge。网络/SQLite/页面服务使用 Python 标准库，配置及密码窗口需要 Tkinter。查看既有结果无需 CUDA、模型权重或 GPU。开发验证使用的 Playwright 不是日常使用依赖。

在项目根目录执行（本机 `E:/cstudy/AIGC/vlmassistance`）：

```powershell
python --version
python -m pip install -r image-eval-review/requirements.txt
python -c "import PIL, tkinter; print(PIL.__version__, tkinter.TkVersion)"
python -m unittest discover -s image-eval-review/tests -v
```

若 `python` 不在 PATH，可替换为 `& 'D:\PYTHON313\python.exe'`。CMD 入口使用 PATH 上的 `python`；本机已核对指向该解释器。2026-11-09 全量 267 项测试通过，测试使用隔离数据/本机模拟，不调用付费接口。

## 数据位置与复现范围

仓库包含代码、固定清单/规则、提示词版本和文字报告。照片、生成图、人工标签、数据库、原始响应和详细导出留在本机。克隆代码不会自动获得这些数据，也没有随仓库提供替代正式素材的演示数据。

```text
项目根目录/
  README.md、H01_结果报告.md、G04_真实复核案例.md
  启动人工标注.cmd、启动开发集分歧查看.cmd、启动保留集复核.cmd
  image-eval-review/
    data/、prompts/、web/、tests/
    requirements.txt、config.example.json
    config.local.json          # 本机地址/模型设置，Git 忽略
    private/                   # 数据库、冻结参考、证据及导出，Git 忽略
```

以下路径相对 `image-eval-review/`：

| 位置 | 用途 |
| --- | --- |
| `data/sample_sources.jsonl`、`data/reference_assets.json` | 原图片路径映射，素材在仓库外 |
| `private/annotations.sqlite3` | 人工当前值、初始快照及历史 |
| `private/human_references/human_reference_v1/` | 冻结初始人工参考，主要指标来源 |
| `private/runs.sqlite3`、`private/runs/inputs/`、`private/runs/evidence/` | 批次、尝试、缓存、计数及请求/响应证据 |
| `private/holdout_g01/` | 两版登记、结果、完成回执与失败 |
| `private/holdout_metrics/g02_20261109T143234_2aeb44d4/` | metrics.json、items.jsonl、五份 CSV 和指标说明 |
| `private/adjudications.sqlite3`、`private/adjudication_exports/` | 独立的作者复核结论及追加历史 |
| `private/g04_cases_20261109/index.html` | 内嵌原图及完整证据的五个案例 |
| `private/h_stage_20261109/` | H 阶段来源、启动和交付检查回执 |

两张参考原图位于 `E:/cstudy/AIGC/flux/flux/refer/`，文件名为 `IMG_20261001_164239.jpg` 和 `IMG_20261001_164251.jpg`。新机器可先运行隔离测试；完整查看旧实验需恢复私有包及原素材。不能直接修改冻结路径/哈希迁移旧实验，迁移应保留旧材料并登记新来源/实验版本。

## 查看既有结果：无需密钥

| 入口 | 地址/文件 | 功能 |
| --- | --- | --- |
| 双击 `启动人工标注.cmd` | http://127.0.0.1:8501 | 匿名初标及初始快照；当前参考已冻结，后续修改留历史 |
| 双击 `启动开发集分歧查看.cmd` | http://127.0.0.1:8502 | 开发初标/V1，只读筛选及图片放大，后端拦截保留集 |
| 双击 `启动保留集复核.cmd` | http://127.0.0.1:8503 | 保留集初标/V1/V2，独立作者确认与修订 |
| 打开 `image-eval-review/private/g04_cases_20261109/index.html` | 本机静态 HTML | 五个真实案例，离线可看 |

启动器复用已运行的本机服务，否则后台启动。查看不调用模型；8502/8503 核验来源可能需 1–3 分钟。8503 有保存功能，查看无需保存；修订不替换冻结参考或 G02。联调及开发结果也保存在本机 `private/reports/`。

需要终端日志或自行停止服务时可前台运行；已有端口占用时用其他空闲端口，或确认并关闭对应旧服务：

```powershell
python image-eval-review/annotation_app.py --port 8501
python image-eval-review/review_app.py --port 8502
python image-eval-review/holdout_review.py --port 8503
```

前台按 Ctrl+C 停止。不要为了释放端口终止其他项目进程。日志为 `private/annotation_server.log`、`private/review_server.log`、`private/holdout_review_server.log`。三个服务均有 `/api/health`；健康检查只证明服务响应，完整内容还需核对页面。

## 接口配置与密钥

查看、隔离测试和核验无需 Key。新增真实评测才需要 Responses 兼容服务：双击 `配置接口.cmd` 填写 Base URL、模型和本次 Key。地址/模型写入被 Git 忽略的 `config.local.json`，Key 仅保留在窗口进程。连接/多图测试按钮会实际发送请求并可能计费。

[config.example.json](./image-eval-review/config.example.json) 是无密钥结构示例，不会自动读取。新安装可按示例创建本机 JSON 或使用窗口；示例域名不能调用。已有冻结实验的地址/模型已有绑定，不要用示例覆盖当前配置。

| 设置 | 读取方式 |
| --- | --- |
| Base URL | `VLM_BASE_URL` 优先，否则本机 JSON 的 `base_url`；程序追加 `/responses` |
| 模型 | `VLM_MODEL` 优先，否则本机 JSON 的 `model` |
| API Key | CLI 读取进程环境变量 `VLM_API_KEY`；密码窗口临时设置，结束清除 |

Key 不写入示例、代码、截图或命令历史。当前请求/返回模型标识为 `gpt-6.1-terra`，中转实际后端身份未独立验证。正式库占用 70/100 请求名额，包含失败、重试和不确定尝试；没有自动金额停机或实时余额查询。人民币账单未核实，不能从 token 或早期约 0.003 元合成测试读数推算正式费用。

## 本地准备、版本核验与批处理

下面命令不调用模型。不要重建运行库重置计数，或重冻旧规则来让哈希检查通过。

```powershell
# 固定版本与初标核验
python image-eval-review/prompt_versions.py --verify
python image-eval-review/freeze_human_reference.py --verify-registered

# 旧 V1 单图只准备请求，不发送；此入口只接受开发样本
python image-eval-review/judge_client.py --sample-id sample_020

# 完整开发 V1 本地批次准备，复用登记，不发送
python image-eval-review/development_v1.py --prepare

# 保留集两版已经运行，先看状态
python image-eval-review/holdout_runs.py status

# 主要指标完整来源复验，不重算或覆盖
python image-eval-review/holdout_metrics.py --verify image-eval-review/private/holdout_metrics/g02_20261109T143234_2aeb44d4
```

G01 批次为 V1 `batch-ec62d5f6d98d464db9f29164ed58c59c`、V2 `batch-cd00599fab3c4e128c9a86855d1c7e6e`。开发 V1 批次为 `batch-2a50643355544ef49d9c8d5831d8ef61`。旧 `run_tasks.py` 只适用开发 V1；保留集两版使用独立 `holdout_runs.py`，不能把 V2 请求塞入旧执行器。

真实发送入口：开发 V1 使用 `启动开发集V1.cmd`；保留集两版为 `python image-eval-review/holdout_runs.py run`。这些入口可能计费，会核对冻结来源、确认快照、缓存及请求上限。既有最终 524 保留，G01 不自动重开失败。仅查看结果无需再次运行。

旧开发调度命令见 [D03](./D03_运行记录与恢复.md)：`create` 只准备；`run`/`resume` 可能发送，`--retry-failed` 显式重开未达上限的失败；`recover` 本地整理中断。`status` 若库含失败可返回退出码 1，同时输出完整状态，不等于查询失败。鉴权/配额等错误停止批次，有限重试不等于所有 HTTP 错误都会重试。

缓存只有在键与原证据核验通过时才复用；不能确认是否发送的中断仍占名额。恢复、缓存、非法结果重试用真实响应的隔离副本及本机回放验证，没有向中转故意断网做故障注入。

## 常见问题

| 现象 | 下一步 |
| --- | --- |
| 图片不存在或来源哈希不同 | 恢复清单指向的原素材/私有包，不修改哈希凑通过 |
| 8503 尚未就绪 | 查看日志并等待核验，不连续启动多个进程 |
| Pillow/Tk 缺失 | 核对当前 Python，安装 requirements 并使用带 Tk 的解释器 |
| HTTP 524 | 定位原尝试证据，保留工程缺失，不当作视觉不符 |
| 保存版本冲突 | 刷新当前结论后核对，不覆盖另一页面先保存的历史 |
| 克隆仓库后缺正式数据 | 先运行隔离测试；正式页面依赖未公开材料，缺失时不伪造 |

## 实验边界与详细记录

单一主体、4 个 seed 家族，同图条目相关，两侧共享提示词模板。初标含作者批量判断和助手辅助文字，作者复核已看过 AI，不是独立盲评。主体/结构/选定提示词条目分别判断，木桌不评。提示词缺人工不符例，结构仅 1 个不符例，零分母不可计算。

未做多人独立盲评、审核提效对照、多主体泛化、第二模型或同图重复性实验；不能证明 LoRA 训练收益、生产自动审核或后端身份/计费。两天是原排期目标，实际开发跨 2026-11-06 至 11-09，人工投入未记录。

- [实施方案 v2](./生图评测与人工复核工具_两天实施方案_v2.md)和 [任务记录表](./生图评测与人工复核工具_两天任务记录表.md)：范围与任务。
- [E04 人工冻结](./E04_人工参考冻结记录.md)、[F04 V2 冻结](./F04_V2提示词冻结与版本兼容.md)、[G01 两版运行](./G01_保留集两版运行记录.md)：来源与门禁。
- [D03 运行与恢复](./D03_运行记录与恢复.md)、[D04 联调](./D04_三图联调记录.md)：工程验证。
- [Day2 历史交接](./DAY2_新对话交接.md)：保留旧时点，最新状态以任务表为准。

H02 实际启动检查回执在本机 `image-eval-review/private/h_stage_20261109/`。逐任务提交约定见 [AGENTS.md](./AGENTS.md)。
