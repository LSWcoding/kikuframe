# KikuFrame：AI 日语视频学习工具

> 把日语视频转换成可点击、可循环、可纠错、可分析的逐句学习材料。

## 产品功能

KikuFrame 是一款在本机运行的日语视频学习工具。输入单个公开的 YouTube 视频 URL 后，
程序会：

1. 使用 `yt-dlp` 获取视频信息，并下载适合识别的视频流和音频。
2. 使用 FFmpeg 从画面底部的默认字幕区域抽取关键帧。
3. 调用用户配置的 OpenAI-compatible 视觉模型识别画面中的烧录字幕。
4. 下载同语言的 YouTube 人工字幕或自动字幕，作为说话内容和读音参考。
5. 让文本模型结合 OCR 结果、前后文和读音参考纠正错字；安全检查会拦截大幅增删和跨段搬运文字。
6. 根据语义和字幕时间轴整理断句，并把每句话与对应音频时间对齐。
7. 生成原始字幕、综合校正版、逐句整理版和视频音频。
8. 在本地播放器中逐句显示字幕，支持点击跳转、上一句、下一句、当前句循环和自动高亮。
9. 支持手动修改前后句边界；修改后会重新计算句子对应的音频时间。
10. 支持单句翻译、词汇、搭配、平假名读音和文法分析；分析结果保存在本机并可重新生成。
11. 保存提取历史、OCR 检查点和整理检查点；同一视频可以复用已有结果，降低中断后的重复处理和 API 消耗。

每个视频通常产生以下文件：

```text
output/
├── 视频标题.md                         # 原始 OCR，包含 metadata 和时间戳
├── 视频标题（综合校正版）.md           # OCR + YouTube 读音参考校正结果
├── 视频标题（综合校正版）（整理版）.md # 只保留一句一行的字幕
└── 视频标题.m4a                        # 字幕播放器使用的音频
```

视频帧会发送给用户配置的云端视觉模型。API Key、浏览器 Cookie、`.env`、下载媒体和学习
记录只保存在本机，不会写入导出的 Markdown，也已被 Git 排除。

## 下载方法

项目地址：[LSWcoding/kikuframe](https://github.com/LSWcoding/kikuframe)

### macOS

1. 在项目页面点击 **Code → Download ZIP**，下载后解压。
2. 安装 [Homebrew](https://brew.sh/)，然后打开“终端”安装运行环境：

```bash
brew install python@3.12 ffmpeg deno
```

3. 打开解压后的 `kikuframe` 文件夹，双击 `start-submd.command`。
4. 第一次启动会在项目文件夹中创建 `.venv` 并安装 Python 依赖，完成后自动打开本地网页。

如果 macOS 阻止启动，请右键 `start-submd.command` 并选择“打开”。如果文件没有执行权限，
在项目目录运行：

```bash
chmod +x start-submd.command uninstall-submd.command
```

### Windows

1. 在项目页面点击 **Code → Download ZIP**，下载后解压。
2. 安装 Python 3.12、FFmpeg 和 Deno，并确保三者已加入系统 `PATH`。
3. 双击 `start-submd.bat`。第一次启动会自动创建 `.venv` 并安装 Python 依赖。

项目不需要安装 PaddleOCR，但必须准备一个支持图片输入、采用 OpenAI-compatible API 的云端
视觉模型，以及对应的 Base URL、模型名称和 API Key。

## 使用方法

### 启动

- macOS：双击 `start-submd.command`。
- Windows：双击 `start-submd.bat`。
- 命令行：在已经安装依赖的项目目录运行 `.venv/bin/submd ui`。

默认界面地址为 `http://127.0.0.1:8765`。程序运行期间请保持启动脚本的终端窗口开启。

### 配置和提取

在前端填写：

- **YouTube 视频 URL**：单个公开 YouTube 视频地址。
- **OCR API Base URL**：云厂商的 OpenAI-compatible API 地址，例如 `https://厂商地址/v1`。
- **视觉模型名称**：能够接收图片的模型名称。
- **API Key**：云厂商提供的密钥，只保存在项目根目录的 `.env`。
- **浏览器 Cookie**：YouTube 要求登录验证时填写 `chrome`，并先在 Chrome 中登录 YouTube。
- **文本模型（可选）**：用于字幕综合纠错和语义断句；留空时复用视觉模型。
- **语言学习模型（可选）**：用于单句翻译、词汇和文法分析；留空时复用文本模型或视觉模型。

点击“提取字幕”后，页面会显示处理阶段。成功后可以下载结果文件，也会自动进入字幕音频
播放器。处理失败时，动画会停止并显示 `❌` 和失败原因。

如果同一视频已经完成视觉 OCR，再次提交相同视频 URL 时会优先复用已有字幕、音频、
YouTube 字幕和检查点。URL 中不同的 `?si=` 参数不会影响视频识别。

### 字幕播放器

- 点击任意句子：音频跳到该句开始时间并播放。
- “上一句”或“下一句”：切换当前句子。
- “单句循环”：在当前句的开始和结束时间之间循环播放。
- 提取历史：点击以前处理过的视频，重新打开其字幕、音频和学习记录。

这里使用烧录字幕出现时间进行句子级对齐，不进行逐单词 ASR 对齐。

### 修改断句

长按一句字幕进入断句纠错，再点击另一句即可选择连续范围；也可以先选中当前句，再点击
“断句纠错”。在弹窗中增删或移动句号“。”，预览确认后保存。程序会保留原文字并重新计算
每个新句子的音频开始和结束时间。手动修改记录保存在 `workspace/manual-edits/`。

### 单句分析

点击句子右下角的“分析”，模型会按照以下结构生成学习内容：

```text
整句翻译
词汇以及搭配（包含日文汉字的平假名读音）
文法
```

分析结果保存在 `workspace/learning/`。再次点击同一句会读取上一次结果；点击“重新分析”
才会重新调用模型并在成功后覆盖旧结果。如果重新分析失败，旧结果仍然保留。

## 删除方法

### macOS

1. 关闭 KikuFrame 页面和运行它的终端窗口。
2. 双击 `uninstall-submd.command`。
3. 在确认窗口点击“移到废纸篓”。

脚本会确认目标是完整的 KikuFrame 项目，停止由该项目启动的本地服务，然后把整个项目
文件夹移入废纸篓。程序、`.env`、API Key、字幕、音频、历史、缓存、检查点和 `.venv`
会一起移除；系统中的 Python、FFmpeg、Deno、Chrome 和 Homebrew 不会被删除。清空废纸篓
后才会永久删除这些项目数据。

也可以先关闭程序，再把整个 `kikuframe` 文件夹手动移到废纸篓。

### Windows

关闭网页和启动脚本的终端窗口，然后删除整个 `kikuframe` 文件夹。项目没有注册系统服务
或浏览器扩展；Python、FFmpeg 和 Deno 需要自行决定是否从系统卸载。

## 项目架构解释

```text
YouTube URL
    │
    ▼
网页 UI ──► 本地任务服务
              │
              ├──► YouTube 获取层：元数据、Cookie、视频、音频和参考字幕
              ├──► 媒体层：ffprobe、FFmpeg 抽帧、ROI 裁切和关键帧筛选
              ├──► 云 OCR 层：图片编码、视觉模型请求、重试和响应校验
              ├──► 字幕 Pipeline：检查点、相邻帧去重和时间段合并
              ├──► 综合纠错器：OCR + 读音参考、文本模型和安全改写检查
              ├──► 语义整理器：字符级断句、参考句界、字符守恒和时间映射
              ├──► 手动纠错器：连续选句、句号重排、音频时间重算和编辑历史
              ├──► 学习分析器：翻译、词汇读音、搭配、文法和本地缓存
              ├──► Markdown 导出器：原始版、综合校正版和逐句整理版
              └──► 学习播放器：音频传输、点击跳转、循环播放和当前句高亮
```

```text
kikuframe/
├── start-submd.command       # macOS 一键安装和启动
├── start-submd.bat           # Windows 一键安装和启动
├── uninstall-submd.command   # macOS 一键移入废纸篓
├── .env.example              # 外部配置示例，不包含真实密钥
├── pyproject.toml            # Python 版本、依赖、CLI 和打包配置
├── src/submd/
│   ├── cli.py                # `submd extract`、`submd organize`、`submd ui` 入口
│   ├── web.py                # 本地 HTTP 服务、任务、历史、播放器 API 和音频传输
│   ├── ui/                   # 网页结构、样式和浏览器交互
│   ├── youtube.py            # yt-dlp 元数据、下载、Cookie 和字幕解析
│   ├── media.py              # ffprobe、FFmpeg、ROI、关键帧和音频导出
│   ├── ocr/                  # 云端视觉模型 OCR 接口和实现
│   ├── pipeline.py           # 从下载到导出的主流程和断点恢复
│   ├── segments.py           # 相邻帧去重和字幕时间段合并
│   ├── fusion.py             # OCR 与 YouTube 读音参考的综合纠错
│   ├── organize.py           # 字符级语义断句和句子时间映射
│   ├── editing.py            # 手动断句和音频重新对齐
│   ├── learning.py           # 单句语言分析和缓存
│   ├── exporters.py          # 文件名处理和 Markdown 导出
│   ├── models.py             # Pydantic 配置与领域模型
│   ├── json_io.py            # JSON 检查点安全写入
│   ├── text.py               # OCR 文本规范化
│   └── errors.py             # 下载、媒体、OCR 和整理错误类型
├── tests/                    # 自动化测试
├── workspace/                # 运行时数据、检查点、字幕轨、历史和学习缓存
└── output/                   # Markdown 和 M4A 输出文件
```

`workspace/segments.json` 保存原始 OCR 合并结果，`youtube_captions.json` 保存独立的读音
参考，`corrected_segments.json` 保存综合校正结果，`organized_segments.json` 保存逐句文字和
音频时间。各层数据相互独立，不会静默覆盖，便于检查、恢复和重新整理。
