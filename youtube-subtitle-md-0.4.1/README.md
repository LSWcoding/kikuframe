# SubMD：YouTube 烧录字幕提取工具

## 为什么要做这个程序

YouTube 提供的人工字幕和自动字幕经常与视频中的实际说话内容不完全一致，尤其容易出现
漏字、错字、断句错误和时间轴偏移。

很多视频已经把更准确的字幕直接烧录在画面里，但这类字幕属于视频图像的一部分，无法
像普通 YouTube 字幕一样复制或下载。SubMD 的目标就是识别视频画面中的烧录字幕，并将
它们保存成可以编辑、搜索和继续处理的 Markdown 文档。

## 有什么效果

输入一个公开的 YouTube 视频 URL 后，SubMD 会：

1. 使用 `yt-dlp` 获取视频信息并下载适合识别的视频流。
2. 使用 FFmpeg 从画面底部的默认字幕区域抽取视频帧。
3. 筛选字幕发生变化的关键帧，减少重复的云端请求。
4. 调用用户提供 API Key 的 OpenAI-compatible 视觉模型识别烧录字幕。
5. 对相邻帧文字进行去重、纠错投票和时间段合并。
6. 使用文本模型根据语义重新断句。
7. 输出两个以视频标题命名的 Markdown 文件：

```text
output/
├── 视频标题.md             # 原始版：包含 metadata、来源和时间戳
└── 视频标题（整理版）.md   # 整理版：没有 metadata 和时间戳，只保留一句话一行的字幕
```

如果没有填写文本模型，程序会默认复用视觉模型。假如视觉模型无法处理纯文本，原始版
仍然会保留并出现在前端；补填文本模型后再次提交同一视频，程序会直接复用原始版，不
重新下载视频、抽帧或执行视觉 OCR。同一视频通过 YouTube 视频 ID 判断，因此 URL 中
不同的 `?si=` 参数不会影响复用。

处理失败时，前端会停止转动动画、显示 `❌`，恢复提取按钮并说明失败原因。任务历史、
OCR 检查点和语义断句检查点都会保存在本机，意外中断后可以继续。

## 如何安装

### macOS

1. 在 GitHub 项目页面点击 **Code → Download ZIP**，下载后解压。
2. 安装 Homebrew，并安装运行环境：

```bash
brew install python@3.12 ffmpeg deno
```

3. 项目不需要安装 PaddleOCR。第一次启动时，启动脚本会在项目内部创建 `.venv`
   虚拟环境并安装所需 Python 依赖。

如果 macOS 没有保留脚本的执行权限，在项目目录运行一次：

```bash
chmod +x start-submd.command uninstall-submd.command
```

### Windows

1. 下载并解压 GitHub ZIP。
2. 安装 Python 3.12、FFmpeg 和 Deno，并确保三者已经加入 `PATH`。
3. 第一次启动时，`start-submd.bat` 会在项目内部创建虚拟环境并安装 Python 依赖。

项目附带 `.env.example`，但不需要提前手动创建 `.env`；第一次在前端保存配置时会
自动生成。`.env` 已被 Git 忽略，不会随正常提交上传到 GitHub。

## 如何启动

### macOS

双击：

```text
start-submd.command
```

如果 macOS 首次阻止打开，可以右键该文件并选择“打开”。启动成功后会用 Chrome 打开：

```text
http://127.0.0.1:8765
```

字幕处理期间需要保持启动脚本的终端窗口开启。

### Windows

双击：

```text
start-submd.bat
```

也可以在已经创建虚拟环境的项目目录中手动启动：

```bash
.venv/bin/submd ui
```

## 如何使用

在前端填写：

- **YouTube 视频 URL**：单个公开 YouTube 视频地址。
- **OCR API Base URL**：云厂商提供的 OpenAI-compatible API 地址，例如
  `https://厂商地址/v1`。
- **视觉模型名称**：能够接收图片的模型名称。
- **API Key**：云厂商提供的密钥，只保存在本机 `.env`。
- **浏览器 Cookie**：YouTube 要求登录验证时填写 `chrome`；请先在 Chrome 中登录
  YouTube。
- **文本模型地址和名称（可选）**：用于语义断句；留空时复用 OCR 地址和视觉模型。

点击“提取字幕”后，页面会显示当前阶段。成功时会出现原始版和整理版两个下载链接，
并写入提取历史。

如果只生成了原始版，说明语义断句模型调用失败。此时展开“整理版字幕模型”，填入可
处理纯文本的模型，再用相同视频 URL 提交。程序会跳过视觉 OCR，直接从已有原始
Markdown 生成整理版。

视频帧会发送到用户配置的云厂商进行 OCR；API Key、浏览器 Cookie 和 `.env` 不会
写入 Markdown 或 JSON 中间结果。默认字幕区域是画面底部 35%，需要其他区域时可以
通过 CLI 的 `--roi "x,y,width,height"` 参数调整。

## 如何删除

### macOS 一键删除

1. 关闭 SubMD 页面和运行它的终端窗口。
2. 双击：

```text
uninstall-submd.command
```

3. 在确认窗口点击“移到废纸篓”。

脚本会进行项目标记检查，停止由当前项目启动的 8765 本地服务，然后将整个项目文件夹
移入废纸篓。项目代码、`.env`、API Key、字幕结果、缓存、检查点和 `.venv` 会一起
移除；系统中的 Python、FFmpeg 和 Deno 不会被删除。清空废纸篓后才是永久删除。

### Windows

关闭 SubMD 窗口和终端后，删除整个项目文件夹即可。虚拟环境、配置、缓存和输出都在
项目文件夹内，没有注册系统服务或浏览器扩展。

## 项目结构、架构和各单元功能

```text
YouTube URL
    │
    ▼
网页 UI ──► 本地任务服务
              │
              ├──► YouTube 获取层：元数据、Cookie、视频下载
              ├──► 媒体层：ffprobe、FFmpeg 抽帧、ROI 裁切、关键帧筛选
              ├──► 云 OCR 层：图片编码、API 请求、重试、响应校验
              ├──► 字幕 Pipeline：检查点、相邻帧去重、时间段合并
              ├──► Markdown 导出：原始字幕文件
              └──► 语义整理器：纯文本断句、字符守恒校验、整理版文件
```

```text
youtube-subtitle-md/
├── start-submd.command       # macOS 一键启动
├── start-submd.bat           # Windows 一键启动
├── uninstall-submd.command   # macOS 一键移入废纸篓
├── .env.example              # 外部变量示例，不包含真实密钥
├── pyproject.toml            # Python 版本、依赖、CLI 和打包配置
├── src/submd/
│   ├── cli.py                # `submd extract`、`submd organize`、`submd ui` 命令入口
│   ├── web.py                # 本地 HTTP 服务、后台任务、历史和结果下载
│   ├── ui/
│   │   ├── index.html        # 前端页面结构
│   │   ├── styles.css        # 页面、进度、成功和失败状态样式
│   │   └── app.js            # 配置提交、轮询、弹窗和双文件显示
│   ├── youtube.py            # yt-dlp 元数据读取、Cookie 配置和视频下载
│   ├── media.py              # ffprobe、FFmpeg 抽帧、ROI 和图像变化筛选
│   ├── ocr/
│   │   ├── base.py           # OCR 引擎统一接口
│   │   └── openai_compatible.py
│   │                         # 视觉模型请求、图片压缩、限流重试和 JSON 校验
│   ├── pipeline.py           # 下载到导出的主流程和断点恢复
│   ├── segments.py           # 相邻帧文本相似度、去重和字幕时间段合并
│   ├── organize.py           # 从原始 Markdown 提取纯字幕并用模型语义断句
│   ├── exporters.py          # 安全文件名和原始 Markdown 导出
│   ├── models.py             # Pydantic 配置与领域数据模型
│   ├── json_io.py            # JSON 检查点的安全写入
│   ├── text.py               # OCR 文本规范化
│   └── errors.py             # 下载、媒体、OCR、整理等错误类型
├── tests/                    # 下载器、OCR、Pipeline、整理器和网页 UI 自动化测试
├── workspace/                # 运行时生成：视频级 JSON、预览、缓存和检查点
└── output/                   # 运行时生成：原始版与整理版 Markdown
```

`workspace/` 中的 `segments.json` 是合并后的事实数据源；Markdown 是可重新生成的
导出结果。`workspace/`、`output/`、`.env` 和 `.venv` 均不会进入正常 Git 提交。
