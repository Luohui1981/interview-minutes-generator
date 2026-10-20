# 本地音频转写流程

本文件说明如何使用豆包录音文件极速版，把本地访谈音频转换为可审计的《转写原稿》。

## 1. 数据边界

- 音频会发送到豆包语音识别服务。
- 原始音频永不删除、移动或覆盖。
- 转码文件、切片和片段检查点位于本次访谈输出目录内的隐藏工作目录。
- 成功后默认删除临时文件；失败时保留以便恢复。
- 访谈提纲和背景资料只用于提取 ASR 提示词，不是访谈内容来源。

## 2. 依赖与凭据

需要：

- Python 3
- `ffmpeg` 与 `ffprobe`
- 豆包录音文件极速版权限

凭据只从环境变量读取：

- 新控制台：`DOUBAO_STT_API_KEY` 或 `VOLC_ASR_API_KEY`
- 旧控制台：`DOUBAO_STT_APP_ID` 加 `DOUBAO_STT_ACCESS_KEY`
- 极速版资源：`DOUBAO_STT_FLASH_RESOURCE_ID` 或 `VOLC_ASR_FLASH_RESOURCE_ID`
- 资源默认值：`volc.bigasr.auc_turbo`

不要打印、写入或回显密钥。

## 3. 输入 JSON

### 访谈元信息

```json
{
  "title": "存储芯片产品与产能访谈",
  "date": "2026-08-04",
  "industry": "半导体",
  "interviewers": [
    {"name": "访谈者姓名", "organization": "机构", "title": "职位"}
  ],
  "interviewees": [
    {"name": "受访者姓名", "organization": "公司", "title": "职位"}
  ],
  "companies": ["长鑫存储"],
  "products": ["DDR5"],
  "technologies": ["HBM3E"],
  "units": ["万片/月", "Gb", "nm"]
}
```

### ASR 上下文输入

```json
{
  "background": ["本次访谈围绕存储芯片产品、工艺和产能口径。"],
  "terms": [
    {
      "canonical": "长鑫存储",
      "category": "公司",
      "aliases": ["CXMT"],
      "priority": "high",
      "source": "用户提供"
    }
  ],
  "corrections": {
    "长心存储": "长鑫存储",
    "吉比特": "Gb"
  },
  "sources": ["用户提供的访谈提纲"]
}
```

仅把用户明确确认的错误形式放入 `corrections`。不确定的同音词只作为候选术语，不做自动替换。

## 4. 运行命令

```powershell
python "C:\Users\骆慧\.codex\skills\interview-minutes-generator\scripts\audio_to_transcript.py" "D:\path\访谈录音.m4a" `
  --out-dir "D:\path\访谈纪要\公司_受访者_20260804" `
  --metadata-file "D:\path\访谈元信息.json" `
  --context-file "D:\path\ASR上下文输入.json"
```

选项：

- `--max-context-terms 80`：限制发送的高价值术语数量
- `--force`：重建该音频的派生产物，不影响源音频
- `--keep-temp`：成功后保留转码、切片和检查点
- `--retry-ambiguous`：重试网络结果不明的片段，可能产生重复计费
- `--no-asr-context`：不发送背景、热词或纠错信息
- `--timeout-seconds 1800`：单个极速版请求的超时时间

## 5. 转码与切段

- 豆包原生支持且文件小于安全直传阈值时，直接读取源音频并 Base64 提交。
- 其他情况转换为 16kHz、单声道、64kbps MP3。
- 默认以约30分钟为目标切段，并优先选择目标点前后60秒内的静音结束位置。
- 找不到合适静音点时按目标时间切分。
- 每段独立保存请求 ID、上下文模式、原始结果、时长和源时间偏移。
- 合并时把片段时间戳换算回原音频时间轴。
- 多片段的豆包说话人编号可能重新编号，因此使用“片段01-说话人1”一类标签，不跨片段强行认定身份。

## 6. 上下文降级

每个片段依次尝试：

1. `hotwords`、`correct_words` 与 `context_data`
2. 仅 `context_data`
3. 无上下文

只有接口明确拒绝上下文字段时才降级。处理清单和 `ASR上下文_*.json` 必须记录实际生效模式。

自动纠错规则：

- 豆包原始结果永久保留，不在原始 JSON 中改写
- 《转写原稿》只应用用户明确给出的 `corrections`
- 记录每种替换的出现次数
- 背景资料中的事实不得被补入逐字稿
- 无法确定的专名、数字或单位由整理阶段标为“待核听”

## 7. 断点恢复与失败处理

- 每个成功片段立即写入隐藏工作目录的检查点
- 重新运行时复用成功片段，不重复提交
- 服务明确返回繁忙时，最多按5秒、15秒、30秒退避重试
- 连接中断或响应不完整时标记 `ambiguous`，默认停止
- 只有用户接受可能重复计费时才使用 `--retry-ambiguous`
- 失败时保留临时切片、任务记录和非敏感错误信息

常见权限错误：

- `45000030 requested resource not granted`：检查是否开通录音文件极速版，并确认资源 ID 为 `volc.bigasr.auc_turbo`
- `401 Invalid X-Api-Key`：检查密钥是否来自豆包语音控制台且仍有效

## 8. 输出与验收

脚本生成：

- `音频元信息_*.json`
- `ASR上下文_*.json`
- `豆包ASR任务_*.json`
- `豆包ASR原始结果_*.json`
- `转写原稿_*.md`
- `处理清单_*.json`

验收：

- 源文件 SHA-256 前后相同
- 每个片段状态成功，或有明确失败/结果不明记录
- ASR 识别时长与片段时长接近
- 最后有效时间戳到达录音结尾；异常尾差必须核听
- 关键人名、公司、产品、技术、英文缩写和单位口径已抽查
- 《转写原稿》尾部完整
- 源音频存在，临时文件删除状态明确
