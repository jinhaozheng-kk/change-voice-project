# AI Cover 声音转换与翻唱工作流

这是一个基于 Demucs、RVC 和 Pedalboard 的本地 AI 翻唱工作流，适用于 NVIDIA GPU（已按 RTX 4070 环境整理）。

## 工作流程

```text
输入歌曲
  ↓
Demucs 分离人声与伴奏
  ↓
RVC 转换人声音色
  ↓
Pedalboard 后期处理与混音
  ↓
导出 WAV / MP3
```

## 目录结构

```text
AI_Cover_4070_Workflow.ipynb   逐步骤调试和完整工作流
AI_Cover_OneClick.ipynb        一键处理工作流
ai-cover-project/
├─ input/                       原始歌曲
├─ stems/                       Demucs 分离结果
├─ vc_model/                    RVC 模型和索引文件
├─ vc_output/                   RVC 转换后的人声
├─ mix/                         最终混音结果
└─ docs/                        F0 等中间数据
```

## 环境要求

- Windows
- Python 3.9 或兼容的 Conda 环境
- NVIDIA GPU 和可用的 CUDA 环境
- Demucs
- RVC 推理仓库及其运行环境
- librosa、soundfile、pedalboard、PyTorch 等依赖

Notebook 默认使用的 RVC 仓库路径是：

```text
D:\5992\change voice\RVC1006Nvidia
```

迁移到其他电脑时，请修改 Notebook 配置区中的 `BASE`、`RVC_DIR`、输入文件和模型名称。

## 使用方法

1. 将待处理歌曲放入 `ai-cover-project/input/`。
2. 将 `.pth` 模型和可选的 `.index` 文件放入 `ai-cover-project/vc_model/`。
3. 打开 `AI_Cover_OneClick.ipynb`。
4. 修改配置区中的输入文件、模型名和 RVC 路径。
5. 按顺序运行 Notebook 单元。
6. 在以下目录查看结果：

```text
ai-cover-project/vc_output/   转换后的人声
ai-cover-project/mix/         最终混音
```

## RVC 参数

常用配置包括：

```python
F0_METHOD  = "crepe"  # 也可使用 rmvpe
F0_KEY     = 0        # 半音升降
INDEX_RATE = 0.0      # 0 表示关闭索引检索
```

如果启用 `.index` 文件，需要确认索引路径正确，并安装对应的 FAISS 依赖。

## 注意事项

- 当前 Notebook 会为 RVC runtime 安装或修复部分依赖，首次运行可能需要较长时间。
- Demucs 和 RVC 会产生较大的中间音频文件，请预留足够磁盘空间。
- RVC 输出通常为单声道，最终混音会统一到 48 kHz 双声道。
- 运行前请确认模型、输入音频和输出音频拥有合法的使用权。
- 本仓库使用 Git LFS 保存模型和音频等大文件。

## 当前示例

示例模型为 `XXXTENTACION.pth`，示例输出位于：

```text
ai-cover-project/vc_output/vocals_rvc.wav
ai-cover-project/mix/final_mix.wav
```

## 许可证与责任

本项目仅用于合法的个人研究、音频制作和技术实验。请遵守声音模型、原始音乐、歌手肖像/声音以及相关平台的版权和使用规定。
