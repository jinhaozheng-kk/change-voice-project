# AI Cover — 本地 AI 翻唱 / 声音转换工作流

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%20%2F%203.12-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.3.1%2Bcu121-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch CUDA">
  <img src="https://img.shields.io/badge/Demucs-mdx__extra-blueviolet" alt="Demucs">
  <img src="https://img.shields.io/badge/RVC-inference-orange" alt="RVC">
  <img src="https://img.shields.io/badge/Git_LFS-enabled-important" alt="Git LFS">
</p>

一套跑在 **Windows + NVIDIA GPU** 上的端到端 AI 翻唱流水线：把一首歌拆成人声与伴奏，用人声音色模型（RVC）把歌手音色换成目标音色，再用 Pedalboard 做 EQ / 压缩 / 混响 / 限幅混音，最后导出 WAV/MP3。

```text
输入歌曲
  ↓
① Demucs 分离：vocals / no_vocals（GPU，失败自动回退 CPU / 小模型）
  ↓
②（可选）torchcrepe 提取 F0 基频，方便检查音高
  ↓
③ RVC 推理：用 .pth 音色模型转换人声（可用 .index 检索提升相似度）
  ↓
④ Pedalboard 混音：高通 → De-esser → 压缩 → 混响 → 人声/伴奏电平匹配 → Limiter
  ↓
⑤（可选）响度归一化到 -14 LUFS、导出 320 kbps MP3
```

仓库提供两个 Jupyter Notebook：

| Notebook | 定位 |
| --- | --- |
| `AI_Cover_OneClick.ipynb` | **一键流水线**（17 个 cell）：配置区 → Demucs → RVC runtime 准备 → RVC 推理 → 混音 → LUFS。适合环境已就绪后成片 |
| `AI_Cover_4070_Workflow.ipynb` | **逐步调试工作流**（42 个 cell）：环境检测、目录初始化、20 秒片段快速迭代、F0 提取、RVC 依赖/补丁排错、混音、MP3 导出。适合首次搭建和排查问题 |

> 本仓库按 RTX 4070 环境整理，同时兼容其他 NVIDIA 显卡（显存不足时可回退 CPU）。

---

## 目录

- [1. 环境要求](#1-环境要求)
- [2. 快速开始](#2-快速开始)
- [3. 工作原理与各步骤细节](#3-工作原理与各步骤细节)
- [4. 目录结构](#4-目录结构)
- [5. RVC 参数与调音建议](#5-rvc-参数与调音建议)
- [6. 常见问题排查](#6-常见问题排查)
- [7. 仓库中的示例数据](#7-仓库中的示例数据)
- [8. 法律、版权与道德](#8-法律版权与道德)
- [9. 已知限制](#9-已知限制)
- [10. 参考](#10-参考)

---

## 1. 环境要求

### 硬件

- Windows 10/11
- NVIDIA GPU（示例环境 RTX 4070；显存 8GB 以上较稳）
- 足够的磁盘空间：一首歌的 Demucs stems 约 60–130 MB，RVC 模型 + 索引约 90 MB，成品 30 MB+

### 软件（两套 Python 环境）

这套流程实际涉及**两个独立的 Python 环境**，这是最容易踩坑的地方：

| 环境 | 用途 | 关键依赖 |
| --- | --- | --- |
| **主环境**（如 conda `aicover`） | 跑 Jupyter、Demucs 分轨、F0 提取、混音导出 | Python 3.9+、PyTorch(CUDA)、demucs、librosa、soundfile、pedalboard、torchcrepe、pydub、pyloudnorm |
| **RVC runtime**（RVC 仓库自带，如 `RVC1006Nvidia/runtime/python.exe`） | 只负责 RVC 推理 | PyTorch 2.3.1+cu121、numpy 1.26.4、scipy 1.10.1、faiss-cpu（启用 index 时）、fairseq、praat-parselmouth、pyworld、av |

Notebook 里对 RVC runtime 的依赖安装/补丁是**写死的命令**，首次运行会花较长时间（下载 CUDA 版 PyTorch 等）。

### 需要自备

- 一个 **RVC 推理仓库**（示例路径 `D:\5992\change voice\RVC1006Nvidia`，需带 `runtime\python.exe` 与 `tools\infer_cli.py`）
- 目标音色的 **RVC 模型** `.pth`，以及可选的 `.index` 检索文件
- 有合法使用权的音频素材

## 2. 快速开始

### 2.1 克隆（需要 Git LFS）

仓库用 Git LFS 管理模型和音频，克隆前先装好 `git-lfs`：

```powershell
git lfs install
git clone https://github.com/jinhaozheng-kk/change-voice-project.git
cd change-voice-project
git lfs pull
```

如果已经 clone 但大文件是 130 字节左右的指针文件，执行 `git lfs pull` 即可拉取真实内容。

### 2.2 准备目录与素材

```text
ai-cover-project/
├─ input/       把歌曲（wav/mp3）放进来
├─ stems/       Demucs 输出（自动创建）
├─ vc_model/    放 .pth 与 .index
├─ vc_output/   RVC 转换结果（自动创建）
├─ mix/         最终混音（自动创建）
└─ docs/        f0.npy 等中间数据
```

### 2.3 修改 Notebook 配置

打开 `AI_Cover_OneClick.ipynb`，只改配置区（第 2 个代码 cell）：

```python
BASE     = Path(r"D:\change voice\ai-cover-project")        # 项目根目录（含 input/stems/…）
RVC_DIR  = Path(r"D:\5992\change voice\RVC1006Nvidia")      # RVC 仓库根目录
INPUT    = BASE / "input" / "source.wav"                    # 待处理歌曲
MODEL_FN = "XXXTENTACION.pth"                               # 放在 RVC 仓库 assets\weights\ 下的文件名
INDEX    = BASE / "vc_model" / "XXXTENTACION.index"         # 可选，不用就留空

F0_METHOD  = "crepe"   # crepe / rmvpe
F0_KEY     = 0         # 升/降调半音数
INDEX_RATE = 0.0       # 0 关闭索引；0.78 启用
```

> `AI_Cover_4070_Workflow.ipynb` 里的 `BASE` / `RVC_DIR` 也同样是硬编码路径，迁移电脑时逐 cell 改。

### 2.4 运行

按顺序从上到下运行 cell。首次运行会自动：

1. 把非 WAV / 含中文路径的输入转成 ASCII 立体声 WAV；
2. 安装 Demucs 并优先用 CUDA 分轨；
3. 给 RVC runtime 安装 CUDA 版 PyTorch 与依赖，并打两个兼容补丁；
4. 调用 `infer_cli.py` 完成音色转换；
5. 混音导出到 `ai-cover-project/mix/final_mix.wav`。

### 2.5 查看结果

```text
ai-cover-project/stems/mdx_extra/source_stereo/vocals.wav     # 分离出的人声
ai-cover-project/vc_output/vocals_rvc.wav                     # 换声后的人声
ai-cover-project/mix/final_mix.wav                            # 最终成品
ai-cover-project/mix/final_mix_lufs-14.wav                    # 响度归一化版本
```

## 3. 工作原理与各步骤细节

### 3.1 输入预处理

- 用 `librosa.load(sr=44100, mono=False)` 读入，单声道会复制成双声道；
- 转成 `(N, 2)` 后写出 `input/source_stereo.wav`；
- **只要路径含非 ASCII 字符或不是 WAV，就会重新编码**——这是为了绕开 Demucs / RVC 在 Windows 上对中文路径的不稳定处理。

### 3.2 Demucs 分轨

```powershell
python -m demucs.separate -d cuda --two-stems vocals -n mdx_extra <source_stereo.wav> -o <stems>
```

- `--two-stems vocals`：只分离出人声（`vocals.wav`）和伴奏（`no_vocals.wav`），比四轨模式快；
- 回退链：`cuda + mdx_extra` → `cpu + mdx_extra` → `cpu + mdx_q`（需要 `diffq`，会自动尝试安装）；4070 工作流里还提供 `htdemucs` 作为不依赖 diffq 的备选；
- 输出目录形如 `stems/<模型名>/<文件名>/vocals.wav`，Notebook 会按修改时间找最新结果，并兼容 `accompaniment.wav` / `instrumental.wav` / `other.wav` 等伴奏命名。

### 3.3 20 秒片段快速迭代（可选）

`AI_Cover_4070_Workflow.ipynb` 用 librosa 截取人声与伴奏前 20 秒，写入 `stems/<stem>/clips/vocals_20s.wav` 与 `inst_20s.wav`。调 RVC 参数时先用片段跑，能省下大量时间。

### 3.4 F0 提取（可选）

用 `torchcrepe` 在 16 kHz 单声道上提取基频：

```python
f0 = torchcrepe.predict(audio, 16000, hop_length=160, fmin=50, fmax=1100,
                        model='full', device='cuda', batch_size=2048,
                        decoder=torchcrepe.decode.viterbi)
```

结果保存到 `docs/f0.npy`，可用于检查音高曲线、判断是否需要升降调（`F0_KEY`）。

### 3.5 RVC runtime 准备与补丁

这一步把 RVC 仓库自带的 runtime 修成可用的 CUDA 推理环境：

| 步骤 | 内容 |
| --- | --- |
| PyTorch | 卸载现有版本，安装 `torch==2.3.1+cu121` / `torchvision==0.18.1+cu121` / `torchaudio==2.3.1+cu121` |
| 数值库 | `numpy==1.26.4`、`scipy==1.10.1`（`--only-binary=:all:` 避免源码编译），避免 SciPy ABI / numba 冲突 |
| 常用依赖 | `python-dotenv`、`soundfile`、`praat-parselmouth`、`pyworld`、`av`、`faiss-cpu`（索引）、`onnxruntime-gpu`（rmvpe） |
| fairseq | 需要时安装预编译 Windows cp39 wheel（`fairseq-0.12.2-cp39-cp39-win_amd64.whl`） |
| 补丁 1 | 把 `infer/lib/audio.py` 的 `load_audio` 替换为 librosa/soundfile 实现，绕开 PyAV 的 `rb`/`wb` 模式兼容问题 |
| 补丁 2 | 给 `infer/modules/vc/modules.py` 的 `file_index.strip()` 加 `None` 防护，避免不传索引时报错 |

补丁前会自动备份为 `.bak` / `.bak2` / `.bak_index`（这些备份名在 `.gitignore` 里）。

### 3.6 RVC 推理

模型需要放在 RVC 仓库的 `assets/weights/` 下（Notebook 会自动复制），然后调用：

```powershell
python tools\infer_cli.py ^
  --input_path  <vocals.wav> ^
  --opt_path    <vocals_rvc.wav> ^
  --model_name  XXXTENTACION.pth ^
  --index_rate  0 ^
  --f0method    crepe ^
  --f0up_key    0 ^
  --device      cuda ^
  --index_path  ""
```

- Notebook 会设置 `PYTHONPATH` 并把 `cwd` 切到 RVC 仓库；
- 输出小于 2 KB 视为失败，自动**回退使用原始人声**继续混音，保证流水线不中断；
- 运行前会清理之前产生的 0 字节坏文件。

### 3.7 Pedalboard 混音

1. 人声与伴奏统一重采样到 **48 kHz**，转成 `(channels, n)` 并按较短长度对齐；
2. **电平匹配**：按 RMS 把人声缩放到伴奏的 90%，避免换声后忽大忽小；
3. 人声处理链：

| 插件 | 参数 |
| --- | --- |
| `HighpassFilter` | 80 Hz（去低频隆隆声） |
| De-esser | `Deesser()`；不可用时降级 `PeakFilter(7500Hz, -4.5dB, Q1.2)` → `HighShelfFilter(7000Hz, -4dB)` → 直通 |
| `Compressor` | threshold -18 dB，ratio 2.5 |
| `Reverb` | room 0.12，wet 0.08，dry 0.92（很轻的空间感） |
| `Gain` | 0 dB |

4. 混音：`mix = inst + 0.9 * vox_proc`；
5. `Limiter(threshold_db=-1.0)` 逐通道限幅，峰值超过 0.999 时整体缩放；
6. 用 `pedalboard.io.AudioFile` 写出 `mix/final_mix.wav`（float32，双声道）。

### 3.8 响度归一化与 MP3（可选）

- `pyloudnorm` 测量 integrated loudness，增益到 **-14 LUFS**，输出 `final_mix_lufs-14.wav`；
- 若 PATH 中有 ffmpeg，用 pydub 导出 **320 kbps MP3**；没有 ffmpeg 则只保留 WAV。

## 4. 目录结构

```text
change-voice-project/
├─ AI_Cover_OneClick.ipynb        # 一键流水线（推荐成片用）
├─ AI_Cover_4070_Workflow.ipynb   # 逐步调试工作流（首次搭建用）
├─ ai-cover-project/
│  ├─ input/                      # 原始歌曲
│  ├─ stems/                      # Demucs 分离结果（含 clips/ 试听片段）
│  ├─ vc_model/                   # RVC .pth 与 .index
│  ├─ vc_output/                  # 换声后的人声
│  ├─ mix/                        # 最终混音
│  ├─ docs/                       # f0.npy 等中间数据
│  └─ .gitignore                  # 子目录专用忽略规则
├─ .gitattributes                 # Git LFS：*.pth / *.index / *.wav / *.mp3 / *.exe
├─ .gitignore
└─ README.md
```

## 5. RVC 参数与调音建议

| 参数 | 默认 | 说明与建议 |
| --- | --- | --- |
| `F0_METHOD` | `crepe` | 基频提取算法。`crepe` 质量好；`rmvpe` 更快更稳但需要 `onnxruntime-gpu` |
| `F0_KEY` | `0` | 升降调半音。男女声互转通常 ±3~±12；先听片段再定 |
| `INDEX_RATE` | `0.0` | 0 = 关闭索引检索；0.70–0.85 更贴近目标音色，但过高会有"念经感" |
| `--device` | `cuda` | 显存不足可换 `cpu`（会慢很多） |
| 20s 片段 | 开 | 调参阶段用 `clips/` 快速试听，定稿后跑整首 |

经验：

- **先保人声干净**：Demucs 分离质量直接决定换声效果，伴奏太响或人声有残留时，先换模型/再分一遍；
- **音域差太大再升降调**：`F0_KEY` 改动要配合试听，别一次 ±12；
- **索引不是必须**：没有 `.index` 用 `INDEX_RATE=0` 也能跑；启用索引必须装 `faiss-cpu` 且 `INDEX` 路径存在；
- **混音不要太湿**：默认 reverb 很轻，翻唱要的是"像原曲里的人在唱"，不是 KTV。

## 6. 常见问题排查

| 症状 | 原因与处理 |
| --- | --- |
| `CUDA available: False` | 当前环境装的是 CPU 版 torch；按 Notebook 里的命令装 `torch==2.3.1+cu121`；确认 `nvidia-smi` 正常 |
| `SciPy` / `numba` ABI 报错 | 固定 `numpy==1.26.4`、`scipy==1.10.1`、`llvmlite==0.41.1`、`numba==0.58.1` |
| `av.open(... "rb")` 相关报错 | RVC 的 PyAV 用法不兼容；跑 Notebook 的 "补丁 1" 把 `load_audio` 换成 librosa/soundfile 实现 |
| `file_index` 为 None 报 `AttributeError` | 跑 "补丁 2" 加 `None` 防护，或确保 `INDEX_RATE=0` 时传空字符串 |
| `faiss` 导入失败 | 启用 `.index` 时必须 `pip install faiss-cpu` |
| `fairseq` 装不上 | 用预编译 cp39 wheel；或手动下载后用 `runtime\python.exe -m pip install <whl>` |
| RVC 输出 0 字节 / 小于 2 KB | 查看 cell 的 STDERR；Notebook 已自动清理坏文件并回退原人声 |
| Demucs CUDA 显存不足 | 自动回退 CPU / 小模型；也可手动 `-d cpu -n htdemucs` |
| Demucs 找不到输出目录 | 不同版本目录层级不同（`stems/<model>/<stem>` vs `stems/<stem>`），Notebook 已按多个候选路径查找 |
| MP3 导出被跳过 | PATH 中没有 ffmpeg；安装 ffmpeg 或只保留 WAV |
| 中文路径报错 | 输入会被重编码为 `source_stereo.wav`；但 `BASE` / `RVC_DIR` 本身也建议全英文路径 |
| 首次运行特别慢 | 会给 RVC runtime 重装 CUDA 版 PyTorch 与多个依赖，属正常现象 |

## 7. 仓库中的示例数据

为了展示完整链路，仓库里保留了一次真实运行的输入、中间产物与成品（均由 Git LFS 存储）：

| 文件 | 大小 | 说明 |
| --- | --- | --- |
| `input/Bryant Barnes,d4vd - I'd Rather Pretend.mp3` | ~7.8 MB | 示例原曲 |
| `input/source.wav` / `source_stereo.wav` | 15.5 / 31.1 MB | 预处理后的 44.1 kHz 音频 |
| `input/vs_BuildTools.exe` | 4.3 MB | 环境折腾时误放的工具安装包（**建议清理**，见第 9 节） |
| `stems/mdx_extra/source_stereo/vocals.wav` / `no_vocals.wav` | 各 31.1 MB | Demucs 分离结果 |
| `stems/.../clips/vocals_20s.wav` / `inst_20s.wav` | 各 1.8 MB | 20 秒试听片段 |
| `vc_model/XXXTENTACION.pth` / `.index` | 52.5 / 37.5 MB | 示例音色模型与检索索引 |
| `vc_output/vocals_rvc.wav` | 14.1 MB | RVC 换声后的人声 |
| `mix/final_mix.wav` | 33.8 MB | 最终混音成品 |
| `docs/f0.npy` | ~10 KB | 示例 F0 数据 |

> 示例模型与歌曲仅用于演示流程；请勿将示例音色/歌曲用于商业发行。

## 8. 法律、版权与道德

这类项目天然涉及多重权利，请务必遵守：

- **音频素材**：只使用你拥有版权、已获授权或可合法使用的歌曲；
- **声音模型**：只使用你本人、已获明确授权、或公开允许使用的音色模型；
- **歌手人格权**：AI 翻唱可能涉及声音/肖像/姓名权，公开发布前确认当地法律与平台规则；
- **平台规则**：不要用于伪造、诈骗、冒充或规避 DRM；
- 本项目仅用于合法的个人研究、音频制作与技术实验。

## 9. 已知限制

- Notebook 中路径是**硬编码**的（`D:\change voice\...`），换机器必须逐处修改；
- 依赖两个环境（主环境 + RVC runtime），且 RVC runtime 需要打补丁，可移植性一般；
- RVC 仓库、模型、索引不随本仓库提供，需要自备；
- 混音参数是固定的一条链，没有做母带处理（只做到限幅 + 可选 LUFS）；
- 没有自动化测试，也没有 CLI 脚本，流程依赖按顺序运行 Notebook；
- `input/vs_BuildTools.exe` 是与流程无关的安装包，建议从仓库移除；
- 示例数据让仓库体积很大（克隆需 LFS + 数百 MB），只想跑流程的话可以只保留两个 notebook；
- 未附带 LICENSE 文件。

## 10. 参考

- [Demucs（Facebook Research）](https://github.com/facebookresearch/demucs)
- [RVC-Project / Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- [Spotify Pedalboard](https://github.com/spotify/pedalboard)
- [torchcrepe](https://github.com/maxrmorrison/torchcrepe)
- [pyloudnorm](https://github.com/csteinmetz1/pyloudnorm)
- [Git LFS](https://git-lfs.com/)
