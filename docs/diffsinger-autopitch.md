# OpenUtau ── DiffSinger 自动音高（Auto Pitch）原理

> 调研对象：[`stakira/OpenUtau`](https://github.com/stakira/OpenUtau) 的 DiffSinger 渲染器。
> 目的：为 `msst-api` 的 DiffSinger(SVS) 接入提供参考——尤其是"如何从 `note + lyric` 得到 `f0_seq`"。

## 0. 结论

OpenUtau 的 DiffSinger "自动音高" = **用声库自带的音高预测模型**
（`dspitch/` 下的 `linguistic.onnx` + `pitch.onnx`），条件于音素编码与音符 MIDI/时长，
用**扩散 / 整流流**采样生成一条逐帧 F0 曲线（半音单位），再把曲线换算成
**PITD（音高偏差，cents）**写回工程，供显示与编辑。

它**不是**"用固定滑音把音符连起来"的规则算法，而是学习式生成模型。

它是**两步式**：

1. 先生成音高曲线（Auto Pitch / "Load rendered pitch"）；
2. 再正常渲染（声学模型读 PITD 曲线）。

> 渲染本身**不会**自动运行音高模型。自动音高是一个独立的预处理步骤。

---

## 1. 三层结构

```
① 基础音高曲线 (确定性)
   音符 tone ×100(cents) + vibrato + 手绘 pitch bend
   → RenderPhrase.pitches / pitchesBeforeDeviation      (RenderPhrase.cs)
                │
② 音高预测模型 (学习式, 两步 ONNX)
   linguistic.onnx : tokens (+ word_div/word_dur | ph_dur) (+ languages)
                     → encoder_out, x_masks
   pitch.onnx      : encoder_out + note_midi + note_dur + ph_dur + pitch + retake
                     (+ expr / note_rest / spk_embed) (+ steps|speedup|depth)
                     → pitch_out (逐帧, 半音)              (DsPitch.Process)
                │
③ 回写 / 重录
   半音 ×100 - basePitch(cents) → PITD 音高偏差曲线 (SetCurveCommand)
   绝对音高 → RPIT 渲染音高曲线 (显示 / 表达图)
   支持 retake (只重录改动音符) 与实时 (编辑后自动重算)
```

---

## 2. 触发条件与入口

- **声库必须带音高预测器**：`DiffSingerSinger.HasPitchPredictor` 判定
  `<voicebank>/dspitch/dsconfig.yaml` 是否存在。这是**独立于声学声库的第二个导出**：
  `dspitch/` 里自带 `dsconfig.yaml`、`linguistic.onnx`、`pitch.onnx`、
  `phonemes.json`、`dsdict.yaml`。
- 缺失时报错 `errors.editing.autopitch.unsupported`：
  "当前声库不支持生成音高线。请使用 Diffsinger 或 Enunu 声库。"
- 触发方式：
  1. 钢琴卷帘菜单 **"Load rendered pitch / 加载音高渲染结果"**
     → `LoadRenderedPitch` BatchEdit（`NoteBatchEdits.cs`）。
  2. **实时**：`RealTimePitchGenerationService`（Live Pitch `Off` / `Normal` / `Fast`），
     监听编辑命令，防抖后自动重算。
  3. 显示："View Final Pitch to Render (R)"。

---

## 3. 输入构造（`DsPitch.Process`）

| 步骤 | 说明 | 代码 |
|---|---|---|
| 乐句切分 | 按音素切 phrase；前后各 padding `headFrames = tailFrames = 8` 帧 | `DiffSingerUtils.PaddedSegments` / `PaddedPhoneDurations` |
| 音素 `tokens` | 音素 → id（`phonemes.json`），并校验 `dsdict.yaml` 覆盖 | `PhonemeTokenize` |
| 基础 pitch 输入 | 全量预测时 `pitch` = **常数 60 占位**、`retake` = **全 true**；只有 retake 时 `pitch` 才填现有曲线 | `DsPitch.Process` |
| 音符信息 | `note_midi`（每段音符 MIDI；head/tail/gap 为 rest 段，用邻近音符填充）、`note_dur`（帧）、`note_rest` | |
| 语言 | 多语言声库带 `languages` id | |
| 采样控制 | `steps`（连续加速）或 `speedup`（离散步）或 shallow `depth` + `steps` | `AddPitchSamplingInputs`，`DiffusionTimesteps = 1000` |

编码器走"词级"还是"音素级"由 `dsconfig.predict_dur` 决定
（`word_div`/`word_dur` vs `ph_dur`），与声学/方差模型共用同一种 `linguistic.onnx` 导出格式。

---

## 4. 输出 → 回写（`LoadRenderedPitch` BatchEdit）

`pitch.onnx` 返回 `RenderPitchResult { ticks, tones, retakeMask, voiced }`：

- `tones`：**半音**单位，逐帧 F0；
- `voiced`：`PaddedVoicedMask` 标记 padding、音素间空隙、清音帧 —— 这些帧的模型输出是
  伪影，**不写回**，避免句首 / 空隙出现尖刺；
- 换算写回 PITD：

  ```text
  y         = tones[i] * 100 - basePitch
  basePitch = phrase.pitchesBeforeDeviation[idx]   // 不含偏差的基准, cents
  SetCurveCommand(PITD, x, y, ...)
  ```

- 绝对音高同时写入 **RPIT（rendered pitch 掩码曲线）**，给表达图 / 显示读；
- 只清空**本乐句自己的 tick 区间**（`ClampRanges`），不擦相邻乐句；
- `PITD` 下限受表达式定义约束（默认 `min = -1200` cents）；
- 结果可经 "Convert PITD to pitch control points"（`BakePitch`）变成可手绘锚点。

> `PITD` 是"相对基准音符的偏差"，不是绝对 F0；这正是它可被手绘、可叠加用户意图的原因。

---

## 5. Retake：只重录改动的音符

- `DiffSingerRetake.MapSelectedPositionsToNoteIndexes`：把选中音符位置映射成音符索引；
- `BuildRetakeFrameMask`：给每个 padded 段映射回真实音符，改动帧 `true`；
- 重录时把**现有曲线作为 `pitch` 输入**、`retake` mask 传入，模型**只重写改动区域**，
  其余帧 `pitch[i] = existingPitch[i]` 保留；
- 好处：编辑几个音符不必重算整句，快且稳定。

---

## 6. 实时音高（Live Pitch）

`LivePitchMode = Off | Normal | Fast`：

| 模式 | steps | 防抖 | 说明 |
|---|---|---|---|
| Normal | 2 | 200 ms | 常规实时 |
| Fast | 0.1（走 shallow depth） | 80 ms | 最快，走 `ApplyTransient`（不记 undo） |

- 监听的命令：`AddNote` / `RemoveNote` / `ResizeNote` / `MoveNote` /
  `PhonemeOffset` / `PhonemePreutter` / `PhonemeOverlap` / `ClearPhonemeTiming` /
  `ChangePhonemeAlias`，以及 `ChangeNoteLyric`（等 `PhonemizedNotification`
  音素化完成后再算）；
- 异步 + 可取消（`CancellationToken`），防抖合并；
- 仅对 **DiffSinger 轨道**生效
  （`SupportsRenderPitch && SingerType == DiffSinger`）。

---

## 7. 为什么需要模型（与"音符连线"的区别）

传统做法是音符 tone + 固定 portamento / vibrato，机械。学习式音高模型能给出：

- 音素 / 时长相关的**自然微变化**；
- 颤音、滑音、跳音、连音过渡的细节；
- 清浊切换、句首句尾的收敛；
- 通过 `note_rest`（melody encoder）与 `expr`（表情 / 表现力）可控的演唱细节。

它本质上是 DiffSinger 方差 / 声学模型同族的**扩散（DDPM）或整流流（reflow）**生成器，
只是输出换成 F0。

---

## 8. 关键代码位置

| 文件 | 作用 |
|---|---|
| `OpenUtau.Core/DiffSinger/DiffSingerPitch.cs` | `DsPitch.Process` —— 音高预测核心（编码器 + `pitch.onnx` + retake + 采样控制） |
| `OpenUtau.Core/DiffSinger/RealTimePitchGenerationService.cs` | 实时自动重算（命令订阅、防抖、Live 设置） |
| `OpenUtau.Core/DiffSinger/DiffSingerRetake.cs` | retake 帧 mask / 区间 |
| `OpenUtau.Core/Editing/NoteBatchEdits.cs` (#440–668) | `LoadRenderedPitch` —— 结果 → PITD / RPIT 回写 |
| `OpenUtau.Core/DiffSinger/DiffSingerRenderer.cs` | `LoadRenderedPitch*` / `BuildRenderedRealCurves` / `ShouldRefreshRealCurvesOnCurveEdit` |
| `OpenUtau.Core/DiffSinger/DiffSingerSinger.cs` | `HasPitchPredictor`（`dspitch/dsconfig.yaml`） |
| `OpenUtau.Core/DiffSinger/DiffSingerUtils.cs` | `SampleCurve` / `Padded*` / `headFrames = tailFrames = 8` |
| `OpenUtau.Core/Render/RenderPhrase.cs` (`pitches`, `pitchesBeforeDeviation`) | 基础音高（音符 + vibrato + bend）构成 |
| `OpenUtau.Core/ExpressionGraph/PhrasePitch.cs` | 音高来源（pitch_bend / vibrato / mod_plus / notes） |

---

## 9. 对 msst-api 接入的启发

1. **这正好补上 v2m 输出的缺口**：v2m 导出的 `txt`
   （`onset offset pitch lyric`）与 `ustx`（notes，且通常**没有 `pitd` 曲线**）
   只有"音符 + 歌词"，缺 DiffSinger 需要的 `f0_seq`。
   **DiffSinger 的自动音高（`dspitch/`）就是用来从 note + lyric 生成 F0 的**。
2. 在 Python 侧复刻"自动音高"需要实现：
   - `linguistic.onnx` + `pitch.onnx` 两次 ONNX 推理；
   - head/tail padding、`PaddedSegments` / `PaddedPhoneDurations`、
     `note_midi` / `note_dur` / `note_rest` 构造；
   - retake mask 与 `pitch` 初值（全量 = 60 占位 / 局部 = 现有曲线）；
   - 半音 ↔ cents 换算与 PITD 写回；
   - `voiced` 掩码，避免在静音 / 清音帧写伪影。
3. **前提是声库带 `dspitch/`**；若没有，只能退化为"由音符 tone 合成 F0"
   （质量差，无颤音 / 滑音细节）。
4. 这也印证了选择 **OpenUtau 兼容声库**是对的：它同时带了声学模型、方差模型，
   以及这个 `dspitch/` 音高模型。

---

## 10. 一句话总结

> OpenUtau 的 DiffSinger 自动音高是一个**独立于声学渲染的、可手动触发 / 可实时重算的
> "音高生成步骤"**：它用声库 `dspitch/` 里的语言编码器 + 扩散 / 整流流音高模型，
> 从音素与音符信息预测逐帧 F0，转换成 PITD 偏差曲线写回工程；用户随后正常渲染时，
> 声学模型就基于这条曲线合成歌声。支持"只重录改动音符"和"编辑即重算"两种工作流。
