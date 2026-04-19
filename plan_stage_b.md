下面是基于我刚刚实际阅读的仓库内容整理出的 **grounded implementation plan**。  
我把内容分成两类：

- `已确认`：能直接从当前仓库代码里看到的事实
- `建议 / 待定`：基于当前代码结构给出的最小实现建议，不等于已经存在

**1. 当前仓库里与 Stage B 最相关的文件**

- `已确认`：[main_per_image.py](/root/autodl-fs/watermark/anti-t2i/UnivIntruder_ap/main_per_image.py) 负责产生 per-image 样本目录；每个样本会保存 `original.png`、`protected.png`、`delta_vis.png`、`compare.png`、`delta.pth`、`log.json`，而 run 目录名已经带有 `eps / img / target / aug / tv / par / timestamp`。
- `已确认`：[evaluate_per_image.py](/root/autodl-fs/watermark/anti-t2i/UnivIntruder_ap/evaluate_per_image.py) 会输出 `evaluation_summary.json`，里面有 `results`、`source_group_summaries`、target-class proxy 指标，以及质量指标 `L2 / Linf / PSNR / SSIM`。`prepare_i2v_pilot.py` 目前就是靠这些结果挑 sample。
- `已确认`：[prepare_i2v_pilot.py](/root/autodl-fs/watermark/anti-t2i/UnivIntruder_ap/prepare_i2v_pilot.py) 已经是 Stage B 的 protocol skeleton。它会：
  - 读取两个候选 run 的 `evaluation_summary.json`
  - 选择 easy / hard source classes 和代表样本
  - 生成 `sample_subset_manifest.json`
  - 生成 `annotation_template.csv`
  - 生成 `pilot_protocol.md`
  - 可选复制 still-image 资产到 `samples/<sample_id>/...`
- `已确认`：[summarize_i2v_pilot.py](/root/autodl-fs/watermark/anti-t2i/UnivIntruder_ap/summarize_i2v_pilot.py) 只依赖人工标注 CSV，不依赖视频内部细节；它按 `(i2v_model_id, candidate_config_id)` 聚合，并计算 severe-rate 与默认 signal gate。
- `已确认`：参考侧最直接相关的是：
  - [ref/I2VWM/utils/video_generation.py](/root/autodl-fs/watermark/anti-t2i/UnivIntruder_ap/ref/I2VWM/utils/video_generation.py)
  - [ref/I2VWM/test_I2V.py](/root/autodl-fs/watermark/anti-t2i/UnivIntruder_ap/ref/I2VWM/test_I2V.py)
  - [ref/VideoShield/watermark_embedding_and_extraction.py](/root/autodl-fs/watermark/anti-t2i/UnivIntruder_ap/ref/VideoShield/watermark_embedding_and_extraction.py)
  - [ref/VideoShield/utils.py](/root/autodl-fs/watermark/anti-t2i/UnivIntruder_ap/ref/VideoShield/utils.py)
  - [ref/SIGMark-release/main.py](/root/autodl-fs/watermark/anti-t2i/UnivIntruder_ap/ref/SIGMark-release/main.py)
  - [ref/SIGMark-release/prompt_set/vbench2_prompt_set.py](/root/autodl-fs/watermark/anti-t2i/UnivIntruder_ap/ref/SIGMark-release/prompt_set/vbench2_prompt_set.py)

**2. 准备复用哪些已有脚本 / 数据结构**

- `已确认`：Stage B 最应该复用的是 `prepare_i2v_pilot.py` 产出的 `sample_subset_manifest.json`。它已经定义好了：
  - `protocol`
  - `samples`
  - `clean_image_path`
  - `candidates`
  - `planned_outputs`
- `已确认`：`annotation_template.csv` 和 `summarize_i2v_pilot.py` 可以原样保留，不需要为了 first runner 重写总结逻辑。
- `已确认`：`evaluation_summary.json -> source_group_summaries -> easy/hard sample selection` 这条上游链已经存在，因此 Stage B 不需要再发明一套 sample subset selection。
- `已确认`：当前 manifest 已经给每个 sample 规划了 `clean.mp4 / cand_a.mp4 / cand_b.mp4` 的目标路径，runner 最自然的做法就是按这些路径写入。
- `建议`：first runner 应该把 `prepare_i2v_pilot.py` 当作唯一上游入口，而不是自己重新扫 per-image 输出目录。

**3. 我建议新增 / 修改哪些文件**

- `建议新增`：[run_i2v_pilot.py](/root/autodl-fs/watermark/anti-t2i/UnivIntruder_ap/run_i2v_pilot.py)  
  作用：读取 `sample_subset_manifest.json`，按固定 protocol 生成 `clean / cand_a / cand_b` 视频，并把运行元数据落盘。
- `建议新增`：[utils/i2v_backends.py](/root/autodl-fs/watermark/anti-t2i/UnivIntruder_ap/utils/i2v_backends.py)  
  作用：放 backend registry 和各 backend adaptor。第一版只实现 `CogVideoX`，`SVD` 可以先留 stub 或第二步再接。
- `建议尽量不改`：
  - [main_per_image.py](/root/autodl-fs/watermark/anti-t2i/UnivIntruder_ap/main_per_image.py)
  - [evaluate_per_image.py](/root/autodl-fs/watermark/anti-t2i/UnivIntruder_ap/evaluate_per_image.py)
  - [summarize_i2v_pilot.py](/root/autodl-fs/watermark/anti-t2i/UnivIntruder_ap/summarize_i2v_pilot.py)
- `建议可选小改`：[prepare_i2v_pilot.py](/root/autodl-fs/watermark/anti-t2i/UnivIntruder_ap/prepare_i2v_pilot.py)  
  仅当你希望把 `fps / negative_prompt / backend_model_path / dtype` 也写进 manifest 时再改；第一版也可以不改，而让 runner 自己通过 CLI 补充这些字段。

**4. 最小 CogVideoX runner 的输入 / 输出契约**

- `已确认`：I2VWM 的 `test_CogVideoX(...)` 形式是：
  - 输入目录 `input_image_path`
  - 输出目录 `out_video_path`
  - `resolution`
  - `prompt json`
  - `guidance_scale / steps / fps / quality / device`
- `建议输入契约`：
  - `--pilot_dir` 或 `--manifest_path`
  - `--model_key CogVideoX`
  - `--device`
  - `--hf_model_id`，默认可指向 `THUDM/CogVideoX-5b-I2V`
  - `--fps`
  - `--negative_prompt`
  - `--overwrite`
  - `--limit_samples`
- `建议输出契约`：
  - 严格写到 manifest 里的 `planned_outputs[model_id]` 路径
  - 每个 sample 目录额外保存一个 `run_metadata.json`
  - pilot 根目录额外保存一个 `generation_run.json` 或 `generation_log.jsonl`
- `建议运行粒度`：以 `sample_id` 为最小单位，每个 sample 生成三条视频：
  - `clean`
  - `primary candidate`
  - `secondary candidate`

**5. backend registry 应该如何组织**

- `已确认`：I2VWM 用的是很轻量的 `choose_test(["CogVideoX"]) -> test_CogVideoX` 风格，没有复杂框架。
- `建议`：当前仓库也保持轻量，不要上大类层次。最小形式就够：
  - `BACKEND_REGISTRY = {"CogVideoX": CogVideoXBackend(...)}`
- `建议`：每个 backend adaptor 只暴露两件事：
  - `load_pipeline(...)`
  - `generate(...)`
- `建议`：不要把 Stage B 做成大一统 engine；第一版 registry 只解决“按 model_id 找到正确 runner”。
- `建议`：manifest 中 `protocol.i2v_model_ids` 用什么字符串，registry 就直接接收什么字符串，避免再做一层不透明映射。

**6. manifest / prompt / metadata / output directory 的落盘方案**

- `已确认`：当前 manifest 已包含这些 protocol 字段：
  - `primary_candidate_id`
  - `secondary_candidate_id`
  - `candidate_config_ids`
  - `i2v_model_ids`
  - `prompt`
  - `seed`
  - `num_inference_steps`
  - `guidance_scale`
  - `num_frames`
  - `frame_resolution`
  - `selection_metric`
- `已确认`：每个 sample 已包含：
  - `sample_id`
  - `image_id`
  - `source_bucket`
  - `source_class_index`
  - `source_class_text`
  - `clean_image_path`
  - `candidates[...].protected_image_path / delta_vis_path / compare_image_path`
  - `planned_outputs[...]`
- `建议`：第一版 runner 不修改原 manifest，只新增一个 `generation_protocol_resolved.json`，把 backend-specific 的字段补进去，比如：
  - `hf_model_id`
  - `negative_prompt`
  - `fps`
  - `dtype`
  - `device`
- `建议`：输出目录沿用当前 skeleton：
  - `pilot_dir/samples/<sample_id>/...` 放 still-image 资产
  - `pilot_dir/generated/CogVideoX/<sample_id>/...` 放 mp4 和运行元数据
- `建议`：不要把 prompt 再拆成 per-sample prompt；第一版严格使用 manifest 里的单一固定 `protocol.prompt`。

**7. clean / cand_a / cand_b 的成对生成如何保证可复现**

- `已确认`：当前 protocol 已经固定了 `prompt / seed / num_inference_steps / guidance_scale / num_frames / frame_resolution`。
- `建议`：对同一个 sample，三条视频必须共享以下完全相同的条件：
  - 同一个 backend
  - 同一个 prompt
  - 同一个 seed
  - 同一个 step 数
  - 同一个 guidance scale
  - 同一个 frame 数
  - 同一个分辨率
  - 同一个 negative prompt
  - 同一个 dtype / scheduler / model revision
- `建议`：每次 `clean -> cand_a -> cand_b` 调用前，都重新创建一个新的 `torch.Generator(device).manual_seed(seed)`，而不是复用已被消耗状态的 generator。
- `建议`：保存 `run_metadata.json`，明确记录“除了输入图像路径不同，其余参数完全一致”。
- `猜测 / 待验证`：diffusers + CUDA 环境下未必保证跨机器 bitwise 一致，但过程级复现和实验协议复现是可以严格做的。

**8. 第一版明确不做哪些东西**

- `建议不做`：watermark extraction / decode
- `建议不做`：tamper localization
- `建议不做`：latent watermark / inversion
- `建议不做`：多模型大一统调度框架
- `建议不做`：SVD 第二 backend
- `建议不做`：新的视频自动指标大扩展
- `建议不做`：mp4/gif 后处理花活
- `建议不做`：prompt suite / VBench 风格大规模 protocol
- `建议不做`：重写 `prepare_i2v_pilot.py` / `summarize_i2v_pilot.py` 现有骨架

**9. 潜在风险点**

- `已确认`：当前 manifest 没有 backend-specific 字段，比如 `fps`、`negative_prompt`、`hf_model_id`、`dtype`；这意味着第一版要么 runner 自己补 CLI 参数，要么小改 manifest。
- `已确认`：`prepare_i2v_pilot.py` 当前协议是“所有样本共用一个固定 prompt”；这对早期 pilot 是优点，但也可能让某些图像与 prompt 不匹配，从而稀释异常信号。
- `已确认`：`model_ids` 是自由字符串；如果 manifest 写的是 `CogVideoX`，registry 也必须用同样名字，否则 `planned_outputs` 和后续 summary 会脱节。
- `已确认`：`--skip_asset_copy` 模式下，manifest 里的图像路径可能指向原实验目录，而不是 pilot 工作区；runner 需要小心处理。
- `猜测 / 待验证`：CogVideoX 的显存和速度成本可能明显高于当前 skeleton 预期，尤其如果一开始就按 `clean + cand_a + cand_b` 三倍运行。
- `猜测 / 待验证`：如果 pipeline 在同一个进程内长期复用，是否会受内部缓存或 generator 消耗顺序影响，需要 first smoke test 验证。

**10. 一个尽量小的 patch plan**

- `建议新增`：[run_i2v_pilot.py](/root/autodl-fs/watermark/anti-t2i/UnivIntruder_ap/run_i2v_pilot.py)  
  读取 manifest，校验 protocol，按 sample 逐个生成 `clean / cand_a / cand_b`，写入 `planned_outputs` 指定路径，并保存运行元数据。
- `建议新增`：[utils/i2v_backends.py](/root/autodl-fs/watermark/anti-t2i/UnivIntruder_ap/utils/i2v_backends.py)  
  放最小 backend registry；第一版只接 `CogVideoX` adaptor，结构参考 I2VWM 的 `test_CogVideoX`。
- `建议暂不改`：[summarize_i2v_pilot.py](/root/autodl-fs/watermark/anti-t2i/UnivIntruder_ap/summarize_i2v_pilot.py)  
  因为它已经完成了人工标签汇总闭环。
- `建议暂不改`：[evaluate_per_image.py](/root/autodl-fs/watermark/anti-t2i/UnivIntruder_ap/evaluate_per_image.py)  
  它的职责已经清晰，继续负责 proxy / grouped / quality 上游评估即可。
- `建议可选小改`：[prepare_i2v_pilot.py](/root/autodl-fs/watermark/anti-t2i/UnivIntruder_ap/prepare_i2v_pilot.py)  
  只有在你希望把 `fps / negative_prompt / hf_model_id` 提前固化到 manifest 时再加字段；否则第一版不动更稳。
- `建议不动`：
  - [main.py](/root/autodl-fs/watermark/anti-t2i/UnivIntruder_ap/main.py)
  - [evaluate.py](/root/autodl-fs/watermark/anti-t2i/UnivIntruder_ap/evaluate.py)
  - [main_siglip.py](/root/autodl-fs/watermark/anti-t2i/UnivIntruder_ap/main_siglip.py)
  - [main_image_bind.py](/root/autodl-fs/watermark/anti-t2i/UnivIntruder_ap/main_image_bind.py)

一句话概括这个 plan：  
**最小改动方案就是：保留现有 Stage B skeleton，不碰原主线，只新增一个 `run_i2v_pilot.py` 和一个 `utils/i2v_backends.py`，先把 `prepare_i2v_pilot.py` 产出的 manifest 接到一个真实的 CogVideoX backend 上，严格生成 clean / cand_a / cand_b 三元对照。**