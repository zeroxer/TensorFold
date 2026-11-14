Experimental native Inkling2 text target and DSpark for TensorFold

PUBLIC SNAPSHOT LIMITATION (2026-11-14)
A later audit traced the actual vLLM attention_prologue to _kv_conv_tile: it rounds K/V convolution
outputs to BF16 before K clipping and V normalization. This TensorFold experiment and its historical
FP64 oracle retain the earlier unrounded-convolution assumption. That boundary is not qualified as
numerically equivalent to the deployed vLLM model. The historical receipts below are retained with
this explicit limitation; they are not current whole-model fidelity or production-serving approval.

Current development result (2026-11-14): original target weights plus all 84 MTP BF16 tensors are loaded natively.
DSpark runs its five-layer eight-row masked block and seven-token Markov walk through shared lanes.Engine/Drafted.
The full TP4/EP4 target passes eight paired 63-token plain/drafted runs, fixed width-eight acceptance controls
(exactly 1, 4 and 8 input rows), next-token rollback, draft-cache commit, cancellation and reset checks.
Latest batch-target medians on a 26-input/63-output high-effort math prompt: shared plain 17.8277 token/s,
DSpark 32.6517 token/s, drafted first token 0.5151 s. The earlier scalar target measured plain 18.6921 and
DSpark 15.6072 token/s. Batch verification doubles drafted throughput; one-row wrapper overhead remains.
Target prefill and verification now run windows of up to eight rows with host grouping of whole experts.
All four ranks pass complete-logit, all-eight-feature and KV/convolution byte parity against scalar execution
for widths 8, 3+5 and 1+7, including feature preservation when the intermediate prompt chunk skips its head.
Family host tests: 67 passed. Updated CUDA host suite: 395/395 build steps, 6526 tests passed, 33 skipped.
Normalized FP4 lane checks pass: 8x4608 reconstructed values, width 1/8 bit equality, FP64 dot max error zero.
Primary vLLM DSpark probes match tested softcap, Markov state/latent, feature mixing and 48 real fuse-alpha values.
Rare target AttnRes BF16-boundary differences remain; matching text is not full numerical equivalence.

Earlier precision changes removed BF16 rounds in AttnRes, KV convolution, residual addition and target head;
correct convolution order/FMA association, dynamic-tap dtype, expert outer-scale association and MTP alpha dtype.
The development CPU FP64 oracle shares that earlier KV-convolution assumption; it is not an exact oracle for the deployed vLLM fused path.
Three full-vocabulary rows match top1, with relative L2 0.007954672, 0.055645874, 0.031734741 and max absolute
0.444662, 1.745196, 1.469464. Quantizer threshold changes can amplify small normalization errors without router flips.
Full-trial minimum available RAM remains above 20 GiB. Never load this model beside an existing full model.

Base: ashhart/TensorFold aaac299d1d90c2625e54cb50f3f0d1494c1c33ea.
Experimental fork branch: inkling2-spark-experimental-20261114. Zig exactly 0.17.0.
Checkpoint: thinkingmachines/Inkling-2-Small e242a97e271095c11bc33ddb3850d37e22917343.
The unchanged HF NVFP4/MXFP8/BF16/F32 bytes are loaded through core.Checkpoint.
Attention is whole-head TP4 (12 Q / 2 KV heads per rank); experts are EP4 (96 full experts per rank).
Never split an NVFP4 expert's intermediate activation row: its outer amax spans all 2304 columns.

Build host checks and development executable:
  zig build test-inkling2-host tf-inkling2-cuda -Doptimize=safe -j1
Build inkling2_rows, inkling2_quant, inkling2_attention, inkling2_draft and inkling2_bridge as fatbins:
  nvcc -fatbin -O3 --fmad=false --ftz=false --prec-div=true --prec-sqrt=true -arch=sm_121 -std=c++17 SOURCE.cu -o NAME.fatbin
Run the development executable on all four ranks with the same NCCL_COMM_ID=LEADER_IPV4:PORT.
Its shared TensorFold tp_watch uses PORT+1, shares NCCL's unique ID and checks checkpoint metadata identity.
  tf-inkling2-cuda FATBIN_DIR MODEL_DIR RANK 48 IDS_JSON OUTPUT_DIR 128
IDS_JSON accepts an array for teacher forcing, or {"ids":[...],"prompt":[...]} for a separate greedy prompt.
Add "shared_dspark":true to use the shared scheduler's paired benchmark, fixed controls and cancellation checks.
The fast/DSpark path also requires qmmf and modelopt_pack fatbins built with their existing shared build flags.
Context is capped at 1024, generation at 128 new tokens. No native HTTP server registration is added.

Earlier pre-DSpark development milestones (historical measurements, not the current speed):
- Unmodified ARM64 baseline: 393/393 steps; 6526 tests passed, 33 skipped.
- Added host target: 56 tests passed, including all 1348 text headers and 12 independent Jinja2 template cases.
- Independent CPU layer0 reference: three real-checkpoint tokens, all 4608 output values exactly equal.
- SM121 operator probes: 77 attention comparisons and 96 row/quant/glue comparisons, 250 repeat runs.
- CPU activation quantization reference matches GPU bitwise in tested NVFP4/MXFP8 row cases.
- Real full-model four-rank run: 48/48 layers, 97,480,059,200 loaded bytes per rank, all ranks exit 0.
- High-effort 17*23 generation: 63 raw tokens including original special tokens; final answer 391.
- One measured greedy decode: 62 forwards in 4.979172255 s = 12.45186887 token/s, without DSpark.
- Full-trial minimum available memory: 22.5688 GiB. The production model was stopped before this load.

Scope and remaining work:
This is a finite text-inference development CLI, not a replacement production HTTP engine.
Native DSpark, a shared scheduler adapter and tested cancellation/reset are present in the finite greedy CLI.
Vision, concurrent streams, sampling, grammar, logprobs and prefix resume are not supported by this adapter.
The shared native family registry and production capabilities intentionally do not advertise inkling2.
The CUDA probes are development evidence, not all five upstream kernel-contract gates or release qualification.
Launch records, planted controls, the full precision envelope and maintainer release gates remain unqualified.
The target graph uses FP32 buffers carrying BF16-rounded activations and cache values (larger than BF16 storage).
Shared qmmf provides lossless packed matrix operations; one host routing synchronization per layer/window remains.
Original vLLM deployment and model bytes remain the rollback baseline; production restoration is verified separately.
See ABI.txt for equations, rounding, stored layouts and external-reference provenance.

Earlier full-logit and matched deployment findings (before the precision fixes and native DSpark above):
Three 200058-vocabulary teacher-forced rows agree on top1; they are not numerically identical to FP64.
Relative L2 differences are 0.0082481, 0.0647194 and 0.0480274 (max absolute 0.422, 2.1613 and 1.9790).
The later per-operator analysis and fixes above supersede this initial unlocalized discrepancy.
Numeric equivalence and release qualification still must not be marked passed.
Matched high-effort prompt: 26 input, 63 output tokens for both engines.
Native CLI single run: 12.4519 decode token/s; prompt plus decode 6.9845 s.
Restored vLLM + DSpark7, three API runs: median decode 43.5088 token/s, median total 2.001 s.
These compare a development CLI without drafts against the deployed API with drafts; no speedup is claimed.
Production vLLM was restored healthy with 32K/C1/DSpark7/full decode graphs and the original weights.
