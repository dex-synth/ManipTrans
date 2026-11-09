# h2r_bench 接入

`maniptrans_h2r` 实现通用单手 imitator 预训练和逐任务 residual 训练。训练框架为 **mjlab 1.6.0 的公共 `Simulation`**，物理引擎为 MJWarp；控制 60 Hz、物理 120 Hz、decimation=2，默认 4096 个环境。原生 `lib/rl` 的 PPO、网络、normalizer 与奖励函数复用；环境直接组合 `Simulation`，自行管理 reward、RSI 和 terminal，避免引入 ManagerBasedRlEnv 的自动 reset 和 dt 奖励缩放。

本版本身份为 `maniptrans-mjlab / virtual-wrist-position-v3`。这是保留原生训练主体的共同任务适配：数据集、机器人、simulator 和腕控制均与论文环境不同，不能称为论文结果的精确复现。原生 IsaacGym 入口仍保留。方法代码不导入 bench；bench 独立生成任务、推进评估世界和计分。阶段 1 的上游差异与参数见 [imitator 适配说明](h2r_phase1_imitator_adaptation.md)。

## 独立环境

```bash
cd /path/to/ManipTrans
bash scripts/h2r_environment.sh /path/to/maniptrans-env python3.11
MT_PY=/path/to/maniptrans-env/bin/python
```

锁文件固定 Torch 2.9、MuJoCo/MJWarp 3.11、Warp 1.16、mjlab 1.6.0、rl-games 1.6.1、SMPL-X 0.1.28 和 chumpy 0.70。MANO 官方模型只在阶段 1 参考准备时需要。chumpy 构建需要预装 NumPy/pip；脚本使用 `--no-build-isolation`。`requirements-h2r-overrides.txt` 明确覆盖 pygame 和 wandb：原 rl-games 可选 wandb logger 的旧上界与 mjlab 冲突，此入口仅使用 TensorBoard，不使用 wandb logger。更新锁文件：

```bash
uv pip compile --python 3.11 requirements-h2r.txt \
  --override requirements-h2r-overrides.txt -o requirements-h2r.lock
```

这些依赖不进入 bench 核心环境，也不覆盖原生 IsaacGym 环境。安装可编辑包使用 `MANIPTRANS_PROFILE=h2r`；默认 `setup.py` 仍是原生配方。

## 阶段 1：通用单手预训练

先在 bench 环境运行 `export-hand-motions`，从 ARCTIC、OakInk2、HOT3D、TACO 的全部可用源录制导出共同手部 corpus，不受 manipulation `included` 清单限制。该 corpus 只含来源、MANO 参数、关键点、有效区间与质量记录，不含机器人 IK。

最短训练片段规则由本方法的 `pretrain prepare` 执行：源帧少于 4 帧或时长不足 .5 秒（容差 `1e-10s`）的片段在 MANO 插值及合成前排除。`clip_selection.json` 保存阈值、选中数量和排除片段的来源；这是训练选择，不是参考构造错误，不会因此将完整 corpus 标为 partial。筛选规则同时写入 prepared 的 `recipe.clip_filter`。Bench 新导出保留单帧等短片段，旧 corpus 已在导出时执行过同一筛选，其有效训练片段不变。

共同双手 task 须预先以 `--physics-hz 120 --control-hz 60 --contacts` 封存。旧 task 不会被修改。两侧从相同 task 抽取独立单手模型，移除物体、另一只手和外部接触，只保留相关固定祖先。实际编译后的腕／手指增益、惯量、碰撞及限幅经审计后继承，dampratio 被物化为实际 affine actuator 参数。

```bash
"$MT_PY" -m maniptrans_h2r.pretrain prepare \
  --corpus /path/to/hand-corpus --task /path/to/sealed-task \
  --mano-root /path/to/mano/models --side both --workers 8 \
  --output /path/to/pretrain
for side in right left; do
  "$MT_PY" -m maniptrans_h2r.pretrain train \
    --prepared "/path/to/pretrain/$side" --output "/path/to/base-$side" \
    --device cuda:0
 done
```

准备阶段按片段使用 CPU 多进程，`--workers` 默认 8，可按 CPU 预算调整；每个 worker 的 PyTorch／BLAS 线程数为 1。任务队列最多保留 `2 × workers` 个片段，worker 写临时分片，主进程按源顺序合并并清理，保持窗口顺序和每片段随机种子。MANO hash 每次准备只核验一次，源插值器按片段复用，腕参考按 worker／目标侧复用。

`--side both` 共用源姿态重建检查和真实／合成 MANO 解码，再分别镜像及生成左右机器人参考，输出 `OUTPUT/right`、`OUTPUT/left` 两个独立 prepared。`--side right|left` 仍直接输出到 `OUTPUT`。准备阶段使用 CPU；GPU 用于后续训练。CPU 并行数不进入训练配方身份，prepare 的 schema、窗口和失败处理语义保持不变；没有新增依赖。

H2R 明确预算为默认 10000 epochs、4096 worlds；沿用原生 horizon 32、minibatch 1024、mini-epochs 5、seed 42。选定入口为 `main/cfg/config.yaml` 加各阶段 task／rl_train 默认，learning_rate=5e-4、moving_average=1；不混用 README 的 .4/2e-4 override。原生 score_to_win=10000、early_stop_epochs=9999999999999、save_best_after=10（imitator）／2（residual）保留，有限 epoch 预算单独记录停止原因。配置集中在 `config/imitator.yaml`。四数据集等概率、源左右侧等概率，90% 真实／10% MANO 旋转 SLERP 合成；异侧源镜像到目标手。真实序列全长切成至多 2000 帧且共享边界的窗口，按有效秒数抽样，CPU mmap 与每环境 GPU cache 按需取数，不预先把全集搬上显卡。

Reset 采用原生 `.99` RSI 法则、手指／腕噪声与参考速度。真实时间导数、连续腕 Euler 分支及 Jacobian 速度换算在方法侧处理；阶段 1 不做手指 IK。原生奖励与阈值课程保持，课程按控制调用计数，终止保留窗口最后一帧。

阶段 1 的虚拟腕初始化先核验未扰动参考及其 Euler 分支，再按原噪声分布最多尝试 64 次。仅限位越界会触发重采；耗尽后使用已核验的同一参考腕姿态，保留手指和速度初始化规则。参考非法或数值／FK 错误仍立即失败。这是受限虚拟腕的显式初始化适配：回退时取消腕姿态噪声，不等同于原生自由浮动腕的初始化分布。实际配方记录 `wrist_rsi`；`sampling.json` 和 resume sampler 状态保存累计 reset、拒绝样本及回退次数，续训延续计数。

阶段 1 的 `windowing=wrist-feasible-v1` 先在完整连续片段上计算世界姿态、速度和无界腕角，再按限位选择至多2000样本的最长合法窗口。每窗只选择固定等价分支及 `2π` 偏移，不在 episode 内换圈；新分支只在窗口 reset 时生效。全部源转移恰好覆盖一次，世界姿态、速度和时间不因切窗重算。相邻窗口保存各自的边界样本与腕角，用 `source_sample_start` 保留源索引。短末窗在可行时均分，限位强制的短窗保留并计入 `short_window_count`。池内仍按真实转移时长加权。单个转移无法表示或腕平移越界仍报 `robot_reference_invalid`，不删帧或扩大共同限位。

阶段 2 仍为完整任务选择连续腕角，不使用预训练窗口重置任务时钟。RSI 以抽中帧的参考腕角锚定分支；噪声越界时重采，不跳转到另一圈。

部分 corpus 必须显式使用 `--allow-incomplete`。功能检查可加 `--small-network --worlds 8 --epochs 2 --diagnostic-every 0`；该选项仅关闭周期诊断。实际训练结束时均执行均值和 RSI 随机策略诊断，包括达到 score 或其他原生提前停止条件；最终检查不依赖是否达到 epoch 上限。同一 epoch 已完成的均值诊断不重复执行，最终随机诊断单独去重。小网络或部分数据产物标记 `smoke=true`，不能用于默认正式 residual 入口。

`pretrain check --imitator BASE --prepared PRETRAIN --output CHECK --device cuda:0` 单独执行固定诊断，拒绝与当前控制配方不兼容的 imitator。报告名为 `training-corpus tracking diagnostic`，与训练数据重叠；严格 scale=0.7，报告 wrist、指尖／关节组、功率、有效前缀和时域覆盖。`wrist_action_saturation` 为 soft clamp 后六维腕动作中绝对值至少 .95 的比例，按有效步平均。它不是 manipulation SR 或未见序列泛化结果。

## 阶段 2：完整任务 residual

```bash
"$MT_PY" -m maniptrans_h2r.prepare \
  --task /path/to/sealed-task --output /path/to/prepared \
  --iterations 4000 --device cuda:0

"$MT_PY" -m maniptrans_h2r.train \
  --prepared /path/to/prepared \
  --imitator-right /path/to/base-right --imitator-left /path/to/base-left \
  --output /path/to/deployment --device cuda:0
```

默认入口只训练 residual，不隐式重训 base。单手只需对应侧，双手必须同时提供两侧。当前具名映射针对共同 Sharpa Wave。兼容性检查包括侧别、实际机器人动力学、控制／参考契约和特征维度；未知机器人或不支持的约束明确拒绝。

每手主要部件按人手指尖与计分物体的中位距离确定，也可用 `--object right=PART --object left=PART` 显式指定。主要部件只决定特征绑定，**全部 scoring parts 均参与 reward 和 failure**；物体项总权重为手数。动态上下文、自由根、hinge、spring 和固定支撑均保留在单个完整场景中。

weighted-keypoint IK 默认以 4000 次为上限，为 RSI 生成候选，腕／手指 Adam 学习率分别为 .0008/.0004。每 100 次检查完整序列平均加权 loss，改善小于 1e-5 则按原生条件停止；256 帧分块仅限制 FK 图内存，不分块决定停止，也不改变全序列 loss 归一化。`rsi.json` 记录实际迭代数、停止原因、更新前 loss、更新后输出及耗时。全部 roots、hinges、双手在相同源 tick 初始化；碰撞、限位和接触候选检查失败会记录，合法起帧为空则拒绝训练。评估始终提交优化后的第 0 帧机器人姿态，速度由 bench 置零，不从 RSI 选择最好起帧。

`config/residual.yaml` 默认 4096 环境、10000 epochs、32/1024/5 PPO 配方；残差 init gain=.01、logstd=-1，保留原生 warmup 调用顺序。全部 base 参数和 normalizer buffers 冻结并在保存时核验；PPO 每次 minibatch forward 按原生行为重新采样 base，不传入 rollout 时的固定 base 样本；部署的每 episode RNG 隔离仍保留。

Residual 私有训练恢复 `ResDexHand.yaml` 的原生物理课程：重力从 0 倍、物体滑动摩擦从 3 倍起，在 1920 个控制步（32 秒）内 linear decay。更新只在 reset 调用时检查：全局间隔至少 32 步，逐 world age 达到 32 且正在 reset 才更新摩擦；首次全部应用。保留 native age 消耗顺序及绝对摩擦 [1, 6) 的 250 桶量化，终点不保证等于未量化名义值。所有交互物体部件及后代 geoms 都覆盖，机器人／环境摩擦、扭转／滚动摩擦、质量、弹簧和 servo gains 保持名义值。mjlab 扩展 per-world friction 并重建 CUDA graph；随后原位写入 GPU，weight 特征使用当前训练重力。保存并回读审计课程状态；恢复时还原各 world 摩擦历史、age 和全局门限，再按重新 reset worlds 的原生调度继续。

阶段 2 默认在 epoch 0、每 100 epochs 和最终 epoch 执行 `full-task tracking diagnostic`，写入 `diagnostics/epoch-N.json`。独立 nominal world 使用固定首帧机器人姿态、task 物体初态、零速度、seed 43、随机 base／residual 均值和严格 scale=.7；从头运行至原生失败或真实末端，记录有效前缀覆盖、终止 tick 及各计分部件误差。首帧碰撞不通过则明确记录初始化失败。这些诊断不更新 normalizer、训练 RNG、训练 world 或课程，也不是 bench SR。`--diagnostic-every 0` 可关闭中间诊断，最终诊断仍执行。

## 控制、物理与产物

动作经原生 soft clamp 后，每手解码为：

```text
wrist = measured_wrist_q(t) + scale * (base + 2 * residual)
finger = lower + (clamp(base + 2 * residual, -1, 1) + 1) / 2 * (upper - lower)
```

平移 scale=.12 m、旋转 scale=1 rad；阶段 1 没有 residual。基准来自控制步开始时的实际关节位置，不累加到上一次目标，也不叠加参考轨迹。每个 60 Hz 控制步计算一次目标，在两个 120 Hz 物理子步内保持，不额外乘 dt。moving average=1，无额外平滑；目标再按启用的 actuator ctrlrange 和手指限位裁剪，腕角不做周期回绕。参考仍用于观测、奖励和 RSI。参数统一定义在 `control.py`，写入训练 recipe 和产物的 `control` 字段。

零腕动作以本步实测位置为目标；运动仍由共同位置执行器产生。该控制让策略主动产生腕运动，但不等同于上游直接施加 wrench：实际作用力依赖共同增益、阻尼及惯量。尺度是位置目标偏置，不是每步实际位移，也不是上游力幅值的等价换算。没有新增参考前馈、速度补偿或物体辅助力。

接触／约束缓冲容量分别在两份 YAML 中固定：单手 128/1024，完整场景 512/2048。容量不足明确报错，不截断接触或自动减少环境数。

episode 的 `completed_controls` 独立于物理 tick／参考 tick，reward 使用当前调用之前的计数：residual 第 8/9 次传 7/8，imitator 第 20/21 次传 19/20；reward 后递增，部分 reset 仅清对应 world。每个控制区间只采样／解码一次动作，调用两次 `Simulation.step()`；任务末端剩一个 physics tick 时仅执行一次。功率使用实际最后一次 solve 的执行力乘该物理步积分后的速度，与原生 reward 和训练计划一致；提前到末端的 world 保留自身最后一步的功率。积分后仅刷新运动学，避免额外 full forward 覆盖接触求解结果。每指接触 sensor 精确匹配该 body 的碰撞 geoms，包含全部接触伙伴；在线状态使用 bench 真实 contact records。新配方要求 `contact_coverage=robot-all-v1`，包含手内与双手接触；schema 相同但覆盖缺失的旧 task 不能用于新训练。

训练核验封存 CPU 模型、sensor 模型和 GPU 实际参数，显式恢复 MJWarp 的 solver tolerance。Bench MJWarp runtime 可能保留 float32 的 `1e-6` 下限；相同物理参数不意味着长轨迹逐位等价。

| 产物 | 内容 |
| --- | --- |
| pretrain prepared | 单手场景、资产、兼容签名、片段筛选记录、motion mmap、窗口／诊断 manifest |
| `imitator.json` | 侧别、数据与物理身份、网络／normalizer、smoke 标识及文件 hash |
| residual prepared | 原封存 `task/`、`reference.npz`、`rsi.npz/json`、首帧 `initialization.json` |
| `deployment.json` | 任务身份、相对路径 base、residual checkpoint、动作与特征契约 |
| `model.pth` | 仅模型 tensor，供部署加载 |
| `resume.pth` | 可信本地模型／优化器／normalizer／scheduler／RNG／sampler／课程状态 |
| `resolved_config.yaml`, `ppo_config.json`, `source.json`, `training.json` | 实际配方、源码 hash、依赖／GPU 版本与训练步数 |

输出目录必须不存在。续训传 `--resume /old/output/resume.pth --output /new/output --epochs TOTAL`，其余影响学习的参数与输入身份保持一致。只在完整 PPO epoch 边界恢复，重新 reset 训练 worlds，记录 `world_state_restarted=true`，不承诺与不中断运行逐位一致。

H2R 入口在原生 residual 的 `nan_to_num` 之前检查网络输出和 Gaussian 标准差，并检查梯度、更新后参数、恢复和保存的模型状态。非有限训练结果产生 `failure.json` 并停止，不生成成功 deployment；原生 IsaacGym 默认入口不启用这些检查。

## 统一评估与批次

Bench 按同一 task/spec 启动 `evaluate-server` 后连接：

```bash
"$MT_PY" -m maniptrans_h2r.evaluate \
  --deployment /path/to/deployment --socket /path/to/eval.sock --device cpu
```

客户端不创建 simulator，只消费真实反馈并输出具名位置目标。推理保留随机 base／residual mean，每个 episode 使用 bench 的 controller seed。初始化准入、完整时域、sim/sim-dr、MuJoCo/MJWarp 和评分分母由 bench 掌握。

`scripts/h2r_oakink2_batch.py --manifest INPUT.json --phase prepare|smoke|pilot|train|eval --output OUTPUT` 接受显式路径 manifest，核对固定十条 included 快照及 SHA。Manifest 字段见脚本说明。smoke 选择 O4/O2/O6/O9/O10，pilot 选择 O4/O2/O9/O10（200 epochs），train 从对应 pilot 续训到 1000 epochs；其他任务直接训练到 1000 epochs。eval 要求每条明确提供 MuJoCo/MJWarp × sim/sim-dr × seeds 11/23 八项 server/score 命令，保留 80 个预定项分母及失败记录。批次脚本不生成 preprocess bundle，也不把流程返回码当成任务 SR。

## 验证与兼容

```bash
"$MT_PY" -m pytest tests/h2r -q
"$MT_PY" -m maniptrans_h2r.check \
  --deployment /path/to/deployment --output /path/to/check.json --device cuda:0
```

`check` 按部署版本选择诊断：v2 检查 masked reset、60 次控制推进 120 ticks、末端单 tick和 nominal 物理；`--run` 使用同任务名义 bench 的真实 `contact_observations.jsonl` 检查初态特征／动作和首区间物理，初始 forward 显式使用 bench reset 的零 ctrl。`--contact-probe` 可读取 bench 同状态探针（task_id、cases 中的 qpos/qvel/ctrl/frame），验证非零手内接触；不再用训练合力伪造物体接触来证明一致性。没有提供 bench 观测时明确记录 parity 未检查。单元测试包含课程原生 helper／reset 调度对照、8/9 与 20/21 门限、完整序列 IK 分块与停止条件；`H2R_NATIVE_RL_SOURCE` 指向只读上游 lib/rl 时，还对照原始 forward 与连续 PPO 更新。

旧版显式使用 `prepare --legacy-v1` 与 `train --train-local-imitators`；旧 `virtual-wrist-position-v1/v2` deployment 仍按原参考加偏置方式评估，保留各自平滑和时钟。v1 的完整场景 imitator／逐示范训练不作为通用预训练。旧 base 不能用于 v3 residual 训练；旧 imitator 的独立诊断须使用其原 checkout。原生 wrench checkpoint 与位置控制版本均不兼容。

当前物理／特征签名仍为 `hand-reference-v3` 和 `maniptrans-hand-physics-v2`，包括具名几何、关节、执行器和封存物理 options；未覆盖的手部 tendon、equality 或显式 contact pair 明确拒绝。阶段 1 prepared 采用 `preparation_revision=2`、`windowing=wrist-feasible-v1`；满足这些条件的现有全量 prepared 可复用，更旧的单手参考需重新 prepare。prepared 兼容签名中的历史 `control_schema=virtual-wrist-position-v2` 保留为位置执行器契约，策略动作解码由训练产物单独封存。训练采用 `training_revision=7`，绑定 v3 控制配方，同时保留受限腕 RSI 和最终诊断进度；新训练必须重新训练两侧 base，旧 resume 不跨版本续接。阶段 2 的完整任务参考语义未变，满足当前签名和 `preparation_revision=3` 的 residual prepared 可复用。启动时须明确选择当前源码目录；已有运行的冻结 worktree 不会随主 checkout 更新。

运行验证记录保存在 bench 的 `runs/checks/`，按各记录中的源码版本和控制配方解释结果。短跑不代表全集覆盖、完整预算收敛或论文成绩；其他机器人与未实际运行的批次组合不据此宣称已验证。
