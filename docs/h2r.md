# h2r_bench 接入

`maniptrans_h2r` 实现通用单手 imitator 预训练和逐任务 residual 训练。唯一的任务输入是 h2r_bench 的 **task 目录**（`h2r-task-v1`，独立打包或位于 suite 内），由仓库内随包分发的加载器 `maniptrans_h2r/h2r_io/` 读取；该目录由 `h2r-bench export-loader` 写入，不手工修改，`h2r-bench check-loader maniptrans_h2r/h2r_io` 核对它与 bench 一致。训练框架为 **mjlab 1.6.0 的公共 `Simulation`**，物理引擎为 MJWarp。单次训练的默认环境数为阶段 1 的 8192 和阶段 2 的 4096。原生 `lib/rl` 的 PPO、网络、normalizer 与奖励函数复用；环境直接组合 `Simulation`，自行管理 reward、RSI 和 terminal，避免引入 ManagerBasedRlEnv 的自动 reset 和 dt 奖励缩放。

本版本身份为 `maniptrans-mjlab / virtual-wrist-position-v3`。这是保留原生训练主体的共同任务适配：数据集、机器人、simulator 和腕控制均与论文环境不同，不能称为论文结果的精确复现。原生 IsaacGym 入口仍保留。方法代码不导入 bench；bench 独立生成任务、推进评估世界和计分。阶段 1 的上游差异与参数见 [imitator 适配说明](h2r_phase1_imitator_adaptation.md)。

## 时钟

方法的时钟集中定义在 `clock.py`，以秒为单位：

| 时钟 | 值 | 归属 |
| --- | --- | --- |
| 控制周期 | 1/60 s | 方法；训练与部署相同，部署时写入 `start.requirements.control_dt_s` |
| 训练物理步长 | 1/120 s（每控制周期 2 步） | 方法的私有训练模型；显式写入模型，**不继承 task 的名义步长**（当前 300 Hz） |
| 评估物理步长 | `hello["dt_s"]`（公共默认 1/300 s） | 评估方；客户端由它换算每控制周期的物理步数（300 Hz 时为 5），产物中不保存 tick 数 |
| 参考 | 30 Hz 共同参考，按秒插值到控制网格 | bench 提供，方法重采样 |

奖励累计、失败宽限、接触历史、阈值课程和物理课程都按控制步计数；控制周期未变，它们的物理时间含义与原生一致。训练模型若带有其他步长，环境构造时直接报错。任务时长须是训练物理步长的整数倍，否则拒绝，不取整。

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

这些依赖不进入 bench 核心环境，也不覆盖原生 IsaacGym 环境。安装可编辑包使用 `MANIPTRANS_PROFILE=h2r`；默认 `setup.py` 仍是原生配方。环境未安装本仓库时，从仓库根目录以 `python -m maniptrans_h2r.<入口>` 运行。全部步骤（阶段 1、阶段 2、评估客户端和测试）使用这一个环境；评估客户端只需 CPU，不加载任何 simulator。

## 阶段 1：通用单手预训练

先在 bench 环境运行 `export-hand-motions`，从 ARCTIC、OakInk2、HOT3D、TACO 的全部可用源录制导出共同手部 corpus，不受 manipulation `included` 清单限制。该 corpus 只含来源、MANO 参数、关键点、有效区间与质量记录，不含机器人 IK。corpus 格式为 `h2r-hand-motion-corpus-v2`，四个数据集的指尖统一取 MANO 顶点 `744,320,443,554,671`；更早格式的 corpus 会被拒绝，须重新导出。合成片段的 MANO 重新解码经 `h2r_io.mano.decode` 完成，并按片段记录处理左手形状基镜像。

最短训练片段规则由本方法的 `pretrain prepare` 执行：源帧少于 4 帧或时长不足 .5 秒（容差 `1e-10s`）的片段在 MANO 插值及合成前排除。`clip_selection.json` 保存阈值、选中数量和排除片段的来源；这是训练选择，不是参考构造错误，不会因此将完整 corpus 标为 partial。筛选规则同时写入 prepared 的 `recipe.clip_filter`。Bench 的导出保留单帧等短片段。

`--task` 给出机器人：两侧从同一个 task 的装配模型抽取独立单手模型，移除物体、另一只手和外部接触，只保留相关固定祖先，并把步长设为方法的训练步长。实际编译后的腕／手指增益、惯量、碰撞及限幅经审计后继承，dampratio 被物化为实际 affine actuator 参数。`--side` 须是该 task 的参与手；`both` 需要双手 task。

```bash
"$MT_PY" -m maniptrans_h2r.pretrain prepare \
  --corpus /path/to/hand-corpus --task /path/to/task \
  --mano-root /path/to/mano/models --side both --workers 8 \
  --output /path/to/pretrain
for side in right left; do
  "$MT_PY" -m maniptrans_h2r.pretrain train \
    --prepared "/path/to/pretrain/$side" --output "/path/to/base-$side" \
    --device cuda:0
 done
```

base 与任务按**手部物理签名**（`maniptrans-hand-physics-v3`）配对：具名的 body、关节、几何体、执行器参数与求解器选项，加上方法的训练步长与控制周期。同一只手在双手包和单手包中编译出的导出量（如由 dampratio 得到的阻尼）可能相差最后一位，比较时数值容许 `1e-9` 的相对误差，其余须相同。task 把与物体接触的机器人几何体改用场景统一的接触参数，因此接触参数不同的场景需要各自的 base。四个数据集使用同一组场景参数（滑动摩擦 2.0），当前 suite 的 40 个 task 只需一组 base；右手单手任务可使用由双手 task 准备的右手 base。

准备阶段按片段使用 CPU 多进程，`--workers` 默认 8，可按 CPU 预算调整；每个 worker 的 PyTorch／BLAS 线程数为 1。任务队列最多保留 `2 × workers` 个片段，worker 写临时分片，主进程按源顺序合并并清理，保持窗口顺序和每片段随机种子。MANO hash 每次准备只核验一次，源插值器按片段复用，腕参考按 worker／目标侧复用。

`--side both` 共用源姿态重建检查和真实／合成 MANO 解码，再分别镜像及生成左右机器人参考，输出 `OUTPUT/right`、`OUTPUT/left` 两个独立 prepared。`--side right|left` 仍直接输出到 `OUTPUT`。准备阶段使用 CPU；GPU 用于后续训练。CPU 并行数不进入训练配方身份，prepare 的 schema、窗口和失败处理语义保持不变；没有新增依赖。

阶段 1 默认与上游有效配方对齐：8192 worlds、最大 9999999999999 epochs（全局 `max_iterations`），horizon 32、minibatch 1024、mini-epochs 5、seed 42。选定入口为 `main/cfg/config.yaml` 加 imitator task／rl_train 默认，learning_rate=5e-4、moving_average=1；不混用 README 的 .4/2e-4 override。保留 score_to_win=10000、early_stop_epochs=9999999999999、save_best_after=10；默认 epoch 上限近乎无限，不代表收敛。可用 `--epochs N` 显式限制运行预算，停止原因单独记录。配置集中在 `config/imitator.yaml`。改变 worlds 需要新开训练，不能将旧环境数的 resume 直接续接。

四数据集等概率、源左右侧等概率，90% 真实／10% MANO 旋转 SLERP 合成；异侧源镜像到目标手。真实序列全长切成至多 2000 帧且共享边界的窗口，按有效秒数抽样，CPU mmap 与每环境 GPU cache 按需取数，不预先把全集搬上显卡。

Reset 采用原生 `.99` RSI 法则、手指／腕噪声与参考速度。真实时间导数、连续腕 Euler 分支及 Jacobian 速度换算在方法侧处理；阶段 1 不做手指 IK。原生奖励与阈值课程保持，课程按控制调用计数，终止保留窗口最后一帧。

阶段 1 的虚拟腕初始化先核验未扰动参考及其 Euler 分支，再按原噪声分布最多尝试 64 次。仅限位越界会触发重采；耗尽后使用已核验的同一参考腕姿态，保留手指和速度初始化规则。参考非法或数值／FK 错误仍立即失败。这是受限虚拟腕的显式初始化适配：回退时取消腕姿态噪声，不等同于原生自由浮动腕的初始化分布。实际配方记录 `wrist_rsi`；`sampling.json` 和 resume sampler 状态保存累计 reset、拒绝样本及回退次数，续训延续计数。

阶段 1 的 `windowing=wrist-feasible-v1` 先在完整连续片段上计算世界姿态、速度和无界腕角，再按限位选择至多2000样本的最长合法窗口。每窗只选择固定等价分支及 `2π` 偏移，不在 episode 内换圈；新分支只在窗口 reset 时生效。全部源转移恰好覆盖一次，世界姿态、速度和时间不因切窗重算。相邻窗口保存各自的边界样本与腕角，用 `source_sample_start` 保留源索引。短末窗在可行时均分，限位强制的短窗保留并计入 `short_window_count`。池内仍按真实转移时长加权。单个转移无法表示或腕平移越界仍报 `robot_reference_invalid`，不删帧或扩大共同限位。

阶段 2 仍为完整任务选择连续腕角，不使用预训练窗口重置任务时钟。RSI 以抽中帧的参考腕角锚定分支；噪声越界时重采，不跳转到另一圈。

部分 corpus 必须显式使用 `--allow-incomplete`。功能检查可加 `--small-network --worlds 8 --epochs 2 --diagnostic-every 0`；该选项仅关闭周期诊断。实际训练结束时均执行均值和 RSI 随机策略诊断，包括达到 score 或其他原生提前停止条件；最终检查不依赖是否达到 epoch 上限。同一 epoch 已完成的均值诊断不重复执行，最终随机诊断单独去重。小网络或部分数据产物标记 `smoke=true`，不能用于默认正式 residual 入口。

`pretrain check --imitator BASE --prepared PRETRAIN --output CHECK --device cuda:0` 单独执行固定诊断，拒绝与当前控制配方不兼容的 imitator。报告名为 `training-corpus tracking diagnostic`，与训练数据重叠；严格 scale=0.7，报告 wrist、指尖／关节组、功率、有效前缀和时域覆盖。`wrist_action_saturation` 为 soft clamp 后六维腕动作中绝对值至少 .95 的比例，按有效步平均。它不是 manipulation SR 或未见序列泛化结果。

## 阶段 2：完整任务 residual

```bash
"$MT_PY" -m maniptrans_h2r.prepare \
  --task /path/to/task --output /path/to/prepared \
  --iterations 4000 --device cuda:0

"$MT_PY" -m maniptrans_h2r.train \
  --prepared /path/to/prepared \
  --imitator-right /path/to/base-right --imitator-left /path/to/base-left \
  --output /path/to/deployment --device cuda:0
```

默认入口只训练 residual，不隐式重训 base。只为 task 的参与手提供 base：右手任务只传 `--imitator-right`，左手任务只传 `--imitator-left`，双手须同时提供两侧。关键点对应表按机器人名称给出，当前只有 `sharpa_wave`；手的集合、关节、腕部关节和执行器来自 robot 包的 semantics，不比较变体名称。兼容性检查包括侧别、手部物理签名、控制／参考契约和特征维度；未知机器人或不支持的约束明确拒绝。

每手主要部件按人手指尖与计分物体的中位距离确定，也可用 `--object right=PART --object left=PART` 显式指定。主要部件只决定特征绑定，**全部 scoring parts 均参与 reward 和 failure**；物体项总权重为手数。多个物体、动态上下文、自由根、hinge、spring、额外的固定支撑与固定 mesh 均保留在单个完整场景中；物体和关节的名称、顺序取自 `task.object_binding`，四个数据集走同一段代码。

weighted-keypoint IK 默认以 4000 次为上限，为 RSI 生成候选，腕／手指 Adam 学习率分别为 .0008/.0004。每 100 次检查完整序列平均加权 loss，改善小于 1e-5 则按原生条件停止；256 帧分块仅限制 FK 图内存，不分块决定停止，也不改变全序列 loss 归一化。`rsi.json` 记录实际迭代数、停止原因、更新前 loss、更新后输出及耗时。全部 roots、hinges、参与手在相同参考时刻初始化；碰撞、限位和接触候选检查失败会记录，合法起帧为空则拒绝训练。穿透预检阈值是方法常量 `MAX_PENETRATION_M = 0.02`（`physics.py`），与评估方的准入阈值无关，task 中没有该值。

**训练与评估的初态不同。** 训练保留原生 RSI：每次 reset 按原生分布 `floor(.98·N·U)` 在合法参考帧中抽取起点，带参考速度（第 0 帧被抽中的概率约为 1/N）。评估总是从参考第 0 帧开始，物体取 task 的起始状态，全部速度为零，没有 warm-up；客户端提交优化后的第 0 帧机器人姿态，不从 RSI 中挑选起帧。这一差异是保留原生训练配方的结果，方法没有为评估初态增加训练项。

`config/residual.yaml` 默认 4096 环境、10000 epochs、32/1024/5 PPO 配方；残差 init gain=.01、logstd=-1，保留原生 warmup 调用顺序。全部 base 参数和 normalizer buffers 冻结并在保存时核验；PPO 每次 minibatch forward 按原生行为重新采样 base，不传入 rollout 时的固定 base 样本；部署的每 episode RNG 隔离仍保留。

Residual 私有训练恢复 `ResDexHand.yaml` 的原生物理课程：重力从 0 倍、物体滑动摩擦从 3 倍起，在 1920 个控制步（32 秒）内 linear decay。更新只在 reset 调用时检查：全局间隔至少 32 步，逐 world age 达到 32 且正在 reset 才更新摩擦；首次全部应用。保留 native age 消耗顺序及绝对摩擦 [1, 6) 的 250 桶量化，终点不保证等于未量化名义值。所有交互物体部件及后代 geoms 都覆盖，机器人／环境摩擦、扭转／滚动摩擦、质量、弹簧和 servo gains 保持名义值。mjlab 扩展 per-world friction 并重建 CUDA graph；随后原位写入 GPU，weight 特征使用当前训练重力。保存并回读审计课程状态；恢复时还原各 world 摩擦历史、age 和全局门限，再按重新 reset worlds 的原生调度继续。

阶段 2 默认在 epoch 0、每 100 epochs 和最终 epoch 执行 `full-task tracking diagnostic`，写入 `diagnostics/epoch-N.json`。独立 nominal world 使用固定首帧机器人姿态、task 物体初态、零速度、seed 43、随机 base／residual 均值和严格 scale=.7；从头运行至原生失败或真实末端，记录有效前缀覆盖、终止时刻及各计分部件误差。首帧碰撞不通过则明确记录初始化失败。这些诊断不更新 normalizer、训练 RNG、训练 world 或课程，也不是 bench SR。`--diagnostic-every 0` 可关闭中间诊断，最终诊断仍执行。

## 控制、物理与产物

动作经原生 soft clamp 后，每手解码为：

```text
wrist = measured_wrist_q(t) + scale * (base + 2 * residual)
finger = lower + (clamp(base + 2 * residual, -1, 1) + 1) / 2 * (upper - lower)
```

平移 scale=.12 m、旋转 scale=1 rad；阶段 1 没有 residual。基准来自控制步开始时的实际关节位置，不累加到上一次目标，也不叠加参考轨迹。每个 60 Hz 控制步计算一次目标，并在该控制周期的全部物理子步内保持（训练为两个 120 Hz 子步，300 Hz 评估为五个），不额外乘 dt。moving average=1，无额外平滑；目标再按启用的 actuator ctrlrange 和手指限位裁剪，腕角不做周期回绕。参考仍用于观测、奖励和 RSI。参数统一定义在 `control.py`，写入训练 recipe 和产物的 `control` 字段。

零腕动作以本步实测位置为目标；运动仍由共同位置执行器产生。该控制让策略主动产生腕运动，但不等同于上游直接施加 wrench：实际作用力依赖共同增益、阻尼及惯量。尺度是位置目标偏置，不是每步实际位移，也不是上游力幅值的等价换算。没有新增参考前馈、速度补偿或物体辅助力。

接触／约束缓冲默认容量在两份 YAML 中配置：单手 128/1024，完整场景 512/2048。训练可用 `--contact-capacity`／`--constraint-capacity` 显式覆盖，训练及周期／最终诊断共用所选容量。MJWarp 接触容量是每 world 的平均分配预算，实际使用跨 world 总池；约束上限按 world 生效。默认任何 overflow 立即失败，不截断接触或自动减少环境数；阶段 1 可显式启用下述局部 EPA 恢复。错误报告保留 MJWarp overflow 位名、受影响 world 数量、前 16 个 world 的编号／位掩码／约束数，以及实际分配上限；未知位也会报错。训练期间的报告写入 `failure.json` 的 `overflow` 字段，不能仅从通用“容量超限”推断具体缓冲。

episode 的 `completed_controls` 独立于物理 tick／参考 tick，reward 使用当前调用之前的计数：residual 第 8/9 次传 7/8，imitator 第 20/21 次传 19/20；reward 后递增，部分 reset 仅清对应 world。每个控制区间只采样／解码一次动作，训练中调用两次 `Simulation.step()`；任务末端剩一个训练物理步时仅执行一次。功率使用实际最后一次 solve 的执行力乘该物理步积分后的速度，与原生 reward 和训练计划一致；提前到末端的 world 保留自身最后一步的功率。积分后仅刷新运动学，避免额外 full forward 覆盖接触求解结果。每指接触 sensor 精确匹配该 body 的碰撞 geoms，包含全部接触伙伴；在线状态使用 bench 真实 contact records 累加出的逐指尖净力，因此评估须提供 `contact_coverage=robot-all-v1`（含手内与双手接触）：客户端核对 hello 中声明的覆盖，遇到公共默认的缩减覆盖时拒绝 start，不补零。

训练按 task 的 `compiled.json` 核验 CPU 模型，再核验 sensor 模型和 GPU 实际参数（步长为方法的训练步长，其余为 task 的封存值），显式恢复 MJWarp 的 solver tolerance。Bench MJWarp runtime 可能保留 float32 的 `1e-6` 下限；相同物理参数不意味着长轨迹逐位等价。

| 产物 | 格式 | 内容 |
| --- | --- | --- |
| pretrain prepared | `maniptrans-h2r-pretrain-v2` | 单手场景、资产、手部物理签名、片段筛选记录、motion mmap、窗口／诊断 manifest |
| `imitator.json` | `maniptrans-h2r-imitator-v2` | 侧别、数据与物理身份、网络／normalizer、smoke 标识及文件 hash |
| residual prepared | `maniptrans-h2r-prepared-v3` | `task/`（`h2r_io.export_task` 导出的独立 task，`task_id` 不变）、`reference.npz`、`rsi.npz/json`、首帧 `initialization.json`；绑定 `task_id`、`demo_id`、`robot_id` |
| `deployment.json` | `maniptrans-h2r-deployment-v3` | `task_id`、控制周期（秒）、相对路径 base、residual checkpoint、动作与特征契约、实际完成的训练进度 |
| `model.pth` | | 仅模型 tensor，供部署加载 |
| `resume.pth` | | 可信本地模型／优化器／normalizer／scheduler／RNG／sampler／课程状态 |
| `resolved_config.yaml`, `ppo_config.json`, `source.json`, `training.json` | | 实际配方（含 `control_dt_s`、`physics_dt_s`）、源码 hash（含 `h2r_io/`）、依赖／GPU 版本与训练步数 |

此前格式的产物（面向 human／robot bundle、旧 task 和旧协议）一律不读取：加载时报出文件的格式名和当前读取的格式名，须重新 prepare 并重训。`training_revision=8` 表示训练自设物理步长并按 `task_id` 绑定任务；更早 revision 的 resume 被拒绝。

输出目录必须不存在。续训传 `--resume /old/output/resume.pth --output /new/output --epochs TOTAL`，其余影响学习的参数与输入身份保持一致。只在完整 PPO epoch 边界恢复，重新 reset 训练 worlds，记录 `world_state_restarted=true`，不承诺与不中断运行逐位一致。

仅扩容续训需显式加 `--allow-capacity-increase`，允许上述两个容量增加，仍严格校验 prepared／base 身份、环境数、seed、控制、物理和 PPO 配方；不允许减少容量、增删配方字段或跨训练 revision 恢复。既有的 epochs／保存／诊断周期例外保持不变。checkpoint 内嵌完整 recipe，其 hash 必须与 checkpoint 的绑定一致。恢复记录 `resume.json` 保存 checkpoint hash、原／新 binding、容量差异和恢复 epoch。原生 reward-best `training/imitator/nn/imitator.pth` 也包含完整恢复状态；`model.pth` 和 `epoch-N.pth` 只有模型参数，不可作为 resume。

例如将已确认需要增加的缓冲显式设为 256/2048，并提高保存频率：

```bash
"$MT_PY" -m maniptrans_h2r.pretrain train \
  --prepared "$PRETRAIN" --resume "$OLD_BASE/training/imitator/nn/imitator.pth" \
  --allow-capacity-increase \
  --contact-capacity 256 --constraint-capacity 2048 \
  --save-every 100 --output "$NEW_BASE" --device cuda:0
```

扩容增加显存需求，不修改名义物理参数或学习算法；具体值需结合 overflow 类型与 GPU 检查选择。碰撞算法或传感器的其他 overflow 不一定能由这两个参数消除。未指定 `--epochs` 时仍沿用原生停止预算。此恢复能力不改变 `training_revision` 或既有默认容量。

阶段 1 的 `--recover-epa` 启用 `epa-episode-skip-v1`。仅物理 step 中的纯 `EPA_HORIZON` 可恢复：恢复该 world 的子步前状态、隔离剩余子步，结束当前 episode，下次按原采样规则 reset；其他 worlds 完成原定物理步数。故障 transition 从 PPO loss、KL、输入／value 归一化统计中剔除，GAE 在该处截断，故障前最后一条有效 transition 从有效的故障前状态 value bootstrap。仿真截断不进入训练游戏成功／失败及奖励均值；此前有效轨迹仍可用于学习。这是显式的仿真故障处理适配，会改变遇到故障后的采样分布，不宣称等价于原生无故障训练，也不提高 EPA 容量。

恢复策略同时绑定固定保护阈值：一次控制步最多影响 `max(1, floor(worlds*0.001))` 个 worlds（8192 环境时为 8）；滚动 1000 个控制步内最多跳过 32 个 episodes。同一 world 在同一控制步的两个子步报错只计一个 episode。超过阈值、reset／初始化溢出、混合或未知 overflow、非有限状态和 CUDA 异常仍失败。阶段 2 和 bench 评估不启用该恢复。

`epa-recovery/events.jsonl` 记录完整受影响 world 列表、控制步、参考 window／clip／sample 和错误；对应 `event-N.pth` 保存子步前 `qpos/qvel/act/ctrl/time/qacc_warmstart`，`summary.json` 记录累计次数和滚动窗口。恢复计数随完整 checkpoint 续接。`frame/transitions` 仍是尝试的转移数，实际跳过数单独记录，不能将其当作有效训练样本。方法侧固定诊断也记录 EPA 截断，保留原窗口分母并按失败报告，有效覆盖不含故障步；这些诊断不是 bench 指标。

从未启用恢复的 checkpoint 续接时，`--recover-epa` 同时明确授权加入这一个版本化策略；旧 recipe hash、prepared／base 和其他配方仍严格核验，额外扩容仍需 `--allow-capacity-increase`。`resolved_config.yaml` 绑定 `epa_recovery`，`resume.json` 记录 `recovery_changes`。已启用恢复的 checkpoint 须保持同一策略，不能静默关闭或修改阈值。例如在保留原运行参数的续训命令上增加 `--recover-epa`，并使用新的输出目录；默认原生停止条件不变。

H2R 入口在原生 residual 的 `nan_to_num` 之前检查网络输出和 Gaussian 标准差，并检查梯度、更新后参数、恢复和保存的模型状态。非有限训练结果产生 `failure.json` 并停止，不生成成功 deployment；原生 IsaacGym 默认入口不启用这些检查。

## 统一评估与批次

Bench 按同一 task/spec 启动 `evaluate-server contact_sensing.include_robot_robot=true ...` 后连接（`declare-evaluation` 的 `--override` 只接受 `dr.*`、`physics_hz` 和 `initialization.*`，该观测设置只传给 `evaluate-server`）：

```bash
"$MT_PY" -m maniptrans_h2r.evaluate \
  --deployment /path/to/deployment --socket /path/to/eval.sock --device cpu
```

客户端使用 `h2r-evaluation-v2` 协议（`h2r_io.client`），不创建 simulator，只消费真实反馈并输出具名位置目标：

- 以 `task_id` 核对评估方运行的就是 deployment 所准备的任务；不再逐项比较 hello 与任务副本。
- 声明自己的 `control_dt_s = 1/60`，每控制周期的物理步数由 `hello["dt_s"]` 换算；不能整除时在 start 前报错。
- 关节与物体关节按名称从 hello 的顺序映射到方法顺序；请求所跟踪的手部 body、全部物体部件 body 和接触观测。
- 要求 `hello.observation.contact_coverage == "robot-all-v1"`，否则拒绝并提示所需的 `evaluate-server` 设置。
- binding 的方法身份为 `maniptrans-mjlab`，`controller` 上报 deployment 身份、checkpoint 路径与 sha256、训练 seed 及实际完成的 epoch／transition 数。
- 参考按秒采样：观测当前时刻，跟踪一个控制周期之后的参考（截断在任务终点）。

推理保留随机 base／residual mean，每个 episode 使用 bench 的 controller seed。物理步长、初始化准入、完整时域、sim/sim-dr、MuJoCo/MJWarp 和评分分母由 bench 掌握。

整个 suite 用 `maniptrans_h2r.suite` 按分发顺序逐个 case 准备并训练：

```bash
"$MT_PY" -m maniptrans_h2r.suite --suite /path/to/suite --output /path/to/output \
  --imitator /path/to/base-right /path/to/base-left /path/to/taco-base-right /path/to/taco-base-left \
  --device cuda:0 [--case CASE_ID ...] [--iterations N] [-- 传给 maniptrans_h2r.train 的选项]
```

每个 case 的产物在 `OUTPUT/<name>/prepared` 与 `OUTPUT/<name>/deployment`，`<name>` 为 `case_id` 中 `/` 换成 `-`、`+` 换成 `_`（与 `h2r-bench declare-evaluation --suite` 的命名相同）。各手的 base 按侧别和手部物理签名在给出的 base 中选择，没有匹配时该 case 记为失败并说明缺哪一侧。已完成的步骤跳过；每个 case 的结果写入 `OUTPUT/<name>/result.json`，失败的 case 不中断其后的 case，存在失败时返回非零。`--case` 互不重叠的多次运行可以共用同一个 `OUTPUT`，例如按 GPU 分片。脚本不启动评估，也不把流程返回码当成任务 SR。

## 验证与兼容

```bash
"$MT_PY" -m pytest tests/h2r -q
"$MT_PY" -m maniptrans_h2r.check \
  --deployment /path/to/deployment --output /path/to/check.json --device cuda:0
```

`check` 在方法的私有训练世界中检查一个 deployment：只 reset 部分 world 时其余 world 不变、一秒任务时间对应 60 次控制和 120 个训练物理步、任务末端只执行剩余的一个物理步、功率取积分后的速度，以及 GPU 参数等于 task 的封存值（步长为训练步长）。它不连接 bench，也不比较训练物理与 300 Hz 评估物理的逐步轨迹。单元测试用加载器的封存函数生成小型 task（`tests/h2r/synthetic.py`），覆盖双手、仅右手、仅左手的映射、训练模型的步长、v2 握手、接触覆盖与旧产物的拒绝；另含课程原生 helper／reset 调度对照、8/9 与 20/21 门限、完整序列 IK 分块与停止条件。`H2R_IMITATOR_PREPARED`（部分 corpus 另加 `H2R_IMITATOR_ALLOW_PARTIAL=1`）启用 CUDA 环境检查；`H2R_NATIVE_RL_SOURCE` 指向只读上游 lib/rl 时，还对照原始 forward 与连续 PPO 更新。

当前参考／物理签名为 `hand-reference-v3` 和 `maniptrans-hand-physics-v3`，包括具名几何、关节、执行器、封存的求解器选项和方法的训练步长；未覆盖的手部 tendon、equality 或显式 contact pair 明确拒绝。阶段 1 prepared 的窗口规则为 `wrist-feasible-v1`。训练采用 `training_revision=8`，绑定 v3 控制配方、受限腕 RSI 和最终诊断进度。没有对此前格式的兼容读取：旧 prepared、base、deployment、resume 和按数据集取指尖的 corpus 都须重新生成；原生 wrench checkpoint 与位置控制版本不兼容。启动时须明确选择当前源码目录；已有运行的冻结 worktree 不会随主 checkout 更新。

运行验证记录保存在 bench 的 `runs/checks/`，按各记录中的源码版本和控制配方解释结果。短跑不代表全集覆盖、完整预算收敛或论文成绩；其他机器人与未实际运行的批次组合不据此宣称已验证。
