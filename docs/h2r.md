# h2r_bench 接入

`maniptrans_h2r` 实现通用单手 imitator 预训练和逐任务 residual 训练。训练框架为 **mjlab 1.6.0 的公共 `Simulation`**，物理引擎为 MJWarp；控制 60 Hz、物理 120 Hz、decimation=2，默认 4096 个环境。原生 `lib/rl` 的 PPO、网络、normalizer 与奖励函数复用；环境直接组合 `Simulation`，自行管理 reward、RSI 和 terminal，避免引入 ManagerBasedRlEnv 的自动 reset 和 dt 奖励缩放。

本版本身份为 `maniptrans-mjlab / virtual-wrist-position-v2`。这是保留原生训练方法的共同任务适配：数据集、机器人、simulator 和腕控制均与论文环境不同，不能称为论文结果的精确复现。原生 IsaacGym 入口仍保留。方法代码不导入 bench；bench 独立生成任务、推进评估世界和计分。

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

共同双手 task 须预先以 `--physics-hz 120 --control-hz 60 --contacts` 封存。旧 task 不会被修改。两侧从相同 task 抽取独立单手模型，移除物体、另一只手和外部接触，只保留相关固定祖先。实际编译后的腕／手指增益、惯量、碰撞及限幅经审计后继承，dampratio 被物化为实际 affine actuator 参数。

```bash
for side in right left; do
  "$MT_PY" -m maniptrans_h2r.pretrain prepare \
    --corpus /path/to/hand-corpus --task /path/to/sealed-task \
    --mano-root /path/to/mano/models --side "$side" \
    --output "/path/to/pretrain-$side"
  "$MT_PY" -m maniptrans_h2r.pretrain train \
    --prepared "/path/to/pretrain-$side" --output "/path/to/base-$side" \
    --device cuda:0
 done
```

默认 10000 epochs、horizon 32、minibatch 1024、mini-epochs 5、seed 42。配置集中在 `config/imitator.yaml`。四数据集等概率、源左右侧等概率，90% 真实／10% MANO 旋转 SLERP 合成；异侧源镜像到目标手。真实序列全长切成至多 2000 帧且共享边界的窗口，按有效秒数抽样，CPU mmap 与每环境 GPU cache 按需取数，不预先把全集搬上显卡。

Reset 采用原生 `.99` RSI 法则、手指／腕噪声与参考速度。真实时间导数、连续腕 Euler 分支及 Jacobian 速度换算在方法侧处理；阶段 1 不做手指 IK。原生奖励与阈值课程保持，课程按控制调用计数，终止保留窗口最后一帧。

部分 corpus 必须显式使用 `--allow-incomplete`。功能检查可加 `--small-network --worlds 8 --epochs 2 --diagnostic-every 0`；即使跳过中间诊断，最终仍做均值和 RSI 随机策略诊断。小网络或部分数据产物标记 `smoke=true`，不能用于默认正式 residual 入口。

`pretrain check --imitator BASE --prepared PRETRAIN --output CHECK --device cuda:0` 单独执行固定诊断。报告名为 `training-corpus tracking diagnostic`，与训练数据重叠；严格 scale=0.7，报告 wrist、指尖／关节组、功率、有效前缀和时域覆盖。它不是 manipulation SR 或未见序列泛化结果。

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

4000 步 weighted-keypoint IK 为 RSI 生成候选，腕／手指 Adam 学习率分别为 .0008/.0004。全部 roots、hinges、双手在相同源 tick 初始化；碰撞、限位和接触候选检查失败会记录，合法起帧为空则拒绝训练。评估始终提交优化后的第 0 帧机器人姿态，速度由 bench 置零，不从 RSI 选择最好起帧。

`config/residual.yaml` 默认 4096 环境、10000 epochs、32/1024/5 PPO 配方；残差 init gain=.01、logstd=-1，保留原生 warmup 调用顺序。全部 base 参数和 normalizer buffers 冻结并在保存时核验；PPO 重算使用 rollout 时保存的 base 样本。物体质量、重力、接触、弹簧和 servo gains 在整个课程中保持 task 名义值。

## 控制、物理与产物

动作经原生 soft clamp 后，每手解码为：

```text
wrist = reference_wrist(t_next) + scale * (base + 2 * residual)
finger = lower + (clamp(base + 2 * residual, -1, 1) + 1) / 2 * (upper - lower)
```

平移 scale=.05 m、旋转 scale=.3 rad；阶段 1 没有 residual。v2 moving average=1，无额外平滑；目标再按启用的 actuator ctrlrange 和手指限位裁剪。腕的位置目标由统一物理执行器执行，没有额外 wrench、PID 或物体辅助力。

接触／约束缓冲容量分别在两份 YAML 中固定：单手 128/1024，完整场景 512/2048。容量不足明确报错，不截断接触或自动减少环境数。

每个控制区间只采样／解码一次动作，调用两次 `Simulation.step()`；任务末端剩一个 physics tick 时仅执行一次。实际最后一次 solve 的力与求解前速度构成功率；积分后仅刷新运动学，避免额外 full forward 覆盖接触求解结果。每指接触 sensor 精确匹配该 body 的碰撞 geoms，包含全部接触伙伴；在线状态使用 bench 真实 contact records。

训练核验封存 CPU 模型、sensor 模型和 GPU 实际参数，显式恢复 MJWarp 的 solver tolerance。Bench MJWarp runtime 可能保留 float32 的 `1e-6` 下限；相同物理参数不意味着长轨迹逐位等价。

| 产物 | 内容 |
| --- | --- |
| pretrain prepared | 单手场景、资产、兼容签名、motion mmap、窗口／诊断 manifest |
| `imitator.json` | 侧别、数据与物理身份、网络／normalizer、smoke 标识及文件 hash |
| residual prepared | 原封存 `task/`、`reference.npz`、`rsi.npz/json`、首帧 `initialization.json` |
| `deployment.json` | 任务身份、相对路径 base、residual checkpoint、动作与特征契约 |
| `model.pth` | 仅模型 tensor，供部署加载 |
| `resume.pth` | 可信本地模型／优化器／normalizer／scheduler／RNG／sampler／课程状态 |
| `resolved_config.yaml`, `source.json`, `training.json` | 实际配方、源码 hash、依赖／GPU 版本与训练步数 |

输出目录必须不存在。续训传 `--resume /old/output/resume.pth --output /new/output --epochs TOTAL`，其余影响学习的参数与输入身份保持一致。只在完整 PPO epoch 边界恢复，重新 reset 训练 worlds，记录 `world_state_restarted=true`，不承诺与不中断运行逐位一致。

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

`check` 按部署版本选择诊断：v2 检查 train/client 特征与动作、masked reset、60 次控制推进 120 ticks、末端单 tick、名义物理不变；`--run` 可增加同任务名义 bench 首区间对照。单元测试包含窗口覆盖、腕分支、采样质量、全物体 reward 和条件 base 采样。

旧版显式使用 `prepare --legacy-v1` 与 `train --train-local-imitators`；旧 `virtual-wrist-position-v1` deployment 仍可评估，保留其平滑和时钟。v1 的完整场景 imitator／逐示范训练不作为通用预训练。原生 wrench checkpoint 与两个位置控制版本都不兼容。

四真实数据集样例、短 PPO 与共同评估的具体结果保存在 bench `runs/checks/maniptrans-mjlab-20261108/`。短跑不代表全集覆盖、完整预算收敛或论文成绩；其他机器人与未实际运行的批次组合不据此宣称已验证。
