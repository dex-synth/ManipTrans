# h2r_bench 接入

H2R 入口在 `maniptrans_h2r/`。它读取共同 preprocess／robot／task，使用 **MJWarp 训练 imitator 与 residual**，再通过常驻客户端消费 bench 的实际观测。Bench 独立推进世界和计分。

这是 `maniptrans-mjwarp / virtual-wrist-position-v1` 适配版本。保留原生 weighted-keypoint IK 目标、特征融合网络、两阶段 normalizer、冻结 imitator、残差 PPO 和奖励函数；腕部动作改为共同六关节位置目标。旧 wrench／PID checkpoint 不兼容。原生 IsaacGym 入口仍独立保留。

## 独立环境

在已有开发容器内创建方法自己的 Python 3.11 环境，不向 bench 或其他方法环境安装这些依赖：

```bash
cd /path/to/ManipTrans
MT_ENV=/path/to/independent-maniptrans-env
uv venv --python 3.11 "$MT_ENV"
uv pip sync --python "$MT_ENV/bin/python" requirements-h2r.lock
MANIPTRANS_PROFILE=h2r uv pip install --python "$MT_ENV/bin/python" --no-deps -e .
```

锁定环境包括 Torch 2.9、MuJoCo/MJWarp 3.11、Warp 1.16、rl-games 1.6.1。Gym 的可选 renderer 通过 `requirements-h2r-overrides.txt` 使用有 Python 3.11 wheel 的 pygame；方法不调用该 renderer。更新锁文件使用 `uv pip compile --python 3.11 requirements-h2r.txt --override requirements-h2r-overrides.txt -o requirements-h2r.lock`。`MANIPTRANS_PROFILE=h2r` 只影响安装入口；未设置时 `setup.py` 保留原生 Python 3.8 配方。

H2R 不依赖 IsaacGym、SMPL-X、MANO、PyTorch3D 或 Chamfer CUDA 扩展。128 点 BPS 使用与原 `bps_torch` grid-sphere 相同的确定性基点和最近采样点的**欧氏距离**；1000 个物体表面点固定种子为 0。独立验证核对了全部 128 个基点。

## 准备参考

先由 bench 准备包含 contacts 的共同 task。同一比较组使用相同任务；不能为方法调低物体质量、改变 Kp/Kd 或放宽初态穿透阈值。

```bash
"$MT_ENV/bin/python" -m maniptrans_h2r.prepare \
  --bundle /path/to/human-bundle --robot-bundle /path/to/robot-bundle \
  --task /path/to/task --output /path/to/prepared \
  --object right=tool --object left=target --iterations 2000 --device cuda:0
```

也可以只传自包含 `--task`，仍会验证任务身份和所有封存依赖。多部件任务必须显式给每只手选择奖励与特征绑定的 part；例如 ARCTIC 为 `right=top`、`left=bottom`，双手共持同一刚体则两手绑定同一个 part。每个计分 part 都必须被绑定；当前最多为每手一个主要 part。非计分交互物体、被动铰链、固定支撑和环境全部留在原物理场景中。未知机器人、非标量 hinge 关节或遗漏计分 part 会明确拒绝。

当前具名机器人映射覆盖共同 Sharpa Wave 右手与双手。完整验证覆盖双手；右手映射已实现，尚未进行独立的端到端验证。

准备过程先核对 URDF FK 与 task MJCF，再拟合完整机器人关节。参考保持 task 的时间范围；物体尺寸、世界坐标、根数量和参考时钟均不改写。产物包括：

| 文件 | 用途 |
| --- | --- |
| `prepared.json` | 输入身份、方法参考及文件 hash |
| `task/` | 自包含的原始封存 task |
| `reference.npz` | IK、真实时间导数、人手关键点、部件参考、BPS、指尖距离 |
| `robot_mapping.json` | 全部关节、tracking/contact body、每手主要 part 和名义质量／COM |
| `initialization.json` | 方法 IK 首帧按 task 穿透阈值的诊断 |

初态诊断失败会保留明确记录；训练不会替换物体初态，eval 最终由 bench 独立准入，失败仍保留在评估分母中。增加 IK 迭代属于方法求解，修改任务阈值不属于此入口。

## 两阶段训练

```bash
"$MT_ENV/bin/python" -m maniptrans_h2r.train \
  --prepared /path/to/prepared --output /path/to/deployment \
  --device cuda:0 --worlds 256 --epochs 10000 --horizon 32 --seed 0
```

程序依次训练每手 imitator，然后加载并冻结全部 imitator 和其 normalizer，训练单手或双手 residual。`--epochs` 是**每个阶段**的预算；缺少任一 base checkpoint 不会回退零动作。输出根目录必须不存在。

快速验证可显式使用 `--small-network --worlds 4 --epochs 2 --horizon 12 --mini-epochs 2`。`--small-network` 会缩小网络并记录为诊断变体，不能当成原生完整训练配方或任务成功证据。

默认完整网络使用原生 512 维、3 层 feature encoder 和 `[256,512,128,64]` MLP；PPO 由原 `lib/rl/agent.py`、`base.py` 执行。残差 minibatch 复用 rollout 时采样的 base action，使 PPO likelihood 的条件与采集时一致。任务末端是有限时域的真实 terminal，不用旧状态 value 做 timeout bootstrap；未结束的 PPO horizon 继续正常 bootstrap。原生奖励函数移至 `lib/rl/rewards.py`，函数体保持不变。

第一版采用 task 首帧、零速度 reset，不启用原生 RSI、课程学习或私有物理 DR。单手 imitator 阶段也保留完整共同机器人和全部物体，另一只手跟随方法 IK。原生失败 grace period 与 3 帧 contact history 按 60 Hz 换算为物理时间；同一 task 最后不足一个控制周期的区间只记录到其终点。一个 world 的 reset 保持其他 world 的 q/dq、warm start、接触与目标历史。

### 动作和物理

每手 base 与 residual 各输出 `6+n_finger`。player 的 sigmoid soft clamp 之后：

```text
wrist = IK_wrist(t_next) + scale * (base + 2 * residual)
finger = lower + (clamp(base + 2 * residual, -1, 1) + 1) / 2 * (upper - lower)
```

腕平移幅度为 0.05 m、旋转幅度为 0.3 rad；imitator 阶段没有 residual。这些是方法动作幅度，**不是 servo gain**。腕和手指目标共用 moving average；时间常数由原生 60 Hz 下 `alpha=0.4` 换算，30 Hz 下为 0.64，首个历史等于提交初态。完整关节与 actuator 按名称映射；机械限位不被当成无限 ctrlrange 的线性映射区间。

训练直接加载 task MJCF，只在内存中添加观测 sensor。`physics.py` 对照封存编译记录、原始 CPU 模型、sensor 注入模型和 GPU 实际数组，检查 Kp/Kd、gear、力／ctrl 限幅、被动参数、质量惯量、接触和 timestep。腕功率从各物理子步的实际广义执行器力乘以求解前速度获得，再在控制区间平均。没有额外腕 wrench、PID 或物体辅助力。

MJWarp 默认将 solver tolerance 下限设为 `1e-6`；训练显式恢复 task 的值。当前 bench MJWarp runtime 可能仍采用这个 float32 下限，因此不能声称所有数值选项或长轨迹完全等价。首区间的状态与力对照由 `check` 单独验证，物理审计保存实际值。

### 观测与产物

训练和部署共用 `observations.py`、`policy.py`、`control.py`。actor 使用真实关节、body 原点世界速度、物体部件与 hinge 状态；target 使用方法参考。接触只聚合协议允许的 robot–object／environment 子集，不冒充原生包括自碰撞的完整净力。每手 contact body 为五个 `*_DP`；不将无碰撞的 fingertip marker 当成 force sensor。mass／COM 特征始终采用共同名义参数，sim-dr 不读取真实随机参数。

`deployment.json` 封存模型配置、动作契约、参考身份、checkpoint 和全部文件 hash。根目录包含各 imitator、residual checkpoint、完整 normalizer、每阶段训练记录和 `*_physics.json`。在线只载入 tensor model state，不恢复训练 simulator 或 optimizer state。保留整个目录；base checkpoint 使用相对路径，可以整体搬迁。

## 统一评估

在 bench 环境启动已有 `evaluate-server`，给定同一个 task、metric spec、backend、mode、seeds 和 socket；再在方法环境启动：

```bash
"$MT_ENV/bin/python" -m maniptrans_h2r.evaluate \
  --deployment /path/to/deployment --socket /path/to/eval.sock \
  --device cpu --diagnostics /path/to/controller-diagnostics.json
```

客户端核对身份、时钟、关节单位、观测能力以及 contact solve tick；只为活跃 episode 输出全部具名位置目标。默认保留原生推理语义：base 随机采样、residual 取 mean。每个 episode 的 RNG 取 bench `controller_seed`，滤波历史独立，其他 episode 结束不影响它。CPU 推理不导入或创建 MuJoCo、MJWarp、Warp 或 IsaacGym runtime。

`sim`、`sim-dr` 的真实 rollout、终止、SR／ADD-AUC 和失败分母由 bench 的 `score-evaluation` 与汇总入口处理。客户端不持有评估世界，不能提交物体状态、reset 或提前成功。

## 验证

```bash
"$MT_ENV/bin/python" -m pytest tests/h2r -q
"$MT_ENV/bin/python" -m maniptrans_h2r.check \
  --deployment /path/to/deployment --run /path/to/nominal-mjwarp-rollout \
  --output /path/to/check.json --device cuda:0
```

`check` 验证 train/client 同状态、噪声和历史下的特征／目标一致性、masked reset、末端观测和缩短控制区间。可选 `--run` 回放 bench 首个控制区间的实际目标，对比关节与执行器力；它不把自己生成的训练轨迹交给 scorer。

本次本地检查使用 TACO `20231002_063`、Sharpa 双手、两个自由刚体、240 Hz 物理／30 Hz 控制，覆盖两阶段真实梯度更新、初态准入、CPU 无 simulator 推理和共同评估。运行证据保存在 bench 的 `runs/checks/maniptrans-h2r-20261108/`。完整预算训练的收敛与成功率、其他机器人、右手单独闭环及完整数据集覆盖均未验证；局部运行记录不扩大支持范围。
