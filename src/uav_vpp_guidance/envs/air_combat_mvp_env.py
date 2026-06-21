"""空战 MVP 环境（AirCombatMVPEnv）— air-combat-mvp 特性。

本模块在现有 ``CloseRangeTrackingEnv`` 之上，构建空战决策问题的最小可行版本
（MVP）第一阶段：单向击杀闭环（探测→锁定→发射→PNG 制导→命中/失效）。

坐标系与单位约定（与本特性所有新建模块一致）：
    JSBSim 坐标系 X=North（北）、Y=Up（上）、Z=East（东），单位为米/秒/弧度。
    状态向量约定 pos = [north, up, east]、vel = [v_north, v_up, v_east]。

实现进度：
    - 任务 11（本文件）：核对父类 ``CloseRangeTrackingEnv.step`` 结构并记录所选
      编排方案（见下方“父类 step 结构核对结论与编排方案决策”注释块）。
    - 任务 12 / 13：实现 ``AirCombatMVPEnv`` 的构造、实体管理、观测扩展与
      step 八步编排、终止/事件检测（当前文件仅含决策记录与类占位骨架）。
"""

# =============================================================================
# 父类 CloseRangeTrackingEnv.step 结构核对结论与编排方案决策（任务 11，关键决策点）
# 需求：3.1, 3.5 ；设计：§3.3.4（“super().step 调用时序风险”/“备选方案”/“任务阶段的决策点”）
#
# ---------------------------------------------------------------------------
# A) 父类 step 结构核对结论（已逐行核对 tracking_env.py）
# ---------------------------------------------------------------------------
# 1. CloseRangeTrackingEnv.step(action, command_override=None) 是一个**单体（monolithic）**
#    方法（约 tracking_env.py 第 497–885 行）。其内部线性流程为：
#       - 递增 self.current_step，推进仿真时间 self._sim_time_s；
#       - 读取当前状态、计算相对几何 rel_state；
#       - （可选）更新轨迹预测器；
#       - 生成虚拟追踪点 / 制导指令（VPP / LOS-rate / PN / 直接指令），
#         经 CBF 滤波、限幅、滤波、作动器动力学处理；
#       - 第 7 步：推进后端动力学（_step_jsbsim 或 _step_simple）；
#       - 第 8 步：读取 **post-step** 状态 own_state_post / target_state_post / rel_state_post；
#       - 第 9 步：调用 self._check_done(own_post, target_post, rel_post)（**多态分派**）；
#       - 第 10 步：调用 self._compute_reward(own_post, target_post, rel_post,
#         filtered_command, term_info)（**多态分派**，内部注入 terminal_reward）；
#       - 第 11 步：self._get_observation()；
#       - 返回 (obs, reward, terminated, truncated, info)。
#
# 2. 父类 _check_done(own, target, rel) 仅委托 self.termination_checker.check(
#    own, target, rel, self.current_step)，**只读取几何/相对状态（距离、当前步数）**，
#    **不读取任何导弹状态**。
#
# 3. 父类 _compute_reward(own, target, rel, command, term_info) 使用
#    self.reward_calculator.compute(info) 并按 term_info 注入终端奖励，
#    **不依赖任何导弹状态**。
#
# 4. 父类 step 在 terminated/truncated 之后**不执行任何清理/复位**
#    （current_step、各状态机的复位仅发生在 reset() 中）。因此“调用 super().step()
#    后在子类中重新计算终止/奖励”**不会造成重复清理**。
#
# ---------------------------------------------------------------------------
# B) 所选编排方案：方案 A（直接调用 super().step() 后编排），编排方法重命名
# ---------------------------------------------------------------------------
# 采用设计 §3.3.4 的 **方案 A**：AirCombatMVPEnv.step(action) 先调用
# super().step(action) 完成“本机机动 + 目标机动 + 基础飞行/制导动力学推进 + 基础记账”，
# 随后在子类中编排空战各步：
#     更新雷达 → 检查发射 → 导弹 PNG 制导 step → 命中/失效检查 →
#     空战终止判定 → 稀疏奖励计算。
#
# 该方案安全的理由（消除设计 §3.3.4 标注的“super().step 调用时序风险”）：
#   - super().step() 内部多态调用的 _check_done / _compute_reward **只读取几何与
#     reward_calculator**、**不读取导弹状态**，因此不存在“读到尚未更新的导弹状态”
#     而得出错误终止/奖励结论的时序错位风险（设计所担心的前提在此父类实现下不成立）；
#   - Missile3DoF.check_hit 只需“当前相对距离”，无需导弹先行积分，命中判定不受
#     super().step 与子类第 ④ 步 missile.step(dt) 的先后顺序影响；
#   - 父类 step 在终止时**不做清理**，故子类在 super().step() 返回后重新计算
#     terminated/truncated/reward 是安全的、幂等的（不会重复复位 current_step 或状态机）。
#
# ---------------------------------------------------------------------------
# C) 重要约束：禁止以不兼容签名覆盖父类的 _check_done / _compute_reward
# ---------------------------------------------------------------------------
# 由于父类 step 在其内部**多态调用** self._check_done(own, target, rel) 与
# self._compute_reward(own, target, rel, command, term_info)，若子类以不同签名
# （例如设计 §3.3.5 的 _check_done(own, target, hit, expired)）覆盖它们，会在
# super().step() 执行期间因签名不匹配而崩溃，或产生语义错位。
#
# 因此，空战的终止/事件/奖励编排一律使用**与父类不冲突的、独立命名**的方法，
# 不遮蔽（shadow）父类的 _check_done / _compute_reward：
#     - _check_air_combat_done(own, target, hit, expired)  —— 空战终止判定
#     - _detect_events(own, target, radar_state, launched, hit, expired)  —— 事件检测
#     - _compute_sparse_reward(events, term_info)  —— 稀疏奖励计算
# 当 use_sparse=True 时，super().step() 返回的 base reward/terminated/truncated 被**丢弃**，
# 由子类用上述独立方法重新计算最终的 reward/terminated/truncated。
#
# 说明（相对设计 §3.3.5 的“合理偏差”）：设计 §3.3.5 以 _check_done(own,target,hit,expired)
# 命名空战终止方法；本实现将其重命名为 _check_air_combat_done(...) 以避免与父类多态调用的
# _check_done 产生签名冲突。这是一处**有充分理由的最小偏差**，与设计 §3.3.4
# “任务阶段的决策点”所要求的“据父类 step 实际结构选择并精确控制调用时机”相一致。
# =============================================================================

import numpy as np

from ..sensors.radar import FireControlRadar
from ..weapons.missile import Missile3DoF, _extract_position
from .sparse_reward_air_combat import AirCombatEvent, AirCombatSparseReward
from .tracking_env import CloseRangeTrackingEnv


class AirCombatMVPEnv(CloseRangeTrackingEnv):
    """空战 MVP 环境（第一阶段：单向击杀闭环）。

    继承 ``CloseRangeTrackingEnv``，在保留飞行动力学与制导链路的前提下引入
    本机导弹（Missile3DoF）与火控雷达（FireControlRadar），并以事件稀疏奖励
    （AirCombatSparseReward）训练/评估空战决策。

    具体实现见任务 12（构造/实体管理/观测扩展）与任务 13（step 八步编排/终止/事件）。

    任务 12（本提交）：实现 __init__（实体管理）、reset 复位扩展、
    _get_observation 尾部 6 维空战特征拼接。
    任务 13（后续）：step 八步编排、终止/事件检测。
    """

    # 空战观测特征维度（尾部追加到父类 observation_vector，见设计 §3.3.3 / §5.1）。
    _AIR_COMBAT_FEATURE_DIM = 6

    # time_to_impact 归一化分母（设计 §3.3.3：tti/20.0 裁剪到 [0,1]）。
    _TTI_NORM_DIVISOR = 20.0

    # 雷达状态默认值（reset 前 / 首帧观测的安全兜底）。
    _DEFAULT_RADAR_STATE = {
        "locked": False,
        "in_view": False,
        "azimuth_deg": 0.0,
        "elevation_deg": 0.0,
        "range_m": 0.0,
    }

    # ==================================================================
    # 坐标系适配器（NEU ↔ [North, Up, East]）—— air-combat-mvp 关键修正
    # ------------------------------------------------------------------
    # WHY（两模块使用不同的轴序约定，必须在环境边界处隔离转换）：
    #   - 宿主环境 ``CloseRangeTrackingEnv`` 的简化/JSBSim 后端
    #     （``_get_current_states`` 返回的状态）采用 **NEU** 轴序：
    #         position_m         = [north, east, up]   （Up = 索引 2）
    #         velocity_vector_mps = [v_north, v_east, v_up]
    #     （已确认：``SimplePointMassEnv`` 动力学以 ``vz = speed*sin(pitch)`` 作为
    #      第 3 分量，``altitude_m = pos[2]``。）
    #   - 本特性的 ``Missile3DoF`` 与 ``FireControlRadar`` 则按设计的
    #     **[north, up, east]** 轴序构建并通过单测验证：
    #         · 导弹重力作用于索引 1：``a_gravity = [0, -GRAVITY, 0]``；升力沿索引 1
    #           的 Up 轴正交化；
    #         · 雷达俯仰角用 ``rel_r[1]`` 作为 Up 分量。
    #   - 影响：距离（range）与 ATA 对轴序置换不变，故 ``can_launch`` 与 PNG
    #     叉乘的量值不受影响；但导弹重力/升力与雷达俯仰若直接作用在 NEU 状态上，
    #     会施加到**错误的轴**（导弹横向偏移而非下坠、俯仰门限错误）→ 物理不正确、
    #     命中率低下。
    #
    # 解决：在 ``AirCombatMVPEnv.step`` 边界处，将环境 NEU 状态转换为
    #   missile/radar 期望的 [north, up, east] 轴序后再传入；转换仅是交换索引 1↔2，
    #   因此其逆变换也是交换 1↔2（自反）。该隔离保留了经充分单测验证的
    #   missile/radar 内部实现，不改动其任何逻辑。
    #
    # 不变性说明：range/ATA 轴序不变 ⇒ ``can_launch`` 不受转换影响；统一在边界处
    #   转换可同时让 elevation/gravity/lift 落在正确的 Up 轴上。
    # ==================================================================

    @staticmethod
    def _neu_to_nue(vec3) -> np.ndarray:
        """将 NEU 向量 ``[n, e, u]`` 转换为 [north, up, east] 向量 ``[n, u, e]``。

        即交换索引 1（east）与索引 2（up）。该变换为自反（其逆为自身）。

        Args:
            vec3: 长度为 3 的 NEU 序列 / ndarray。

        Returns:
            np.ndarray: 形状 (3,) 的 [north, up, east] 向量。
        """
        v = np.asarray(vec3, dtype=float)
        return np.array([v[0], v[2], v[1]], dtype=float)

    @staticmethod
    def _nue_to_neu(vec3) -> np.ndarray:
        """将 [north, up, east] 向量 ``[n, u, e]`` 转换回 NEU 向量 ``[n, e, u]``。

        与 :meth:`_neu_to_nue` 相同（均为交换索引 1↔2），互为逆变换。

        Args:
            vec3: 长度为 3 的 [north, up, east] 序列 / ndarray。

        Returns:
            np.ndarray: 形状 (3,) 的 NEU 向量。
        """
        v = np.asarray(vec3, dtype=float)
        return np.array([v[0], v[2], v[1]], dtype=float)

    def _state_neu_to_nue(self, state: dict) -> dict:
        """将一个 NEU 状态字典转换为 missile/radar 期望的 [north, up, east] 状态。

        仅转换 missile/radar 实际读取的字段（位置与速度向量），其余键（如
        ``altitude_m``——它是与轴序无关的 Up 标量）原样保留，便于
        ``_is_out_of_bounds`` 等沿用显式高度字段。

        转换的键：

        - ``position_m``：NEU ``[n,e,u]`` → ``[n,u,e]``。
        - ``velocity_vector_mps``：NEU 速度向量 → ``[n,u,e]`` 速度向量。
        - ``velocity_mps``：仅当其为 3 元向量时转换（场景字典中可能为标量速率，
          此时保留不动）。

        Args:
            state: NEU 状态字典（来自 ``_get_current_states``）。

        Returns:
            dict: 新的 [north, up, east] 状态字典（浅拷贝 + 转换后的位置/速度）。
        """
        out = dict(state)
        pos = state.get("position_m")
        if pos is not None:
            out["position_m"] = self._neu_to_nue(pos)
        vel_vec = state.get("velocity_vector_mps")
        if vel_vec is not None:
            out["velocity_vector_mps"] = self._neu_to_nue(vel_vec)
        vel = state.get("velocity_mps")
        if vel is not None:
            arr = np.asarray(vel, dtype=float)
            if arr.shape == (3,):
                out["velocity_mps"] = self._neu_to_nue(arr)
        return out

    # ------------------------------------------------------------------
    # 任务 12.1：构造与实体管理（需求 3.1 / 3.2 / 3.3 / 3.12；设计 §3.3.4）
    # ------------------------------------------------------------------
    def __init__(self, config: dict):
        """构造空战 MVP 环境（需求 3.1 / 3.2 / 3.3 / 3.12）。

        在父类 ``CloseRangeTrackingEnv.__init__`` 之后追加空战实体：

        - ``missile_ego``：始终创建的本机 ``Missile3DoF``（优先使用，不实例化
          ``EngagementTracker``，需求 3.12）。
        - ``radar_ego``：始终创建的本机 ``FireControlRadar``。
        - ``missile_target`` / ``radar_target``：仅当
          ``air_combat.missile_config.target.enabled`` 为真时创建，否则为 ``None``
          （需求 3.3；第一阶段单向击杀，默认停用）。

        奖励：读取 ``air_combat.reward.use_sparse``（默认 True），为真时创建
        ``AirCombatSparseReward``，否则 ``_sparse_reward=None``（回退父类密集奖励）。

        同时初始化事件检测与发射包线状态缓存（``_prev_event_state``、
        ``_prev_in_envelope``）与最近一次雷达状态缓存（``_last_radar_state``）。

        Args:
            config: 完整环境与制导配置字典。
        """
        super().__init__(config)

        ac = config.get("air_combat", {})

        # 本机导弹与雷达：始终创建（需求 3.2 / 3.12）。
        missile_cfg = ac.get("missile_config", {})
        self.missile_ego = Missile3DoF(config=missile_cfg.get("ego", {}))
        self.radar_ego = FireControlRadar(config=ac.get("radar_config", {}))

        # 目标导弹/雷达：仅当配置启用才创建，否则为 None（需求 3.3）。
        tgt_cfg = missile_cfg.get("target", {})
        if tgt_cfg.get("enabled", False):
            self.missile_target = Missile3DoF(config=tgt_cfg)
            self.radar_target = FireControlRadar(config=ac.get("radar_config", {}))
        else:
            self.missile_target = None
            self.radar_target = None

        # 稀疏奖励接入（需求 4 / 7.2）。
        # 说明：reward.py 提供 create_reward_calculator(config) 工厂（任务 15），
        # 在 air_combat.reward.use_sparse=True 时返回 AirCombatSparseReward，
        # 否则返回 RewardCalculator。此处采用任务 15 认可的“直接持有稀疏奖励实例”
        # 等价方案——直接构造并持有 self._sparse_reward，无需经由工厂转发。
        reward_cfg = ac.get("reward", {})
        self._use_sparse = reward_cfg.get("use_sparse", True)
        self._sparse_reward = (
            AirCombatSparseReward(reward_cfg) if self._use_sparse else None
        )

        # 事件检测前一步状态缓存（跳变检测用，任务 13）。
        self._prev_event_state = {}
        # 发射包线状态（需求 3.13，任务 13）。
        self._prev_in_envelope = False
        # 最近一次雷达状态缓存（供观测 radar_state 字段；reset 前用默认值）。
        self._last_radar_state = dict(self._DEFAULT_RADAR_STATE)

        # 逐步事件轨迹累加器（供回合结束后的轨迹级重标 finalize() 使用）。
        # 每个元素为对应 step 触发的事件名称列表；reset 时清空。
        self._episode_step_events = []

        # ------------------------------------------------------------------
        # 缺陷 2 修复：发射决策动作维（可选，默认关闭以保持向后兼容）。
        # ------------------------------------------------------------------
        # air_combat.action.use_launch_action（默认 False）：
        #   - False（默认，第一阶段）：动作空间为 3 维 VPP 偏移，导弹发射由规则
        #     自动触发（锁定 + 包线 + 无在飞弹），与既有行为完全一致。
        #   - True（第二阶段）：动作空间扩展为 4 维 = [VPP 偏移(3) + 发射决策(1)]。
        #     第 4 维为连续值，经阈值 launch_action_threshold 二值化为"策略是否
        #     请求发射"。实际发射仍由**物理可行性**门控（锁定 + 无在飞弹 +
        #     can_launch），即策略学习"何时发射"，但不能违反物理约束发射。
        action_cfg = ac.get("action", {})
        self._use_launch_action = bool(action_cfg.get("use_launch_action", False))
        # 发射决策阈值：策略第 4 维动作 > 阈值 ⇒ 请求发射（动作归一化区间约 [-1,1]）。
        self._launch_action_threshold = float(
            action_cfg.get("launch_action_threshold", 0.0)
        )
        # 发射时机奖励塑形（可选）：对"在物理可行窗口内请求发射"给予小幅塑形奖励，
        # 缓解发射决策的稀疏性。默认 0.0（关闭，纯稀疏），仅在显式配置时生效。
        self._launch_shaping_reward = float(
            action_cfg.get("launch_shaping_reward", 0.0)
        )
        # 最近一步发射决策诊断缓存（供 info / 奖励塑形；step 前为默认值）。
        self._last_launch_feasible = False
        self._last_policy_wants_launch = False

    @property
    def action_dim(self) -> int:
        """策略动作维度：启用发射决策维时为 4（VPP 偏移 3 + 发射 1），否则 3。

        训练脚本可读取本属性配置策略网络输出维度（与 obs_dim 类似由环境暴露）。
        """
        return 4 if self._use_launch_action else 3

    # ------------------------------------------------------------------
    # 任务 12.2：reset 复位扩展（需求 3.1）
    # ------------------------------------------------------------------
    def reset(self, scenario=None, seed=None) -> dict:
        """复位环境并扩展复位空战实体与事件状态（需求 3.1）。

        流程：

        1. 调用父类 ``reset`` 完成基础复位（状态机、后端、观测构建器等）。
           父类 ``reset`` 末尾会调用 ``self._get_observation()``——由于本子类
           在 ``__init__`` 中已先创建 ``missile_ego`` / ``radar_ego`` /
           ``_last_radar_state``，且 ``_get_observation`` 自带防御性 guard，
           故无论调用时机如何均安全。
        2. 复位本机导弹与雷达：雷达调用 ``reset()``；导弹通过清零其飞行状态
           （``in_flight=False`` 并清零计时/计程/命中/失效/缓存）复位为
           "未发射/未在飞"。
        3. 复位目标导弹/雷达（若存在）。
        4. 复位事件检测与发射包线状态（``_prev_event_state={}``、
           ``_prev_in_envelope=False``）与 ``_last_radar_state``。
        5. 复位稀疏奖励（若存在，调用其 ``reset()``）。
        6. 返回一份**新的**观测（经由 ``self._get_observation()``，已包含尾部
           6 维空战特征）。

        说明（目标初始位置随机化）：目标按场景/配置随机化由**基类 reset 的场景
        采样**（``_reset_simple`` / ``_reset_jsbsim`` 配合 ``scenario`` 与域随机化）
        与 pilot 配置（任务 16）共同驱动；本方法不在此处重复实现场景级随机化，
        保持复位逻辑最小且不与基类职责重叠。

        Args:
            scenario: 可选场景对象/字典（透传给父类）。
            seed: 可选随机种子（透传给父类）。

        Returns:
            dict: 初始观测字典（含尾部 6 维空战特征与 radar_state/missile_state）。
        """
        # 1) 基础复位（父类 reset 末尾会调用 _get_observation，已被本类 guard 保护）。
        super().reset(scenario, seed)

        # 2) 复位本机雷达与导弹。
        self.radar_ego.reset()
        self._reset_missile(self.missile_ego)

        # 3) 复位目标导弹/雷达（若存在）。
        if self.radar_target is not None:
            self.radar_target.reset()
        if self.missile_target is not None:
            self._reset_missile(self.missile_target)

        # 4) 复位事件检测与发射包线状态、雷达状态缓存。
        self._prev_event_state = {}
        self._prev_in_envelope = False
        self._last_radar_state = dict(self._DEFAULT_RADAR_STATE)

        # 4b) 清空逐步事件轨迹累加器（供 finalize() 轨迹级重标）。
        self._episode_step_events = []
        # 4c) 复位发射决策诊断缓存。
        self._last_launch_feasible = False
        self._last_policy_wants_launch = False

        # 5) 复位稀疏奖励（若启用）。
        if self._sparse_reward is not None:
            self._sparse_reward.reset()

        # 6) 返回一份新的、包含空战特征的观测。
        return self._get_observation()

    @staticmethod
    def _reset_missile(missile: Missile3DoF) -> None:
        """将一枚 ``Missile3DoF`` 复位为"未发射/未在飞"的干净初始状态。

        直接清零飞行状态量（不重建实例，避免丢失 config 覆盖的常量），
        与 ``Missile3DoF.__init__`` 构造后、``launch`` 之前的初始状态一致。

        Args:
            missile: 待复位的导弹实例。
        """
        missile.position_m = np.zeros(3, dtype=float)
        missile.velocity_mps = np.zeros(3, dtype=float)
        missile.mass_kg = float(missile.INITIAL_MASS)
        missile.flight_time_s = 0.0
        missile.flight_distance_m = 0.0
        missile.in_flight = False
        missile.hit = False
        missile.expired = False
        missile._last_target_state = None

    # ------------------------------------------------------------------
    # 任务 12.3：观测尾部拼接 6 维空战特征（需求 3.4 / 7.1；设计 §3.3.3 / §5.1）
    # ------------------------------------------------------------------
    def _get_observation(self) -> dict:
        """构建观测并在 ``observation_vector`` 尾部拼接 6 维空战特征。

        在父类基础观测之上**非破坏性**地追加（不修改父类 / observation.py）：

        - ``observation_vector``：父类向量 + 6 维空战特征（``np.concatenate`` 后
          转 ``float32``）。特征顺序（设计 §3.3.3）：

          ====  ==========================  ====================================
          索引  字段                        取值
          ====  ==========================  ====================================
          +0    radar_locked                ``float(radar_ego.locked)`` ∈ {0,1}
          +1    radar_in_view               ``float(radar_ego.in_view)`` ∈ {0,1}
          +2    missile_in_flight           ``float(missile_ego.in_flight)`` ∈ {0,1}
          +3    missile_time_to_impact      ``clip(tti/20.0, 0, 1)``（999→1）
          +4    incoming                    ``0.0``（第一阶段恒 0）
          +5    incoming_distance           ``1.0``（第一阶段无来袭→归一化 1）
          ====  ==========================  ====================================

        - ``radar_state``：最近一次雷达状态字典（``self._last_radar_state``）。
        - ``missile_state``：本机导弹状态字典（见 :meth:`_missile_state_dict`）。

        **防御性 guard（父类初始化/复位调用时机安全）**：父类 ``reset`` 末尾会调用
        ``self._get_observation()``。虽然本子类 ``__init__`` 已在 ``super().__init__``
        之后创建 ``radar_ego`` / ``missile_ego``，但为保证无论父类构造/复位顺序如何
        都不崩溃，此处仅当 ``radar_ego`` 与 ``missile_ego`` 均已创建时才追加空战特征；
        否则原样返回父类观测（不追加），保证构造期安全。

        Returns:
            dict: 观测字典，含扩展后的 ``observation_vector`` 与
            ``radar_state`` / ``missile_state``（当空战实体可用时）。
        """
        obs = super()._get_observation()

        # 防御性 guard：空战实体尚未创建时（理论上不会发生，但保证构造期安全），
        # 原样返回父类观测，不追加空战特征。
        if getattr(self, "radar_ego", None) is None or \
                getattr(self, "missile_ego", None) is None:
            return obs

        features = self._build_air_combat_features()
        obs["observation_vector"] = np.concatenate(
            [obs["observation_vector"], features]
        ).astype(np.float32)
        obs["radar_state"] = self._last_radar_state
        obs["missile_state"] = self._missile_state_dict()
        return obs

    def _build_air_combat_features(self) -> np.ndarray:
        """构建 6 维空战观测特征（设计 §3.3.3，顺序敏感）。

        Returns:
            np.ndarray: 形状 (6,) 的 ``float32`` 特征向量。
        """
        radar_locked = float(self.radar_ego.locked)
        radar_in_view = float(self.radar_ego.in_view)
        missile_in_flight = float(self.missile_ego.in_flight)

        # time_to_impact 归一化：tti/20.0 裁剪到 [0,1]（999 → 1）。
        tti = float(self.missile_ego.time_to_impact)
        missile_tti = float(np.clip(tti / self._TTI_NORM_DIVISOR, 0.0, 1.0))

        # 来袭导弹：第一阶段目标不发射，incoming 恒 0、incoming_distance 归一化恒 1。
        incoming = 0.0
        incoming_distance = 1.0

        return np.array(
            [
                radar_locked,
                radar_in_view,
                missile_in_flight,
                missile_tti,
                incoming,
                incoming_distance,
            ],
            dtype=np.float32,
        )

    def _missile_state_dict(self) -> dict:
        """返回本机导弹的状态字典（设计 §4.1）。

        Returns:
            dict: 含 ``position_m`` / ``velocity_mps`` / ``mass_kg`` /
            ``flight_time_s`` / ``flight_distance_m`` / ``in_flight`` / ``hit`` /
            ``expired`` / ``time_to_impact``。
        """
        m = self.missile_ego
        return {
            "position_m": np.array(m.position_m, dtype=float, copy=True),
            "velocity_mps": np.array(m.velocity_mps, dtype=float, copy=True),
            "mass_kg": float(m.mass_kg),
            "flight_time_s": float(m.flight_time_s),
            "flight_distance_m": float(m.flight_distance_m),
            "in_flight": bool(m.in_flight),
            "hit": bool(m.hit),
            "expired": bool(m.expired),
            "time_to_impact": float(m.time_to_impact),
        }

    # ------------------------------------------------------------------
    # 任务 13.1：step 八步编排（需求 3.5 / 3.6 / 3.7；设计 §3.3.4 方案 A）
    # ------------------------------------------------------------------
    def step(self, action=None):
        """空战 MVP 环境步进——八步编排（设计 §3.3.4，方案 A）。

        采用本文件顶部决策记录的**方案 A**：先调用 ``super().step(action)`` 完成
        “① 本机机动 + ⑥ 目标机动 + 基础飞行/制导动力学推进”，随后在子类中编排
        雷达更新、发射检查、导弹 PNG 制导、命中/失效、终止判定与稀疏奖励计算。

        八步顺序（对齐需求 3.5）：

            ① 本机机动      —— ``super().step(action)`` 推进本机/目标动力学。
            ② 更新雷达      —— ``radar_ego.update(own, target)``，缓存到
                               ``_last_radar_state``。
            ③ 检查发射      —— 组合条件（需求 3.7）：``radar.locked AND
                               (not missile.in_flight) AND missile.can_launch``，
                               满足则 ``launch`` 并置 ``launched=True``。
            ④ 导弹 PNG 制导 —— 若在飞则 ``missile_ego.step(dt, target)``。
            ⑤ 命中/失效     —— ``hit = in_flight and check_hit(target)``；
                               ``expired = (not hit) and in_flight and is_expired()``
                               （expired 加 ``in_flight`` 守卫，避免导弹失效后每步
                               重复触发 missile_miss，见下方实现处注释）。
            ⑥ 目标机动      —— 已在 ``super().step`` 中由场景目标动力学推进
                               （第一阶段直线/简单机动，无需额外处理）。
            ⑦ 检查终止      —— ``_check_air_combat_done(own, target, hit, expired)``。
            ⑧ 计算奖励      —— ``_detect_events(...)`` + ``_compute_sparse_reward``
                               （``use_sparse=False`` 时回退父类 base reward）。

        说明（相对父类返回值的取舍）：当 ``use_sparse=True`` 时，``super().step``
        返回的 base reward / terminated / truncated 被**丢弃**，由本方法用空战独立
        方法重新计算最终的 reward / terminated / truncated（理由见文件顶部决策记录
        C 节：父类 step 在终止时不做清理，重算是安全且幂等的）。

        Args:
            action: 策略动作（透传给父类）。动作维度由 ``action_dim`` 属性决定：

                - **3 维（默认，规则发射）**：``[VPP 偏移 x, y, z]``。导弹发射由规则
                  自动触发（锁定 + 包线 + 无在飞弹），不占用动作维度（需求 3.6）。
                - **4 维（启用 ``air_combat.action.use_launch_action``，缺陷 2 修复）**：
                  ``[VPP 偏移 x, y, z, 发射决策]``。前 3 维透传父类作 VPP 偏移；
                  第 4 维经阈值 ``launch_action_threshold`` 二值化为"策略是否请求发射"。
                  实际发射 = 物理可行（锁定 + 无在飞弹 + can_launch）AND 策略请求发射，
                  即策略学习"何时发射"但不能违反物理约束发射。

        Returns:
            tuple: ``(obs, reward, terminated, truncated, info)``。``info`` 含
            ``events`` / ``termination_info`` / ``missile_in_flight`` /
            ``time_to_impact``，并合并父类 ``base_info``。
        """
        # ① 本机机动：复用父类制导/飞行链路推进本机与目标动力学。
        #    发射决策维（缺陷 2）：启用 use_launch_action 时动作为 4 维
        #    [VPP 偏移(3) + 发射决策(1)]；父类只消费前 3 维 VPP 偏移，故在此切片，
        #    并将第 4 维二值化为"策略是否请求发射"。未启用时动作为 3 维，
        #    policy_wants_launch 恒为 True（等价于纯规则发射，向后兼容）。
        if self._use_launch_action and action is not None:
            action_arr = np.asarray(action, dtype=np.float64)
            vpp_action = action_arr[:3]
            launch_decision_raw = (
                float(action_arr[3]) if action_arr.shape[0] >= 4 else 0.0
            )
            policy_wants_launch = launch_decision_raw > self._launch_action_threshold
        else:
            vpp_action = action
            policy_wants_launch = True

        obs, _base_reward, _term, _trunc, base_info = super().step(vpp_action)

        # super().step 已推进动力学；读取 post-step 当前状态（NEU 轴序）。
        own_neu, target_neu = self._get_current_states()
        dt = self.env_config.get("high_level_dt", 0.2)

        # 坐标系适配（关键修正）：将环境 NEU 状态转换为 missile/radar 期望的
        # [north, up, east] 轴序后再传入。range/ATA 轴序不变 ⇒ can_launch 不受影响，
        # 但 elevation/gravity/lift 会因此落在正确的 Up 轴上（见类顶部适配器注释）。
        # 终止/事件检测方法接收**已转换**的状态，保持其与父类多态调用解耦且
        # 轴序一致（_is_out_of_bounds 用 altitude_m 标量，轴序无关，仍正确）。
        own = self._state_neu_to_nue(own_neu)
        target = self._state_neu_to_nue(target_neu)

        # ② 更新雷达（缓存供观测 radar_state 字段使用）。
        radar_state = self.radar_ego.update(own, target)
        self._last_radar_state = radar_state

        # ③ 检查发射（环境层组合条件，需求 3.7）：
        #    物理可行性门控 = 锁定 AND 无在飞弹 AND 几何允许发射。
        #    缺陷 2：实际发射 = 物理可行 AND 策略请求发射（policy_wants_launch）。
        #    - 规则发射模式（默认）：policy_wants_launch 恒 True ⇒ 物理可行即发射。
        #    - 发射决策模式：策略学习"何时发射"，但不能违反物理约束发射
        #      （可行窗口外即使请求也不发射）。
        launch_feasible = (
            radar_state["locked"]
            and not self.missile_ego.in_flight
            and self.missile_ego.can_launch(own, target)
        )
        launched = False
        # 记录"策略在可行窗口内请求/放弃发射"以供奖励塑形与诊断。
        self._last_launch_feasible = bool(launch_feasible)
        self._last_policy_wants_launch = bool(policy_wants_launch)
        if launch_feasible and policy_wants_launch:
            self.missile_ego.launch(own)
            launched = True

        # ④ 导弹 PNG 制导步进（仅在飞时）。
        if self.missile_ego.in_flight:
            self.missile_ego.step(dt, target)

        # ⑤ 命中/失效检查。
        #    说明（相对任务 13.1 字面公式的最小修正）：任务字面给出
        #      hit     = missile.in_flight and check_hit(target)
        #      expired = (not hit) and missile.is_expired()
        #    其中 hit 以 in_flight 为前置守卫，但 expired 字面未守卫 in_flight。
        #    然而 Missile3DoF.is_expired() 在失效后**每步恒返回 True**（飞行时间/射程
        #    条件持续成立），若 expired 不守卫 in_flight，会在导弹失效后的每一步重复
        #    触发 missile_miss（每步重复 -1.0 奖励），与设计 §3.4.2 “本步 is_expired
        #    返回 True”（指**失效发生的那一步**，即跳变）的语义相悖。
        #    因此与 hit 对称地加上 in_flight 守卫：失效当步 in_flight 仍为 True，
        #    is_expired() 置 in_flight=False；后续步因短路（in_flight=False）不再误触
        #    missile_miss。此为修正字面公式所揭示 bug 的有充分理由的最小偏差。
        hit = self.missile_ego.in_flight and self.missile_ego.check_hit(target)
        expired = (
            (not hit)
            and self.missile_ego.in_flight
            and self.missile_ego.is_expired()
        )

        # ⑥ 目标机动：已在 super().step 中完成（第一阶段直线/简单机动）。

        # ⑦ 检查终止（空战独立终止判定，不遮蔽父类 _check_done）。
        terminated, truncated, term_info = self._check_air_combat_done(
            own, target, hit, expired
        )

        # ⑧ 计算奖励：事件检测 + 稀疏奖励（或回退父类 base reward）。
        events = self._detect_events(
            own, target, radar_state, launched, hit, expired
        )
        # 累加本步事件到回合轨迹（供回合结束后的 finalize() 轨迹级重标）。
        self._episode_step_events.append(list(events))
        if self._use_sparse:
            reward = self._compute_sparse_reward(events, term_info)
        else:
            reward = _base_reward

        # 发射时机奖励塑形（缺陷 2，可选）：仅当启用发射决策维且配置了非零塑形值时
        # 生效。对"在物理可行窗口内策略请求并成功发射"给予 +shaping；对"可行窗口内
        # 策略放弃发射"给予 -shaping（鼓励抓住可行发射窗口）。默认 shaping=0（关闭），
        # 保持纯稀疏奖励语义不变。
        if self._use_launch_action and self._launch_shaping_reward != 0.0:
            if self._last_launch_feasible:
                if launched:
                    reward += self._launch_shaping_reward
                elif not self.missile_ego.in_flight:
                    # 可行但策略放弃发射（且当前无在飞弹）→ 轻微惩罚错失窗口。
                    reward -= self._launch_shaping_reward

        obs = self._get_observation()
        info = {
            **base_info,
            "events": events,
            "termination_info": term_info,
            "missile_in_flight": self.missile_ego.in_flight,
            "time_to_impact": self.missile_ego.time_to_impact,
            "launch_feasible": self._last_launch_feasible,
            "policy_wants_launch": self._last_policy_wants_launch,
            "launched": launched,
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # 任务 13.2：空战终止条件（需求 3.8 / 3.9 / 3.10 / 3.11；设计 §3.3.5）
    # ------------------------------------------------------------------
    def _check_air_combat_done(self, own, target, hit, expired):
        """空战终止判定（设计 §3.3.5）。

        独立命名以避免遮蔽父类多态调用的 ``_check_done``（见文件顶部决策记录 C 节）。

        判定优先级：

        1. **命中成功**（需求 3.8）：``hit`` 为真 → terminated，
           ``reason="missile_hit"``、``is_success=True``。
        2. **被来袭导弹命中失败**（需求 3.9）：仅当 ``missile_target`` 存在且在飞
           且 ``check_hit(own)`` 为真 → terminated，``reason="being_hit"``、
           ``is_being_hit=True``（第一阶段 ``missile_target=None``，恒不触发）。
        3. **越界坠毁**（需求 3.11）：本机绝对位置越过场景边界 → terminated，
           ``reason="out_of_bounds"``、``is_out_of_bounds=True``。
        4. **超时**（需求 3.10）：``current_step >= max_steps`` → truncated，
           ``reason="timeout"``、``is_timeout=True``。
        5. 否则不终止。

        Args:
            own: 本机当前状态字典（含 ``position_m``）。
            target: 目标当前状态字典。
            hit: 本步本机导弹是否命中目标。
            expired: 本步本机导弹是否失效（命中优先，命中时 expired 必为 False）。

        Returns:
            tuple: ``(terminated, truncated, info)``。``info`` 含 ``reason`` 与
            各布尔标志（``is_success`` / ``is_being_hit`` / ``is_timeout`` /
            ``is_out_of_bounds``）。
        """
        info = {
            "reason": None,
            "is_success": False,
            "is_being_hit": False,
            "is_timeout": False,
            "is_out_of_bounds": False,
        }

        # 1) 命中成功（需求 3.8）。
        if hit:
            info["reason"] = "missile_hit"
            info["is_success"] = True
            return True, False, info

        # 2) 被来袭导弹命中（需求 3.9）；第一阶段 missile_target=None，不触发。
        if (
            self.missile_target is not None
            and self.missile_target.in_flight
            and self.missile_target.check_hit(own)
        ):
            info["reason"] = "being_hit"
            info["is_being_hit"] = True
            return True, False, info

        # 3) 越界坠毁（需求 3.11，本机绝对位置越过场景边界）。
        if self._is_out_of_bounds(own):
            info["reason"] = "out_of_bounds"
            info["is_out_of_bounds"] = True
            return True, False, info

        # 4) 超时（需求 3.10）。
        if self.current_step >= self.max_steps:
            info["reason"] = "timeout"
            info["is_timeout"] = True
            return False, True, info

        # 5) 不终止。
        return False, False, info

    def _is_out_of_bounds(self, own) -> bool:
        """判定本机绝对位置是否越过场景边界（需求 3.11；设计 §3.3.5）。

        越界条件（满足任一即越界）：

        - 高度（altitude）< ``min_altitude_m`` 或 > ``max_altitude_m``；
        - 到场景中心（原点，pilot 配置场景中心在原点）的水平距离
          ``sqrt(north^2 + east^2)`` > ``max_range_m``。

        边界值从配置 ``scenario.bounds`` 读取（``min_altitude_m`` /
        ``max_altitude_m`` / ``max_range_m``）。缺失时采用宽松默认值
        （``min_alt=0`` / ``max_alt=1e9`` / ``max_range=1e9``），避免在未配置边界时
        因默认值过紧而误触发终止。

        **坐标约定与跨约定鲁棒性（重要）**：设计 §3.3.5 以本特性约定的 JSBSim 坐标系
        ``position_m = [north, up, east]``（Up=索引 1）定义本判据；但宿主环境
        ``CloseRangeTrackingEnv`` 的简化/JSBSim 后端实际采用 **NEU**
        （``position_m = [north, east, up]``，Up=索引 2），并始终额外提供显式的
        ``altitude_m`` 标量字段。为同时兼容两种约定、避免因轴序错配而误判越界，
        本实现：

        1. **高度**优先取状态字典中的显式 ``altitude_m`` 字段（环境恒提供、语义无歧义）；
           缺失时回退到设计约定的 ``position_m[1]``（[north,up,east] 的 Up 轴）。
        2. **水平距离**用 ``sqrt(|pos|^2 - altitude^2)`` 计算——该式等于
           ``north^2 + east^2`` 且**与 Up 轴所在索引无关**（无论 NEU 还是 [north,up,east]
           均成立），从而对两种坐标约定都给出正确的水平距离。

        Args:
            own: 本机当前状态字典（含 ``position_m`` 与可选 ``altitude_m``）。

        Returns:
            bool: 是否越界（越过场景边界）。
        """
        pos = _extract_position(own)

        # 高度：优先用环境显式提供的 altitude_m（无轴序歧义），回退到设计约定 idx 1。
        altitude_field = own.get("altitude_m") if isinstance(own, dict) else None
        altitude = (
            float(altitude_field) if altitude_field is not None else float(pos[1])
        )

        # 水平距离：sqrt(|pos|^2 - altitude^2) == sqrt(north^2 + east^2)，
        # 与 Up 轴索引（NEU 的 idx2 或 [north,up,east] 的 idx1）无关。
        pos_norm_sq = float(np.dot(pos, pos))
        horizontal_dist = float(np.sqrt(max(0.0, pos_norm_sq - altitude * altitude)))

        bounds = self.config.get("scenario", {}).get("bounds", {})
        min_alt = bounds.get("min_altitude_m", 0.0)
        max_alt = bounds.get("max_altitude_m", 1e9)
        max_range = bounds.get("max_range_m", 1e9)

        if altitude < min_alt or altitude > max_alt:
            return True
        if horizontal_dist > max_range:
            return True
        return False

    # ------------------------------------------------------------------
    # 任务 13.3：发射包线判定与事件检测（需求 3.13 / 4.8；设计 §3.3.5 / §3.4.2）
    # ------------------------------------------------------------------
    def _is_in_launch_envelope(self, own, target) -> bool:
        """判定目标是否处于本机发射包线内（需求 3.13；设计 §3.3.5）。

        发射包线定义：目标相对本机距离 ∈ [500, 5000] m 且 |ATA| < 30°。

        **复用决策**：直接复用 ``self.missile_ego.can_launch(own, target)``。
        理由——设计 §3.3.5 对“发射包线”的几何定义与任务 3.1 ``can_launch`` 的几何
        判定（``LAUNCH_RANGE_MIN <= range <= LAUNCH_RANGE_MAX`` 且
        ``|ATA| < LAUNCH_ATA_MAX_DEG``）**完全一致**；``can_launch`` 仅做纯几何判定
        （不含雷达/在飞弹状态），正是发射包线谓词，故直接复用以保证两者定义恒等、
        避免几何逻辑重复实现导致的不一致。

        Args:
            own: 本机当前状态字典。
            target: 目标当前状态字典。

        Returns:
            bool: 目标是否在发射包线内。
        """
        return self.missile_ego.can_launch(own, target)

    def _detect_events(self, own, target, radar_state, launched, hit, expired):
        """基于相邻两步状态跳变（rising/falling edge）检测空战事件（设计 §3.4.2）。

        事件触发表（设计 §3.4.2）：

        ===================  ===================================================
        事件                 触发条件
        ===================  ===================================================
        radar_lock           locked: False → True
        radar_lock_lost      locked: True → False
        missile_launch       本步发生 launch（``launched=True``）
        missile_hit          本步 ``check_hit`` 返回 True（``hit=True``）
        missile_miss         本步 ``is_expired`` 返回 True 且未命中（``expired=True``）
        being_locked         目标雷达 locked: False→True（第一阶段恒不触发，跳过）
        being_hit            来袭导弹命中本机（第一阶段恒不触发，跳过）
        out_of_envelope      发射包线内 True → False（目标离开包线）
        ===================  ===================================================

        边界（首步）：``_prev_event_state`` 在 reset/init 为 ``{}``，
        ``_prev_in_envelope`` 为 ``False``，均以 ``.get(..., False)`` 安全取默认，
        保证首步跳变检测正确（prior 视为 False）。

        方法在返回前更新 ``_prev_event_state``（存 ``{"locked": radar_state["locked"]}``）
        与 ``_prev_in_envelope``，供下一步跳变检测使用。

        Args:
            own: 本机当前状态字典。
            target: 目标当前状态字典。
            radar_state: 本步雷达状态字典（含 ``locked``）。
            launched: 本步是否发生发射。
            hit: 本步本机导弹是否命中。
            expired: 本步本机导弹是否失效（未命中）。

        Returns:
            list[str]: 本步触发的事件名称列表（取值匹配 ``AirCombatEvent`` 成员值）。
        """
        events = []

        # 雷达锁定跳变（radar_lock / radar_lock_lost）。
        prev_locked = bool(self._prev_event_state.get("locked", False))
        curr_locked = bool(radar_state["locked"])
        if curr_locked and not prev_locked:
            events.append(AirCombatEvent.RADAR_LOCK.value)
        elif prev_locked and not curr_locked:
            events.append(AirCombatEvent.RADAR_LOCK_LOST.value)

        # 发射 / 命中 / 失效（本步布尔信号直接映射）。
        if launched:
            events.append(AirCombatEvent.MISSILE_LAUNCH.value)
        if hit:
            events.append(AirCombatEvent.MISSILE_HIT.value)
        elif expired:
            events.append(AirCombatEvent.MISSILE_MISS.value)

        # being_locked / being_hit：第一阶段（目标无雷达/导弹）恒不触发，跳过。

        # 脱离发射包线（out_of_envelope）：包线内 True → False。
        curr_in_envelope = self._is_in_launch_envelope(own, target)
        if self._prev_in_envelope and not curr_in_envelope:
            events.append(AirCombatEvent.OUT_OF_ENVELOPE.value)

        # 更新前一步状态缓存（供下一步跳变检测）。
        self._prev_event_state = {"locked": curr_locked}
        self._prev_in_envelope = curr_in_envelope

        return events

    def _compute_sparse_reward(self, events, term_info) -> float:
        """计算本步稀疏奖励（需求 4.8；设计 §3.4）。

        调用 ``AirCombatSparseReward.compute_step(events)`` 对本步触发的所有事件
        奖励求和，作为环境 step 的逐步即时奖励。

        说明（per-step vs 轨迹级重标）：``AirCombatSparseReward`` 另提供
        ``relabel_trajectory``（线性核 + 高斯核）做**回合结束后**的轨迹级奖励
        重分配（relabelling）；该重标作用于完整轨迹（post-episode），**不**在单步
        环境奖励中应用。因此 step 的逐步奖励统一使用 ``compute_step``。

        Args:
            events: 本步触发的事件名称列表。
            term_info: 终止信息字典（当前 ``compute_step`` 不直接使用，保留以备
                未来在 step 内注入终端奖励/扩展；轨迹级终端奖励由 relabel 处理）。

        Returns:
            float: 本步事件奖励之和。
        """
        return float(self._sparse_reward.compute_step(events))

    # ------------------------------------------------------------------
    # 轨迹级重标接口（缺陷 1 修复：暴露 relabel_trajectory 给训练流程）
    # ------------------------------------------------------------------
    def finalize(self, terminal_reward: float = 0.0):
        """回合结束后的轨迹级奖励重标（线性核 + 高斯核）。

        本方法把 ``AirCombatSparseReward.relabel_trajectory`` 暴露给训练流程，使
        逐步即时事件奖励（``compute_step``）可被替换为**轨迹级重标后**的逐步奖励
        序列：结局奖励用线性核在整条轨迹上均匀分配，事件奖励用高斯核围绕事件步
        反向衰减分配，两部分均总量守恒（设计 §3.4.3）。

        **使用场景与算法适配说明（缺陷 1）**：

        - on-policy 算法（如本仓库 ``train_*_ppo`` 使用的 PPO）没有回放缓冲，
          逐步奖励在 rollout 中即时消费，advantage 由 GAE 在时间维度上隐式回传，
          因此 PPO 训练**默认不调用**本方法（轨迹级信用分配由 GAE 承担）。
        - off-policy 算法（如 SAC）或需要显式轨迹级信用再分配的训练流程，可在
          回合结束后调用 ``finalize(terminal_reward)`` 获取重标奖励序列，
          回填到其样本/回放缓冲中，替换原始逐步奖励。

        即：本方法是为 off-policy / 显式重标流程**预留**的接口；其存在不改变
        PPO 的现有逐步奖励行为（``step`` 仍返回 ``compute_step`` 的即时奖励）。

        Args:
            terminal_reward: 本回合结局奖励（成功为正、失败为负），用于线性核
                均匀分配。默认 0.0（仅重标事件奖励）。

        Returns:
            np.ndarray | None: 长度等于本回合步数的重标逐步奖励数组；当未启用稀疏
            奖励（``_sparse_reward is None``）或本回合无步数时返回 ``None``。
        """
        if self._sparse_reward is None:
            return None
        episode_length = len(self._episode_step_events)
        if episode_length == 0:
            return None
        return self._sparse_reward.relabel_trajectory(
            step_events=self._episode_step_events,
            terminal_reward=terminal_reward,
            episode_length=episode_length,
        )
