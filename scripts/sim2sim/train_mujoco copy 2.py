# train_mujoco.py

import os
import numpy as np
import torch
import torch.nn as nn
from typing import Dict
from berkeley_humanoid_lite.environments import MujocoSimulator, Cfg

# -------------------------------
# Actor-Critic (IsaacLab 형태 맞춤)
# -------------------------------
class ActorCritic(nn.Module):
    # IsaacLab 로그에 맞춰 256-128-128 3층으로 맞춤 (중요!)
    def __init__(self, obsDim: int, actDim: int, hiddenDims=(256, 128, 128)):
        super().__init__()
        self.obsDim = obsDim
        self.actDim = actDim

        layers = []
        inDim = obsDim
        for h in hiddenDims:
            layers += [nn.Linear(inDim, h), nn.ELU()]
            inDim = h
        self.backbone = nn.Sequential(*layers)

        self.policyHead = nn.Linear(inDim, actDim)
        self.valueHead = nn.Linear(inDim, 1)

        # log std를 학습 파라미터로 (IsaacLab PPO와 동일한 컨벤션)
        self.logStd = nn.Parameter(torch.zeros(actDim))

    def forward(self, obs: torch.Tensor):
        x = self.backbone(obs)
        mean = self.policyHead(x)            # 정책 평균(미스케일 상태)
        value = self.valueHead(x).squeeze(-1)
        return mean, value

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        mean, _ = self.forward(obs)
        std = torch.exp(self.logStd)
        dist = torch.distributions.Normal(mean, std)
        if deterministic:
            action = mean
        else:
            action = dist.sample()
        # 주의: tanh-squash를 쓰면 log_prob 보정이 필요하지만,
        # 여기서는 rollouts에선 mean을 tanh만 적용해 env에 넣고,
        # ratio 계산은 동일하게 유지(간단화를 위해)한다.
        logProb = dist.log_prob(action).sum(dim=-1)
        return action, logProb

    def evaluateActions(self, obs: torch.Tensor, actions: torch.Tensor):
        mean, value = self.forward(obs)
        std = torch.exp(self.logStd)
        dist = torch.distributions.Normal(mean, std)
        logProb = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return value, logProb, entropy


# -------------------------------
# IsaacLab ↔ Policy 어댑터
# -------------------------------
class ObsActionAdapter:
    """play 파이프라인과 1:1로 맞춘 관측/행동 어댑터."""
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.k = cfg.num_actions
        self.default = np.array(cfg.default_joint_positions, np.float32)[cfg.action_indices]
        self.prev_actions = np.zeros((self.k,), dtype=np.float32)
        self._g = np.array([0., 0., -1.], np.float32)

    @staticmethod
    def quat_rotate_inverse(q, v=np.array([0., 0., -1.], np.float32)):
        w, x, y, z = q
        qv = np.array([x, y, z], np.float32)
        a = v * (2.0 * w * w - 1.0)
        b = np.cross(qv, v) * (2.0 * w)
        c = qv * (np.dot(qv, v) * 2.0)
        return a - b + c

    def build_policy_obs(self, raw55: np.ndarray) -> np.ndarray:
        # raw55: [quat(4), ang(3), q(k), dq(k), mode(1), cmd(3)]
        quat = raw55[0:4]
        ang  = raw55[4:7]
        k    = self.k
        qpos = raw55[7:7+k]
        qvel = raw55[7+k:7+2*k]
        cmd  = raw55[7+2*k+1:7+2*k+4]  # mode 건너뜀

        proj_g = self.quat_rotate_inverse(quat, self._g)
        q_rel = (qpos - self.default).astype(np.float32)

        obs75 = np.concatenate(
            [cmd.astype(np.float32),
             ang.astype(np.float32),
             proj_g.astype(np.float32),
             q_rel,
             qvel.astype(np.float32),
             self.prev_actions],
            axis=0
        )
        return obs75

    def map_policy_action_to_abs_target(self, a_rel: np.ndarray) -> np.ndarray:
        # 정책 출력을 [-1,1]로 tanh 후, scale+default로 절대 관절 목표로 변환
        a = np.tanh(a_rel)
        abs_target = a * self.cfg.action_scale + self.default
        self.prev_actions = a.copy()
        return abs_target


# -------------------------------
# MuJoCo 래퍼 (보상/종료 정의 포함)
# -------------------------------
class BhlMujocoRLWrapper:
    """MujocoSimulator를 PPO용 (obs, reward, done, info) API로 감싼 래퍼."""
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.sim = MujocoSimulator(cfg)   # 내부에서 뷰어 시작, 키패드 스레드 등
        self.stepCount = 0
        self.maxEpisodeSteps = 1000

        first = self.sim.reset()
        self.rawDim = int(first.shape[0])     # 55
        self.obsDim = self.rawDim             # rollouts에는 raw를 쌓고, policy_obs는 adapter가 생성
        self.actDim = len(cfg.action_indices) # 22

    def reset(self):
        self.stepCount = 0
        o = self.sim.reset()
        return np.array(o.cpu().numpy() if isinstance(o, torch.Tensor) else o, dtype=np.float32)

    def step(self, abs_target: np.ndarray):
        """abs_target: 절대 조인트 목표(22,)"""
        self.stepCount += 1
        act_t = torch.from_numpy(abs_target).float()
        # print(act_t)
        # act_t = torch.tensor([0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 3, 3, 3, 3, 0, 0, 3, 3, 0, 3, 3, 3])
        # act_t[18] = -3
        print(act_t)
        next_raw = self.sim.step(act_t)
        if isinstance(next_raw, torch.Tensor):
            next_raw = next_raw.cpu().numpy()
        next_raw = np.array(next_raw, dtype=np.float32)

        # 리워드/종료(직립 + 안정성)
        w, x, y, z = next_raw[0:4]
        ang = next_raw[4:7]
        k = self.actDim
        dq = next_raw[7 + k:7 + 2 * k]

        up_z = 1.0 - 2.0 * (x * x + y * y)
        upright = np.clip(up_z, -1.0, 1.0)
        ang_pen = float(np.linalg.norm(ang) ** 2)
        dq_pen = float(np.linalg.norm(dq) ** 2)

        reward = 1.0 * upright - 0.01 * ang_pen - 0.001 * dq_pen
        done = bool(up_z < 0.0 or self.stepCount >= self.maxEpisodeSteps)
        info = {"up_z": up_z, "ang_norm2": ang_pen, "dq_norm2": dq_pen}
        return next_raw, float(reward), done, info


# -------------------------------
# Rollout (obs/reward/done + logprob/value)
# -------------------------------
@torch.no_grad()
def collectRollout(env: BhlMujocoRLWrapper,
                   model: nn.Module,
                   device: torch.device,
                   adapter: ObsActionAdapter,
                   numSteps=1024,
                   gamma=0.99,
                   gaeLambda=0.95):
    rawList, polObsList, actList, oldLogProbList, rewList, doneList, valList = [], [], [], [], [], [], []

    raw = env.reset()                              # 55
    pol = adapter.build_policy_obs(raw)            # 75
    pol_t = torch.from_numpy(pol).float().to(device)

    for _ in range(numSteps):
        # 정책 출력
        mean, _ = model(pol_t.unsqueeze(0))        # (1,22)
        # tanh 후 env로 보낼 절대 타겟 생성
        a_rel = mean.squeeze(0).cpu().numpy()
        abs_target = adapter.map_policy_action_to_abs_target(a_rel)

        # logProb/value는 ratio/GAE 계산용으로 저장
        std = torch.exp(model.logStd)
        dist = torch.distributions.Normal(mean, std)
        logProb = dist.log_prob(mean).sum(dim=-1)  # 단순화: mean 기준
        value, _, _ = model.evaluateActions(pol_t.unsqueeze(0), mean)

        # print(f"action : {abs_target}")
        # abs_target = np.zeros(22)
        # print(f"action : {abs_target}")

        next_raw, reward, done, _ = env.step(abs_target)
        next_pol = adapter.build_policy_obs(next_raw)

        # 버퍼 적재 (policy 관측/행동 저장)
        rawList.append(raw)
        polObsList.append(pol)
        actList.append(a_rel)
        oldLogProbList.append(logProb.item())
        rewList.append(reward)
        doneList.append(done)
        valList.append(value.item())

        raw = env.reset() if done else next_raw
        pol = adapter.build_policy_obs(raw)
        pol_t = torch.from_numpy(pol).float().to(device)

    # 텐서화
    polObs = torch.tensor(np.array(polObsList), dtype=torch.float32, device=device)  # (T,75)
    actions = torch.tensor(np.array(actList), dtype=torch.float32, device=device)    # (T,22)
    oldLogP = torch.tensor(np.array(oldLogProbList), dtype=torch.float32, device=device)
    rewards = torch.tensor(np.array(rewList), dtype=torch.float32, device=device)
    dones   = torch.tensor(np.array(doneList), dtype=torch.float32, device=device)
    values  = torch.tensor(np.array(valList), dtype=torch.float32, device=device)

    # GAE
    lastValue = model.forward(pol_t.unsqueeze(0))[1].item()
    gae = 0.0
    returns = []
    for i in reversed(range(numSteps)):
        nextNonTerm = 1.0 - dones[i] if i == numSteps - 1 else 1.0 - dones[i+1]
        nextValue   = lastValue if i == numSteps - 1 else values[i+1]
        delta = rewards[i] + gamma * nextValue * nextNonTerm - values[i]
        gae = delta + gamma * gaeLambda * nextNonTerm * gae
        returns.insert(0, gae + values[i])
    returns = torch.stack(returns).to(device)

    return {
        "obs": polObs,              # policy 입력(75)
        "actions": actions,         # 상대행동(22, tanh 전 mean)
        "oldLogProbs": oldLogP,
        "returns": returns,
        "values": values,
    }


# -------------------------------
# PPO Update
# -------------------------------
def ppoUpdate(model, optimizer, batch: Dict[str, torch.Tensor],
              clipEps=0.2, valueCoef=0.5, entropyCoef=0.0,
              ppoEpochs=5, miniBatchSize=256, maxGradNorm=0.5):

    obs = batch["obs"]
    actions = batch["actions"]
    oldLogProbs = batch["oldLogProbs"]
    returns = batch["returns"]
    values = batch["values"]

    advantages = returns - values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    batchSize = obs.size(0)
    for _ in range(ppoEpochs):
        idxs = np.arange(batchSize)
        np.random.shuffle(idxs)
        for start in range(0, batchSize, miniBatchSize):
            mb = idxs[start:start + miniBatchSize]
            mbObs = obs[mb]
            mbAct = actions[mb]
            mbOld = oldLogProbs[mb]
            mbRet = returns[mb]
            mbAdv = advantages[mb]

            valuesPred, newLogProbs, entropy = model.evaluateActions(mbObs, mbAct)
            ratio = torch.exp(newLogProbs - mbOld)
            surr1 = ratio * mbAdv
            surr2 = torch.clamp(ratio, 1.0 - clipEps, 1.0 + clipEps) * mbAdv

            policyLoss = -torch.min(surr1, surr2).mean()
            valueLoss = (mbRet - valuesPred).pow(2).mean()
            loss = policyLoss + valueCoef * valueLoss - entropyCoef * entropy.mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), maxGradNorm)
            optimizer.step()


# -------------------------------
# Checkpoint Loader
# -------------------------------
def loadPretrained(model: nn.Module, ckptPath: str, device: torch.device):
    ckpt = torch.load(ckptPath, map_location=device)
    candidates = [
        "model_state_dict", "actor_critic", "state_dict"
    ]
    loaded = False
    for k in candidates:
        if k in ckpt:
            missing, unexpected = model.load_state_dict(ckpt[k], strict=False)
            print(f"[load] missing={len(missing)}, unexpected={len(unexpected)}")
            loaded = True
            break
    if not loaded:
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        print(f"[load] missing={len(missing)}, unexpected={len(unexpected)}")
    return model


# -------------------------------
# Main
# -------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Cfg.from_arguments()

    env = BhlMujocoRLWrapper(cfg)
    adapter = ObsActionAdapter(cfg)

    obsDim = 75                 # 정책 입력 차원(고정)
    actDim = cfg.num_actions    # 22
    model = ActorCritic(obsDim, actDim).to(device)
    model.train()

    # IsaacLab PPO 체크포인트 로드(배우만 맞춰짐/비슷하게 매핑됨)
    model = loadPretrained(model, "./logs/rsl_rl/humanoid/2025-11-03_09-51-40/model_282199.pt", device)

    # 약간의 탐험을 주고 싶다면 초기 logStd를 낮게 세팅 (예: -2.0)
    with torch.no_grad():
        model.logStd.fill_(-2.0)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    os.makedirs("./checkpoints", exist_ok=True)

    numIters = 10
    for it in range(numIters):
        batch = collectRollout(env, model, device, adapter, numSteps=1024)
        ppoUpdate(model, optimizer, batch)
        print(f"[MuJoCo] iter {it+1}/{numIters} | returns mean={batch['returns'].mean().item():.3f}")

        torch.save({"model_state_dict": model.state_dict()},
                   f"./checkpoints/finetune_mujoco_{it+1:03d}.pt")

if __name__ == "__main__":
    main()
