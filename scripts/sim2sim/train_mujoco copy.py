import os
import torch
import torch.nn as nn
import numpy as np

from berkeley_humanoid_lite.environments import MujocoSimulator, Cfg

# ===================== 1. 모델 =====================
class ActorCritic(nn.Module):
    def __init__(self, obsDim: int, actDim: int, hiddenDims=(256, 128)):
        super().__init__()
        layers = []
        inDim = obsDim
        for h in hiddenDims:
            layers.append(nn.Linear(inDim, h))
            layers.append(nn.ELU())
            inDim = h
        self.backbone = nn.Sequential(*layers)
        self.policyHead = nn.Linear(inDim, actDim)
        self.valueHead = nn.Linear(inDim, 1)
        self.logStd = nn.Parameter(torch.zeros(actDim))

    def forward(self, obs: torch.Tensor):
        x = self.backbone(obs)
        mean = self.policyHead(x)
        value = self.valueHead(x).squeeze(-1)
        return mean, value

    def act(self, obs: torch.Tensor):
        mean, _ = self.forward(obs)
        std = torch.exp(self.logStd)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        logProb = dist.log_prob(action).sum(dim=-1)
        return action, logProb, dist, mean, std

    def evaluateActions(self, obs: torch.Tensor, actions: torch.Tensor):
        mean, value = self.forward(obs)
        std = torch.exp(self.logStd)
        dist = torch.distributions.Normal(mean, std)
        logProb = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return value, logProb, entropy


def loadPretrained(model: nn.Module, ckptPath: str, device: torch.device):
    ckpt = torch.load(ckptPath, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    elif "actor_critic" in ckpt:
        model.load_state_dict(ckpt["actor_critic"], strict=False)
    elif "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    return model

# ===================== 2. MujocoSimulator → Gym 래퍼 =====================
class BhlMujocoRLWrapper:
    """네가 가진 MujocoSimulator를 PPO가 먹을 수 있게 감싼 래퍼."""

    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.sim = MujocoSimulator(cfg)   # 이 순간 뷰어도 같이 뜸
        self.stepCount = 0
        self.maxEpisodeSteps = 1000   # 일단 기본값, 바꿔도 됨

        # reset 한 번 해서 obs / act 차원 파악
        firstObs = self.sim.reset()   # torch.Tensor
        if isinstance(firstObs, torch.Tensor):
            firstObs = firstObs.cpu().numpy()
        self.obsDim = firstObs.shape[0]

        # 액션은 cfg.action_indices 길이만큼
        self.actDim = len(self.cfg.action_indices)

    def reset(self):
        self.stepCount = 0
        obs = self.sim.reset()
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        return np.array(obs, dtype=np.float32)

    def step(self, action: np.ndarray):
        """PPO가 기대하는 형식: (obs, reward, done, info)"""
        self.stepCount += 1

        # 1) 액션을 토치로 변환
        if isinstance(action, np.ndarray):
            action_t = torch.from_numpy(action).float()
        else:
            action_t = action.float()

        # 2) 시뮬레이터에 한 스텝
        nextObs = self.sim.step(action_t)
        if isinstance(nextObs, torch.Tensor):
            nextObs = nextObs.cpu().numpy()
        nextObs = np.array(nextObs, dtype=np.float32)

        # 3) 리워드/종료는 우리가 임시로 만든다
        #    - 지금 관찰에 base quat, ang vel, joint들이 다 들어오니까
        #      높이 같은 건 없고, "살아있다" 리워드만 우선 줘도 됨
        reward = 0.0

        # 에피소드 길이로만 종료
        done = self.stepCount >= self.maxEpisodeSteps

        info = {}
        return nextObs, reward, done, info


# ===================== 3. rollout / ppo =====================
def collectRollout(env, model, device, numSteps=1024, gamma=0.99, gaeLambda=0.95):
    obsList, actList, logProbList, rewardList, doneList, valueList = [], [], [], [], [], []
    obs = env.reset()
    obsTensor = torch.from_numpy(obs).float().to(device)

    for _ in range(numSteps):
        with torch.no_grad():
            action, logProb, _, _, _ = model.act(obsTensor.unsqueeze(0))
            value = model.evaluateActions(obsTensor.unsqueeze(0), action)[0]

        actionNp = action.squeeze(0).cpu().numpy()
        nextObs, reward, done, _ = env.step(actionNp)

        obsList.append(obs)
        actList.append(actionNp)
        logProbList.append(logProb.item())
        rewardList.append(reward)
        doneList.append(done)
        valueList.append(value.item())

        if done:
            obs = env.reset()
        else:
            obs = nextObs
        obsTensor = torch.from_numpy(obs).float().to(device)

    obsTensorAll = torch.tensor(np.array(obsList), dtype=torch.float32, device=device)
    actTensorAll = torch.tensor(np.array(actList), dtype=torch.float32, device=device)
    oldLogProbTensorAll = torch.tensor(np.array(logProbList), dtype=torch.float32, device=device)
    rewardTensorAll = torch.tensor(np.array(rewardList), dtype=torch.float32, device=device)
    doneTensorAll = torch.tensor(np.array(doneList), dtype=torch.float32, device=device)
    valueTensorAll = torch.tensor(np.array(valueList), dtype=torch.float32, device=device)

    with torch.no_grad():
        lastValue = model.forward(obsTensor.unsqueeze(0))[1].item()

    gae = 0.0
    returns = []
    for i in reversed(range(numSteps)):
        if i == numSteps - 1:
            nextNonTerminal = 1.0 - doneTensorAll[i]
            nextValue = lastValue
        else:
            nextNonTerminal = 1.0 - doneTensorAll[i + 1]
            nextValue = valueTensorAll[i + 1]
        delta = rewardTensorAll[i] + gamma * nextValue * nextNonTerminal - valueTensorAll[i]
        gae = delta + gamma * gaeLambda * nextNonTerminal * gae
        returns.insert(0, gae + valueTensorAll[i])

    return {
        "obs": obsTensorAll,
        "actions": actTensorAll,
        "oldLogProbs": oldLogProbTensorAll,
        "returns": torch.stack(returns).to(device),
        "values": valueTensorAll,
    }

def ppoUpdate(model, optimizer, batch,
              clipEps=0.2, valueCoef=0.5, entropyCoef=0.0,
              ppoEpochs=5, miniBatchSize=256, maxGradNorm=0.5):
    obs = batch["obs"]
    actions = batch["actions"]
    oldLogProbs = batch["oldLogProbs"]
    returns = batch["returns"]
    values = batch["values"]

    adv = returns - values
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    batchSize = obs.size(0)
    for _ in range(ppoEpochs):
        idxs = np.arange(batchSize)
        np.random.shuffle(idxs)
        for start in range(0, batchSize, miniBatchSize):
            end = start + miniBatchSize
            mb = idxs[start:end]

            mbObs = obs[mb]
            mbAct = actions[mb]
            mbOld = oldLogProbs[mb]
            mbRet = returns[mb]
            mbAdv = adv[mb]

            valuesPred, newLogProbs, entropy = model.evaluateActions(mbObs, mbAct)
            ratio = torch.exp(newLogProbs - mbOld)

            surr1 = ratio * mbAdv
            surr2 = torch.clamp(ratio, 1.0 - clipEps, 1.0 + clipEps) * mbAdv
            policyLoss = -torch.min(surr1, surr2).mean()
            valueLoss = (mbRet - valuesPred).pow(2).mean()
            loss = policyLoss + valueCoef * valueLoss - entropyCoef * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), maxGradNorm)
            optimizer.step()

# ===================== 4. main =====================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Cfg.from_arguments()

    env = BhlMujocoRLWrapper(cfg)

    model = ActorCritic(env.obsDim, env.actDim).to(device)
    model = loadPretrained(
        model,
        "./logs/rsl_rl/humanoid/2025-11-03_09-51-40/model_282199.pt",
        device,
    )

    os.makedirs("./checkpoints", exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    numIters = 50
    for it in range(numIters):
        batch = collectRollout(env, model, device, numSteps=512)
        ppoUpdate(model, optimizer, batch)
        if (it + 1) % 5 == 0:
            print(f"[MuJoCo] iter {it+1}/{numIters}")
        torch.save(
            {"model_state_dict": model.state_dict()},
            f"./checkpoints/finetune_mujoco_{it+1:03d}.pt",
        )

if __name__ == "__main__":
    main()
