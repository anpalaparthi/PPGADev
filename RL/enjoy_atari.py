import torch
import envpool
import cv2

from envs.wrappers.common_wrappers import EnvPoolTorchWrapper
from models.actor_critic import DiscreteActor
from models.vectorized import VectorizedDiscreteVisualActor


def enjoy():
    env_name = 'SpaceInvaders-v5'
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    env = envpool.make(task_id=env_name,
                       env_type='gym',
                       num_envs=1,
                       episodic_life=True,
                       reward_clip=True)
    env = EnvPoolTorchWrapper(env, device='cuda' if torch.cuda.is_available() else 'cpu')

    normalize_obs = True

    vec_policy_cp = 'checkpoints/SpaceInvaders-v5_envpool_policy_checkpoint.pt'
    policy = DiscreteActor(env.observation_space.shape, env.action_space.n, normalize_obs=True, normalize_returns=True).to(device)
    vec_policy = VectorizedDiscreteVisualActor([policy], DiscreteActor, env.observation_space.shape, env.action_space.n, True, True).to(device)
    vec_policy.load_state_dict(torch.load(vec_policy_cp)['model_state_dict'])

    vec_policy.eval()

    total_reward = 0
    traj_length = 0
    num_steps = 1000

    # create video from images
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('policy_eval.mp4', fourcc, 20, (84, 84), 0)

    if normalize_obs:
        mean, var = vec_policy.obs_normalizers[0].obs_rms.mean, vec_policy.obs_normalizers[0].obs_rms.var

    obs, _ = env.reset()
    done = False
    while not done:
        with torch.no_grad():
            if normalize_obs:
                obs = (obs - mean) / (torch.sqrt(var) + 1e-8)

            action, _, _ = vec_policy.get_action(obs)
            obs, reward, done, trunc, info = env.step(action.cpu().numpy())
            total_reward += reward.item()

            image = obs.squeeze(0)[1:2].permute(1, 2, 0).to(torch.uint8).cpu().numpy()
            video.write(image)

    video.release()
    print(f'Total reward: {total_reward}')


if __name__ == '__main__':
    enjoy()
