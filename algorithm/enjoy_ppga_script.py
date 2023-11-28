import os
from pathlib import Path
project_root = os.path.join(str(Path.home()) + '/Documents', 'PPGADev')
os.chdir(project_root)
# %pwd # should be PPGA root dir

import pickle

import numpy as np
from attrdict import AttrDict
from RL.ppo import *
from utils.utilities import log
from envs.brax_custom.brax_env import make_vec_env_brax
from models.actor_critic import Actor, PGAMEActor
from pandas import DataFrame

from IPython.display import HTML, Image
from IPython.display import display
from brax.io import html, image

print("enjoy ppga script:")
# params to config
device = torch.device('cuda')
#env_name = 'humanoid'
env_name = 'walker2d'
# env_name = 'walker2d'
seed = 1111
normalize_obs = True
normalize_rewards = True
# non-configurable params
obs_shapes = {
    'humanoid': (227,),
    'ant': (87,),
    'halfcheetah': (18,),
    'walker2d': (17,)
}
action_shapes = {
    'humanoid': (17,),
    'ant': (8,),
    'halfcheetah': (6,),
    'walker2d': (6,)
}

# define the final config objects
actor_cfg = AttrDict({
        'obs_shape': obs_shapes[env_name],
        'action_shape': action_shapes[env_name],
        'normalize_obs': normalize_obs,
        'normalize_rewards': normalize_rewards,
})
env_cfg = AttrDict({
        'env_name': env_name,
        'env_batch_size': None,
        'num_dims': 2 if not 'ant' in env_name else 4,
        'envs_per_model': 1,
        'seed': seed,
        'num_envs': 1,
        'clip_obs_rew': False,
        'is_energy_measures': True
})

if env_cfg.is_energy_measures:
    extra = 1 if not 'walker2d' in env_name else 2
    env_cfg.num_dims += extra
print("saved cfg, loading archive/scheduler...")
# now lets load in a saved archive dataframe and scheduler
# change this to be your own checkpoint path
# archive_path = 'experiments/paper_ppga_ant/1111/checkpoints/cp_00001390/archive_df_00001390.pkl'
# scheduler_path = 'experiments/paper_ppga_ant/1111/checkpoints/cp_00001390/scheduler_00001390.pkl'

# archive_path = 'experiments/paper_ppga_walker2d/1111/checkpoints/cp_00001700/archive_df_00001700.pkl'
# scheduler_path = 'experiments/paper_ppga_walker2d/1111/checkpoints/cp_00001700/scheduler_00001700.pkl'

archive_path = 'experiments/paper_ppga_humanoid/1111/checkpoints/cp_00002000/archive_df_00002000.pkl'
scheduler_path = 'experiments/paper_ppga_humanoid/1111/checkpoints/cp_00002000/scheduler_00002000.pkl'

with open(archive_path, 'rb') as f:
    archive_df = pickle.load(f)
# with open(scheduler_path, 'rb') as f:
#     scheduler = pickle.load(f)
print("archive loaded, add solutions...")

from ribs.archives import GridArchive
all_solutions = archive_df.solution_batch()
all_objectives = archive_df.objective_batch()
all_measures = archive_df.measures_batch()
all_metadata = archive_df.metadata_batch()
print(all_solutions.shape)

archive = GridArchive(
    solution_dim=all_solutions.shape[1],  # Dimensionality of solutions in the archive.
    dims=[20, 20, 20],  # 50 cells along each dimension.
    ranges=[(0.0, 1.0), (0.0, 1.0), (0.0, 16.0)],  # (-1, 1) for x-pos and (-3, 0) for y-vel.
    qd_offset=0.0,  # See the note below.
)

# ant offset = 3.24, 5 dims - 4 (0, 1.0), 1 (0, 8.0), grid 10x10x10....
# walker offset = 1.413, 3 dims - 2 (0, 1.0), 1 (0, 6.0), grid 50x50x50
# humanoid offset = 0.0, 3 dims - 2 (0, 1.0), 1 (0, 16.0) grid 20x20x20

# archive.add(all_solutions, all_objectives, all_measures)

archive.add(all_solutions[0:1000], all_objectives[0:1000], all_measures[0:1000], all_metadata[0:1000])
i = 1001
end = min(i + 1000, all_solutions.shape[0])
while i < all_solutions.shape[0]:
    end = min(i + 1000, all_solutions.shape[0])
    archive.add(all_solutions[i:end], all_objectives[i:end], all_measures[i:end], all_metadata[i:end])
    print(i, " to ", end)
    i += 1000
    
print("solutions added, creating env...")
# create the environment
env = make_vec_env_brax(env_cfg)

def get_best_elite():
    best_elite = archive.best_elite
    print(f'Loading agent with reward {best_elite.objective} and measures {best_elite.measures}')
    agent = Actor(obs_shape=actor_cfg.obs_shape[0], action_shape=actor_cfg.action_shape, normalize_obs=normalize_obs, normalize_returns=normalize_rewards).deserialize(best_elite.solution).to(device)
#     print("elite metadata")
#     print(best_elite.metadata)
    if actor_cfg.normalize_obs:
        norm = best_elite.metadata['obs_normalizer']
        if isinstance(norm, dict):
            agent.obs_normalizer.load_state_dict(norm)
        else:
            agent.obs_normalizer = norm
    return agent

def get_random_elite():
#     elite = scheduler.archive.sample_elites(1)
    elite = archive.sample_elites(1)
    print(f'Loading agent with reward {elite.objective[0]} and measures {elite.measures[0]}')
    agent = Actor(obs_shape=actor_cfg.obs_shape[0], action_shape=actor_cfg.action_shape, normalize_obs=normalize_obs, normalize_returns=normalize_rewards).deserialize(elite.solution_batch.flatten()).to(device)
    if actor_cfg.normalize_obs:
        norm = elite.metadata['obs_normalizer']
        if isinstance(norm, dict):
            agent.obs_normalizer.load_state_dict(norm)
        else:
            agent.obs_normalizer = norm
    return agent

def get_elite(measures):
#     elite = scheduler.archive.elites_with_measures_single(measures)
    elite = archive.elites_with_measures_single(measures)
    print(f'Loading agent with reward {elite.objective} and measures {elite.measures}')
    agent = Actor(obs_shape=actor_cfg.obs_shape[0], action_shape=actor_cfg.action_shape, normalize_obs=normalize_obs, normalize_returns=normalize_rewards).deserialize(elite.solution.flatten()).to(device)
    print("elite")
    print(elite)
#     print("elite metadata")
#     print(elite.metadata)
    if actor_cfg.normalize_obs:
        norm = elite.metadata['obs_normalizer']
        if isinstance(norm, dict):
            agent.obs_normalizer.load_state_dict(norm)
        else:
            agent.obs_normalizer = norm
    return agent

def enjoy_brax(agent, render=True, deterministic=True):
    if actor_cfg.normalize_obs:
        obs_mean, obs_var = agent.obs_normalizer.obs_rms.mean, agent.obs_normalizer.obs_rms.var
        print(f'{obs_mean=}, {obs_var=}')

    obs = env.reset()
    rollout = [env.unwrapped._state]
    total_reward = 0
    measures = torch.zeros(env_cfg.num_dims).to(device)
    done = False
    while not done:
        with torch.no_grad():
            obs = obs.unsqueeze(dim=0).to(device)
            if actor_cfg.normalize_obs:
                obs = (obs - obs_mean) / torch.sqrt(obs_var + 1e-8)

            if deterministic:
                act = agent.actor_mean(obs)
            else:
                act, _, _ = agent.get_action(obs)
            act = act.squeeze()
            obs, rew, done, info = env.step(act.cpu())
            measures += info['measures']
            rollout.append(env.unwrapped._state)
            total_reward += rew
    if render:
        i = HTML(html.render(env.unwrapped._env.sys, [s.qp for s in rollout]))
        # display(i)
        html_data = i.data
        with open('html_file.html', 'w') as f:
            f.write(html_data)
        print(f'{total_reward=}')
        print(f' Rollout length: {len(rollout)}')
        measures /= len(rollout)
        print(f'Measures: {measures.cpu().numpy()}')
    return total_reward.detach().cpu().numpy()

print("env complete, getting best elite...")
# agent = get_random_elite()
print("line")
print("line")
agent = get_best_elite()
enjoy_brax(agent, render=True, deterministic=True)
print("best elite rendered, creating plots...")

import matplotlib.pyplot as plt
from ribs.visualize import parallel_axes_plot
plt.figure(figsize=(8, 6))
parallel_axes_plot(archive)
# plt.show
plt.savefig('parallel_axes_plot.png')

# visualizing archive - WALKER2D
# walker2d
from ribs.archives import CVTArchive
cvt_archive1 = CVTArchive( #measure 1 vs energy
    solution_dim=all_solutions.shape[1],  # Dimensionality of solutions in the archive.
    cells=1000,
    ranges=[(0.0, 1.0), (0.0, 6.0)]
)

cvt_archive2 = CVTArchive( #measure 2 vs energy
    solution_dim=all_solutions.shape[1],  # Dimensionality of solutions in the archive.
    cells=1000,
    ranges=[(0.0, 1.0), (0.0, 6.0)]
)

cvt_archive12 = CVTArchive( #measure 1 vs measure 2
    solution_dim=all_solutions.shape[1],  # Dimensionality of solutions in the archive.
    cells=1000,
    ranges=[(0.0, 1.0), (0.0, 1.0)]
)


cvt_archive1z = CVTArchive( #measure 1 vs z height
    solution_dim=all_solutions.shape[1],  # Dimensionality of solutions in the archive.
    cells=1000,
    ranges=[(0.0, 1.0), (0.0, 3.0)]
)


cvt_archive2z = CVTArchive( #measure 2 vs z height
    solution_dim=all_solutions.shape[1],  # Dimensionality of solutions in the archive.
    cells=1000,
    ranges=[(0.0, 1.0), (0.0, 3.0)]
)


cvt_archiveez = CVTArchive( #energy vs z height
    solution_dim=all_solutions.shape[1],  # Dimensionality of solutions in the archive.
    cells=1000,
    ranges=[(0.0, 6.0), (0.0, 3.0)]
)

print("cvt archives created, loading solutions...")

cvt_archive1.add(all_solutions[0:1000], all_objectives[0:1000], all_measures[0:1000][:, [0, 2]], all_metadata[0:1000])
cvt_archive2.add(all_solutions[0:1000], all_objectives[0:1000], all_measures[0:1000][:, [1, 2]], all_metadata[0:1000])
cvt_archive12.add(all_solutions[0:1000], all_objectives[0:1000], all_measures[0:1000][:, [0, 1]], all_metadata[0:1000])

cvt_archive1z.add(all_solutions[0:1000], all_objectives[0:1000], all_measures[0:1000][:, [0, 3]], all_metadata[0:1000])
cvt_archive2z.add(all_solutions[0:1000], all_objectives[0:1000], all_measures[0:1000][:, [1, 3]], all_metadata[0:1000])
cvt_archiveez.add(all_solutions[0:1000], all_objectives[0:1000], all_measures[0:1000][:, [2, 3]], all_metadata[0:1000])
i = 1001
end = min(i + 1000, all_solutions.shape[0])
while i < all_solutions.shape[0]:
    end = min(i + 1000, all_solutions.shape[0])
#     archive.add(all_solutions[i:end], all_objectives[i:end], all_measures[i:end], all_metadata[i:end])

    cvt_archive1.add(all_solutions[i:end], all_objectives[i:end], all_measures[i:end][:, [0, 2]], all_metadata[i:end])
    cvt_archive2.add(all_solutions[i:end], all_objectives[i:end], all_measures[i:end][:, [1, 2]], all_metadata[i:end])
    cvt_archive12.add(all_solutions[i:end], all_objectives[i:end], all_measures[i:end][:, [0, 1]], all_metadata[i:end])
        
    cvt_archive1z.add(all_solutions[i:end], all_objectives[i:end], all_measures[i:end][:, [0, 3]], all_metadata[i:end])
    cvt_archive2z.add(all_solutions[i:end], all_objectives[i:end], all_measures[i:end][:, [1, 3]], all_metadata[i:end])
    cvt_archiveez.add(all_solutions[i:end], all_objectives[i:end], all_measures[i:end][:, [2, 3]], all_metadata[i:end])
    print(i, " to ", end)
    i += 1000

print("solutions added, plotting...")
import matplotlib.pyplot as plt
from ribs.visualize import cvt_archive_heatmap
cvt_archive_heatmap(cvt_archive1)
plt.title("Measure 1 vs Energy")
# plt.show()
plt.savefig('m1_vs_energy.png')
cvt_archive_heatmap(cvt_archive2)
plt.title("Measure 2 vs Energy")
# plt.show()
plt.savefig('m2_vs_energy.png')
cvt_archive_heatmap(cvt_archive12)
plt.title("Measure 1 vs Measure 2")
# plt.show()
plt.savefig('m1_vs_m2.png')
plt.title("Measure 1 vs Z")
# plt.show()
plt.savefig('m1_vs_z.png')
cvt_archive_heatmap(cvt_archive2)
plt.title("Measure 2 vs Z")
# plt.show()
plt.savefig('m2_vs_z.png')
cvt_archive_heatmap(cvt_archive12)
plt.title("Energy vs Z")
# plt.show()
plt.savefig('energy_vs_z.png')

print("done")




















