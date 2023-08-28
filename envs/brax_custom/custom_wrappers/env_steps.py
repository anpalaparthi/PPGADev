import jax
import brax
from jax import numpy as jp
from brax import math
from brax.envs.env import Env, State

def ant_step(env: Env, state: State, action: jp.ndarray) -> State:
    """Run one timestep of the environment's dynamics."""
    # print("ant step")

    # print("state.metrics before")
    # print(state.metrics)
    qp, info = env.sys.step(state.qp, action)

    velocity = (qp.pos[0] - state.qp.pos[0]) / env.sys.config.dt
    forward_reward = velocity[0]

    # min_z, max_z = env._healthy_z_range
    # is_healthy = jp.where(qp.pos[0, 2] < min_z, x=0.0, y=1.0)
    # is_healthy = jp.where(qp.pos[0, 2] > max_z, x=0.0, y=is_healthy)
    # if env._terminate_when_unhealthy:
    #   healthy_reward = env._healthy_reward
    # else:
    #   healthy_reward = env._healthy_reward * is_healthy
    ctrl_cost = jp.sum(jp.square(action))
    contact_cost = (env._contact_cost_weight *
                    jp.sum(jp.square(jp.clip(info.contact.vel, -1, 1))))
    obs = env._get_obs(qp, info)
    # reward = forward_reward + healthy_reward - ctrl_cost - contact_cost
    #reward = forward_reward - contact_cost
    reward = forward_reward
    # done = 1.0 - is_healthy if env._terminate_when_unhealthy else 0.0

    # print("state.metrics")
    # print(state.metrics)
    zero, temp = jp.zeros(2)
    state.metrics.update(
        reward_forward=forward_reward,
        reward_survive=zero,
        reward_ctrl=ctrl_cost,
        reward_contact=-contact_cost,
        x_position=qp.pos[0, 0],
        y_position=qp.pos[0, 1],
        distance_from_origin=jp.linalg.norm(qp.pos[0]),
        x_velocity=velocity[0],
        y_velocity=velocity[1],
        forward_reward=forward_reward,
    )

    # print("state.metrics.update")
    # print(state.metrics)
    newstate = state.replace(qp=qp, 
                         obs=obs, 
                         reward=reward, 
                         done=zero
                         )
    
    #newstate.info["measures"] = jp.array([ctrl_cost, contact_cost]).astype(jp.float32)
    newstate.info["measures"] = jp.array([ctrl_cost]).astype(jp.float32)
    # print("newstate.metrics")
    # print(newstate.metrics)
    return newstate

def walker2d_step(env: Env, state: State, action: jp.ndarray) -> State:
    """Run one timestep of the environment's dynamics."""
    qp, _ = env.sys.step(state.qp, action)

    x_velocity = (qp.pos[0, 0] - state.qp.pos[0, 0]) / env.sys.config.dt
    # forward_reward = env._forward_reward_weight * x_velocity
    forward_reward = x_velocity

    # min_z, max_z = env._healthy_z_range
    # min_angle, max_angle = env._healthy_angle_range
    # ang_y = math.quat_to_euler(qp.rot[0])[1]
    # is_healthy = jp.where(qp.pos[0, 2] < min_z, x=0.0, y=1.0)
    # is_healthy = jp.where(qp.pos[0, 2] > max_z, x=0.0, y=is_healthy)
    # is_healthy = jp.where(ang_y > max_angle, x=0.0, y=is_healthy)
    # is_healthy = jp.where(ang_y < min_angle, x=0.0, y=is_healthy)
    # if env._terminate_when_unhealthy:
    #     healthy_reward = env._healthy_reward
    # else:
    #     healthy_reward = env._healthy_reward * is_healthy
    # ctrl_cost = env._ctrl_cost_weight * jp.sum(jp.square(action))
    
    ctrl_cost = jp.sum(jp.square(action))
    z_height = jp.array(qp.pos[0,2]) 

    measures = jp.array([ctrl_cost, z_height]).astype(jp.float32)
    
    obs = env._get_obs(qp)
    # reward = forward_reward + healthy_reward - ctrl_cost
    reward = forward_reward
    # done = 1.0 - is_healthy if env._terminate_when_unhealthy else 0.0
    zero, _ = jp.zeros(2)
    state.metrics.update(
        reward_forward=forward_reward,
        reward_ctrl=-ctrl_cost,
        reward_healthy=zero,
        x_position=qp.pos[0, 0],
        x_velocity=x_velocity)

    newstate = state.replace(qp=qp, obs=obs, reward=reward, done=zero)
    newstate.info["measures"] = measures
    return newstate

def humanoid_step(env: Env, state: State, action: jp.ndarray) -> State:
    """Run one timestep of the environment's dynamics."""
    qp, info = env.sys.step(state.qp, action)

    com_before = env._center_of_mass(state.qp)
    com_after = env._center_of_mass(qp)
    velocity = (com_after - com_before) / env.sys.config.dt
    # forward_reward = env._forward_reward_weight * velocity[0]
    forward_reward = velocity[0]
    

    # min_z, max_z = env._healthy_z_range
    # is_healthy = jp.where(qp.pos[0, 2] < min_z, x=0.0, y=1.0)
    # is_healthy = jp.where(qp.pos[0, 2] > max_z, x=0.0, y=is_healthy)
    # if env._terminate_when_unhealthy:
    #   healthy_reward = env._healthy_reward
    # else:
    #   healthy_reward = env._healthy_reward * is_healthy

    # ctrl_cost = env._ctrl_cost_weight * jp.sum(jp.square(action))
    ctrl_cost = jp.sum(jp.square(action))

    obs = env._get_obs(qp, info, action)
    # reward = forward_reward + healthy_reward - ctrl_cost
    reward = forward_reward

    # done = 1.0 - is_healthy if env._terminate_when_unhealthy else 0.0

    zero, _ = jp.zeros(2)
    state.metrics.update(
        forward_reward=forward_reward,
        reward_linvel=forward_reward,
        reward_quadctrl=-ctrl_cost,
        reward_alive=zero,
        x_position=com_after[0],
        y_position=com_after[1],
        distance_from_origin=jp.linalg.norm(com_after),
        x_velocity=velocity[0],
        y_velocity=velocity[1],
    )

    # return state.replace(qp=qp, obs=obs, reward=reward, done=done)

    newstate = state.replace(qp=qp, obs=obs, reward=reward, done=zero)
    newstate.info["measures"] = jp.array([ctrl_cost]).astype(jp.float32)
    return newstate