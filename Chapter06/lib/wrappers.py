import cv2
import gym
import gym.spaces
import numpy as np
import collections

class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    # 
    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
        obs, _, done, _, _ = self.env.reset(1)
        if done:
            self.env.reset()
        obs, _, done, _, _ = self.env.reset(2)
        if done:
            self.env.reset()
        return obs

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4) -> None:
        #return only every skip-th frame
        super(MaxAndSkipEnv, self).__init__(env)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, _, info = self.env.reset()
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        #stack the frames on top of each other and return the max among them
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs

class ProcessFrame84(gym.ObservationWrapper):
    def __init_(self, env=None):
        super(ProcessFrame84, self).__init__(env):
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 48, 1), dtype=np.uint8)
        
