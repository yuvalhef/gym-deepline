import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='deepline-v0',
    entry_point='gym_deepline.envs:AutomlEnv',
)
