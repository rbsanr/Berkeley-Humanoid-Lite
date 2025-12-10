import numpy as np
import torch

from berkeley_humanoid_lite_lowlevel.policy.rl_controller import RlController
from berkeley_humanoid_lite.environments import MujocoSimulator, Cfg
from berkeley_humanoid_lite_lowlevel.policy.keypad import Se2Keypad
from wholebody_operator import wholebody_operator

# Load configuration
cfg = Cfg.from_arguments()

if not cfg:
    raise ValueError("Failed to load config.")

# Main execution block
def main():
    """Main execution function for the MuJoCo simulation environment."""
    # Initialize environment
    
    # obs = robot.reset()

    # Initialize and start policy controller
    # try:
    #     keypad = Se2Keypad()
    #     keypad.run()
    # except IOError as e:
    #     print(e)
    #     return

    # controller = RlController(cfg)
    # controller.load_policy()

    # Default actions for fallback
    # default_actions = np.array(cfg.default_joint_positions, dtype=np.float32)[robot.cfg.action_indices]

    robot.calibration_robot()

    # robot.reset()
        

robot = wholebody_operator(cfg)
if __name__ == "__main__":
    try:
        main()
    except:
        pass

    print("done")
    robot.set_IDLE()
    robot.thread_run = False