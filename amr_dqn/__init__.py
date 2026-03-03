"""amr_dqn — DQN/DDQN path-planning for Autonomous Mobile Robots.

Package layout
--------------
agents.py          DQN/DDQN/PDDQN agent (TD learning, target network, DQfD expert loss)
env.py             Gymnasium environments: AMRGridEnv (8-dir grid) + AMRBicycleEnv (Ackermann bicycle)
networks.py        Q-network architectures: MLPQNetwork, CNNQNetwork
replay_buffer.py   Uniform experience-replay buffer with DQfD demo-preservation
reward_norm.py     Welford online reward normalizer (mean/std + clip)
forest_policy.py   Unified admissible-action selection pipeline (train & infer share same logic)
schedules.py       Epsilon-decay schedules (linear, adaptive sigmoid)
smoothing.py       Chaikin corner-cutting path smoother
metrics.py         Path KPI helpers: length, curvature, corners
config_io.py       JSON config loading + argparse integration
runtime.py         PyTorch/CUDA/Matplotlib backend setup
runs.py            Timestamped experiment-run directory management

Sub-packages
------------
cli/               Entry points: train.py, infer.py, benchmark.py, config.py, precompute_forest_paths.py
maps/              Map definitions: forest (procedural A-D), realmap (PGM), precomputed expert paths
baselines/         Classical planners: Hybrid A*, RRT* (wrapper around third_party.pathplan)
third_party/       Vendored path-planning library: hybrid_a_star, rrt, geometry utilities
"""
