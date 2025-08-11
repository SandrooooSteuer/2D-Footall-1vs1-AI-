2D Soccer – Self-Play DQN Agents ⚽🤖

Parallel self-play training with resume, rendering, and safe interrupt

This project implements a 1v1 2D soccer game in pygame, powered by two Deep Q-Network (DQN) agents learning entirely through self-play.
It supports parallel environments for faster training, automatic saving/resuming, and live rendering while training.
✨ Features

    All-in-one script – just run AI.py.

    Parallel training – run multiple matches at once for faster learning.

    Safe interrupt – stop anytime with CTRL+C, progress is saved automatically.

    Resume training – pick up exactly where you left off.

    Watch while training – render a match every N episodes.

    Play mode – watch the trained agents compete (no further training).

    Simple physics & collisions – realistic ball and player interactions.

📦 Installation

pip install torch pygame numpy

🚀 Usage
Train from scratch

python AI.py --episodes 1000

Runs 1000 training episodes using 5 parallel games (default).
Continue training

python AI.py --episodes 1000 --resume

Resumes from the last saved checkpoint.
Render every N episodes

python AI.py --episodes 2000 --render-every 50

Shows one match every 50 episodes.
Increase parallel environments

python AI.py --episodes 1000 --num-envs 8

Uses 8 matches running in parallel.
Watch only (no training)

python AI.py --play

Loads saved models and plays continuously.
⚙ Command-line Options
Option	Description
--play	Play only, no training (uses saved models)
--episodes	Number of training episodes
--render-every	Render every N episodes (Env #0 only)
--prefix	Filename prefix for saved models
--resume	Resume training from saved models
--num-envs	Number of parallel environments
--checkpoint-every	Save every N episodes
--end-on-goal	End an episode immediately after a goal
--max-steps	Maximum steps per episode
--batch-size	Batch size for DQN updates
--buffer-capacity	Replay buffer capacity
--goal-reward	Reward per goal
--shape-scale	Scaling factor for shaping rewards
🧠 How It Works

    Environment (SoccerEnv)

        Simple 1v1 soccer field with goals, ball physics, collisions, and scoring.

        State space: Ball + player positions/velocities (12 values, normalized).

        Action space:
        0 noop, 1 up, 2 down, 3 left, 4 right, 5 kick.

    Agents

        Two independent DQNAgent instances (left & right players).

        Neural network: 2 fully-connected hidden layers (128 units each, ReLU).

        Uses an experience replay buffer and target networks.

    Training Loop

        Multiple environments run in parallel for efficient data collection.

        Each agent selects an action via ε-greedy exploration.

        Transitions are stored in replay buffers, and networks are updated via mini-batch SGD.

        Target networks are softly updated to stabilize learning.

    Rewards

        Positive reward for scoring, negative for conceding.

        Small shaping rewards for ball pursuit and directional ball movement.

📷 Gameplay Preview

During early training, agents move randomly.
Over time, they learn to:

    Chase the ball

    Block shots

    Aim kicks towards the opponent’s goal

In play mode, both agents use their learned greedy policies.
📌 Notes

    Supports both CPU and GPU (auto-detected).

    Press R during a match to reset scores.

    Works best when running at least 5–8 parallel environments.

