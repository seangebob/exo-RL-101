# Exo RL 101: the Control Team Onboarding Task (PyTorch + RL Basics)

Welcome! This is a short, hands-on task to get you comfortable with **PyTorch** and
**reinforcement learning (RL)** basics before you touch the team's real `exo-RL` code.

No prior ML experience needed. You should know basic Python (loops, functions, classes).

**Total time:** about 1.5-2.5 hours. You can split it across a few sittings.

---

## 0. Setup (5-10 min)

```bash
# from inside this folder
python3 -m venv venv
source venv/bin/activate        # on Windows: run .\venv\Scripts\activate
pip install -r requirements.txt
```

That installs PyTorch, Gymnasium (the RL environment library), NumPy, and Matplotlib.

---

## Part 1: PyTorch Basics (~30 min)

File: `01_pytorch_basics.py`

This trains a tiny model to learn the line `y = 2x + 1` from noisy data. It's the
"hello world" of PyTorch: tensors → forward pass → loss → backward pass → update.

1. Open `01_pytorch_basics.py`.
2. Read through it top to bottom — every step is commented.
3. Fill in the **4 short TODOs** (each is one line).
4. Run it:
   ```bash
   python 01_pytorch_basics.py
   ```
5. You should see the loss go down each epoch, and a plot pop up showing the model's
   line getting closer to the real data.

**What you should take away:** a model is just a function with adjustable numbers
(parameters). Training = repeatedly nudging those numbers to reduce error.

---

## Part 2: Meet a Gymnasium Environment (~10-15 min)

File: `02_explore_cartpole.py`

No TODOs here — just run it and read the output.

```bash
python 02_explore_cartpole.py
```

This runs the classic **CartPole** environment with a *random* agent (it just picks
random actions) so you can see the shape of an RL problem: an **observation**, an
**action**, a **reward**, and a **done** signal, repeated in a loop. This exact
pattern (reset → step → reward → repeat) is what we use to train our exo.

**What you should take away:** RL environments all share this same loop. Once you
know it, you can plug in any environment: to train a game, a robot, or an exoskeleton.

---

## Part 3: Train an RL Agent (REINFORCE) (~45-75 min)

Files: `03_reinforce_cartpole.py`

This is the main event. You'll implement a small but real policy-gradient RL
algorithm called **REINFORCE**, and use it to train an agent that balances the pole.

**The idea:** the agent has a small neural network (the *policy*)
that looks at the cart's state and outputs probabilities for "push left" vs "push
right." It plays a full episode, collecting rewards. At the end, it looks back:
actions taken in episodes that scored well get "reinforced" (made more likely next
time), and actions in episodes that scored badly get discouraged. Repeat this over
many episodes and the policy gets better.

1. Open `03_reinforce_cartpole.py` and read the comments: the whole training loop is there and explained.
2. Fill in the **3 TODOs**:
   - Computing the discounted return for each step in an episode
   - The policy loss formula
   - Picking an action by sampling from the policy's output distribution
3. Run it:
   ```bash
   python 03_reinforce_cartpole.py
   ```
4. Training takes a few minutes. You'll see the average reward per episode
   printed, and it should climb from ~20 (random flailing) toward 200+ (balancing
   well) over a few hundred episodes. A plot of the reward curve saves to
   `training_curve.png`.

**What you should take away:** this "try things → see what worked → do more of
that" loop is the whole idea behind reinforcement learning. We use a fancier
version of this same idea (PPO, running in Isaac Sim) to teach the exoskeleton to
assist a person's gait.

---

## What to Turn In
Submit the following files:
- A link to your completed code (push this folder to a personal GitHub repo) or ZIP files of the completed project.
- `training_curve.png` image from Part 3
- 2-3 sentences: what was confusing, and what clicked

Post it in the team's onboarding channel or attach it to your Onboarding Submission.

## Extra Challenges (optional)

- Swap `CartPole-v1` for `Acrobot-v1` or `LunarLander-v3` in Part 3 and see how
  training changes
- Try increasing/decreasing the learning rate and see what happens to training
  stability
- Skim `exo-RL`'s training code and understands the training pipeline
