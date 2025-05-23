## How to Run the Models with Different Reward Shaping Approaches

To run each model variant, follow the instructions below to replace the necessary files and update configurations as required:

---

### 🎯 Reward Shaping Approaches without CSR:

- **GAT-based Reward Shaping**  
  Copy `utilssparsestaticgat.py` into `utils.py`.

- **GraphSAGE-based Reward Shaping**  
  Copy `utilsfinalsparseminsage.py` into `utils.py`.

- **Node2Vec-based Reward Shaping**  
  Copy `utilsfinalN2Vsparse.py` into `utils.py`.

- For all of the above models, also copy `state_trackerorg.py` into `state_tracker.py`.

---

### 🧩 Reward Shaping Approaches with CSR:

- **GAT + CSR-based Reward Shaping**  
  Copy `utilssparsestaticpartialgat.py` into `utils.py`.

- **GraphSAGE + CSR-based Reward Shaping**  
  Copy `utilspartialsage.py` into `utils.py`.

- **Node2Vec + CSR-based Reward Shaping**  
  Copy `utilsN2Vsparsepartial.py` into `utils.py`.

- For all of these models, also copy `state_trackerw2vfinal.py` into `state_tracker.py`.

---

### 🤖 Base Model Execution

- **To run the D3QN model:**
  - Copy `d3qn.py` into `dqn_agent.py`.
  - In the `constants.json` file, change the value of `"vanilla": true,` to `false`.

- **To run the DQN model:**
  - Copy `dqn_agentdqn.py` into `dqn_agent.py`.
  - Keep the value of `"vanilla": true,` set to `true` in the `constants.json` file.

---

> ⚠️ Please ensure all file replacements and configuration changes are completed before executing any model.
