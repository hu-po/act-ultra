lambdalabs H100 @ $3.29 / hr

```bash
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.124.06             Driver Version: 570.124.06     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA H100 80GB HBM3          On  |   00000000:06:00.0 Off |                    0 |
| N/A   25C    P0             75W /  700W |       1MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
Architecture:             x86_64
  CPU op-mode(s):         32-bit, 64-bit
  Address sizes:          52 bits physical, 57 bits virtual
  Byte Order:             Little Endian
CPU(s):                   26
  On-line CPU(s) list:    0-25
Vendor ID:                GenuineIntel
  Model name:             Intel(R) Xeon(R) Platinum 8480+
    CPU family:           6
    Model:                143
    Thread(s) per core:   2
    Core(s) per socket:   13
Memory block size:         1G
Total online memory:     225G
Total offline memory:      0B
PRETTY_NAME="Ubuntu 22.04.5 LTS"
```

setup dependencies

```bash
ssh -i ~/.ssh/id_ed25519.pub ubuntu@192.222.52.183
git clone https://github.com/hu-po/act-ultra.git
cd act-ultra
curl -sSL https://install.python-poetry.org | python3 -
export PATH="/home/ubuntu/.local/bin:$PATH"
poetry lock
poetry install
poetry run wandb login
```

use scripted policy to generate dataset

```bash
mkdir -p /local_data/sim_insertion_scripted
MUJOCO_GL=egl NUM_EPISODES=50 poetry run python3 record_sim_episodes.py \
--task_name sim_insertion_scripted \
--dataset_dir /local_data/sim_insertion_scripted \
--num_episodes ${NUM_EPISODES}
```