#!/usr/bin/env python3
"""Deploy DQN8 project to remote server and run ablation experiments."""
import paramiko
import sys
import time

HOST = "117.50.216.203"
USER = "ubuntu"
PASS = "g7TXK26Q85Jp493f"

def run_cmd(ssh, cmd, timeout=300, print_output=True):
    """Execute command and print output in real-time."""
    print(f"\n>>> {cmd}")
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode()
    err = stderr.read().decode()
    if print_output and out.strip():
        print(out.rstrip())
    if err.strip():
        # Filter paramiko noise
        for line in err.strip().split('\n'):
            if 'NoneType' not in line and 'paramiko' not in line:
                print(f"  [stderr] {line}")
    return out, err, stdout.channel.recv_exit_status()

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"Connecting to {HOST}...")
    ssh.connect(HOST, username=USER, password=PASS, timeout=10)
    print("Connected!")

    # Step 1: Install Miniconda
    print("\n=== Step 1: Install Miniconda ===")
    out, _, _ = run_cmd(ssh, "which conda 2>/dev/null && echo HAS_CONDA || echo NO_CONDA")
    if "NO_CONDA" in out:
        run_cmd(ssh, "wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh", timeout=120)
        run_cmd(ssh, "bash /tmp/miniconda.sh -b -p $HOME/miniconda3", timeout=120)
        run_cmd(ssh, 'echo \'export PATH="$HOME/miniconda3/bin:$PATH"\' >> ~/.bashrc')
        run_cmd(ssh, "$HOME/miniconda3/bin/conda init bash", timeout=30)

    conda = "$HOME/miniconda3/bin/conda"

    # Step 2: Create conda environment
    print("\n=== Step 2: Create conda env ros2py310 ===")
    out, _, _ = run_cmd(ssh, f"{conda} env list | grep ros2py310 || echo NO_ENV")
    if "NO_ENV" in out:
        run_cmd(ssh, f"{conda} create -n ros2py310 python=3.10 -y", timeout=300)

    # Step 3: Upload project
    print("\n=== Step 3: Upload project ===")
    run_cmd(ssh, "mkdir -p /root/DQN8/runs")

    ssh.close()
    print("\n=== Upload via rsync ===")
    import subprocess
    # Use rsync to upload project (excluding large dirs)
    rsync_cmd = [
        "rsync", "-avz", "--progress",
        "--exclude", "runs/", "--exclude", "__pycache__/",
        "--exclude", "*.pyc", "--exclude", ".git/",
        "--exclude", "DQN8_Papers/", "--exclude", "paperdqn8.3/",
        "--exclude", "*.pptx",
        "-e", f"ssh -o StrictHostKeyChecking=no",
        "/home/sun/phdproject/dqn/DQN8/",
        f"{USER}@{HOST}:/root/DQN8/"
    ]
    print(f"rsync command: {' '.join(rsync_cmd[:5])}...")

    # We can't use rsync with password directly, use paramiko sftp instead
    print("Using SFTP upload instead (rsync needs sshpass)...")

    ssh2 = paramiko.SSHClient()
    ssh2.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh2.connect(HOST, username=USER, password=PASS, timeout=10)

    # Use tar + pipe approach
    import os
    proj_dir = "/home/sun/phdproject/dqn/DQN8"

    # Create local tarball excluding large dirs
    print("Creating tarball...")
    tar_path = "/tmp/dqn8_deploy.tar.gz"
    os.system(
        f"cd {proj_dir} && tar czf {tar_path} "
        f"--exclude='runs' --exclude='__pycache__' --exclude='*.pyc' "
        f"--exclude='.git' --exclude='DQN8_Papers' --exclude='paperdqn8.3' "
        f"--exclude='*.pptx' --exclude='wandb' "
        f"."
    )
    tar_size = os.path.getsize(tar_path) / (1024*1024)
    print(f"Tarball: {tar_size:.1f} MB")

    # Upload via SFTP
    sftp = ssh2.open_sftp()
    print("Uploading tarball...")
    sftp.put(tar_path, "/tmp/dqn8_deploy.tar.gz")
    sftp.close()
    print("Upload complete!")

    # Extract on server
    run_cmd(ssh2, "mkdir -p /root/DQN8 && cd /root/DQN8 && tar xzf /tmp/dqn8_deploy.tar.gz")

    # Step 4: Install Python dependencies
    print("\n=== Step 4: Install dependencies ===")
    run_cmd(ssh2, f"{conda} run -n ros2py310 pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -5", timeout=600)
    run_cmd(ssh2, f"{conda} run -n ros2py310 pip install gymnasium numpy pandas matplotlib openpyxl scipy scikit-image 2>&1 | tail -5", timeout=300)

    # Step 5: Verify CUDA
    print("\n=== Step 5: Verify setup ===")
    run_cmd(ssh2, f"{conda} run --cwd /root/DQN8 -n ros2py310 python -m amr_dqn.cli.train --self-check")

    # Step 6: Launch ablation experiments
    print("\n=== Step 6: Launch ablation training ===")
    run_cmd(ssh2, "mkdir -p /root/DQN8/runs")

    # Launch both training jobs
    run_cmd(ssh2, f"""
{conda} run --cwd /root/DQN8 -n ros2py310 python -m amr_dqn.cli.train --profile ablation_20260311_cnn_drop_edt > /root/DQN8/runs/ablation_drop_edt.log 2>&1 &
echo "drop_edt PID=$!"
""")

    run_cmd(ssh2, f"""
{conda} run --cwd /root/DQN8 -n ros2py310 python -m amr_dqn.cli.train --profile ablation_20260311_cnn_keep_edt > /root/DQN8/runs/ablation_keep_edt.log 2>&1 &
echo "keep_edt PID=$!"
""")

    time.sleep(3)
    run_cmd(ssh2, "ps aux | grep 'amr_dqn.cli.train' | grep -v grep")

    print("\n=== DEPLOYMENT COMPLETE ===")
    print("Monitor: ssh ubuntu@117.50.216.203 'tail -f /root/DQN8/runs/ablation_*.log'")

    ssh2.close()

if __name__ == "__main__":
    main()
