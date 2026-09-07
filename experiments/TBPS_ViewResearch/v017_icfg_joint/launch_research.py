import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def dump(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(str(temporary), str(path))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["audit", "train", "e0-audit", "e0-train"], required=True)
    p.add_argument("--record-dir", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--worker", action="store_true")
    args = p.parse_args()
    record, root = Path(args.record_dir).resolve(), Path(__file__).resolve().parent
    if not args.worker:
        if record.exists():
            raise FileExistsError("Preserve old launch records")
        record.mkdir(parents=True)
        with open(record / "launcher.log", "x") as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"],
                         cwd=str(root), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                         start_new_session=True)
        (record / "worker.pid").write_text(str(process.pid))
        print(json.dumps({"worker_pid": process.pid, "record": str(record)}))
        return
    lock = open('/root/autodl-tmp/.tbps_research_gpu.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    running = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if running:
        raise RuntimeError("GPU process already exists: " + running)
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            argv = (entry / 'cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        names = [Path(a.decode(errors='replace')).name for a in argv if a and len(a) < 512]
        if 'train.py' in names or ('train' in names and any(x in names for x in ('run_research.py', 'run_view4.py', 'run_rstp_e0.py', 'run_cuhk_e0.py', 'run_icfg_e0.py'))):
            # This launcher's parent arguments contain no training entry script.
            raise RuntimeError("Other training command exists: " + entry.name)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3", OMP_NUM_THREADS="4",
               MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", NCCL_P2P_DISABLE="1", NCCL_IB_DISABLE="1",
               PYTHONUNBUFFERED="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    baseline = args.stage.startswith('e0-')
    stage = args.stage[3:] if baseline else args.stage
    entry = 'run_icfg_e0.py' if baseline else 'run_research.py'
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
               str(root / entry), stage, '--config', str(Path(args.config).resolve()),
               '--output-dir', str(Path(args.output_dir).resolve())]
    if stage == 'audit':
        command.extend(['--steps', '8'])
    info = dict(stage=args.stage, command=command, root=str(root), started_at=time.time(),
                environment={k: env[k] for k in ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'NCCL_P2P_DISABLE', 'NCCL_IB_DISABLE')})
    dump(record / 'command.json', info)
    with open(record / 'nohup.log', 'x') as log:
        child = subprocess.Popen(command, cwd=str(root), env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT)
        (record / 'pid').write_text(str(child.pid))
        dump(record / 'status.json', dict(info, status='running', pid=child.pid))
        code = child.wait()
    dump(record / 'status.json', dict(info, status='complete' if code == 0 else 'failed',
                                    pid=child.pid, exit_code=code, ended_at=time.time()))
    raise SystemExit(code)


if __name__ == '__main__':
    main()
