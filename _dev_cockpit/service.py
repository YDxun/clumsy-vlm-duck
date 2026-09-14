#!/usr/bin/python3
"""Small, single-instance supervisor for the original MicroDuck Web server."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT = Path('/root/microduck_sim')
RUNTIME = Path('/run/qiaolong-microduck')
LOGS = ROOT / 'service-logs'
PIDFILE = RUNTIME / 'supervisor.pid'
SELF = Path(__file__).resolve()
stopping = False


def health():
    try:
        with urlopen('http://127.0.0.1:8080/healthz', timeout=2) as response:
            return response.status == 200 and json.load(response).get('ok') is True
    except Exception:
        return False


def supervisor_pid():
    try:
        pid = int(PIDFILE.read_text())
        cmd = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        if str(SELF).encode() in cmd and b'run' in cmd:
            return pid
    except (OSError, ValueError):
        pass
    return None


def terminate_child(child):
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=8)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


def run():
    global stopping
    RUNTIME.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    lock = (RUNTIME / 'supervisor.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return 0
    PIDFILE.write_text(str(os.getpid()))

    def stop_signal(*_):
        global stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    child = None
    backoff = 2
    try:
        if health():
            print('Port 8080 already serves an unmanaged application; stop it explicitly first.', flush=True)
            return 1
        while not stopping:
            log = LOGS / 'server.log'
            if log.exists() and log.stat().st_size > 5 * 1024 * 1024:
                log.replace(LOGS / 'server.previous.log')
            started = time.monotonic()
            with log.open('ab', buffering=0) as output:
                output.write((f'\nSTART {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}\n').encode())
                env = os.environ.copy()
                env.setdefault('MUJOCO_GL', 'egl')
                env.setdefault('OMP_NUM_THREADS', '1')
                env['DUCKAGENT_ENABLE'] = '1'  # Harness Duck C1: Agent drives the web duck
                _scene_file = ROOT / '.duck_scene_id'
                if _scene_file.exists():
                    env['DUCK_SCENE_ID'] = _scene_file.read_text(encoding='utf-8').strip()
                # Root-only secrets (API keys) live in /root/.duckagent_env, never in code.
                try:
                    for _ln in Path('/root/.duckagent_env').read_text().splitlines():
                        _ln = _ln.strip()
                        if _ln.startswith('export '):
                            _ln = _ln[len('export '):]
                        if '=' in _ln and not _ln.startswith('#'):
                            _k, _v = _ln.split('=', 1)
                            _k, _v = _k.strip(), _v.strip().strip('"').strip("'")
                            if _k:
                                env.setdefault(_k, _v)
                except Exception:
                    pass
                env.setdefault('DUCKAGENT_LLM_URL', 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions')
                env.setdefault('DUCKAGENT_LLM_MODEL', 'qwen3.5-flash')
                if not env.get('DUCKAGENT_LLM_KEY') and env.get('DASHSCOPE_KEY'):
                    env['DUCKAGENT_LLM_KEY'] = env['DASHSCOPE_KEY']
                if env.get('DASHSCOPE_KEY'):
                    env.setdefault('DUCKAGENT_VLM_MODEL', 'qwen3-vl-plus')
                    env.setdefault('DUCKAGENT_VLM_PLAN', '1')
                child = subprocess.Popen(
                    ['/usr/bin/python3', '-u', str(ROOT / 'sim_server.py')],
                    cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=output, stderr=subprocess.STDOUT,
                )
                failures = 0
                while child.poll() is None and not stopping:
                    time.sleep(2)
                    if time.monotonic() - started > 60:
                        failures = 0 if health() else failures + 1
                        if failures >= 3:
                            output.write(b'Health check failed three times; restarting.\n')
                            break
                terminate_child(child)
                output.write(f'EXIT {child.returncode}\n'.encode())
            if stopping:
                break
            backoff = 2 if time.monotonic() - started > 60 else min(backoff * 2, 30)
            for _ in range(backoff):
                if stopping:
                    break
                time.sleep(1)
    finally:
        if child is not None:
            terminate_child(child)
        PIDFILE.unlink(missing_ok=True)
        lock.close()
    return 0


def main():
    action = sys.argv[1] if len(sys.argv) > 1 else 'status'
    if action == 'run':
        return run()
    if action == 'status':
        pid = supervisor_pid()
        ready = health()
        print(json.dumps({'supervisor_pid': pid, 'healthy': ready, 'log': str(LOGS / 'server.log')}))
        return 0 if pid and ready else 1
    if action in ('stop', 'restart'):
        pid = supervisor_pid()
        if pid:
            os.kill(pid, signal.SIGTERM)
            for _ in range(60):
                if supervisor_pid() is None:
                    break
                time.sleep(0.25)
            else:
                print('Supervisor did not stop; refusing to start another.', file=sys.stderr)
                return 1
        if action == 'stop':
            return 0
    if action in ('start', 'restart'):
        if supervisor_pid() is None:
            LOGS.mkdir(parents=True, exist_ok=True)
            with (LOGS / 'supervisor.log').open('ab', buffering=0) as output:
                subprocess.Popen(['/usr/bin/python3', str(SELF), 'run'],
                                 stdin=subprocess.DEVNULL, stdout=output, stderr=output,
                                 start_new_session=True, close_fds=True)
        for _ in range(60):
            if supervisor_pid() and health():
                print('MicroDuck ready on 0.0.0.0:8080')
                return 0
            time.sleep(1)
        print(f'Start failed/timed out: inspect {LOGS}', file=sys.stderr)
        return 1
    print('Usage: ./service.py start|stop|restart|status', file=sys.stderr)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())


