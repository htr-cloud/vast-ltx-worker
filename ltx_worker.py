#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

LTX_DIR = Path(os.environ.get("LTX_HOME", "/opt/LTX-2"))
LTX_PYTHON = Path(
    os.environ.get("LTX_PYTHON", str(LTX_DIR / ".venv/bin/python"))
)

RCLONE = os.environ.get("RCLONE_BIN", "/usr/bin/rclone")
FFMPEG = os.environ.get("FFMPEG_BIN", "/usr/bin/ffmpeg")
NVIDIA_SMI = os.environ.get("NVIDIA_SMI_BIN", "nvidia-smi")

MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "/workspace/models/ltx-2.5"))
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", "/workspace/output/af_jobs"))

TRANSFORMER = MODEL_ROOT / "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
TEXT_ENCODER = MODEL_ROOT / "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
VIDEO_VAE = MODEL_ROOT / "vae/ltx-2.5-video-vae-bf16.safetensors"
AUDIO_VAE = MODEL_ROOT / "vae/ltx-2.5-audio-vae-bf16.safetensors"
SPATIAL = MODEL_ROOT / "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"

HF_REPO = "Lightricks/LTX-2.5"
HF_FILES = [
    "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "vae/ltx-2.5-video-vae-bf16.safetensors",
    "vae/ltx-2.5-audio-vae-bf16.safetensors",
    "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
]

JOB_RE = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")


def emit(obj: dict) -> None:
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def read_json() -> dict:
    value = json.loads(sys.stdin.read() or "{}")
    if not isinstance(value, dict):
        raise ValueError("JSON root muss Objekt sein")
    return value


def executable(path_or_name: str | Path) -> str | None:
    value = str(path_or_name)
    if "/" in value:
        p = Path(value)
        return value if p.is_file() and os.access(p, os.X_OK) else None
    return shutil.which(value)


def required_executable(path_or_name: str | Path, label: str) -> str:
    found = executable(path_or_name)
    if not found:
        raise RuntimeError(f"{label} fehlt auf der VM: {path_or_name}")
    return found


def model_status() -> dict[str, bool]:
    return {
        "transformer": TRANSFORMER.is_file(),
        "text_encoder": TEXT_ENCODER.is_file(),
        "video_vae": VIDEO_VAE.is_file(),
        "audio_vae": AUDIO_VAE.is_file(),
        "spatial": SPATIAL.is_file(),
    }


def health() -> int:
    gpu = "nvidia-smi fehlt"
    nvidia_smi = executable(NVIDIA_SMI)
    if nvidia_smi:
        proc = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            capture_output=True,
        )
        gpu = proc.stdout.strip() or proc.stderr.strip() or f"exit={proc.returncode}"

    ltx_python = executable(LTX_PYTHON)
    rclone = executable(RCLONE)
    ffmpeg = executable(FFMPEG)
    models = model_status()

    ok = bool(
        LTX_DIR.is_dir()
        and ltx_python
        and rclone
        and ffmpeg
    )

    emit(
        {
            "ok": ok,
            "ltx": str(LTX_DIR),
            "ltx_python": ltx_python,
            "rclone": rclone,
            "ffmpeg": ffmpeg,
            "gpu": gpu,
            "models": models,
        }
    )
    return 0 if ok else 1


def hf_cmd() -> list[str]:
    # Im fertigen Image bevorzugen wir das hf-Binary aus der LTX-venv.
    venv_hf = LTX_DIR / ".venv/bin/hf"
    found = executable(venv_hf)
    if found:
        return [found]

    found = shutil.which("hf")
    if found:
        return [found]

    raise RuntimeError(
        "hf CLI fehlt im LTX Runtime-Image "
        "(/opt/LTX-2/.venv/bin/hf oder PATH)"
    )


def prepare() -> int:
    req = read_json()
    token = str(req.get("hf_token", "")).strip()
    if not token:
        emit({"ok": False, "error": "hf_token fehlt"})
        return 2

    missing = [f for f in HF_FILES if not (MODEL_ROOT / f).is_file()]
    if not missing:
        emit({"ok": True, "status": "already_ready"})
        return 0

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["HF_TOKEN"] = token
    cmd = hf_cmd() + [
        "download",
        HF_REPO,
        *missing,
        "--local-dir",
        str(MODEL_ROOT),
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=LTX_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, file=sys.stderr, end="", flush=True)
        rc = proc.wait()
    finally:
        env.pop("HF_TOKEN", None)
        token = ""

    emit(
        {
            "ok": rc == 0,
            "status": "downloaded" if rc == 0 else "failed",
            "exit_code": rc,
        }
    )
    return rc


def validate(req: dict) -> dict:
    job = str(req.get("job_id", "")).strip()
    prompt = str(req.get("prompt", "")).strip()

    if not JOB_RE.fullmatch(job):
        raise ValueError("Ungültige job_id")
    if not prompt:
        raise ValueError("Prompt fehlt")

    frames = int(req.get("frames", 121))
    if frames <= 0 or frames % 8 != 1:
        raise ValueError("frames muss >0 und frames % 8 == 1 sein")

    req.update(
        job_id=job,
        prompt=prompt,
        width=int(req.get("width", 1920)),
        height=int(req.get("height", 1088)),
        frames=frames,
        fps=int(req.get("fps", 24)),
        seed=int(req.get("seed", 1)),
    )
    return req


def write_rclone_config(s3: dict) -> tuple[int, str]:
    required = (
        "access_key",
        "secret_key",
        "region",
        "endpoint",
        "bucket",
    )
    missing = [k for k in required if not str(s3.get(k, "")).strip()]
    if missing:
        raise ValueError("S3-Konfiguration unvollständig: " + ", ".join(missing))

    cfg = f"""[garage]
type = s3
provider = Other
env_auth = false
access_key_id = {s3['access_key']}
secret_access_key = {s3['secret_key']}
region = {s3['region']}
endpoint = {s3['endpoint']}
force_path_style = true
acl = private
"""

    fd, name = tempfile.mkstemp(prefix="af-rclone-", suffix=".conf")
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(cfg)
    return fd, name


def rclone_upload(job_dir: Path, s3: dict, job_id: str) -> None:
    rclone = required_executable(RCLONE, "rclone")
    _, config_name = write_rclone_config(s3)

    try:
        subprocess.run(
            [
                rclone,
                "--config",
                config_name,
                "copy",
                str(job_dir),
                f"garage:{s3['bucket']}/jobs/{job_id}",
                "--transfers",
                "2",
                "--checkers",
                "4",
                "--stats",
                "10s",
                "--stats-one-line",
            ],
            check=True,
        )
    finally:
        try:
            os.unlink(config_name)
        except FileNotFoundError:
            pass


def render() -> int:
    req = validate(read_json())
    s3 = req.pop("s3", None)
    if not isinstance(s3, dict):
        raise ValueError("S3-Konfiguration fehlt")

    ltx_python = required_executable(LTX_PYTHON, "LTX Python")
    required_executable(RCLONE, "rclone")

    missing_models = [
        str(path)
        for path in (TRANSFORMER, TEXT_ENCODER, VIDEO_VAE, AUDIO_VAE, SPATIAL)
        if not path.is_file()
    ]
    if missing_models:
        raise RuntimeError("Modelle fehlen: " + ", ".join(missing_models))

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    job_dir = OUTPUT_ROOT / req["job_id"]
    job_dir.mkdir(parents=True, exist_ok=False)

    (job_dir / "job.json").write_text(
        json.dumps(req, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    raw = job_dir / "output_raw.mp4"
    final = job_dir / "output_1080p.mp4"
    render_log = job_dir / "render.log"
    ffmpeg_log = job_dir / "ffmpeg.log"

    cmd = [
        ltx_python,
        "-m",
        "ltx_pipelines.distilled",
        "--transformer-path",
        str(TRANSFORMER),
        "--text-encoder-path",
        str(TEXT_ENCODER),
        "--video-vae-path",
        str(VIDEO_VAE),
        "--audio-vae-path",
        str(AUDIO_VAE),
        "--spatial-upsampler-path",
        str(SPATIAL),
        "--width",
        str(req["width"]),
        "--height",
        str(req["height"]),
        "--num-frames",
        str(req["frames"]),
        "--frame-rate",
        str(req["fps"]),
        "--seed",
        str(req["seed"]),
        "--quantization",
        str(req.get("quantization", "fp8-cast")),
        "--offload",
        str(req.get("offload", "cpu")),
        "--diffvae-optimization",
        str(req.get("diffvae_optimization", "chunked_eager")),
        "--output-path",
        str(raw),
        "--prompt",
        req["prompt"],
    ]

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    with render_log.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=LTX_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            log.flush()
            print(line, file=sys.stderr, end="", flush=True)
        rc = proc.wait()

    if rc != 0 or not raw.is_file():
        result = {
            "ok": False,
            "job_id": req["job_id"],
            "render_exit_code": rc,
        }
        (job_dir / "result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        emit(result)
        return rc or 1

    selected = raw
    ffmpeg = executable(FFMPEG)
    if req.get("postprocess_1080p") and ffmpeg:
        with ffmpeg_log.open("w", encoding="utf-8") as log:
            ff_rc = subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-i",
                    str(raw),
                    "-vf",
                    "crop=1920:1080:0:4",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-crf",
                    "18",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    str(final),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            ).returncode
        if ff_rc == 0 and final.is_file():
            selected = final

    # Vor dem Upload zunächst einen Status schreiben. Die S3-Credentials sind
    # bereits aus req entfernt und landen damit nie in job.json/result.json.
    result = {
        "ok": True,
        "job_id": req["job_id"],
        "seed": req["seed"],
        "job_dir": str(job_dir),
        "output": str(selected),
        "s3_prefix": f"jobs/{req['job_id']}",
        "uploaded": False,
    }
    result_file = job_dir / "result.json"
    result_file.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("Upload nach S3 via rclone ...", file=sys.stderr, flush=True)
    rclone_upload(job_dir, s3, req["job_id"])

    # Nach erfolgreichem Upload Status aktualisieren und result.json erneut
    # hochladen, damit Garage den finalen Status uploaded=true enthält.
    result["uploaded"] = True
    result_file.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Nur die kleine Statusdatei nochmals übertragen.
    rclone = required_executable(RCLONE, "rclone")
    _, config_name = write_rclone_config(s3)
    try:
        subprocess.run(
            [
                rclone,
                "--config",
                config_name,
                "copyto",
                str(result_file),
                f"garage:{s3['bucket']}/jobs/{req['job_id']}/result.json",
            ],
            check=True,
        )
    finally:
        try:
            os.unlink(config_name)
        except FileNotFoundError:
            pass

    s3.clear()
    emit(result)
    return 0


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: ltx_worker.py {health|prepare|render}", file=sys.stderr)
        return 2

    command = sys.argv[1]
    commands = {
        "health": health,
        "prepare": prepare,
        "render": render,
    }

    if command not in commands:
        print("Unbekanntes Kommando", file=sys.stderr)
        return 2

    try:
        return commands[command]()
    except Exception as exc:
        emit({"ok": False, "error": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
from __future__ import annotations
import json,os,re,shutil,subprocess,sys,tempfile
from pathlib import Path
LTX_DIR=Path('/opt/LTX-2'); MODEL_ROOT=Path('/workspace/models/ltx-2.5'); OUTPUT_ROOT=Path('/workspace/output/af_jobs')
TRANSFORMER=MODEL_ROOT/'diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors'; TEXT_ENCODER=MODEL_ROOT/'text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors'; VIDEO_VAE=MODEL_ROOT/'vae/ltx-2.5-video-vae-bf16.safetensors'; AUDIO_VAE=MODEL_ROOT/'vae/ltx-2.5-audio-vae-bf16.safetensors'; SPATIAL=MODEL_ROOT/'latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors'
HF_REPO='Lightricks/LTX-2.5'; HF_FILES=['diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors','text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors','vae/ltx-2.5-video-vae-bf16.safetensors','vae/ltx-2.5-audio-vae-bf16.safetensors','latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors']; JOB_RE=re.compile(r'^[A-Za-z0-9_.-]{1,100}$')
def emit(o):print(json.dumps(o,ensure_ascii=False),flush=True)
def read_json():
    v=json.loads(sys.stdin.read() or '{}')
    if not isinstance(v,dict):raise ValueError('JSON root muss Objekt sein')
    return v
def health():
    gpu='nvidia-smi fehlt'
    if shutil.which('nvidia-smi'):gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True,capture_output=True).stdout.strip()
    emit({'ok':LTX_DIR.is_dir() and shutil.which('uv') is not None,'ltx':str(LTX_DIR),'uv':shutil.which('uv'),'rclone':shutil.which('rclone'),'ffmpeg':shutil.which('ffmpeg'),'gpu':gpu,'models':{'transformer':TRANSFORMER.exists(),'text_encoder':TEXT_ENCODER.exists(),'video_vae':VIDEO_VAE.exists(),'audio_vae':AUDIO_VAE.exists(),'spatial':SPATIAL.exists()}}); return 0
def hf_cmd():
    if shutil.which('hf'):return ['hf']
    p=LTX_DIR/'.venv/bin/hf'
    if p.exists():return [str(p)]
    return ['uv','run','hf']
def prepare():
    req=read_json(); token=str(req.get('hf_token','')).strip()
    if not token:emit({'ok':False,'error':'hf_token fehlt'}); return 2
    missing=[f for f in HF_FILES if not (MODEL_ROOT/f).is_file()]
    if not missing:emit({'ok':True,'status':'already_ready'}); return 0
    MODEL_ROOT.mkdir(parents=True,exist_ok=True); env=os.environ.copy(); env['HF_TOKEN']=token; cmd=hf_cmd()+['download',HF_REPO]+missing+['--local-dir',str(MODEL_ROOT)]; p=subprocess.Popen(cmd,cwd=LTX_DIR,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1); assert p.stdout is not None
    for line in p.stdout:print(line,file=sys.stderr,end='',flush=True)
    rc=p.wait(); env.pop('HF_TOKEN',None); token=''; emit({'ok':rc==0,'status':'downloaded' if rc==0 else 'failed','exit_code':rc}); return rc
def validate(req):
    job=str(req.get('job_id','')).strip(); prompt=str(req.get('prompt','')).strip()
    if not JOB_RE.fullmatch(job):raise ValueError('Ungültige job_id')
    if not prompt:raise ValueError('Prompt fehlt')
    frames=int(req.get('frames',121))
    if frames<=0 or frames%8!=1:raise ValueError('frames muss >0 und frames % 8 == 1 sein')
    req.update(job_id=job,prompt=prompt,width=int(req.get('width',1920)),height=int(req.get('height',1088)),frames=frames,fps=int(req.get('fps',24)),seed=int(req.get('seed',1))); return req
def rclone_upload(job_dir,s3,job_id):
    if not shutil.which('rclone'):raise RuntimeError('rclone fehlt auf der VM')
    cfg=f"""[garage]\ntype = s3\nprovider = Other\nenv_auth = false\naccess_key_id = {s3['access_key']}\nsecret_access_key = {s3['secret_key']}\nregion = {s3['region']}\nendpoint = {s3['endpoint']}\nforce_path_style = true\nacl = private\n"""
    fd,name=tempfile.mkstemp(prefix='af-rclone-',suffix='.conf'); os.fchmod(fd,0o600)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:f.write(cfg)
        subprocess.run(['rclone','--config',name,'copy',str(job_dir),f"garage:{s3['bucket']}/jobs/{job_id}",'--progress'],check=True)
    finally:
        try:os.unlink(name)
        except FileNotFoundError:pass
def render():
    req=validate(read_json()); s3=req.pop('s3',None)
    if not isinstance(s3,dict):raise ValueError('S3-Konfiguration fehlt')
    miss=[str(x) for x in (TRANSFORMER,TEXT_ENCODER,VIDEO_VAE,AUDIO_VAE,SPATIAL) if not x.is_file()]
    if miss:raise RuntimeError('Modelle fehlen: '+', '.join(miss))
    OUTPUT_ROOT.mkdir(parents=True,exist_ok=True); job_dir=OUTPUT_ROOT/req['job_id']; job_dir.mkdir(parents=True,exist_ok=False); (job_dir/'job.json').write_text(json.dumps(req,indent=2,ensure_ascii=False)+'\n',encoding='utf-8'); raw=job_dir/'output_raw.mp4'; final=job_dir/'output_1080p.mp4'; render_log=job_dir/'render.log'; ffmpeg_log=job_dir/'ffmpeg.log'
    cmd=['uv','run','python','-m','ltx_pipelines.distilled','--transformer-path',str(TRANSFORMER),'--text-encoder-path',str(TEXT_ENCODER),'--video-vae-path',str(VIDEO_VAE),'--audio-vae-path',str(AUDIO_VAE),'--spatial-upsampler-path',str(SPATIAL),'--width',str(req['width']),'--height',str(req['height']),'--num-frames',str(req['frames']),'--frame-rate',str(req['fps']),'--seed',str(req['seed']),'--quantization',str(req.get('quantization','fp8-cast')),'--offload',str(req.get('offload','cpu')),'--diffvae-optimization',str(req.get('diffvae_optimization','chunked_eager')),'--output-path',str(raw),'--prompt',req['prompt']]
    env=os.environ.copy(); env['CC']=env.get('CC','/usr/bin/gcc'); env['CXX']=env.get('CXX','/usr/bin/g++'); env['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    with render_log.open('w',encoding='utf-8') as log:
        p=subprocess.Popen(cmd,cwd=LTX_DIR,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1); assert p.stdout is not None
        for line in p.stdout:log.write(line); log.flush(); print(line,file=sys.stderr,end='',flush=True)
        rc=p.wait()
    if rc!=0 or not raw.is_file():result={'ok':False,'job_id':req['job_id'],'render_exit_code':rc}; (job_dir/'result.json').write_text(json.dumps(result,indent=2)+'\n'); emit(result); return rc or 1
    selected=raw
    if req.get('postprocess_1080p') and shutil.which('ffmpeg'):
        with ffmpeg_log.open('w',encoding='utf-8') as log:ffrc=subprocess.run(['ffmpeg','-y','-i',str(raw),'-vf','crop=1920:1080:0:4','-c:v','libx264','-preset','medium','-crf','18','-c:a','aac','-b:a','192k',str(final)],stdout=log,stderr=subprocess.STDOUT).returncode
        if ffrc==0 and final.is_file():selected=final
    result={'ok':True,'job_id':req['job_id'],'seed':req['seed'],'job_dir':str(job_dir),'output':str(selected),'s3_prefix':f"jobs/{req['job_id']}"}; (job_dir/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8'); print('Upload nach S3 via rclone ...',file=sys.stderr,flush=True); rclone_upload(job_dir,s3,req['job_id']); s3.clear(); emit(result); return 0
def main():
    if len(sys.argv)!=2:print('Usage: ltx_worker.py {health|prepare|render}',file=sys.stderr); return 2
    try:return {'health':health,'prepare':prepare,'render':render}[sys.argv[1]]()
    except KeyError:print('Unbekanntes Kommando',file=sys.stderr); return 2
    except Exception as e:emit({'ok':False,'error':str(e)}); return 1
if __name__=='__main__':raise SystemExit(main())
