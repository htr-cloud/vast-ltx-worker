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
