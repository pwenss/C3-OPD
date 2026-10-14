# C3-OPD final experimental source overlay; deployment paths made configurable.

"""Fixed external C3 teacher/verifier. Training imports only the lightweight client."""
from __future__ import annotations
import json,os,subprocess,time,uuid,urllib.request,urllib.error
from pathlib import Path
ROOT=Path(os.environ.get("C3_PROJECT_ROOT", os.getcwd())).resolve()
IPC=ROOT/"cache/c3-scoring"
def _set_engine_sleeping(llm, sleeping, sleep_level):
    """Restore the same frozen weights before permitting a level-2 score."""
    if sleep_level not in (1, 2):
        raise ValueError("Unsupported Teacher sleep level")
    if sleeping:
        llm.sleep(level=sleep_level)
    else:
        llm.wake_up()
        if sleep_level == 2:
            llm.collective_rpc("reload_weights", timeout=180)

class ExternalScorerClient:
    def __init__(self,cfg):
        self.cfg=dict(cfg)
        self.replicas=[]
        dp=int(self.cfg.get("external_data_parallel_size",1))
        if dp>1:
            from concurrent.futures import ThreadPoolExecutor
            devices=str(self.cfg.get("external_gpus","2,3")).split(",")
            tp=int(self.cfg.get("external_tensor_parallel_size",1))
            if tp!=1 or len(devices)!=dp:
                raise ValueError("External DP requires one dedicated GPU per TP=1 replica")
            def init_replica(rank):
                child=dict(self.cfg,external_data_parallel_size=1,external_gpus=devices[rank].strip(),
                           external_port=int(self.cfg.get("external_port",18881))+rank,external_replica_rank=rank)
                return ExternalScorerClient(child)
            with ThreadPoolExecutor(max_workers=dp) as pool:
                self.replicas=list(pool.map(init_replica,range(dp)))
            self.model=self.replicas[0].model
            self.info=dict(self.replicas[0].info,data_parallel_size=dp,
                           cuda_visible_devices=",".join(devices),replicas=[x.info for x in self.replicas])
            print("[C3 external DP] ready "+json.dumps(self.info),flush=True)
            return
        self.port=int(self.cfg.get("external_port",18881))
        self.base=f"http://127.0.0.1:{self.port}"
        self.model=str(Path(self.cfg["teacher_model_path"]).resolve())
        run_name = str(self.cfg.get("external_run_name", ""))
        if run_name and (Path(run_name).name != run_name or run_name in (".", "..")):
            raise ValueError("Invalid external scorer run name")
        self.ipc = IPC / run_name if run_name else IPC
        self.ipc.mkdir(parents=True,exist_ok=True)
        try:
            health=self.health()
        except (OSError,urllib.error.URLError):
            self._start()
            deadline=time.monotonic()+int(self.cfg.get("external_startup_timeout",1200))
            while time.monotonic()<deadline:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"External scorer exited; see {self.log_path}")
                try:
                    health=self.health();break
                except (OSError,urllib.error.URLError):
                    time.sleep(2)
            else:raise TimeoutError(f"External scorer startup timed out; see {self.log_path}")
        if health["model"]!=self.model:
            raise RuntimeError(f"Scorer model mismatch: {health['model']} != {self.model}")
        if health["tensor_parallel_size"]!=int(self.cfg.get("external_tensor_parallel_size",2)):
            raise RuntimeError("Existing external scorer has a different tensor-parallel setting")
        expected={"prefix_caching":bool(self.cfg.get("external_enable_prefix_caching",False)),
                  "result_cache":bool(self.cfg.get("external_enable_result_cache",False)),
                  "max_model_len":int(self.cfg.get("external_max_model_len",40960)),
                  "max_num_batched_tokens":int(self.cfg.get("external_max_num_batched_tokens",4096)),
                  "teacher_batch_tokens":int(self.cfg.get("external_teacher_batch_tokens",4096)),
                  "verifier_batch_tokens":int(self.cfg.get("external_verifier_batch_tokens",4096)),
                  "scoring_protocol_version":2,
                  "colocated":bool(self.cfg.get("external_colocated",False)),
                  "sleep_level":int(self.cfg.get("external_sleep_level",1))}
        for key,value in expected.items():
            if health.get(key)!=value:raise RuntimeError(f"Existing scorer mismatch for {key}: {health.get(key)} != {value}")
        if self.cfg.get("external_local_skip_mode","individual")=="whole_batch" and health.get("local_skip_mode")!="whole_batch":
            raise RuntimeError("Existing scorer does not preserve complete Local batch boundaries")
        if self.cfg.get("skip_equal_reward_local",False) and not health.get("local_skip_supported",False):
            raise RuntimeError("Existing scorer does not support equal-reward Local skipping")
        self.info=health
        print("[C3 external] ready "+json.dumps(health),flush=True)
    def health(self):
        if self.replicas:
            return dict(self.info,replicas=[x.health() for x in self.replicas])
        with urllib.request.urlopen(self.base+"/health",timeout=3) as response:
            return json.load(response)
    def set_sleeping(self, sleeping):
        if not self.cfg.get("external_colocated", False):
            return
        if self.replicas:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=len(self.replicas)) as pool:
                return list(pool.map(lambda replica: replica.set_sleeping(sleeping), self.replicas))
        endpoint = "/sleep" if sleeping else "/wake_up"
        request = urllib.request.Request(self.base + endpoint, data=b"{}", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=300) as remote:
            status = json.load(remote)
        if not status.get("ok") or status.get("sleeping") != sleeping:
            raise RuntimeError("Scorer phase transition failed: " + str(status))
        print("[C3 scorer phase] " + json.dumps(status), flush=True)
        return status

    def _start(self):
        suffix="" if "external_replica_rank" not in self.cfg else "_dp"+str(self.cfg["external_replica_rank"])
        prefix = str(self.cfg.get("external_run_name", ""))
        self.log_path=ROOT/"runs"/((prefix+"_" if prefix else "")+"c3_external_scorer"+suffix+".log")
        config_file=self.ipc/("service"+suffix+"_config.json")
        config_file.write_text(json.dumps(self.cfg))
        env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(self.cfg.get("external_gpus","2,3")),
                   VLLM_USE_V1="1",VLLM_WORKER_MULTIPROC_METHOD="spawn",
                   HF_HOME=str(ROOT/"hf-cache"),HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
                   XDG_CACHE_HOME=str(ROOT/"cache"),TRITON_CACHE_DIR=str(ROOT/"cache/triton-external"),
                   TORCHINDUCTOR_CACHE_DIR=str(ROOT/"cache/torchinductor-external"),
                   VLLM_CACHE_ROOT=str(ROOT/"cache/vllm-external"),
                   TORCH_EXTENSIONS_DIR=str(ROOT/"cache/torch_extensions-external"),
                   TMPDIR=str(ROOT/"cache/tmp"),TOKENIZERS_PARALLELISM="false",
                   OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
                   PYTHONPATH=os.environ.get("SDAR_ROOT",str(ROOT/"SDAR")))
        python=str(self.cfg.get("external_python",ROOT/".scorer_venv/bin/python"))
        with self.log_path.open("a") as log:
            self.proc=subprocess.Popen([python,str(Path(__file__).resolve()),"--config",str(config_file)],
                cwd=str(ROOT),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (self.ipc/("service"+suffix+".pid")).write_text(str(self.proc.pid))
    def score(self,payload):
        if self.replicas:return self._score_replicas(payload)
        key=uuid.uuid4().hex
        request=self.ipc/(key+".request.json");response=self.ipc/(key+".response.json")
        try:
            request.write_text(json.dumps(payload,ensure_ascii=False))
            body=json.dumps({"request":str(request),"response":str(response)}).encode()
            req=urllib.request.Request(self.base+"/score",data=body,headers={"Content-Type":"application/json"})
            with urllib.request.urlopen(req,timeout=int(self.cfg.get("external_request_timeout",86400))) as remote:
                status=json.load(remote)
            if not status.get("ok"):raise RuntimeError(status)
            return json.loads(response.read_text())
        except urllib.error.HTTPError as exc:
            raise RuntimeError("External scoring failed: "+exc.read().decode()) from exc
        finally:
            request.unlink(missing_ok=True);response.unlink(missing_ok=True)


    def _score_replicas(self,payload):
        from concurrent.futures import ThreadPoolExecutor
        keys=("teacher_messages","teacher_response_ids","local_messages","valid_messages")
        if "local_required" in payload:keys+=("local_required",)
        count=len(payload[keys[0]])
        if not count or any(len(payload[k])!=count for k in keys):raise ValueError("Invalid DP scoring batch")
        shards=[list(range(rank,count,len(self.replicas))) for rank in range(len(self.replicas))]
        started=time.monotonic()
        def run(rank):
            indices=shards[rank]
            if not indices:return None
            child={k:([v[i] for i in indices] if k in keys else v) for k,v in payload.items()}
            return self.replicas[rank].score(child)
        with ThreadPoolExecutor(max_workers=len(self.replicas)) as pool:results=list(pool.map(run,range(len(self.replicas))))
        merged={k:[None]*count for k in ("teacher_log_probs","c","v","teacher_removed_tokens","teacher_cached_tokens","local_scored")}
        merged.update(prompt_lengths={k:[None]*count for k in ("teacher","local","valid")},generated_labels=[],model=self.info,
                      replica_timings=[None if x is None else x["timings"] for x in results],
                      replica_batch_stats=[None if x is None else x["batch_stats"] for x in results])
        for indices,result in zip(shards,results):
            if result is None:continue
            for key in ("teacher_log_probs","c","v","teacher_removed_tokens","teacher_cached_tokens","local_scored"):
                if key=="local_scored" and key not in result:result[key]=[True]*len(indices)
                if len(result[key])!=len(indices):raise ValueError("DP result size mismatch")
                for index,value in zip(indices,result[key]):merged[key][index]=value
            for stage,values in result["prompt_lengths"].items():
                for index,value in zip(indices,values):merged["prompt_lengths"][stage][index]=value
            for label in result["generated_labels"]:
                merged["generated_labels"].append(dict(label,index=indices[label["index"]]))
        merged["timings"]={stage:max(x["timings"][stage] for x in results if x is not None) for stage in ("teacher","local","valid")}
        merged["timings"]["dispatch_wall"]=time.monotonic()-started
        if any(x is None for key in ("teacher_log_probs","c","v") for x in merged[key]):raise ValueError("Incomplete DP results")
        print(f"[C3 external DP] completed rows={count} wall_s={merged['timings']['dispatch_wall']:.3f} replica_timings={merged['replica_timings']}",flush=True)
        return merged

def _serve(cfg):
    import torch
    import vllm
    from vllm import LLM,SamplingParams
    from transformers import AutoTokenizer
    from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
    import threading
    import traceback
    model=str(Path(cfg["teacher_model_path"]).resolve())
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
    student=Path(cfg["student_model_path"])
    a=json.loads((student/"tokenizer.json").read_text())
    b=json.loads((Path(model)/"tokenizer.json").read_text())
    va=dict(a["model"]["vocab"]);vb=dict(b["model"]["vocab"])
    va.update({x["content"]:x["id"] for x in a["added_tokens"]})
    vb.update({x["content"]:x["id"] for x in b["added_tokens"]})
    if va!=vb:raise ValueError("Student/teacher token-ID vocabularies differ; exact token scoring is required")
    label_ids=[tok.encode(x,add_special_tokens=False) for x in ("A","B")]
    if any(len(ids)!=1 for ids in label_ids):raise ValueError("A/B labels must be single tokens")
    label_a,label_b=label_ids[0][0],label_ids[1][0]
    model_len=int(cfg.get("external_max_model_len",40960))
    tp=int(cfg.get("external_tensor_parallel_size",2))
    print("[C3 external] loading model="+model+" TP="+str(tp),flush=True)
    llm=LLM(model=model,tokenizer=model,dtype="bfloat16",tensor_parallel_size=tp,
            distributed_executor_backend="mp",max_model_len=model_len,
            max_num_batched_tokens=int(cfg.get("external_max_num_batched_tokens",4096)),
            max_num_seqs=int(cfg.get("external_max_num_seqs",32)),
            gpu_memory_utilization=float(cfg.get("external_gpu_memory_utilization",.65)),
            enforce_eager=True,enable_chunked_prefill=True,enable_prefix_caching=bool(cfg.get("external_enable_prefix_caching",False)),
            limit_mm_per_prompt={"image":0,"video":0},logprobs_mode="processed_logprobs",
            generation_config="vllm",seed=0,disable_log_stats=True,
            enable_sleep_mode=bool(cfg.get("external_colocated",False)))
    health=dict(model=model,backend="vllm",vllm_version=vllm.__version__,tensor_parallel_size=tp,
                cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],pid=os.getpid(),
                label_ids=[label_a,label_b],vocab_ids_equal=True,max_model_len=model_len,prefix_caching=bool(cfg.get("external_enable_prefix_caching",False)),
                 max_num_batched_tokens=int(cfg.get("external_max_num_batched_tokens",4096)),
                 teacher_batch_tokens=int(cfg.get("external_teacher_batch_tokens",4096)),
                 verifier_batch_tokens=int(cfg.get("external_verifier_batch_tokens",4096)))
    budget=int(cfg.get("external_scoring_batch_tokens",4096))
    request_cap=int(cfg.get("external_scoring_batch_requests",16))
    def render(messages):
        return tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
    def teacher_prompt(messages,response_length):
        ids=render(messages)
        cap=min(int(cfg.get("teacher_prompt_length",8192)),model_len-response_length-1)
        if cap<1:raise ValueError("No space for teacher conditioning")
        if len(ids)<=cap:return ids,0
        # Preserve system PI, Task and chat framing; truncate only the last User's data.
        kept=[dict(x) for x in messages]
        user=next((i for i in range(len(kept)-1,-1,-1) if kept[i]["role"]=="user"),None)
        if user is None:raise ValueError("Cannot trim teacher data without a User message")
        content=tok.encode(kept[user]["content"],add_special_tokens=False)
        empty=[dict(x) for x in kept];empty[user]["content"]=""
        available=cap-len(render(empty))
        if available<1:raise ValueError("Teacher system/PI exceeds context cap")
        removed=0
        while True:
            kept[user]["content"]=tok.decode(content[-available:],skip_special_tokens=False)
            cropped=render(kept)
            if len(cropped)<=cap:break
            available-=max(1,len(cropped)-cap)
            if available<1:raise ValueError("Teacher prompt cannot fit while preserving PI")
        return cropped,len(ids)-len(cropped)
    from collections import OrderedDict
    from array import array
    import hashlib,struct
    from types import SimpleNamespace
    verifier_memo={stage:OrderedDict() for stage in ("local","valid")}
    memo_enabled=bool(cfg.get("external_enable_result_cache",False))
    memo_capacity=int(cfg.get("external_result_cache_batches",8192))
    if memo_capacity<1:raise ValueError("Verifier result cache capacity must be positive")
    local_skip_mode=cfg.get("external_local_skip_mode","individual")
    if local_skip_mode not in ("individual","whole_batch"):raise ValueError("Invalid Local skip mode")
    health.update(result_cache=memo_enabled,result_cache_batches=memo_capacity,scoring_protocol_version=2,local_skip_supported=True,local_skip_mode=local_skip_mode)
    def score(payload):
        teacher_messages=payload["teacher_messages"]
        response_ids=payload["teacher_response_ids"]
        local=payload["local_messages"];valid=payload["valid_messages"]
        if not(len(teacher_messages)==len(response_ids)==len(local)==len(valid)):
            raise ValueError("Teacher/verifier batch sizes differ")
        options=payload.get("benchmark_options",{})
        if options and not cfg.get("external_allow_benchmark_options",False):
            raise ValueError("Benchmark controls are disabled in training")
        if options.get("reset_prefix_cache",False) and health["prefix_caching"]:
            llm.reset_prefix_cache()  # vLLM V1 returns None after the synchronous reset.
        if options.get("reset_result_cache",False):
            for memo in verifier_memo.values():memo.clear()
        local_required=payload.get("local_required",[True]*len(local))
        if len(local_required)!=len(local) or any(type(x) is not bool for x in local_required):raise ValueError("Invalid Local scoring mask")
        local_scored=[True]*len(local)
        temp=float(payload.get("temperature",1))
        if abs(temp-1)>1e-7:raise ValueError("External teacher prompt-logprob scoring currently requires temperature=1")
        teacher_lps=[None]*len(response_ids);cs=[None]*len(local);vs=[None]*len(valid)
        timings={};lengths={"teacher":[],"local":[],"valid":[]};cuts=[];generated=[];teacher_cached_tokens=[];batch_stats={}
        def run(stage,items,params,consume):
            started=time.monotonic();done=0;pending=[];count=0
            stage_budget=int(options.get("teacher_batch_tokens" if stage=="teacher" else "verifier_batch_tokens",cfg.get("external_teacher_batch_tokens" if stage=="teacher" else "external_verifier_batch_tokens",budget)))
            stage_request_cap=int(options.get("batch_requests",request_cap))
            if stage_budget<1 or stage_request_cap<1:raise ValueError("Scoring batch limits must be positive")
            batch_stats[stage]=dict(calls=0,requests=len(items),max_requests_per_call=0,budget=stage_budget,request_cap=stage_request_cap,generate_calls=0,cached_calls=0,cached_requests=0,skipped_requests=0)
            print(f"[C3 external scoring] {stage} start requests={len(items)}",flush=True)
            def flush():
                nonlocal pending,count,done
                if not pending:return
                batch_stats[stage]["calls"]+=1
                batch_stats[stage]["max_requests_per_call"]=max(batch_stats[stage]["max_requests_per_call"],len(pending))
                if stage=="local" and local_skip_mode=="whole_batch" and not any(local_required[i] for i,_,_ in pending):
                    # Preserve all original pack boundaries. Only omit a whole batch whose c values cannot affect q.
                    for i,_,_ in pending:cs[i]=.5;local_scored[i]=False
                    batch_stats[stage]["skipped_requests"]+=len(pending)
                    done+=len(pending);pending=[];count=0
                    return
                # Cache the entire original packed batch: never regroup requests.
                # Only verifier logits are reused; teacher token scoring stays unchanged.
                memo=verifier_memo.get(stage) if memo_enabled and not options.get("disable_result_cache",False) else None
                key=None
                if memo is not None:
                    digest=hashlib.sha256()
                    for _,ids,_ in pending:
                        digest.update(struct.pack("<I",len(ids)))
                        digest.update(array("I",ids).tobytes())
                    key=digest.digest()
                cached=None if memo is None else memo.get(key)
                if cached is not None:
                    memo.move_to_end(key)
                    outputs=[SimpleNamespace(outputs=[SimpleNamespace(logprobs=[{
                        label_a:SimpleNamespace(logprob=la),label_b:SimpleNamespace(logprob=lb)}],
                        token_ids=label)],num_cached_tokens=0,result_cache_hit=True)
                        for la,lb,label in cached]
                    batch_stats[stage]["cached_calls"]+=1
                    batch_stats[stage]["cached_requests"]+=len(pending)
                else:
                    outputs=llm.generate([dict(prompt_token_ids=x[1]) for x in pending],params,use_tqdm=False)
                    batch_stats[stage]["generate_calls"]+=1
                    if memo is not None:
                        memo[key]=tuple((float(out.outputs[0].logprobs[0][label_a].logprob),
                                         float(out.outputs[0].logprobs[0][label_b].logprob),
                                         tuple(out.outputs[0].token_ids)) for out in outputs)
                        if len(memo)>memo_capacity:memo.popitem(last=False)
                if len(outputs)!=len(pending):raise RuntimeError("Unexpected scoring output count")
                for (index,ids,start),out in zip(pending,outputs):consume(index,ids,start,out)
                done+=len(pending)
                print(f"[C3 external scoring] {stage} {done}/{len(items)} elapsed_s={time.monotonic()-started:.1f}",flush=True)
                pending=[];count=0
            for index,item in enumerate(items):
                if stage=="teacher":
                    ids,cut=teacher_prompt(item,len(response_ids[index]))
                    cuts.append(cut);start=len(ids);ids=ids+response_ids[index]
                else:
                    ids=render(item);start=0
                    if len(ids)>model_len-1:raise ValueError(f"Complete {stage} prompt exceeds external context: {len(ids)}")
                lengths[stage].append(len(ids))
                if stage=="local" and local_skip_mode=="individual" and not local_required[index]:
                    # A_i=0 makes q=1, independently of c. The placeholder is never a verifier observation.
                    cs[index]=.5;local_scored[index]=False;done+=1;batch_stats[stage]["skipped_requests"]+=1
                    continue
                if pending and (count+len(ids)>stage_budget or len(pending)>=stage_request_cap):flush()
                pending.append((index,ids,start));count+=len(ids)
            flush();timings[stage]=time.monotonic()-started
        def consume_teacher(index,ids,start,out):
            values=[]
            for pos in range(start,len(ids)):
                entry=out.prompt_logprobs[pos]
                if entry is None or ids[pos] not in entry:raise RuntimeError("Missing teacher response log-prob")
                values.append(float(entry[ids[pos]].logprob))
            teacher_lps[index]=values
            teacher_cached_tokens.append(int(getattr(out,"num_cached_tokens",0) or 0))
        params=SamplingParams(temperature=1,top_p=1,top_k=-1,max_tokens=1,prompt_logprobs=0,
                              detokenize=False,skip_special_tokens=False)
        run("teacher",teacher_messages,params,consume_teacher)
        params=SamplingParams(temperature=1,top_p=1,top_k=-1,max_tokens=1,logprobs=2,
                              allowed_token_ids=[label_a,label_b],detokenize=False)
        def consume_label(target,stage):
            def consume(index,ids,start,out):
                entry=out.outputs[0].logprobs[0]
                if label_a not in entry or label_b not in entry:raise RuntimeError("Missing constrained A/B logits")
                la=float(entry[label_a].logprob);lb=float(entry[label_b].logprob)
                target[index]=float(torch.softmax(torch.tensor([la,lb],dtype=torch.float32),0)[0])
                generated.append(dict(stage=stage,index=index,generated_ids=list(out.outputs[0].token_ids),logprob_A=la,logprob_B=lb,cached_tokens=int(getattr(out,"num_cached_tokens",0) or 0),result_cache_hit=bool(getattr(out,"result_cache_hit",False))))
            return consume
        run("local",local,params,consume_label(cs,"local"));run("valid",valid,params,consume_label(vs,"valid"))
        if not all(all(__import__("math").isfinite(x) for x in seq) for seq in teacher_lps):
            raise RuntimeError("Nonfinite teacher scores")
        return dict(teacher_log_probs=teacher_lps,c=cs,v=vs,timings=timings,prompt_lengths=lengths,
                    teacher_removed_tokens=cuts,teacher_cached_tokens=teacher_cached_tokens,generated_labels=generated,model=health,batch_stats=batch_stats,local_scored=local_scored)
    def checked(path):
        p=Path(path).resolve()
        if not p.is_relative_to(IPC.resolve()):raise ValueError("IPC file must stay in project scoring cache")
        return p
    score_lock=threading.Lock()
    phase={"sleeping":False}
    sleep_level=int(cfg.get("external_sleep_level",1))
    if sleep_level not in (1,2):raise ValueError("Unsupported Teacher sleep level")
    health["sleep_level"]=sleep_level
    health["colocated"]=bool(cfg.get("external_colocated",False))
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def reply(self,status,obj):
            data=json.dumps(obj).encode();self.send_response(status);self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(data)));self.end_headers();self.wfile.write(data)
        def do_GET(self):
            self.reply(200,dict(health,**phase)) if self.path=="/health" else self.reply(404,{"error":"not found"})
        def do_POST(self):
            try:
                if self.path in ("/sleep", "/wake_up"):
                    if not health["colocated"]:raise ValueError("Phase sharing was not enabled")
                    requested=self.path=="/sleep"
                    with score_lock:
                        if requested!=phase["sleeping"]:
                            _set_engine_sleeping(llm,requested,sleep_level)
                            torch.cuda.synchronize();torch.cuda.empty_cache()
                            phase["sleeping"]=requested
                        result=dict(ok=True,pid=os.getpid(),cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),**phase,
                                    allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved())
                        print("[C3 external phase] "+json.dumps(result),flush=True)
                    self.reply(200,result);return
                if self.path!="/score":raise ValueError("Unknown endpoint")
                size=int(self.headers.get("Content-Length","0"))
                if size>8192:raise ValueError("IPC envelope too large")
                command=json.loads(self.rfile.read(size));request=checked(command["request"]);response=checked(command["response"])
                with score_lock:
                    if phase["sleeping"]:raise RuntimeError("Teacher must wake before scoring")
                    result=score(json.loads(request.read_text()))
                response.write_text(json.dumps(result));self.reply(200,{"ok":True})
            except Exception as exc:
                traceback.print_exc();self.reply(500,{"error":str(exc)})
    if health["colocated"]:
        _set_engine_sleeping(llm,True,sleep_level);torch.cuda.synchronize();torch.cuda.empty_cache();phase["sleeping"]=True
    print("[C3 external] READY "+json.dumps(dict(health,**phase)),flush=True)
    ThreadingHTTPServer(("127.0.0.1",int(cfg.get("external_port",18881))),Handler).serve_forever()
if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("--config",required=True);args=parser.parse_args()
    _serve(json.loads(Path(args.config).read_text()))
