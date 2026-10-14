# C3-OPD final experimental source overlay; deployment paths made configurable.
"""Batch HTTP retrieval without changing SDAR's E5 encoder or Flat index."""
from pathlib import Path
import os,time,queue,threading,concurrent.futures,asyncio,json
os.environ.setdefault('OMP_NUM_THREADS','32')
os.environ.setdefault('MKL_NUM_THREADS','32')
# Load the project-local threaded BLAS before Faiss's bundled single-thread BLAS.
import ctypes
ROOT=Path(os.environ.get('C3_PROJECT_ROOT', os.getcwd())).resolve()
blas_runtime={'library': os.environ['C3_OPENBLAS_LIBRARY']}
blas=ctypes.CDLL(blas_runtime['library'],mode=ctypes.RTLD_GLOBAL)
blas.openblas_set_num_threads.argtypes=[ctypes.c_int]
blas.openblas_set_num_threads(32)
blas.openblas_get_num_threads.restype=ctypes.c_int
assert blas.openblas_get_num_threads()==32
import faiss,torch,uvicorn
from fastapi import FastAPI,HTTPException
from retrieval_server import DenseRetriever,Config,QueryRequest
ROOT=Path(os.environ.get('C3_PROJECT_ROOT', os.getcwd())).resolve()
MAX_BATCH=128;WAIT_S=.1
app=FastAPI();pending=queue.Queue(maxsize=2048)
retriever=None;health={};counts={'requests':0,'batches':0,'search_seconds':0.0}

def batch_worker():
 faiss.omp_set_num_threads(64)
 while True:
  batch=[pending.get()];deadline=time.monotonic()+WAIT_S
  while len(batch)<MAX_BATCH:
   try:batch.append(pending.get(timeout=max(0,deadline-time.monotonic())))
   except queue.Empty:break
  grouped={}
  for item in batch:grouped.setdefault(item[0].topk or 3,[]).append(item)
  for topk,items in grouped.items():
   t=time.monotonic()
   try:
    docs,scores=retriever.batch_search([i[0].query for i in items],num=topk,return_score=True)
    # Return transient encoder allocations so the frozen scorer can use the GPU later.
    torch.cuda.empty_cache()
    for (request,future),dd,ss in zip(items,docs,scores):
     result=[{'document':doc,'score':float(score)} for doc,score in zip(dd,ss)] if request.return_scores else dd
     future.set_result({'result':[result]})
    counts['requests']+=len(items);counts['batches']+=1;counts['search_seconds']+=time.monotonic()-t
    print(json.dumps({'batch_queries':len(items),'topk':topk,'seconds':round(time.monotonic()-t,4),'queue_remaining':pending.qsize()}),flush=True)
   except Exception as e:
    for _,future in items:
     if not future.done():future.set_exception(e)

@app.get('/health')
async def get_health():return {**health,'counts':dict(counts),'queue':pending.qsize()}

@app.post('/retrieve')
async def retrieve(request:QueryRequest):
 if not request.query.strip():raise HTTPException(400,'Query must not be empty')
 if request.topk is not None and not 1<=request.topk<=20:raise HTTPException(400,'topk out of range')
 future=concurrent.futures.Future()
 try:pending.put_nowait((request,future))
 except queue.Full:raise HTTPException(503,'Retrieval queue full')
 return await asyncio.wrap_future(future)

if __name__=='__main__':
 t=time.monotonic();faiss.omp_set_num_threads(64)
 config=Config(retrieval_method='e5',retrieval_topk=3,index_path=str(ROOT/'data/search/e5_Flat.index'),corpus_path=str(ROOT/'data/search/wiki-18.jsonl'),faiss_gpu=False,retrieval_model_path=str(ROOT/'data/e5-base-v2'),retrieval_pooling_method='mean',retrieval_query_max_length=256,retrieval_use_fp16=True,retrieval_batch_size=128)
 retriever=DenseRetriever(config)
 assert retriever.index.ntotal==len(retriever.corpus)==21015324
 assert type(retriever.index).__name__=='IndexFlatIP' and retriever.index.d==768
 # Compare the optimized blocked GEMM path with the original direct SIMD path
 # on the same actual E5 embeddings and complete original index.
 import pandas as pd,numpy as np
 rows=pd.read_parquet(ROOT/'data/search/processed/train.parquet',columns=['env_kwargs','data_source'])
 questions=[]
 for tag in ['nq','hotpotqa']:
  questions.extend(x['question'] for x in rows[rows['data_source'].str.contains(tag,case=False)].head(4)['env_kwargs'])
 del rows
 assert len(questions)==8
 embeddings=retriever.encoder.encode(questions)
 faiss.cvar.distance_compute_blas_threshold=20
 baseline_start=time.monotonic();baseline_scores,baseline_ids=retriever.index.search(embeddings,3);baseline_s=time.monotonic()-baseline_start
 faiss.cvar.distance_compute_blas_database_bs=65536
 faiss.cvar.distance_compute_blas_query_bs=128
 faiss.cvar.distance_compute_blas_threshold=1
 optimized_start=time.monotonic();scores,ids=retriever.index.search(embeddings,3);optimized_s=time.monotonic()-optimized_start
 assert np.array_equal(ids,baseline_ids),'Tuned retrieval changed top-3 document IDs'
 assert np.allclose(scores,baseline_scores,rtol=1e-4,atol=2e-5),'Tuned retrieval changed scores beyond tolerance'
 validation={'status':'passed','real_queries':8,'same_e5_embeddings':True,'complete_original_index':True,'top3_ids_equal':True,'max_score_abs_error':float(np.max(np.abs(scores-baseline_scores))),'original_direct_path_s':baseline_s,'threaded_blas_path_s':optimized_s,'blas_threads':blas.openblas_get_num_threads(),'faiss_threads':64,'blas_library':blas_runtime['library'],'database_block':65536,'query_block':128,'blas_threshold':1}
 (ROOT/'runs/c3_search_retriever_startup_validation.json').write_text(json.dumps(validation,indent=2))
 torch.cuda.empty_cache()
 print('NUMERICAL VALIDATION '+json.dumps(validation),flush=True)
 health={'status':'ready','pid':os.getpid(),'index':'original_wiki18_e5_Flat','index_type':type(retriever.index).__name__,'index_device':'cpu','index_threads':64,'blas_threads':32,'blas_library':blas_runtime['library'],'database_block':65536,'query_block':128,'blas_threshold':1,'encoder_device':'cuda','cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'encoder':'e5-base-v2','encoder_fp16':True,'documents':retriever.index.ntotal,'dimension':768,'topk':3,'max_query_length':256,'max_batch':MAX_BATCH,'batch_wait_s':WAIT_S,'cache':False,'initialization_seconds':time.monotonic()-t}
 (ROOT/'runs/c3_search_retriever.json').write_text(json.dumps(health,indent=2));threading.Thread(target=batch_worker,daemon=True).start()
 print('READY '+json.dumps(health),flush=True);uvicorn.run(app,host='127.0.0.1',port=18890,log_level='warning')
