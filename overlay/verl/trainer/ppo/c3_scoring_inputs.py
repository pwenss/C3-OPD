# C3-OPD final experimental source overlay; deployment paths made configurable.

"""Prompt-only external scorer inputs, preserving generated Student token IDs."""
import json,re
from pathlib import Path

def _task_environment(skill_provider):
    return Path(skill_provider.skills_dir).name.lower()

def _privileged_skills(skill_provider, task, prompt='', gamefile=None, data_source=None):
    if _task_environment(skill_provider) == 'search':
        return skill_provider.get_privileged_info_from_data_source(str(data_source or 'unknown'), task or prompt)
    if gamefile is not None:
        return skill_provider.get_privileged_info(str(gamefile))
    return skill_provider.get_privileged_info_from_prompt(task or prompt)
def parse_student_prompt(text):
    marker="<|im_start|>assistant\n"
    if text.endswith(marker):text=text[:-len(marker)]
    # Complete messages only; a left-truncated Student context becomes one User input.
    expression=re.compile(r"<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>\n?",re.S)
    matches=list(expression.finditer(text))
    if matches and not text[:matches[0].start()].strip():
        messages=[dict(role=m.group(1),content=m.group(2)) for m in matches]
        remaining=text[matches[-1].end():]
        if remaining.strip():raise ValueError("Unexpected Student assistant prefill")
        return messages
    text=text.removesuffix("<|im_end|>\n")
    return [dict(role="user",content=text)]
def build_external_teacher_inputs(batch,skill_provider,student_tokenizer):
    bs=len(batch.batch["responses"]);width=batch.batch["responses"].shape[1]
    prompts=[];responses=[];positions=[]
    for i in range(bs):
        ids=batch.batch["input_ids"][i];mask=batch.batch["attention_mask"][i]
        prompt_width=len(ids)-width
        text=student_tokenizer.decode(ids[:prompt_width][mask[:prompt_width].bool()],skip_special_tokens=False)
        messages=parse_student_prompt(text)
        files=batch.non_tensor_batch.get("gamefile")
        task=batch.non_tensor_batch.get("task_description")
        task_text="" if task is None else str(task[i])
        gf=None if files is None else files[i]
        data_source=batch.non_tensor_batch.get("data_source")
        source=None if data_source is None else data_source[i]
        skills=_privileged_skills(skill_provider,task_text,text,gf,source)
        prefix="[Privileged Skill Information]\n"+skills
        if task_text:
            label="Current ALFWorld Task" if _task_environment(skill_provider) == "alfworld" else "Current Task"
            prefix=label+": "+task_text+"\n\n"+prefix
        if messages and messages[0]["role"]=="system":
            messages[0]["content"]+="\n\n"+prefix
        else:messages.insert(0,dict(role="system",content=prefix))
        prompts.append(messages)
        pos=batch.batch["response_mask"][i].bool().nonzero(as_tuple=True)[0]
        positions.append(pos.tolist())
        responses.append(batch.batch["responses"][i][pos].tolist())
    return prompts,responses,positions
def extract_verifier_action(text, environment="alfworld"):
    """Use the environment action syntax; never infer actions from thinking."""
    if environment == "search":
        from agent_system.environments.env_package.search.projection import search_projection
        actions,_=search_projection([str(text)])
        return actions[0] or None
    text=str(text).lower();begin=text.find("<action>");end=text.find("</action>")
    return text[begin+8:end].strip() if 0<=begin<end else None

def build_verifier_messages(trainer,batch,templates):
    import numpy as np
    metadata=batch.non_tensor_batch
    # Follow the Student environment setting; there is no separate verifier window.
    history_length=max(0,int(trainer.config.env.history_length))
    observations=metadata.get("anchor_obs");results=metadata.get("next_anchor_obs")
    if observations is None or results is None:raise ValueError("Verifier requires raw anchor_obs/next_anchor_obs")
    bs=len(batch.batch["responses"]);trajs=metadata.get("traj_uid");turns=metadata.get("turn_step")
    if trajs is None or turns is None:raise ValueError("Missing trajectory/turn identity")
    tasks=metadata.get("task_description",[None]*bs);files=metadata.get("gamefile",[None]*bs)
    actions=metadata.get("admissible_actions",[None]*bs);responses=metadata.get("assistant_text")
    projected=metadata.get("student_action")
    environment=_task_environment(trainer.skill_provider)
    sources=metadata.get("data_source",[None]*bs)
    seen={}
    for i in range(bs):
        uid,j=str(trajs[i]),int(turns[i])
        if observations[i] is None or results[i] is None:raise ValueError(f"Missing raw state {uid}/{j}")
        answer=(str(responses[i]) if responses is not None else trainer._decode_valid(trainer.tokenizer,batch.batch["responses"][i],batch.batch["response_mask"][i]))
        pool=actions[i]
        if isinstance(pool,np.ndarray):pool=pool.tolist()
        record=dict(observation=trainer._task_text(observations[i]),result=trainer._task_text(results[i]),
                    task=trainer._task_text(tasks[i]),action=(extract_verifier_action(answer,environment) if projected is None else projected[i]),
                    commands=None if pool is None else list(pool),gamefile=None if files[i] is None else str(files[i]),data_source=None if sources[i] is None else str(sources[i]))
        prev=seen.setdefault(uid,{}).get(j)
        if prev is not None and prev!=record:raise ValueError(f"Conflicting duplicate turn {uid}/{j}")
        seen[uid][j]=record
    quote=lambda x:json.dumps(x,ensure_ascii=False)
    output=[];skill_cache={}
    for i in range(bs):
        uid,j=str(trajs[i]),int(turns[i]);trajectory=seen[uid];cur=trajectory[j]
        history_start=max(0,j-history_length)
        history_turns=range(history_start,j)
        if any(k not in trajectory for k in history_turns):
            raise ValueError(f"Incomplete recent history {uid}/{j}")
        task=cur["task"] or trajectory.get(0,{}).get("task","")
        if not task:raise ValueError(f"Missing Task {uid}")
        key=(cur["gamefile"],task,cur["data_source"])
        if key not in skill_cache:
            skill_cache[key]=_privileged_skills(trainer.skill_provider,task,gamefile=cur["gamefile"],data_source=cur["data_source"])
        history=[]
        for k in history_turns:
            prev=trajectory[k]
            if prev["result"]!=trajectory[k+1]["observation"]:
                raise ValueError(f"Discontinuous states {uid}/{k}")
            # SearchMemory stores action/result pairs; SimpleMemory (ALFWorld,
            # WebShop) stores observation/action pairs. Match these visible fields.
            if environment != "search":
                history.append(f"Student Turn {k+1} Observation: "+quote(prev["observation"]))
            history.append(f"Student Turn {k+1} Action: "+quote(prev["action"]))
            if environment == "search" and k<j-1:
                history.append("Resulting Observation: "+quote(prev["result"]))
        if not j:
            scope_note="This is the first turn; there are no preceding steps."
        elif not history_length:
            scope_note="History is disabled; earlier steps are not supplied."
        else:
            scope_note=f"Showing {j-history_start} preceding turns (turns {history_start+1}-{j}); at most {history_length}."
            if history_start:
                scope_note+=" Earlier turns are outside the visible window and must not be inferred."
        values=dict(task_json=quote(task),privileged_information_type="task_skills",privileged_information_json=quote(skill_cache[key]),
                    history_length=history_length,history_scope_note=scope_note,
                    preceding_transition_records="\n".join(history) if history else "No preceding history supplied.",
                    current_turn=j+1,current_observation_json=quote(cur["observation"]),
                    current_admissible_actions_json=quote(cur["commands"]),current_student_action_json=quote(cur["action"]),
                    current_resulting_observation_json=quote(cur["result"]))
        local=[dict(role="system",content=templates["local_system"]),dict(role="user",content=templates["local_user"].format(**values))]
        valid=[dict(role="system",content=templates["valid_system"]),dict(role="user",content=templates["valid_user"].format(**values))]
        output.append((local,valid,uid,j,float(metadata.get("episode_rewards",[0]*bs)[i])))
    return output
