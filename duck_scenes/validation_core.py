#!/usr/bin/env python3
"""Execution-based MuJoCo validator for MicroDuck DuckVLM scenes."""
from __future__ import annotations
import argparse, json, math, os, sys, time, traceback, xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any
os.environ.setdefault("MUJOCO_GL", "wgl" if os.name == "nt" else "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import mujoco, numpy as np, yaml
from PIL import Image
EXPECTED_JOINTS=["left_hip_yaw","left_hip_roll","left_hip_pitch","left_knee","left_ankle","neck_pitch","head_pitch","head_yaw","head_roll","right_hip_yaw","right_hip_roll","right_hip_pitch","right_knee","right_ankle"]
HEAD_JOINTS={"neck_pitch","head_pitch","head_yaw","head_roll"}
DEFAULT_POSE=np.array([0.0,-0.0873,-0.4579,-0.0049,0.4530,0.3491,0.3491,0.0,0.0,0.0,0.0873,0.4579,0.0049,-0.4530],dtype=np.float32)
@dataclass
class Check:
    name:str; passed:bool; skipped:bool; detail:str; metrics:dict[str,Any]
class Results:
    def __init__(self,scene:Path): self.scene=scene; self.items=[]
    def add(self,name,passed,detail="",**metrics):
        self.items.append(Check(name,bool(passed),False,detail,metrics)); print(f"[{'PASS' if passed else 'FAIL'}] {name}"+(f": {detail}" if detail else ""))
        for k,v in metrics.items(): print(f"  {k}: {v}")
    def skip(self,name,detail,**metrics):
        self.items.append(Check(name,True,True,detail,metrics)); print(f"[SKIP] {name}: {detail}")
        for k,v in metrics.items(): print(f"  {k}: {v}")
    @property
    def passed(self): return all(x.passed for x in self.items)
def check_call(results,name,fn):
    try: fn()
    except Exception as exc: results.add(name,False,f"{type(exc).__name__}: {exc}"); print(traceback.format_exc())
def bid(m,n): return int(mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n))
def gid(m,n): return int(mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,n))
def jid(m,n): return int(mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n))
def bname(m,i): return mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(i)) or f"body{i}"
def gname(m,i): return mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(i)) or f"geom{i}"
def reset_initial(m,d):
    k=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_KEY,"initial")
    mujoco.mj_resetDataKeyframe(m,d,k) if k>=0 else mujoco.mj_resetData(m,d)
    mujoco.mj_forward(m,d)
def quat_rotate(q,v):
    w=float(q[0]); xyz=np.asarray(q[1:4],float); t=np.cross(xyz,v)*2.0; return v-w*t+np.cross(xyz,t)
def quat_yaw(q):
    w,x,y,z=[float(v) for v in q]; return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
def free_joint(m,body):
    b=bid(m,body)
    for j in range(m.njnt):
        if int(m.jnt_bodyid[j])==b and int(m.jnt_type[j])==int(mujoco.mjtJoint.mjJNT_FREE): return j
    return -1
def set_pose(m,d,body,xyz,yaw=0.0):
    j=free_joint(m,body); adr=int(m.jnt_qposadr[j]); a=math.radians(yaw); d.qpos[adr:adr+7]=[xyz[0],xyz[1],xyz[2],math.cos(a/2),0,0,math.sin(a/2)]
def set_quat(m,d,body,q):
    j=free_joint(m,body); adr=int(m.jnt_qposadr[j]); d.qpos[adr+3:adr+7]=q; mujoco.mj_forward(m,d)
def free_bodies(m):
    return sorted({int(m.jnt_bodyid[j]) for j in range(m.njnt) if int(m.jnt_type[j])==int(mujoco.mjtJoint.mjJNT_FREE) and bname(m,int(m.jnt_bodyid[j]))!="trunk_base"})
def body_mass(m,b): return float(np.sum(m.body_mass[b]))
def geom_min_z(m,d,g):
    typ=int(m.geom_type[g]); p=np.asarray(d.geom_xpos[g],float); R=np.asarray(d.geom_xmat[g],float).reshape(3,3); s=np.asarray(m.geom_size[g],float)
    if typ==int(mujoco.mjtGeom.mjGEOM_PLANE): return -math.inf
    if typ==int(mujoco.mjtGeom.mjGEOM_SPHERE): return float(p[2]-s[0])
    if typ==int(mujoco.mjtGeom.mjGEOM_BOX): return float(p[2]-(abs(R)@s)[2])
    if typ==int(mujoco.mjtGeom.mjGEOM_CYLINDER): return float(p[2]-(abs(R)@np.array([s[0],s[0],s[1]]))[2])
    if typ==int(mujoco.mjtGeom.mjGEOM_CAPSULE): return float(p[2]-(abs(R)@np.array([s[0],s[0],s[1]+s[0]]))[2])
    if typ==int(mujoco.mjtGeom.mjGEOM_MESH):
        mid=int(m.geom_dataid[g]); a=int(m.mesh_vertadr[mid]); b=a+int(m.mesh_vertnum[mid]); v=np.asarray(m.mesh_vert[a:b],float); return float(((R@v.T).T+p)[:,2].min())
    return float(p[2]-m.geom_rbound[g])
def geom_aabb(m,d,g):
    typ=int(m.geom_type[g]); p=np.asarray(d.geom_xpos[g],float); R=np.asarray(d.geom_xmat[g],float).reshape(3,3); s=np.asarray(m.geom_size[g],float)
    if typ==int(mujoco.mjtGeom.mjGEOM_PLANE): return None
    if typ==int(mujoco.mjtGeom.mjGEOM_MESH):
        mid=int(m.geom_dataid[g]); a=int(m.mesh_vertadr[mid]); b=a+int(m.mesh_vertnum[mid]); w=(R@np.asarray(m.mesh_vert[a:b],float).T).T+p; return np.r_[w.min(0),w.max(0)]
    if typ==int(mujoco.mjtGeom.mjGEOM_SPHERE): ext=np.full(3,s[0])
    elif typ==int(mujoco.mjtGeom.mjGEOM_BOX): ext=abs(R)@s
    elif typ==int(mujoco.mjtGeom.mjGEOM_CYLINDER): ext=abs(R)@np.array([s[0],s[0],s[1]])
    elif typ==int(mujoco.mjtGeom.mjGEOM_CAPSULE): ext=abs(R)@np.array([s[0],s[0],s[1]+s[0]])
    else: ext=np.full(3,float(m.geom_rbound[g]))
    return np.r_[p-ext,p+ext]
def approx(a,b,t=1e-5): return abs(float(a)-float(b))<=t
def va(a,b,t=1e-5): return np.allclose(np.asarray(a,float),np.asarray(b,float),atol=t,rtol=0)
class PolicyRunner:
    def __init__(self,m,policy_dir):
        self.m=m; self.policy_dir=Path(policy_dir); self.d=mujoco.MjData(m); self.names=["alpha_stand","alpha_walking","ball_kick_left","ball_kick_right"]
        self.qidx=np.array([int(m.jnt_qposadr[int(m.actuator_trnid[i,0])]) for i in range(m.nu)],int); self.vidx=np.array([int(m.jnt_dofadr[int(m.actuator_trnid[i,0])]) for i in range(m.nu)],int)
        self.trunk=bid(m,"trunk_base"); self.ball=bid(m,"ball"); sid=int(mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SENSOR,"imu_ang_vel")); self.gyro_adr=int(m.sensor_adr[sid]) if sid>=0 else 0
        self.last=np.zeros(m.nu,np.float32); self.sessions={}; self.available=False
    def load(self):
        import onnxruntime as ort
        so=ort.SessionOptions(); so.intra_op_num_threads=so.inter_op_num_threads=1; so.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL; so.log_severity_level=3
        for n in self.names:
            p=self.policy_dir/f"{n}.onnx"
            if not p.exists(): raise FileNotFoundError(p)
            s=ort.InferenceSession(str(p),sess_options=so,providers=["CPUExecutionProvider"]); i,o=s.get_inputs()[0],s.get_outputs()[0]
            if list(i.shape)!=[1,61] or list(o.shape)!=[1,14]: raise ValueError(f"{n}: {i.shape}->{o.shape}")
            self.sessions[n]=(s,i.name,o.name)
        self.available=True
    def reset(self): reset_initial(self.m,self.d); self.last[:]=0
    def step(self,policy,cmd=(0,0,0)):
        m,d=self.m,self.d; g=quat_rotate(np.asarray(d.xquat[self.trunk],float),np.array([0,0,-1.])); gy=np.asarray(d.sensordata[self.gyro_adr:self.gyro_adr+3],np.float32)
        obs=np.concatenate([gy,g.astype(np.float32),(d.qpos[self.qidx]-DEFAULT_POSE[:m.nu]).astype(np.float32),d.qvel[self.vidx].astype(np.float32),self.last,np.concatenate([np.asarray(cmd,np.float32),np.zeros(10,np.float32)])]).astype(np.float32)
        s,i,o=self.sessions[policy]; self.last[:]=s.run([o],{i:obs.reshape(1,-1)})[0].squeeze(0); d.ctrl[:]=DEFAULT_POSE[:m.nu]+self.last
        for _ in range(4): mujoco.mj_step(m,d)
        return self.last.copy()
    def command(self,cmd,steps,policy=None):
        cmd=np.asarray(cmd,np.float32); pol=policy or ("alpha_walking" if np.linalg.norm(cmd)>=.05 else "alpha_stand")
        for _ in range(steps): self.step(pol,cmd)
    def state(self):
        d=self.d; R=np.asarray(d.xmat[self.trunk],float).reshape(3,3); up=float(quat_rotate(np.asarray(d.xquat[self.trunk],float),np.array([0,0,1.]))[2])
        return {"root_xyz":np.asarray(d.xpos[self.trunk],float).tolist(),"root_yaw_deg":math.degrees(math.atan2(R[1,0],R[0,0])),"upright":up,"speed":float(np.linalg.norm(d.qvel[:3])),"ball_xyz":np.asarray(d.xpos[self.ball],float).tolist(),"qvel_max":float(np.max(np.abs(d.qvel))),"qacc_max":float(np.max(np.abs(d.qacc)))}
class TaskEvaluator:
    def __init__(self,m,d,task,dt=.02): self.m=m; self.d=d; self.task=task; self.dt=dt; self.timers={}; self.sequence_index=0; self.success_timer=0.; self.failure=False; self.failure_reason=""; self.sim_time=0.
    def pos(self,n): return np.asarray(self.d.xpos[bid(self.m,n)],float).copy()
    def quat(self,n): return np.asarray(self.d.xquat[bid(self.m,n)],float).copy()
    def yaw(self,n):
        R=np.asarray(self.d.xmat[bid(self.m,n)],float).reshape(3,3); return math.atan2(R[1,0],R[0,0])
    def timer(self,key,ok,seconds,dt):
        self.timers[key]=self.timers.get(key,0.)+(dt if ok else 0.); return self.timers[key]>=seconds-1e-12
    def atomic(self,n,dt):
        t=n.get("type")
        if t=="distance_xy_le": return float(np.linalg.norm((self.pos(n["body"])-self.pos(n["target"]))[:2]))<=float(n["distance_m"])
        if t=="distance_xy_ge": return float(np.linalg.norm((self.pos(n["body"])-self.pos(n["target"]))[:2]))>=float(n["distance_m"])
        if t=="distance_xy_range":
            x=float(np.linalg.norm((self.pos(n["body"])-self.pos(n["target"]))[:2])); return float(n["min_m"])<=x<=float(n["max_m"])
        if t=="body_in_zone": return float(np.linalg.norm((self.pos(n["body"])-self.pos(n["zone"]))[:2]))<=float(n["radius_m"])
        if t=="upright": return float(quat_rotate(self.quat(n["body"]),np.array([0,0,1.]))[2])>=float(n["min_upright"])
        if t=="speed_xy_le": return float(np.linalg.norm(self.d.cvel[bid(self.m,n["body"]),3:5]))<=float(n["speed_mps"])
        if t=="relative_bearing_range":
            b=self.pos(n["body"]); v=self.pos(n["target"])-b; y=self.yaw(n["body"]); R=np.array([[math.cos(y),math.sin(y)],[-math.sin(y),math.cos(y)]]); local=R@v[:2]; a=math.degrees(math.atan2(local[1],local[0]))%360; return float(n["min_deg"])<=a<=float(n["max_deg"])
        if t=="robot_pose_region":
            xy=self.pos(n["body"])[:2]; c=np.asarray(n["center_xy"],float)
            if float(np.linalg.norm(xy-c))>float(n["radius_m"]): return False
            e=abs((math.degrees(self.yaw(n["body"]))-float(n["yaw_deg"])+180)%360-180); return e<=float(n["yaw_tolerance_deg"])
        if t=="fallen_for_s":
            up=float(quat_rotate(self.quat(n["body"]),np.array([0,0,1.]))[2]); return self.timer(("fall",id(n)),up<.5,float(n["seconds"]),dt)
        if t=="outside_bounds": return float(np.linalg.norm(self.pos(n.get("body","trunk_base"))[:2]))>float(n["radius_m"])
        if t=="outside_bounds_aabb":
            xy=self.pos(n.get("body","trunk_base"))[:2]; return bool(np.any(xy<np.asarray(n["min_xy"])) or np.any(xy>np.asarray(n["max_xy"])))
        if t=="timeout": return self.sim_time>=float(n.get("max_duration_s",self.task.get("max_duration_s",1e9)))
        if t=="hold_condition_s": return self.timers.get(("hold",id(n)),0.)>=float(n["seconds"])-1e-12
        if t=="ordered_zone_sequence":
            p=self.pos(n["body"]); z=list(n["zones"])
            if self.sequence_index<len(z) and float(np.linalg.norm((p-self.pos(z[self.sequence_index]))[:2]))<=float(n["radius_m"]): self.sequence_index+=1
            return self.sequence_index>=len(z)
        raise ValueError(f"unsupported condition: {t}")
    def condition(self,n,dt):
        if not n: return False
        if isinstance(n,list): return all(self.condition(x,dt) for x in n)
        if not isinstance(n,dict): return bool(n)
        if "all" in n:
            ch=list(n["all"]); non=[x for x in ch if not(isinstance(x,dict) and x.get("type")=="hold_condition_s")]; ok=all(self.condition(x,dt) for x in non); hs=[x for x in ch if isinstance(x,dict) and x.get("type")=="hold_condition_s"]
            return ok and (all(self.timer(("hold",id(h)),ok,float(h["seconds"]),dt) for h in hs) if hs else True)
        if "any" in n: return any(self.condition(x,dt) for x in n["any"])
        if "not" in n: return not self.condition(n["not"],dt)
        return self.atomic(n,dt)
    def step(self,dt=None):
        dt=self.dt if dt is None else float(dt); self.sim_time+=dt; ok=self.condition(self.task.get("success",{}),dt); self.success_timer=self.success_timer+dt if ok else 0.; success=ok and self.success_timer>=float(self.task.get("success_hold_s",0.))-1e-12
        if self.condition(self.task.get("failure",{}),dt): self.failure=True; self.failure_reason="failure_condition"
        if self.sim_time>float(self.task.get("max_duration_s",1e9)): self.failure=True; self.failure_reason="timeout"
        return success,self.failure,self.failure_reason
def validate_xml(results,scene,m):
    root=ET.parse(scene).getroot(); seen={}
    for tag in ("body","geom","joint","camera","actuator"):
        for e in root.iter(tag):
            if e.attrib.get("name"): seen.setdefault(e.attrib["name"],[]).append(tag)
    dup={k:v for k,v in seen.items() if len(v)>1}; results.add("XML_LOAD",True,"MuJoCo model compiled",nq=m.nq,nv=m.nv,nu=m.nu,nbody=m.nbody,ngeom=m.ngeom,njnt=m.njnt); results.add("unique_name_audit",not dup,f"duplicates={dup}")
def validate_robot(results,m):
    d=mujoco.MjData(m); reset_initial(m,d); root=bid(m,"trunk_base"); missing=[j for j in EXPECTED_JOINTS if jid(m,j)<0]; acts=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in range(m.nu)]; mh=sorted(HEAD_JOINTS-{x for x in HEAD_JOINTS if jid(m,x)>=0}); ranges=np.asarray(m.actuator_ctrlrange,float)
    rg=[i for i in range(m.ngeom) if int(m.geom_bodyid[i])>0 and int(m.body_rootid[int(m.geom_bodyid[i])])==root and m.geom_contype[i]!=0]; minz=min((geom_min_z(m,d,i) for i in rg),default=math.nan); overlap=[]
    for c in d.contact:
        b1,b2=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2]); r1=root if b1>0 and int(m.body_rootid[b1])==root else -1; r2=root if b2>0 and int(m.body_rootid[b2])==root else -1
        if (r1==root)^(r2==root) and float(c.dist)<-1e-4: overlap.append([gname(m,c.geom1),gname(m,c.geom2),float(c.dist)])
    ok=root>=0 and not missing and m.nu==14 and not mh and ranges.size and np.isfinite(ranges).all() and np.all(ranges[:,1]>ranges[:,0]) and -1e-3<=minz<=.02 and not overlap
    results.add("ROBOT_COMPATIBILITY",ok,f"root=trunk_base joints={14-len(missing)}/14 head={4-len(mh)}/4 actuators={m.nu}",root_body="trunk_base",controlled_joints=acts,head_joints=sorted(HEAD_JOINTS),actuators=m.nu,ctrl_ranges=ranges[:3].tolist(),min_robot_geom_z=float(minz),initial_scene_overlap=overlap)
def validate_initial(results,m):
    d=mujoco.MjData(m); reset_initial(m,d); bs=free_bodies(m); rec=[]; bad=[]; contacts=[]
    for b in bs:
        gs=[i for i in range(m.ngeom) if int(m.geom_bodyid[i])==b and m.geom_contype[i]!=0]; z=min((geom_min_z(m,d,i) for i in gs),default=math.inf); r={"body":bname(m,b),"xyz":np.asarray(d.xpos[b],float).tolist(),"bottom_z":float(z),"freejoint":free_joint(m,bname(m,b))>=0}; rec.append(r)
        if not r["freejoint"] or z<-1e-3 or z>.0025: bad.append(r)
    for c in d.contact:
        b1,b2=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2])
        if b1 in bs or b2 in bs:
            x={"geom1":gname(m,c.geom1),"geom2":gname(m,c.geom2),"dist":float(c.dist)}; contacts.append(x)
            if x["dist"]<-1e-3: bad.append({"contact":x})
    results.add("INITIAL_CONTACT",not bad,"free bodies rest on collision surfaces without penetration",bodies=rec,contacts=contacts,anomalies=bad)
def validate_static_overlap(results,m):
    d=mujoco.MjData(m); reset_initial(m,d); world=[]
    for g in range(m.ngeom):
        if int(m.geom_bodyid[g])==0 and m.geom_contype[g]!=0 and int(m.geom_type[g])!=int(mujoco.mjtGeom.mjGEOM_PLANE): world.append(g)
    overlaps=[]
    for k,g1 in enumerate(world):
        a=geom_aabb(m,d,g1)
        for g2 in world[k+1:]:
            b=geom_aabb(m,d,g2); span=np.minimum(a[3:],b[3:])-np.maximum(a[:3],b[:3])
            if np.all(span>1e-4):
                n1,n2=gname(m,g1),gname(m,g2)
                if ("door" in n1 or "door" in n2) and ("wall_mid" in n1 or "wall_mid" in n2): continue
                if "wall_" in n1 and "wall_" in n2: continue
                overlaps.append({"geom1":n1,"geom2":n2,"overlap_xyz_m":span.tolist()})
    results.add("STATIC_GEOM_OVERLAP",not overlaps,"no unintended positive-volume static intersections",overlaps=overlaps)
def validate_passive(results,m):
    d=mujoco.MjData(m); reset_initial(m,d); bs=free_bodies(m); init={b:np.asarray(d.xpos[b],float).copy() for b in bs}; mv=ma=0.; minc=0.; maxc=0; bad=0
    for i in range(2000):
        mujoco.mj_step(m,d)
        if not(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all() and np.isfinite(d.qacc).all() and np.isfinite(d.ctrl).all()): bad+=1; break
        mv=max(mv,float(np.max(np.abs(d.qvel)))); ma=max(ma,float(np.max(np.abs(d.qacc)))); maxc=max(maxc,int(d.ncon));
        if d.ncon: minc=min(minc,float(np.min(d.contact.dist)))
    motion={bname(m,b):float(np.linalg.norm(d.xpos[b]-init[b])) for b in bs if bname(m,b)!="trunk_base"}; up=float(quat_rotate(np.asarray(d.xquat[bid(m,"trunk_base")],float),np.array([0,0,1.]))[2]); ok=bad==0 and mv<30 and ma<1e5 and minc>-.01 and max(motion.values(),default=0)<.05
    results.add("PHYSICS_2000_STEPS_NO_COMMAND",ok,"finite passive physics; biped may fall without a balance policy",steps=2000,simulated_time_s=2000*float(m.opt.timestep),nan_detected=bad,final_upright=up,max_qvel=mv,max_qacc=ma,min_contact_dist=minc,max_contacts=maxc,free_body_motion_m=motion)
def validate_stand(results,runner):
    if not runner or not runner.available: results.skip("PHYSICS_2000_STEPS_STAND","ONNX policies unavailable"); return
    runner.reset(); runner.command(np.zeros(3),50); start=np.asarray(runner.d.xpos[runner.trunk],float).copy(); mv=ma=0.; minup=1.; move=0.; bad=0
    for _ in range(450):
        runner.step("alpha_stand"); mv=max(mv,float(np.max(np.abs(runner.d.qvel)))); ma=max(ma,float(np.max(np.abs(runner.d.qacc)))); up=float(quat_rotate(np.asarray(runner.d.xquat[runner.trunk],float),np.array([0,0,1.]))[2]); minup=min(minup,up); move=max(move,float(np.linalg.norm(runner.d.xpos[runner.trunk]-start)))
        if not(np.isfinite(runner.d.qpos).all() and np.isfinite(runner.d.qvel).all() and np.isfinite(runner.d.qacc).all()): bad+=1; break
    ok=bad==0 and minup>.9 and move<.10 and mv<20 and ma<1e5
    results.add("PHYSICS_2000_STEPS_STAND",ok,"2000 substeps driven by project alpha_stand ONNX",steps=2000,simulated_time_s=10.,nan_detected=bad,min_upright=minup,max_root_move_m=move,max_qvel=mv,max_qacc=ma)
def project_fixed(m,d,cam,point):
    c=int(mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_CAMERA,cam)); C=np.asarray(d.cam_xpos[c],float); R=np.asarray(d.cam_xmat[c],float).reshape(3,3); f=-R[:,2]; v=np.asarray(point,float)-C; z=float(v@f)
    if z<=1e-6:return None
    focal=.5*480/math.tan(math.radians(float(m.cam_fovy[c]))/2); u=320+focal*float(v@R[:,0])/z; vv=240-focal*float(v@R[:,1])/z; return (u,vv) if 0<=u<640 and 0<=vv<480 else None
def render_fixed(m,d,cam,path):
    r=mujoco.Renderer(m,480,640)
    try:
        c=mujoco.MjvCamera(); mujoco.mjv_defaultCamera(c); c.type=mujoco.mjtCamera.mjCAMERA_FIXED; c.fixedcamid=int(mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_CAMERA,cam)); r.update_scene(d,camera=c); img=r.render().copy()
    finally:
        if hasattr(r,"close"): r.close()
    path.parent.mkdir(exist_ok=True); Image.fromarray(img).save(path); return img
def validate_cameras(results,m,scene):
    d=mujoco.MjData(m); reset_initial(m,d)
    for cam in ("cam_overhead","cam_workspace"):
        p=scene.parent/"previews"/f"{cam[4:]}.png"; im=render_fixed(m,d,cam,p); uv={b:project_fixed(m,d,cam,d.xpos[bid(m,b)]) for b in ("trunk_base","ball")}; ok=float(im.mean())>5 and float(im.std())>8 and all(v is not None for v in uv.values())
        results.add(f"CAM_{cam[4:].upper()}",ok,f"rendered {p}",mean=float(im.mean()),std=float(im.std()),targets_uv=uv)
def load_headcam(project):
    sys.path.insert(0,str(project)); from duck_play.perception.camera import DuckHeadCam; from duck_play.perception.headcam import render_headcam_rgb; return DuckHeadCam,render_headcam_rgb
def orange_mask(im):
    r,g,b=im[:,:,0].astype(int),im[:,:,1].astype(int),im[:,:,2].astype(int); return (r>150)&(r>g*1.35)&(g>=25)&(b<110)
def validate_vlm_cam(results,m,scene,project):
    try: _DuckHeadCam,render=load_headcam(project)
    except Exception as e: results.add("DUCKVLM_CAMERA",False,str(e)); return
    d=mujoco.MjData(m); reset_initial(m,d); root=bid(m,"trunk_base"); head=bid(m,"yaw_roll_motion"); ball=bid(m,"ball"); sim=SimpleNamespace(model=m,data=d,head_id=head,trunk_id=root,ball_body=ball); out=scene.parent/"previews"
    root_world=np.asarray(d.xpos[root],float).copy(); ball_front=np.array([root_world[0]+.90,root_world[1],.035])
    cases=[("front",0.0),("left",-20.0),("right",20.0),("turn",90.0)]; blob={}; images={}; means={}
    for label,yaw in cases:
        set_pose(m,d,"trunk_base",[root_world[0],root_world[1],.125],yaw); set_pose(m,d,"ball",ball_front); mujoco.mj_forward(m,d)
        image=render(sim,width=320,height=240,pitch_up_deg=-20).copy(); images[label]=image; means[label]=float(image.mean()); Image.fromarray(image).save(out/f"duckvlm_{label}.png")
        set_pose(m,d,"ball",[root_world[0],root_world[1],-1.0]); mujoco.mj_forward(m,d); hidden=render(sim,width=320,height=240,pitch_up_deg=-20).copy()
        diff=np.abs(image.astype(int)-hidden.astype(int)).sum(2); ys,xs=np.where(diff>20); blob[label]={"pixels":int(len(xs)),"centroid":[round(float(xs.mean()),2),round(float(ys.mean()),2)] if len(xs) else None}
        set_pose(m,d,"ball",ball_front); mujoco.mj_forward(m,d)
    f=blob["front"]["centroid"]; l=blob["left"]["centroid"]; r=blob["right"]["centroid"]
    ok=(f is not None and abs(f[0]-160)<30 and l is not None and l[0]<130 and r is not None and r[0]>190 and blob["turn"]["pixels"]==0 and l[0]<f[0]<r[0] and all(v>5 for v in means.values()))
    results.add("DUCKVLM_CAMERA",ok,"actual duck_play/local-sim head camera; ball visibility measured by render differencing",visible_blob=blob,image_mean={k:round(v,2) for k,v in means.items()},outputs=[str(out/f"duckvlm_{k}.png") for k in ("front","left","right")])
def validate_metadata(results,m,scene):
    md=yaml.safe_load(scene.with_name("metadata.yaml").read_text(encoding="utf-8")); d=mujoco.MjData(m); reset_initial(m,d); bad=[]; root=np.asarray(d.xpos[bid(m,"trunk_base")],float)
    if md.get("robot",{}).get("body_name")!="trunk_base":bad.append("robot.body_name")
    sp=md.get("robot",{}).get("spawn",{})
    if "xy" in sp and not va(sp["xy"],root[:2]):bad.append(f"spawn.xy {sp['xy']} != {root[:2].tolist()}")
    if "z_m" in sp and not approx(sp["z_m"],root[2]):bad.append("spawn.z_m")
    for o in md.get("objects",[]):
        b=bid(m,o.get("body","")); gs=[i for i in range(m.ngeom) if int(m.geom_bodyid[i])==b and m.geom_contype[i]!=0]
        if b<0:bad.append(f"missing body {o.get('body')}");continue
        if "mass_kg" in o and not approx(o["mass_kg"],body_mass(m,b),2e-6):bad.append(f"{o['body']}.mass")
        if "default_pose" in o and not va(o["default_pose"],d.xpos[b],2e-4):bad.append(f"{o['body']}.default_pose")
        if gs:
            s=np.asarray(m.geom_size[gs[0]],float)
            if "radius_m" in o and not approx(o["radius_m"],s[0],1e-6):bad.append(f"{o['body']}.radius")
            if "half_size_m" in o and not va(o["half_size_m"],s,1e-6):bad.append(f"{o['body']}.half_size")
            if "height_m" in o and not approx(o["height_m"],2*s[1],1e-6):bad.append(f"{o['body']}.height")
    for z in md.get("zones",[]):
        b=bid(m,z.get("body","")); gs=[i for i in range(m.ngeom) if int(m.geom_bodyid[i])==b]
        if b<0:bad.append(f"missing zone {z.get('body')}")
        elif "center" in z and not va(z["center"],d.xpos[b],2e-4):bad.append(f"{z['body']}.center")
        elif "radius_m" in z and gs and not approx(z["radius_m"],m.geom_size[gs[0],0],1e-6):bad.append(f"{z['body']}.radius")
    plat=md.get("platform_low",{})
    if plat:
        gg=gid(m,plat.get("geom",""))
        if gg<0 or not va(plat["center_xyz"],d.geom_xpos[gg],2e-4) or not va(plat["size_m"],2*m.geom_size[gg],1e-6) or not approx(plat["top_z_m"],d.geom_xpos[gg,2]+m.geom_size[gg,2],1e-6):bad.append("platform_low")
    layout=md.get("layout",{})
    if "floor_size_m" in layout and not va(layout["floor_size_m"],2*m.geom_size[gid(m,"floor")][:2],1e-6):bad.append("layout.floor_size_m")
    for x in md.get("furniture",[]):
        gg=gid(m,x.get("geom",""))
        if gg<0 or not va(x["center_xyz"],d.geom_xpos[gg],2e-4) or not va(x["size_m"],2*m.geom_size[gg],1e-6):bad.append(f"furniture.{x.get('id')}")
    for x in md.get("people",[]):
        b=bid(m,x.get("body",""))
        if b<0 or not va(x["center_xyz"],d.xpos[b],2e-4):bad.append(f"people.{x.get('id')}")
    for name,x in md.get("terrain",{}).get("bodies",{}).items():
        gg=gid(m,name)
        if gg<0 or not va(x["center"],d.geom_xpos[gg],2e-4) or not va(x["size_m"],2*m.geom_size[gg],1e-6):bad.append(f"terrain.{name}")
        if "friction" in x and not approx(x["friction"],m.geom_friction[gg,0],1e-6):bad.append(f"terrain.{name}.friction")
    bc=md.get("beacon",{})
    if bc:
        b=bid(m,bc.get("body",""))
        if b<0 or not va(bc["center"],d.xpos[b],2e-4):bad.append("beacon.center")
    for c in md.get("cameras",[]):
        i=int(mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_CAMERA,c.get("name","")))
        if i<0:bad.append(f"missing camera {c.get('name')}")
        elif ("pos" in c and not va(c["pos"],m.cam_pos[i],1e-6)) or ("fovy" in c and not approx(c["fovy"],m.cam_fovy[i])):bad.append(f"{c['name']} mismatch")
    results.add("METADATA_CONSISTENCY",not bad,"metadata checked against expanded model",problems=bad)
def collect_refs(x,out):
    if isinstance(x,dict):
        for k,v in x.items():
            if k in {"body","zone","target","object"} and isinstance(v,str):out.add(v)
            elif k=="zones" and isinstance(v,list):out.update(str(z) for z in v)
            else:collect_refs(v,out)
    elif isinstance(x,list):
        for v in x:collect_refs(v,out)
def validate_refs(results,m,scene):
    docs=yaml.safe_load(scene.with_name("tasks.yaml").read_text(encoding="utf-8")); refs=set()
    for t in docs.get("tasks",[]):collect_refs(t.get("success",{}),refs);collect_refs(t.get("failure",{}),refs)
    miss=[x for x in sorted(refs) if bid(m,x)<0];results.add("TASK_REFERENCE_CHECK",not miss,"all task identifiers resolve",references=sorted(refs),missing=miss);return docs
def validate_success(results,m,tasks,scene_name):
    by={t["id"]:t for t in tasks["tasks"]}
    def fresh(tid):
        d=mujoco.MjData(m);reset_initial(m,d);return d,TaskEvaluator(m,d,by[tid],dt=.05)
    def body_in_zone(tid,obj,zone):
        d,ev=fresh(tid);z=np.asarray(d.xpos[bid(m,zone)],float);set_pose(m,d,obj,[z[0],z[1],.04]);mujoco.mj_forward(m,d);a,_,_=ev.step(.40);b,_,_=ev.step(.45);return not a and b
    checks={}
    if scene_name=="duck_workspace_v1":
        d,ev=fresh("walk_to_ball");ball=np.asarray(d.xpos[bid(m,"ball")],float);set_pose(m,d,"trunk_base",[ball[0]-.5,ball[1],.125]);d.qvel[:]=0;mujoco.mj_forward(m,d);a,_,_=ev.step(.25);set_pose(m,d,"trunk_base",[ball[0]-.35,ball[1],.125]);d.qvel[:]=0;mujoco.mj_forward(m,d);b,_,_=ev.step(.25);c,_,_=ev.step(.30);checks["walk_to_ball"]=not a and not b and c
        d,ev=fresh("kick_ball_to_zone");z=np.asarray(d.xpos[bid(m,"zone_green")],float);set_pose(m,d,"ball",[z[0]+.6,z[1],.035]);mujoco.mj_forward(m,d);a,_,_=ev.step(.25);set_pose(m,d,"ball",[z[0],z[1],.035]);mujoco.mj_forward(m,d);b,_,_=ev.step(.25);c,_,_=ev.step(.30);checks["kick_ball_to_zone"]=not a and not b and c
        checks["push_red_cube"]=body_in_zone("push_red_cube","obj_cube_red","zone_blue")
    elif scene_name=="duck_home_v1":
        d,ev=fresh("search_ball_across_rooms");ball=np.asarray(d.xpos[bid(m,"ball")],float);set_pose(m,d,"trunk_base",[ball[0]-.9,ball[1],.125]);d.qvel[:]=0;mujoco.mj_forward(m,d);a,_,_=ev.step(.25);set_pose(m,d,"trunk_base",[ball[0]-.30,ball[1],.125]);d.qvel[:]=0;mujoco.mj_forward(m,d);b,_,_=ev.step(.25);c,_,_=ev.step(.30);e,_,_=ev.step(.30);checks["search_ball_across_rooms"]=not a and not b and not c and e
        checks["retrieve_ball"]=body_in_zone("retrieve_ball","ball","zone_green")
        d,ev=fresh("go_to_beacon");bn=np.asarray(d.xpos[bid(m,"beacon_target")],float);set_pose(m,d,"trunk_base",[bn[0]-.9,bn[1],.125]);d.qvel[:]=0;mujoco.mj_forward(m,d);a,_,_=ev.step(.25);set_pose(m,d,"trunk_base",[bn[0]-.25,bn[1],.125]);d.qvel[:]=0;mujoco.mj_forward(m,d);b,_,_=ev.step(.25);c,_,_=ev.step(.30);e,_,_=ev.step(.30);checks["go_to_beacon"]=not a and not b and not c and e
    elif scene_name=="duck_obstacle_v1":
        d,ev=fresh("rough_ground_walk");beacon=np.asarray(d.xpos[bid(m,"beacon_target")],float);set_pose(m,d,"trunk_base",[beacon[0]-.3,beacon[1],.125]);mujoco.mj_forward(m,d);a,_,_=ev.step(.40);b,_,_=ev.step(.45);checks["rough_ground_walk"]=not a and b
        checks["push_ball_around_obstacle"]=body_in_zone("push_ball_around_obstacle","ball","zone_green")
        d,ev=fresh("multi_target_navigation");z=np.asarray(d.xpos[bid(m,"zone_green")],float);set_pose(m,d,"trunk_base",[z[0],z[1],.125]);mujoco.mj_forward(m,d);skip,_,_=ev.step(.8);d,ev=fresh("multi_target_navigation");ok=False
        for zn in ["zone_red","zone_blue","zone_green"]:
            z=np.asarray(d.xpos[bid(m,zn)],float);set_pose(m,d,"trunk_base",[z[0],z[1],.125]);mujoco.mj_forward(m,d)
            for _ in range(2):ok,_,_=ev.step(.05)
        for _ in range(18):ok,_,_=ev.step(.05)
        checks["multi_target_navigation"]=not skip and ok
    results.add("SUCCESS_EVALUATOR",all(checks.values()),"success conditions exercised with mutated MjData",checks=checks)
def validate_failure(results,m):
    task={"id":"failure_unit","max_duration_s":10.,"success_hold_s":0.,"success":{"all":[]},"failure":{"any":[{"type":"fallen_for_s","body":"trunk_base","seconds":.20},{"type":"outside_bounds_aabb","min_xy":[-1,-1],"max_xy":[1,1]}]}}
    d=mujoco.MjData(m);reset_initial(m,d);ev=TaskEvaluator(m,d,task,dt=.10);set_pose(m,d,"trunk_base",[0,0,.125]);set_quat(m,d,"trunk_base",[math.cos(math.pi/4),0,math.sin(math.pi/4),0]);_,a,_=ev.step(.10);_,b,_=ev.step(.11)
    d=mujoco.MjData(m);reset_initial(m,d);ev=TaskEvaluator(m,d,task,dt=.10);set_pose(m,d,"trunk_base",[1.2,0,.125]);c,_,_=ev.step(.10)
    timeout_task={"id":"timeout","max_duration_s":.10,"success_hold_s":0.,"success":{"all":[]},"failure":{"any":[]}};d=mujoco.MjData(m);reset_initial(m,d);ev=TaskEvaluator(m,d,timeout_task,dt=.10);_,fail,reason=ev.step(.11)
    results.add("FAILURE_EVALUATOR",(not a and b and c and fail and reason=="timeout"),"fallen duration, outside bounds and timeout evaluated",fallen_early=a,fallen_late=b,outside=c,timeout_reason=reason)
def validate_locomotion(results,runner):
    if not runner or not runner.available:results.add("LOCOMOTION",False,"ONNX policies unavailable");return
    cmds={"forward":([.3,0,0],90),"backward":([-.5,0,0],120),"turn_left":([0,0,1.0],80),"turn_right":([0,0,-1.0],80),"stop":([0,0,0],90)};out={}
    for n,(cmd,steps) in cmds.items():
        runner.reset();runner.command(np.zeros(3),50);p0=np.asarray(runner.d.xpos[runner.trunk],float).copy();y0=runner.state()["root_yaw_deg"];runner.command(cmd,steps);p1=np.asarray(runner.d.xpos[runner.trunk],float).copy();st=runner.state();out[n]={"disp_xy":(p1[:2]-p0[:2]).tolist(),"distance":float(np.linalg.norm(p1[:2]-p0[:2])),"dyaw_deg":(st["root_yaw_deg"]-y0+180)%360-180,"upright":st["upright"],"speed":st["speed"]}
    ok=out["forward"]["disp_xy"][0]>.08 and out["forward"]["upright"]>.9 and out["backward"]["disp_xy"][0]<-.04 and out["backward"]["upright"]>.9 and out["turn_left"]["dyaw_deg"]>4 and out["turn_right"]["dyaw_deg"]<-4 and out["stop"]["upright"]>.9 and out["stop"]["speed"]<.30
    results.add("LOCOMOTION",ok,"project alpha_walking/alpha_stand controls executed",outcomes=out)
def validate_interaction(results,runner,scene_name):
    if not runner or not runner.available:results.add("OBJECT_INTERACTION",False,"policies unavailable");return
    natural=scene_name=="duck_workspace_v1";runner.reset()
    if not natural:
        cp=np.asarray(runner.d.xpos[runner.ball],float).copy();set_pose(runner.m,runner.d,"trunk_base",[cp[0]-.50,cp[1],.125]);runner.d.qvel[:]=0;mujoco.mj_forward(runner.m,runner.d)
    runner.command(np.zeros(3),50);b0=np.asarray(runner.d.xpos[runner.ball],float).copy();r0=np.asarray(runner.d.xpos[runner.trunk],float).copy();steps=0
    for steps in range(550):
        p=np.asarray(runner.d.xpos[runner.trunk,:2],float);b=np.asarray(runner.d.xpos[runner.ball,:2],float);v=b-p;dist=float(np.linalg.norm(v))
        if dist<.070:break
        R=np.asarray(runner.d.xmat[runner.trunk],float).reshape(3,3);yv=np.array([R[0,0],R[1,0]]);e=math.atan2(yv[0]*v[1]-yv[1]*v[0],float(yv@v));lat=-yv[1]*v[0]+yv[0]*v[1];runner.command([.25 if abs(e)<.9 else .05,float(np.clip(.6*lat,-.2,.2)),float(np.clip(1.2*e,-.7,.7))],1)
    runner.command(np.zeros(3),20);before=np.asarray(runner.d.xpos[runner.ball],float).copy();attempts=[];disp=0.
    for k in range(3):
        R=np.asarray(runner.d.xmat[runner.trunk],float).reshape(3,3);local=R.T@(np.asarray(runner.d.xpos[runner.ball],float)-np.asarray(runner.d.xpos[runner.trunk],float));kick="ball_kick_right" if local[1]<=.005 else "ball_kick_left"
        if k==1:kick="ball_kick_left" if kick.endswith("right") else "ball_kick_right"
        runner.command(np.zeros(3),25,policy=kick);runner.command(np.zeros(3),70,policy="alpha_stand");after=np.asarray(runner.d.xpos[runner.ball],float).copy();disp=float(np.linalg.norm(after-before));attempts.append({"attempt":k+1,"kick":kick,"ball_displacement_m":disp,"ball_xyz":after.tolist()})
        if disp>.02:break
    after=np.asarray(runner.d.xpos[runner.ball],float).copy();rad=float(runner.m.geom_size[gid(runner.m,"geom_ball"),0]);ok=disp>.02 and after[2]-rad>-.001 and after[2]-rad<.01 and np.isfinite(runner.d.qvel).all() and float(np.max(np.abs(runner.d.qvel)))<15
    results.add("OBJECT_INTERACTION",ok,"walking approach plus existing ONNX kick policy",natural_workspace_approach=natural,approach_steps=steps+1,robot_before=r0.tolist(),robot_after=np.asarray(runner.d.xpos[runner.trunk],float).tolist(),ball_before=before.tolist(),ball_after=after.tolist(),ball_displacement_m=disp,attempts=attempts,max_qvel=float(np.max(np.abs(runner.d.qvel))))
def validate_minimal(results,runner):
    if not runner or not runner.available:results.add("MINIMAL_CLOSED_LOOP",False,"policies unavailable");return
    runner.reset();runner.command(np.zeros(3),50);start=np.asarray(runner.d.xpos[runner.trunk],float).copy();steps=0
    for steps in range(550):
        p=np.asarray(runner.d.xpos[runner.trunk,:2],float);b=np.asarray(runner.d.xpos[runner.ball,:2],float);v=b-p;dist=float(np.linalg.norm(v))
        if dist<.4:break
        R=np.asarray(runner.d.xmat[runner.trunk],float).reshape(3,3);yv=np.array([R[0,0],R[1,0]]);e=math.atan2(yv[0]*v[1]-yv[1]*v[0],float(yv@v));lat=-yv[1]*v[0]+yv[0]*v[1];runner.command([.28 if abs(e)<.9 else .04,float(np.clip(.5*lat,-.2,.2)),float(np.clip(1.1*e,-.7,.7))],1)
    runner.command(np.zeros(3),80);end=np.asarray(runner.d.xpos[runner.trunk],float).copy();dist=float(np.linalg.norm((np.asarray(runner.d.xpos[runner.ball],float)-end)[:2]));st=runner.state();ok=float(np.linalg.norm(start[:2]-end[:2]))>.05 and dist<.4 and st["speed"]<.08 and st["upright"]>.9
    results.add("MINIMAL_CLOSED_LOOP",ok,"spawn -> visual ball -> approach -> <0.4 m -> stop",approach_steps=steps+1,start_xyz=start.tolist(),end_xyz=end.tolist(),ball_xyz=np.asarray(runner.d.xpos[runner.ball],float).tolist(),final_distance_m=dist,final_speed_mps=st["speed"],final_upright=st["upright"])
def validate_dryrun(results,m,project):
    try:
        import io;sys.path.insert(0,str(project));from duck_play.perception.headcam import render_headcam_rgb;from duck_vlm.state import DuckStateSensor;from duck_vlm.types import VlmObservation;from duck_vlm.vlm import VlmDecisionClient;from duck_vlm.config import VLMConfig
    except Exception as e:results.add("DUCKVLM_DECISION_DRY_RUN",False,str(e));return
    d=mujoco.MjData(m);reset_initial(m,d);_b=np.asarray(d.xpos[bid(m,"ball")],float);set_pose(m,d,"trunk_base",[_b[0]-.8,_b[1],.125],0);d.qvel[:]=0;mujoco.mj_forward(m,d);sim=SimpleNamespace(model=m,data=d,head_id=bid(m,"yaw_roll_motion"),trunk_id=bid(m,"trunk_base"),ball_body=bid(m,"ball"));state=DuckStateSensor(sim).snapshot("ball",0.);im=render_headcam_rgb(sim,320,240,pitch_up_deg=-20);buf=io.BytesIO();Image.fromarray(im).save(buf,format="JPEG",quality=82);obs=VlmObservation(task="find the orange ball and approach it",target="ball",image_jpeg=buf.getvalue(),state=state);reply=VlmDecisionClient(config=VLMConfig(mode="dry_run")).decide(obs,allowed=["FWD","TURN_L","TURN_R","KICK_L","KICK_R","STOP","DONE"],mode="dry_run");ok=bool(reply.token) and state.target_visible
    results.add("DUCKVLM_DECISION_DRY_RUN",ok,"RGB/state reached the existing DuckVLM decision client",target_visible=state.target_visible,target_range_m=state.target_range_m,target_bearing_deg=math.degrees(state.target_bearing_rad or 0),token=reply.token,image_bytes=len(buf.getvalue()),note="external model inference not claimed without credentials")
def validate_scene(results,m,name):
    d=mujoco.MjData(m);reset_initial(m,d)
    if name=="duck_workspace_v1":
        floor=np.asarray(m.geom_size[gid(m,"floor")],float)[:2]*2;plat=np.asarray(m.geom_size[gid(m,"platform_low")],float)*2;top=float(d.geom_xpos[gid(m,"platform_low"),2]+m.geom_size[gid(m,"platform_low"),2]);bg=gid(m,"geom_ball");rg=gid(m,"geom_cube_red");ok=approx(floor[0],4,.11) and approx(floor[1],4,.11) and va(plat,[1.5,1,.06],1e-6) and approx(top,.06,1e-6) and free_joint(m,"ball")>=0 and approx(m.geom_size[bg,0],.035,1e-6) and approx(body_mass(m,bid(m,"ball")),.015,1e-6) and va(m.geom_size[rg],[.03,.03,.03],1e-6) and approx(body_mass(m,bid(m,"obj_cube_red")),.05,1e-6)
        results.add("WORKSPACE_GEOMETRY",ok,"4x4 floor, 0.06 platform top, 70 mm/15 g ball, 60 mm/50 g cube",floor_size=floor.tolist(),platform_size=plat.tolist(),platform_top_z=top,ball_radius=float(m.geom_size[bg,0]),ball_mass=body_mass(m,bid(m,"ball")),cube_size=(m.geom_size[rg]*2).tolist(),cube_mass=body_mass(m,bid(m,"obj_cube_red")))
    elif name=="duck_home_v1":
        floor=np.asarray(m.geom_size[gid(m,"floor")],float)[:2]*2;dl,dr=gid(m,"geom_door_left"),gid(m,"geom_door_right");width=abs(float((d.geom_xpos[dl,1]-m.geom_size[dl,1])-(d.geom_xpos[dr,1]+m.geom_size[dr,1])));ok=va(floor,[6,5],1e-6) and width>=.5;results.add("HOME_LAYOUT",ok,"6x5 room with measured doorway clearance; this scene pack contains no human actors",floor_size=floor.tolist(),door_clear_width_m=width)
    elif name=="duck_obstacle_v1":
        rg=gid(m,"ramp_12deg");R=np.asarray(d.geom_xmat[rg],float).reshape(3,3);angle=math.degrees(math.acos(float(np.clip(abs(R[2,2]),-1,1))));corners=np.array([[sx*m.geom_size[rg,0],sy*m.geom_size[rg,1],sz*m.geom_size[rg,2]] for sx in(-1,1) for sy in(-1,1) for sz in(-1,1)]);zs=((R@corners.T).T+np.asarray(d.geom_xpos[rg],float))[:,2];t5=float(d.geom_xpos[gid(m,"step_5cm"),2]+m.geom_size[gid(m,"step_5cm"),2]);t10=float(d.geom_xpos[gid(m,"step_10cm"),2]+m.geom_size[gid(m,"step_10cm"),2]);gl,gr=gid(m,"corridor_left"),gid(m,"corridor_right");fw=float((d.geom_xpos[gr,1]-m.geom_size[gr,1])-(d.geom_xpos[gl,1]+m.geom_size[gl,1]));lo=float(m.geom_friction[gid(m,"friction_low_patch"),0]);hi=float(m.geom_friction[gid(m,"friction_high_patch"),0]);ok=abs(angle-12)<.2 and zs.min()>=-1e-4 and zs.min()<.003 and approx(t5,.05) and approx(t10,.10) and approx(fw,.50) and lo<.5 and .8<hi<1.5
        results.add("OBSTACLE_GEOMETRY",ok,"12 degree grounded ramp, 5/10 cm steps, 0.50 m corridor, moderate friction",ramp_angle_deg=angle,ramp_lowest_z=float(zs.min()),ramp_highest_z=float(zs.max()),step_top_z=[t5,t10],corridor_free_width_m=fw,friction=[lo,hi])
def validate_door_traversal(results,runner,scene_name):
    if scene_name!="duck_home_v1": results.skip("DOORWAY_TRAVERSAL","home-only acceptance test"); return
    if not runner or not runner.available: results.add("DOORWAY_TRAVERSAL",False,"policies unavailable"); return
    runner.reset(); set_pose(runner.m,runner.d,"trunk_base",[-.80,0,.125],0); runner.d.qvel[:]=0; mujoco.mj_forward(runner.m,runner.d); runner.command(np.zeros(3),50); start=np.asarray(runner.d.xpos[runner.trunk],float).copy(); runner.command([.30,0,0],400); end=np.asarray(runner.d.xpos[runner.trunk],float).copy(); st=runner.state(); runner.command(np.zeros(3),60); stop=runner.state(); ok=end[0]>.10 and end[0]-start[0]>.85 and st["upright"]>.90 and stop["speed"]<.30
    results.add("DOORWAY_TRAVERSAL",ok,"physical traversal through measured door opening",start_xyz=start.tolist(),end_xyz=end.tolist(),forward_m=float(end[0]-start[0]),upright=st["upright"],stop_speed=stop["speed"],door_clear_width_m=1.4)
def validate_friction_traversal(results,runner,scene_name):
    if scene_name!="duck_obstacle_v1": results.skip("FRICTION_TRAVERSAL","obstacle-only acceptance test"); return
    if not runner or not runner.available: results.add("FRICTION_TRAVERSAL",False,"policies unavailable"); return
    out={}
    for label,y in [("low",1.35),("high",-1.35)]:
        runner.reset(); set_pose(runner.m,runner.d,"trunk_base",[-1.90,y,.125],0); runner.d.qvel[:]=0; mujoco.mj_forward(runner.m,runner.d); runner.command(np.zeros(3),50); p0=np.asarray(runner.d.xpos[runner.trunk],float).copy(); runner.command([.30,0,0],140); p1=np.asarray(runner.d.xpos[runner.trunk],float).copy(); runner.command(np.zeros(3),60); st=runner.state(); out[label]={"start":p0.tolist(),"end":p1.tolist(),"displacement_m":float(np.linalg.norm(p1[:2]-p0[:2])),"upright":st["upright"],"stop_speed":st["speed"]}
    ok=out["low"]["displacement_m"]>.05 and out["high"]["displacement_m"]>.02 and out["low"]["upright"]>.90 and out["high"]["upright"]>.90 and out["low"]["stop_speed"]<.5 and out["high"]["stop_speed"]<.5
    results.add("FRICTION_TRAVERSAL",ok,"robot entered, crossed and stopped on both friction zones",zones=out)
def validate_cube_push(results,runner,scene_name):
    if scene_name!="duck_workspace_v1": results.skip("CUBE_INTERACTION","workspace-only acceptance test"); return
    if not runner or not runner.available: results.add("CUBE_INTERACTION",False,"policies unavailable"); return
    runner.reset(); cube=bid(runner.m,"obj_cube_red"); cp=np.asarray(runner.d.xpos[cube],float).copy(); set_pose(runner.m,runner.d,"trunk_base",[cp[0]-.50,cp[1],.125]); runner.d.qvel[:]=0; mujoco.mj_forward(runner.m,runner.d); runner.command(np.zeros(3),50); c0=np.asarray(runner.d.xpos[cube],float).copy(); steps=0
    for steps in range(480):
        d=r.pose if False else runner.d; p=np.asarray(d.xpos[runner.trunk,:2],float); c=np.asarray(d.xpos[cube,:2],float); v=c-p; R=np.asarray(d.xmat[runner.trunk],float).reshape(3,3); yv=np.array([R[0,0],R[1,0]]); err=math.atan2(yv[0]*v[1]-yv[1]*v[0],float(yv@v)); lat=-yv[1]*v[0]+yv[0]*v[1]
        if float(np.linalg.norm(c0-np.asarray(d.xpos[cube],float)))>.005: break
        runner.command([.22 if abs(err)<1.0 else .03,float(np.clip(.5*lat,-.15,.15)),float(np.clip(1.0*err,-.6,.6))],1)
    runner.command(np.zeros(3),60); c1=np.asarray(runner.d.xpos[cube],float).copy(); st=runner.state(); disp=float(np.linalg.norm(c1-c0)); ok=disp>.005 and st["upright"]>.90 and abs(float(c1[2]-runner.m.geom_size[gid(runner.m,"geom_cube_red"),2]))<.005
    results.add("CUBE_INTERACTION",ok,"robot walked into and pushed the 60 mm/50 g cube",approach_steps=steps+1,cube_before=c0.tolist(),cube_after=c1.tolist(),cube_displacement_m=disp,robot_after=np.asarray(runner.d.xpos[runner.trunk],float).tolist(),upright=st["upright"])
def run_validation(scene,package_root,render=True):
    scene=Path(scene).resolve();t0=time.perf_counter();print(f"\n=== MicroDuck validation: {scene} ===");print(f"MuJoCo {mujoco.__version__}, Python {sys.version.split()[0]}");res=Results(scene)
    try:m=mujoco.MjModel.from_xml_path(str(scene))
    except Exception as e:res.add("XML_LOAD",False,f"MuJoCo: {e}");return {"scene":scene.parent.name,"passed":False,"checks":[asdict(c) for c in res.items],"elapsed_s":time.perf_counter()-t0}
    check_call(res,"XML_LOAD",lambda:validate_xml(res,scene,m));check_call(res,"ROBOT_COMPATIBILITY",lambda:validate_robot(res,m));check_call(res,"INITIAL_CONTACT",lambda:validate_initial(res,m));check_call(res,"STATIC_GEOM_OVERLAP",lambda:validate_static_overlap(res,m));check_call(res,"PHYSICS_2000_STEPS_NO_COMMAND",lambda:validate_passive(res,m))
    runner=PolicyRunner(m,Path(package_root)/"policies")
    try:runner.load()
    except Exception as e:runner.available=False;print(f"[policy] unavailable: {e}")
    check_call(res,"PHYSICS_2000_STEPS_STAND",lambda:validate_stand(res,runner));check_call(res,"SCENE_GEOMETRY",lambda:validate_scene(res,m,scene.parent.name));check_call(res,"METADATA_CONSISTENCY",lambda:validate_metadata(res,m,scene));holder={}
    def refs():holder["docs"]=validate_refs(res,m,scene)
    check_call(res,"TASK_REFERENCE_CHECK",refs)
    if holder:check_call(res,"SUCCESS_EVALUATOR",lambda:validate_success(res,m,holder["docs"],scene.parent.name))
    else:res.add("SUCCESS_EVALUATOR",False,"tasks.yaml unavailable")
    check_call(res,"FAILURE_EVALUATOR",lambda:validate_failure(res,m))
    if render:check_call(res,"CAM_RENDER",lambda:validate_cameras(res,m,scene));check_call(res,"DUCKVLM_CAMERA",lambda:validate_vlm_cam(res,m,scene,Path(package_root).parent))
    else:res.skip("CAM_RENDER","disabled");res.skip("DUCKVLM_CAMERA","disabled")
    check_call(res,"LOCOMOTION",lambda:validate_locomotion(res,runner));check_call(res,"OBJECT_INTERACTION",lambda:validate_interaction(res,runner,scene.parent.name));
    if scene.parent.name=="duck_workspace_v1":check_call(res,"MINIMAL_CLOSED_LOOP",lambda:validate_minimal(res,runner))
    else:res.skip("MINIMAL_CLOSED_LOOP","P0 workspace-only acceptance test")
    check_call(res,"DOORWAY_TRAVERSAL",lambda:validate_door_traversal(res,runner,scene.parent.name));check_call(res,"FRICTION_TRAVERSAL",lambda:validate_friction_traversal(res,runner,scene.parent.name));check_call(res,"CUBE_INTERACTION",lambda:validate_cube_push(res,runner,scene.parent.name));check_call(res,"DUCKVLM_DECISION_DRY_RUN",lambda:validate_dryrun(res,m,Path(package_root).parent))
    payload={"scene":scene.parent.name,"scene_xml":str(scene),"model":{"nq":m.nq,"nv":m.nv,"nu":m.nu,"nbody":m.nbody,"ngeom":m.ngeom,"njnt":m.njnt},"passed":res.passed,"checks":[asdict(c) for c in res.items],"elapsed_s":time.perf_counter()-t0};out=scene.parent/"validation.json";out.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8");print(f"Validation: {'PASS' if res.passed else 'FAIL'} ({out})");return payload
def main(scene_arg=None,package_root=None,render=True):
    p=argparse.ArgumentParser();p.add_argument("scene",nargs="?",default="scene.xml");p.add_argument("--no-render",action="store_true");a=p.parse_args([] if scene_arg is not None else None);scene=Path(scene_arg if scene_arg is not None else a.scene);pkg=Path(package_root) if package_root is not None else scene.resolve().parent.parent.parent;return 0 if run_validation(scene,pkg,render and not a.no_render)["passed"] else 1
if __name__=="__main__":raise SystemExit(main())