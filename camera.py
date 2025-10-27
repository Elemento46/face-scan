import os
os.environ["GLOG_minloglevel"]="2"
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

import cv2, numpy as np, mediapipe as mp, onnxruntime as ort, time, math, queue, threading, sounddevice as sd, webrtcvad
from collections import deque

ONNX_PATH="models/emotion-ferplus.onnx"
HAVE_EMO=False
emo_sess=None
emo_input=None
try:
    if os.path.exists(ONNX_PATH):
        emo_sess=ort.InferenceSession(ONNX_PATH,providers=["CPUExecutionProvider"])
        emo_input=emo_sess.get_inputs()[0].name
        HAVE_EMO=True
except:
    HAVE_EMO=False

EMO_LABELS=["neutral","happiness","surprise","sadness","anger","disgust","fear","contempt"]

mp_hol=mp.solutions.holistic
hol=mp_hol.Holistic(static_image_mode=False,model_complexity=1,smooth_landmarks=True,refine_face_landmarks=True,min_detection_confidence=0.6,min_tracking_confidence=0.6)

class EMA:
    def __init__(self,a=0.4): self.a=a; self.s=None
    def __call__(self,x):
        if x is None: return None
        x=np.asarray(x,dtype=np.float32)
        if self.s is None: self.s=x
        self.s=self.a*x+(1-self.a)*self.s
        return self.s

ema_pose,ema_face,ema_lh,ema_rh=EMA(0.35),EMA(0.35),EMA(0.5),EMA(0.5)

def lmk_xy(l,iw,ih): return np.array([(p.x*iw,p.y*ih) for p in l],dtype=np.float32)
def eu(a,b): return float(np.linalg.norm(np.array(a,dtype=np.float32)-np.array(b,dtype=np.float32)))
def softmax(x): x=np.asarray(x,dtype=np.float32); x-=np.max(x); e=np.exp(x); return e/np.sum(e)

def norm_shoulders(p):
    ls,rs=p[11],p[12]
    c=(ls+rs)/2.0
    s=max(1.0,float(np.linalg.norm(rs-ls)))
    return (p-c)/s,c,s

def crop_face(frame,face_xy,margin=0.28):
    h,w=frame.shape[:2]
    x0=int(np.clip(np.min(face_xy[:,0]),0,w-1)); y0=int(np.clip(np.min(face_xy[:,1]),0,h-1))
    x1=int(np.clip(np.max(face_xy[:,0]),0,w-1)); y1=int(np.clip(np.max(face_xy[:,1]),0,h-1))
    if x1<=x0 or y1<=y0: return None,None
    cx=(x0+x1)//2; cy=(y0+y1)//2; side=int(max(x1-x0,y1-y0)*(1+margin))
    x0n=int(np.clip(cx-side//2,0,w-1)); y0n=int(np.clip(cy-side//2,0,h-1))
    x1n=int(np.clip(x0n+side,0,w)); y1n=int(np.clip(y0n+side,0,h))
    return frame[y0n:y1n,x0n:x1n],(x0n,y0n,x1n,y1n)

def emo_fer(face_bgr):
    g=cv2.cvtColor(face_bgr,cv2.COLOR_BGR2GRAY)
    g=cv2.resize(g,(64,64),interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
    blob=g[np.newaxis,np.newaxis,:,:]
    v=emo_sess.run(None,{emo_input:blob})[0][0]
    v=softmax(v); i=int(np.argmax(v))
    return EMO_LABELS[i],float(v[i]),v

def iris_center(face_xy,idxs): return np.mean(face_xy[idxs],axis=0)

def eye_contact(face_xy):
    if face_xy is None or len(face_xy)<478: return 0.0
    le_l,le_r,le_t,le_b=face_xy[33],face_xy[133],face_xy[159],face_xy[145]
    re_l,re_r,re_t,re_b=face_xy[362],face_xy[263],face_xy[386],face_xy[374]
    li=iris_center(face_xy,[468,469,470,471,472]); ri=iris_center(face_xy,[473,474,475,476,477])
    def norm(i,l,r,t,b):
        x=(i[0]-l[0])/(r[0]-l[0]+1e-6) if abs(r[0]-l[0])>1e-3 else 0.5
        y=(i[1]-t[1])/(b[1]-t[1]+1e-6) if abs(b[1]-t[1])>1e-3 else 0.5
        return x,y
    lx,ly=norm(li,le_l,le_r,le_t,le_b); rx,ry=norm(ri,re_l,re_r,re_t,re_b)
    cx=1.0-(abs(lx-0.5)+abs(rx-0.5)); cy=1.0-(abs(ly-0.5)+abs(ry-0.5))
    return float(max(0.0,min(1.0,0.55*cx+0.45*cy)))

def head_pose(face_xy,iw,ih):
    if face_xy is None or len(face_xy)<468: return 0.0,0.0,0.0
    idx=[33,263,1,61,291,199]
    pts2d=np.array([face_xy[j] for j in idx],dtype=np.float32)
    pts3d=np.array([(-30,0,30),(30,0,30),(0,0,50),(-20,-20,20),(20,-20,20),(0,-30,10)],dtype=np.float32)
    f=iw
    cam=np.array([[f,0,iw/2],[0,f,ih/2],[0,0,1]],dtype=np.float32)
    dist=np.zeros((4,1),dtype=np.float32)
    ok,rot,tran=cv2.solvePnP(pts3d,pts2d,cam,dist,flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: return 0.0,0.0,0.0
    r,_=cv2.Rodrigues(rot); sy=math.sqrt(r[0,0]**2+r[1,0]**2)
    pitch=np.degrees(math.atan2(-r[2,0],sy)); yaw=np.degrees(math.atan2(r[1,0],r[0,0])); roll=np.degrees(math.atan2(r[2,1],r[2,2]))
    return float(yaw),float(pitch),float(roll)

def arms_crossed(ls,rs,lw,rw,lelb,relb):
    mid=(ls+rs)/2; sw=np.linalg.norm(rs-ls)
    dwr=np.linalg.norm(lw-rw)
    center=np.linalg.norm(lw-mid)<0.85*sw and np.linalg.norm(rw-mid)<0.85*sw
    across=lw[0]>lelb[0] and rw[0]<relb[0]
    chest=(abs(lw[1]-mid[1])<0.65*sw) and (abs(rw[1]-mid[1])<0.65*sw)
    return (dwr<0.85*sw and center and chest) or across

def hand_to_face(nose,lh_pt,rh_pt,scale):
    dL=eu(lh_pt,nose) if lh_pt is not None else 1e9
    dR=eu(rh_pt,nose) if rh_pt is not None else 1e9
    return min(dL,dR)<0.55*scale

def mouth_ratio(face_xy):
    if face_xy is None or len(face_xy)<292: return 0.0
    pL,pR,pT,pB=face_xy[61],face_xy[291],face_xy[13],face_xy[14]
    return float(eu(pT,pB)/max(1.0,eu(pL,pR)))

def draw_line(frame,text,y,color=(0,255,0)):
    tw=max(260,9*len(text))
    cv2.rectangle(frame,(10,y-22),(10+tw,y),color,-1)
    cv2.putText(frame,text,(16,y-6),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,0,0),2,cv2.LINE_AA)
    return y+26

def zero_crossings(x):
    x=np.sign(x); return int(np.sum(x[:-1]*x[1:]<0))

vad=webrtcvad.Vad(2)
SAMPLE_RATE=16000
FRAME_MS=20
BUFFER_BYTES=int(SAMPLE_RATE*FRAME_MS/1000)*2
audio_q=queue.Queue(maxsize=50)
talking_prob=0.0
def audio_cb(indata,frames,time_info,status):
    try:
        audio_q.put(bytes(indata))
    except: pass
def audio_thread():
    global talking_prob
    win=deque(maxlen=25)
    while True:
        try:
            data=audio_q.get(timeout=1)
        except: continue
        step=BUFFER_BYTES
        voiced=[]
        for i in range(0,len(data)-step+1,step):
            voiced.append(1 if vad.is_speech(data[i:i+step],SAMPLE_RATE) else 0)
        if voiced:
            win.append(np.mean(voiced))
            talking_prob=float(np.mean(win))
thr=threading.Thread(target=audio_thread,daemon=True)
stream=sd.RawInputStream(samplerate=SAMPLE_RATE,blocksize=int(SAMPLE_RATE*FRAME_MS/1000),channels=1,dtype="int16",callback=audio_cb)
stream.start(); thr.start()

votes_eye=deque(maxlen=24)
votes_cross=deque(maxlen=24)
votes_slouch=deque(maxlen=24)
votes_handface=deque(maxlen=24)
emo_hist=deque(maxlen=24)
mr_hist=deque(maxlen=24)
yaw_hist=deque(maxlen=36)
pitch_hist=deque(maxlen=36)

base_spine_angles=deque(maxlen=90)
base_dy_vals=deque(maxlen=90)
baseline_ready=False
baseline_spine=0.0
baseline_dy=0.0

cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,360)
t0=time.time(); n=0; fps=0.0

while True:
    ok,frame=cap.read()
    if not ok: break
    frame=cv2.flip(frame,1)
    ih,iw=frame.shape[:2]
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    res=hol.process(rgb)

    pose_xy=face_xy=lh_xy=rh_xy=None
    if res.pose_landmarks: pose_xy=ema_pose(lmk_xy(res.pose_landmarks.landmark,iw,ih))
    if res.face_landmarks: face_xy=ema_face(lmk_xy(res.face_landmarks.landmark,iw,ih))
    if res.left_hand_landmarks: lh_xy=ema_lh(lmk_xy(res.left_hand_landmarks.landmark,iw,ih))
    if res.right_hand_landmarks: rh_xy=ema_rh(lmk_xy(res.right_hand_landmarks.landmark,iw,ih))

    eye_prob=0.0; crossF=False; slouchF=False; handfaceF=False
    emo_txt="N/A"; emo_conf=0.0
    calib=None

    yaw=pitch=0.0
    if face_xy is not None:
        iris_prob=eye_contact(face_xy)
        yaw,pitch,_=head_pose(face_xy,iw,ih)
        yaw_hist.append(yaw); pitch_hist.append(pitch)
        yaw_f=abs(float(np.median(yaw_hist))) if len(yaw_hist)>0 else 0.0
        pitch_f=abs(float(np.median(pitch_hist))) if len(pitch_hist)>0 else 0.0
        head_ok=max(0.0,1.0-min(yaw_f/20.0,1.0))*0.6+max(0.0,1.0-min(pitch_f/15.0,1.0))*0.4
        eye_prob=float(max(0.0,min(1.0,0.6*iris_prob+0.4*head_ok)))
        votes_eye.append(eye_prob)
        mr_hist.append(mouth_ratio(face_xy))

    if pose_xy is not None:
        norm,center,scale=norm_shoulders(pose_xy)
        nose=norm[0]; ls,rs=norm[11],norm[12]
        lw,rw=norm[15],norm[16]
        lelb,relb=norm[13],norm[14]
        hips=(norm[23],norm[24]) if len(norm)>24 else (np.array([0,0]),np.array([0,0]))
        mid=(ls+rs)/2; hx=(hips[0]+hips[1])/2
        spine_angle=np.degrees(np.arctan2(hx[1]-mid[1],hx[0]-mid[0]))
        dy=nose[1]-mid[1]
        if not baseline_ready:
            base_spine_angles.append(spine_angle)
            base_dy_vals.append(dy)
            if len(base_spine_angles)>60 and len(base_dy_vals)>60:
                baseline_spine=float(np.median(base_spine_angles))
                baseline_dy=float(np.median(base_dy_vals))
                baseline_ready=True
            calib="Calibrando postura..."
        dy_thresh=(baseline_dy+0.18) if baseline_ready else 0.35
        spine_thresh=(abs(baseline_spine)+14.0) if baseline_ready else 28.0
        slch=(dy>dy_thresh) or (abs(spine_angle-baseline_spine)>spine_thresh) or (abs(pitch)>18.0)
        cross=arms_crossed(ls,rs,lw,rw,lelb,relb)
        lh_tip=rh_tip=None
        if lh_xy is not None and len(lh_xy)>8: lh_tip=(lh_xy[8]-center)/scale
        if rh_xy is not None and len(rh_xy)>8: rh_tip=(rh_xy[8]-center)/scale
        hface=hand_to_face(nose,lh_tip,rh_tip,1.0)
        votes_cross.append(1 if cross else 0)
        votes_slouch.append(1 if slch else 0)
        votes_handface.append(1 if hface else 0)

    if len(votes_eye)>0: eye_prob=float(np.mean(votes_eye))
    if len(votes_cross)>0: crossF=sum(votes_cross)/len(votes_cross)>0.6
    if len(votes_slouch)>0: slouchF=sum(votes_slouch)/len(votes_slouch)>0.6
    if len(votes_handface)>0: handfaceF=sum(votes_handface)/len(votes_handface)>0.5

    nodding=headshake=False
    if len(yaw_hist)>=24 and len(pitch_hist)>=24:
        py=np.array(pitch_hist,dtype=np.float32)
        px=np.array(yaw_hist,dtype=np.float32)
        nodding = zero_crossings(np.diff(py))>=2 and np.std(py)>2.5
        headshake = zero_crossings(np.diff(px))>=2 and np.std(px)>3.0

    talking = talking_prob>=0.35

    if HAVE_EMO and face_xy is not None:
        roi,rect=crop_face(frame,face_xy,margin=0.28)
        if roi is not None and roi.size>0:
            try:
                e_txt,e_conf,e_vec=emo_fer(roi)
                emo_hist.append(e_vec)
                v=np.mean(np.stack(emo_hist,axis=0),axis=0) if len(emo_hist)>0 else e_vec
                i=int(np.argmax(v)); emo_txt=EMO_LABELS[i]; emo_conf=float(v[i])
                x0,y0,x1,y1=rect; cv2.rectangle(frame,(x0,y0),(x1,y1),(0,255,0),2)
            except: emo_txt,emo_conf="N/A",0.0

    score=0
    if eye_prob>=0.65: score+=1
    if not slouchF: score+=1
    if not crossF: score+=1
    if not handfaceF: score+=1

    n+=1
    if n%12==0:
        t1=time.time(); fps=12.0/max(1e-6,(t1-t0)); t0=t1

    y=30
    y=draw_line(frame,f"FPS {fps:.1f} | Score {score}/4",y,(0,255,0) if score>=3 else (0,165,255) if score==2 else (0,0,255))
    if not baseline_ready and calib: y=draw_line(frame,calib,y,(0,165,255))
    y=draw_line(frame,f"Contato visual {int(eye_prob*100)}%",y)
    y=draw_line(frame,f"Nodding {'Sim' if nodding else 'Nao'} | Head-shake {'Sim' if headshake else 'Nao'}",y)
    y=draw_line(frame,f"Postura {'Curvado' if slouchF else 'Reto'}",y,(0,0,255) if slouchF else (0,255,0))
    y=draw_line(frame,f"Bracos cruzados {'Sim' if crossF else 'Nao'}",y,(0,0,255) if crossF else (0,255,0))
    y=draw_line(frame,f"Mao no rosto {'Sim' if handfaceF else 'Nao'}",y,(0,0,255) if handfaceF else (0,255,0))
    if HAVE_EMO: y=draw_line(frame,f"Emocao {emo_txt} {int(emo_conf*100)}%",y)
    y=draw_line(frame,f"Fala {'Sim' if talking else 'Nao'}",y)

    key=cv2.waitKey(1)
    if key==27: break
    if key==ord('c'):
        base_spine_angles.clear(); base_dy_vals.clear()
        baseline_ready=False

    cv2.imshow("Avaliacao de Entrevista - Tracking Pro",frame)

cap.release()
cv2.destroyAllWindows()
stream.stop(); stream.close()
