import cv2
import numpy as np
import os, math

# ============================================================
# TASK 4 - TRACKING V8
# Persistent identity: local template tracking + prediction +
# strict global reacquisition after prolonged loss.
#
# V8 does NOT use E1 connected components.
# The ORIGINAL template is never replaced.
# ============================================================

VIDEO_PATH = None
OUTPUT_DIR = "task4_v8_output"
START_TIME, END_TIME = 18.0, 34.0

INITIAL_TRACKS = [
    {"id": 1, "center": (173, 620), "size": (196, 196)},
    {"id": 2, "center": (455, 652), "size": (210, 210)},
]

LOCAL_RADIUS = 280
LOCAL_SCALES = [0.80, 0.90, 1.00, 1.10, 1.20]
LOCAL_ACCEPT = 0.55
GLOBAL_ACCEPT = 0.72
GLOBAL_AFTER = 18
MAX_PREDICTION = 30
MAX_LOCAL_DISTANCE = 190
MAX_SIZE_RATIO = 1.55

TEMPLATE_SIZE = 128
UPDATE_SCORE = 0.70
UPDATE_ALPHA = 0.04
HIST_ALPHA = 0.03

FLOW_POINTS = 80
FLOW_REFRESH = 6
FLOW_WIN = (21, 21)
FLOW_LEVELS = 3

def find_video():
    if VIDEO_PATH and os.path.exists(VIDEO_PATH):
        return VIDEO_PATH
    exts = (".mp4",".avi",".mov",".mkv",".webm")
    files = []
    for root, dirs, fs in os.walk("."):
        dirs[:] = [d for d in dirs if "task4_v8_output" not in d]
        for f in fs:
            if f.lower().endswith(exts):
                files.append(os.path.join(root,f))
    if not files:
        raise FileNotFoundError("Video not found. Set VIDEO_PATH.")
    pref = [p for p in files if "Y4JyV94y2" in os.path.basename(p)
            or "Mocomi" in os.path.basename(p)]
    return pref[0] if pref else files[0]

def box(cx,cy,w,h,W,H):
    x1=max(0,min(W-2,int(round(cx-w/2))))
    y1=max(0,min(H-2,int(round(cy-h/2))))
    x2=max(x1+1,min(W,int(round(cx+w/2))))
    y2=max(y1+1,min(H,int(round(cy+h/2))))
    return x1,y1,x2,y2

def crop(im,b):
    return im[b[1]:b[3],b[0]:b[2]]

def gray(im):
    if im is None or im.size==0: return None
    g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    g=cv2.GaussianBlur(g,(5,5),0)
    return cv2.resize(g,(TEMPLATE_SIZE,TEMPLATE_SIZE),interpolation=cv2.INTER_AREA)

def edges(im):
    if im is None or im.size==0: return None
    g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    g=cv2.GaussianBlur(g,(5,5),0)
    e=cv2.Canny(g,50,150)
    return cv2.resize(e,(TEMPLATE_SIZE,TEMPLATE_SIZE),interpolation=cv2.INTER_NEAREST)

def hist(im):
    if im is None or im.size==0: return None
    hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    h=cv2.calcHist([hsv],[0,1],None,[24,16],[0,180,0,256])
    cv2.normalize(h,h)
    return h

def sim(a,b):
    if a is None or b is None:return 0.
    a=a.astype(np.float32); b=b.astype(np.float32)
    a=(a-a.mean())/(a.std()+1e-6)
    b=(b-b.mean())/(b.std()+1e-6)
    return float(np.clip((np.mean(a*b)+1)/2,0,1))

def edge_sim(a,b):
    if a is None or b is None:return 0.
    a=a.astype(np.float32)/255; b=b.astype(np.float32)/255
    d=a.sum()+b.sum()
    return float(2*(a*b).sum()/d) if d>1e-6 else 0.

def hist_sim(a,b):
    if a is None or b is None:return 0.
    return float(np.clip((cv2.compareHist(a,b,cv2.HISTCMP_CORREL)+1)/2,0,1))

def mscore(c,p):
    d=math.hypot(c[0]-p[0],c[1]-p[1])
    return math.exp(-0.5*(d/100.)**2)

class Flow:
    def __init__(self):
        self.prev=None; self.pts=None; self.n=0

    def init(self,g,b):
        self.prev=g.copy()
        mask=np.zeros_like(g)
        x1,y1,x2,y2=b
        mx=max(5,int((x2-x1)*.12)); my=max(5,int((y2-y1)*.12))
        cv2.rectangle(mask,(x1+mx,y1+my),(x2-mx,y2-my),255,-1)
        self.pts=cv2.goodFeaturesToTrack(
            g,maxCorners=FLOW_POINTS,qualityLevel=.01,
            minDistance=7,blockSize=7,mask=mask)
        self.n=0

    def predict(self,g,b,c,v):
        if self.prev is None or self.pts is None or len(self.pts)<5:
            self.init(g,b)
        if self.pts is None or len(self.pts)<5:
            return np.asarray(c,dtype=np.float32)+v,0
        q,s,_=cv2.calcOpticalFlowPyrLK(
            self.prev,g,self.pts,None,winSize=FLOW_WIN,
            maxLevel=FLOW_LEVELS,
            criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
        if q is None or s is None:
            self.init(g,b); return np.asarray(c,dtype=np.float32)+v,0
        ok=s.ravel()==1
        old=self.pts[ok]; new=q[ok]
        if len(old)<5:
            self.init(g,b); return np.asarray(c,dtype=np.float32)+v,len(old)
        A,ins=cv2.estimateAffinePartial2D(
            old,new,method=cv2.RANSAC,ransacReprojThreshold=3,
            maxIters=1000,confidence=.99)
        if A is not None:
            p=np.asarray(c,dtype=np.float32).reshape(1,1,2)
            pred=cv2.transform(p,A)[0,0].astype(np.float32)
            ni=int(ins.sum()) if ins is not None else len(old)
        else:
            pred=np.asarray(c,dtype=np.float32)+np.median(new-old,axis=0)
            ni=len(old)
        pred=.8*pred+.2*(np.asarray(c,dtype=np.float32)+v)
        self.pts=new.reshape(-1,1,2); self.prev=g.copy(); self.n+=1
        if self.n>=FLOW_REFRESH or len(self.pts)<12:self.init(g,b)
        return pred,ni

class Track:
    def __init__(self,t,frame):
        self.id=t["id"]
        self.center=np.array(t["center"],np.float32)
        self.size=np.array(t["size"],np.float32)
        self.velocity=np.zeros(2,np.float32)
        self.state="TRACKING"; self.pred_frames=0
        self.matched=self.predicted=self.lost=self.reacquired=0
        self.score=1.; self.gs=self.es=self.hs=self.ms=1.
        self.last_box=self.get_box(frame.shape[1],frame.shape[0])
        c=crop(frame,self.last_box)
        # Original identity: permanently protected.
        self.orig_g=gray(c); self.orig_e=edges(c); self.orig_h=hist(c)
        self.adapt_g=self.orig_g.copy(); self.adapt_e=self.orig_e.copy()
        self.adapt_h=self.orig_h.copy()
        self.flow=Flow()
        self.flow.init(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),self.last_box)

    def get_box(self,W,H):
        return box(self.center[0],self.center[1],self.size[0],self.size[1],W,H)

    def predict(self,g,W,H):
        p,n=self.flow.predict(g,self.last_box,self.center,self.velocity)
        p[0]=np.clip(p[0],0,W-1); p[1]=np.clip(p[1],0,H-1)
        return p,n

    def evaluate(self,frame,b,pred,motion=True):
        c=crop(frame,b)
        if c.size==0:return None
        cg,ce,ch=gray(c),edges(c),hist(c)
        gs=.6*sim(self.adapt_g,cg)+.4*sim(self.orig_g,cg)
        es=.6*edge_sim(self.adapt_e,ce)+.4*edge_sim(self.orig_e,ce)
        hs=.6*hist_sim(self.adapt_h,ch)+.4*hist_sim(self.orig_h,ch)
        cx=(b[0]+b[2])/2; cy=(b[1]+b[3])/2
        ms=mscore((cx,cy),pred) if motion else .5
        return gs,es,hs,ms

    def local(self,frame,pred):
        H,W=frame.shape[:2]
        x1=max(0,int(pred[0]-LOCAL_RADIUS)); y1=max(0,int(pred[1]-LOCAL_RADIUS))
        x2=min(W,int(pred[0]+LOCAL_RADIUS)); y2=min(H,int(pred[1]+LOCAL_RADIUS))
        s=frame[y1:y2,x1:x2]
        if s.size==0:return None,(x1,y1,x2,y2)
        sg=cv2.cvtColor(s,cv2.COLOR_BGR2GRAY)
        best=None
        for scale in LOCAL_SCALES:
            tw=max(55,int(self.size[0]*scale)); th=max(55,int(self.size[1]*scale))
            if tw>=sg.shape[1] or th>=sg.shape[0]:continue
            templ=cv2.resize(self.adapt_g,(tw,th),interpolation=cv2.INTER_AREA)
            r=cv2.matchTemplate(sg,templ,cv2.TM_CCOEFF_NORMED)
            _,tm,_,loc=cv2.minMaxLoc(r)
            cx=x1+loc[0]+tw/2; cy=y1+loc[1]+th/2
            d=math.hypot(cx-pred[0],cy-pred[1])
            if d>MAX_LOCAL_DISTANCE:continue
            b=box(cx,cy,tw,th,W,H)
            ev=self.evaluate(frame,b,pred,True)
            if ev is None:continue
            gs,es,hs,ms=ev
            score=.25*((tm+1)/2)+.30*gs+.25*es+.15*hs+.05*ms
            if score>=LOCAL_ACCEPT and gs>=.43 and es>=.18 and hs>=.32:
                item={"score":float(score),"gray":gs,"edge":es,"hist":hs,
                      "motion":ms,"box":b,"size":(tw,th),"tm":(tm+1)/2}
                if best is None or score>best["score"]:best=item
        return best,(x1,y1,x2,y2)

    def global_search(self,frame,other_centers):
        H,W=frame.shape[:2]
        sg=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),None,fx=.5,fy=.5,
                      interpolation=cv2.INTER_AREA)
        best=None
        for scale in [.8,.9,1.,1.1,1.2]:
            tw=max(55,int(self.size[0]*scale*.5)); th=max(55,int(self.size[1]*scale*.5))
            if tw>=sg.shape[1] or th>=sg.shape[0]:continue
            templ=cv2.resize(self.orig_g,(tw,th),interpolation=cv2.INTER_AREA)
            r=cv2.matchTemplate(sg,templ,cv2.TM_CCOEFF_NORMED)
            flat=r.ravel(); k=min(20,len(flat))
            if k==0:continue
            ids=np.argpartition(flat,-k)[-k:]
            for idx in ids:
                yy,xx=np.unravel_index(idx,r.shape)
                cx=(xx+tw/2)/.5; cy=(yy+th/2)/.5
                if any(math.hypot(cx-o[0],cy-o[1])<100 for o in other_centers):continue
                b=box(cx,cy,tw/.5,th/.5,W,H)
                ev=self.evaluate(frame,b,(cx,cy),False)
                if ev is None:continue
                gs,es,hs,ms=ev
                tmv=float(np.clip((flat[idx]+1)/2,0,1))
                score=.30*tmv+.35*gs+.25*es+.10*hs
                if score>=GLOBAL_ACCEPT and gs>=.62 and es>=.25 and hs>=.48:
                    item={"score":score,"gray":gs,"edge":es,"hist":hs,
                          "motion":ms,"box":b,"size":(tw/.5,th/.5),"tm":tmv}
                    if best is None or score>best["score"]:best=item
        return best

    def update(self,r,frame,reacquired=False):
        b=r["box"]; nc=np.array([(b[0]+b[2])/2,(b[1]+b[3])/2],np.float32)
        ns=np.array([b[2]-b[0],b[3]-b[1]],np.float32)
        disp=nc-self.center
        self.velocity=.45*disp+.55*self.velocity
        self.center=.70*nc+.30*self.center
        self.size=.65*ns+.35*self.size
        old=self.state; self.state="TRACKING"; self.pred_frames=0
        self.matched+=1
        if reacquired or old!="TRACKING":self.reacquired+=1
        self.score=r["score"]; self.gs=r["gray"]; self.es=r["edge"]
        self.hs=r["hist"]; self.ms=r["motion"]
        self.last_box=self.get_box(frame.shape[1],frame.shape[0])
        if r["score"]>=UPDATE_SCORE:
            c=crop(frame,self.last_box); ng,ne,nh=gray(c),edges(c),hist(c)
            if ng is not None:self.adapt_g=((1-UPDATE_ALPHA)*self.adapt_g.astype(np.float32)+UPDATE_ALPHA*ng).astype(np.uint8)
            if ne is not None:self.adapt_e=ne
            if nh is not None:self.adapt_h=(1-HIST_ALPHA)*self.adapt_h+HIST_ALPHA*nh
        self.flow.init(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),self.last_box)

    def predict_only(self,p,frame):
        p=np.asarray(p,np.float32)
        self.velocity=.2*(p-self.center)+.8*self.velocity
        self.center=p; self.pred_frames+=1
        if self.pred_frames<=MAX_PREDICTION:self.state="PREDICTED";self.predicted+=1
        else:self.state="LOST";self.lost+=1
        self.score=self.gs=self.es=self.hs=self.ms=0.
        self.last_box=self.get_box(frame.shape[1],frame.shape[0])

    def ratio(self):
        n=self.matched+self.predicted+self.lost
        return self.matched/n if n else 0.
    def coverage(self):
        n=self.matched+self.predicted+self.lost
        return (self.matched+self.predicted)/n if n else 0.

def draw(frame,t):
    color=(0,255,0) if t.state=="TRACKING" else ((0,255,255) if t.state=="PREDICTED" else (0,0,255))
    x1,y1,x2,y2=t.last_box
    cv2.rectangle(frame,(x1,y1),(x2,y2),color,3)
    cv2.circle(frame,tuple(map(int,t.center)),5,color,-1)
    cv2.putText(frame,f"Egg {t.id} | {t.state} | S={t.score:.2f}",
                (x1,max(25,y1-10)),cv2.FONT_HERSHEY_SIMPLEX,.65,color,2,cv2.LINE_AA)

def main():
    video=find_video(); os.makedirs(OUTPUT_DIR,exist_ok=True)
    out=os.path.join(OUTPUT_DIR,"task4_tracking_v8_test.mp4")
    cap=cv2.VideoCapture(video)
    if not cap.isOpened():raise RuntimeError(f"Could not open video: {video}")
    fps=cap.get(cv2.CAP_PROP_FPS); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("="*70); print("TASK 4 - TRACKING V8")
    print("Persistent identity + local tracking + global reacquisition")
    print("="*70)
    print(f"Video: {os.path.basename(video)}"); print(f"Resolution: {W} x {H}")
    print(f"FPS: {fps:.2f}"); print(f"Frames: {total}"); print(f"Duration: {total/fps:.2f}s")
    print(f"Testing: {START_TIME:.1f}s -> {END_TIME:.1f}s")
    print("GREEN = validated true track"); print("YELLOW = prediction only / temporary occlusion")
    print("RED = lost")
    print("\nV8 principle:")
    print("  V7 local template tracking during normal operation.")
    print("  Prediction during short failures.")
    print("  Strict global identity search after prolonged loss.")
    print("  Original egg template is permanently protected.")
    for t in INITIAL_TRACKS:print(f"Egg {t['id']}: center={t['center']} size={t['size']}")
    start=int(round(START_TIME*fps)); end=int(round(END_TIME*fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES,start); ok,first=cap.read()
    if not ok:raise RuntimeError("Could not read starting frame.")
    tracks=[Track(t,first) for t in INITIAL_TRACKS]
    writer=cv2.VideoWriter(out,cv2.VideoWriter_fourcc(*"mp4v"),fps,(W,H))
    vis=first.copy()
    for t in tracks:draw(vis,t)
    cv2.putText(vis,"V8 | Persistent Identity Tracker",(30,45),cv2.FONT_HERSHEY_SIMPLEX,.9,(255,255,255),2,cv2.LINE_AA)
    writer.write(vis); lastsec=-1

    for fi in range(start+1,end+1):
        ok,frame=cap.read()
        if not ok:break
        g=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        preds=[]; locals_=[]
        for t in tracks:
            p,ins=t.predict(g,W,H); r,sb=t.local(frame,p)
            preds.append(p); locals_.append((r,sb))
        chosen=[locals_[0][0],locals_[1][0]]
        if chosen[0] is not None and chosen[1] is not None:
            b0,b1=chosen[0]["box"],chosen[1]["box"]
            ix1,iy1=max(b0[0],b1[0]),max(b0[1],b1[1]); ix2,iy2=min(b0[2],b1[2]),min(b0[3],b1[3])
            inter=max(0,ix2-ix1)*max(0,iy2-iy1)
            a0=(b0[2]-b0[0])*(b0[3]-b0[1]); a1=(b1[2]-b1[0])*(b1[3]-b1[1])
            if inter/max(a0+a1-inter,1)>0.45:
                if chosen[0]["score"]>=chosen[1]["score"]:chosen[1]=None
                else:chosen[0]=None
        for i,t in enumerate(tracks):
            r=chosen[i]
            if r is not None:t.update(r,frame)
            else:t.predict_only(preds[i],frame)

        # Strict global reacquisition only after the track has been lost.
        for i,t in enumerate(tracks):
            if t.state!="LOST" or t.pred_frames<GLOBAL_AFTER:continue
            others=[tuple(o.center) for j,o in enumerate(tracks) if j!=i and o.state!="LOST"]
            r=t.global_search(frame,others)
            if r is not None:
                t.update(r,frame,reacquired=True)
                print(f"  REACQUIRED: Egg {t.id} at {fi/fps:.2f}s score={r['score']:.3f}")

        vis=frame.copy()
        for r,sb in locals_:
            x1,y1,x2,y2=sb;cv2.rectangle(vis,(x1,y1),(x2,y2),(255,180,0),1)
        for t in tracks:draw(vis,t)
        cv2.putText(vis,f"V8 | t={fi/fps:.2f}s",(30,45),cv2.FONT_HERSHEY_SIMPLEX,.9,(255,255,255),2,cv2.LINE_AA)
        writer.write(vis)
        sec=int(fi/fps)
        if sec%2==0 and sec!=lastsec:
            lastsec=sec
            print(f"time={fi/fps:.2f}s | Egg 1={tracks[0].state} S={tracks[0].score:.2f} | Egg 2={tracks[1].state} S={tracks[1].score:.2f}")

    writer.release();cap.release()
    print("\n"+"="*70);print("TRACKING V8 COMPLETE");print("="*70)
    for t in tracks:
        print(f"Egg {t.id}:")
        print(f"  Matched / TRUE TRACK: {t.matched}")
        print(f"  Predicted / OCCLUSION: {t.predicted}")
        print(f"  Lost: {t.lost}")
        print(f"  Reacquired: {t.reacquired}")
        print(f"  Algorithmic tracking ratio: {t.ratio()*100:.2f}%")
        print(f"  Visible/predicted coverage: {t.coverage()*100:.2f}%")
    print(f"\nOutput: {out}")
    print("\nIMPORTANT: algorithmic percentages do not prove identity correctness; inspect the video.")

if __name__=="__main__":
    main()
