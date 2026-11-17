import type { LabJob } from "./lab";

type AudioWindow = Window & { webkitAudioContext?:typeof AudioContext };
type JobStatus = Pick<LabJob,"id"|"status">;

export class ExperimentSuccessSound {
  private context:AudioContext|null = null;
  private pending = new Set<string>();
  private finished = new Set<string>();

  // Called synchronously by the user's click/keypress, before awaiting the API.
  arm() {
    try {
      const Constructor=window.AudioContext||(window as AudioWindow).webkitAudioContext;
      if(!Constructor)return;
      if(!this.context||this.context.state==="closed")this.context=new Constructor();
      if(this.context.state==="suspended")void this.context.resume().catch(()=>{});
    } catch { /* Sound must never prevent an experiment from starting. */ }
  }

  track(id:string) {
    if(!this.finished.has(id))this.pending.add(id);
  }

  observe(jobs:JobStatus[]) {
    for(const job of jobs){
      if(job.status==="queued"||job.status==="running"){
        this.track(job.id);
      }else{
        const succeeded=job.status==="completed"&&this.pending.has(job.id)&&!this.finished.has(job.id);
        this.pending.delete(job.id);
        this.finished.add(job.id);
        if(succeeded)void this.play();
      }
    }
  }

  private async play() {
    const context=this.context;
    if(!context||context.state==="closed")return;
    try {
      if(context.state!=="running")await context.resume();
      if(context.state!=="running")return;
      for(let index=0;index<3;index++){
        const start=context.currentTime+.04+index*.28;
        const tone=context.createOscillator();
        const volume=context.createGain();
        tone.type="sine";
        tone.frequency.setValueAtTime(880,start);
        volume.gain.setValueAtTime(0,start);
        volume.gain.linearRampToValueAtTime(.16,start+.01);
        volume.gain.setValueAtTime(.16,start+.12);
        volume.gain.linearRampToValueAtTime(0,start+.16);
        tone.connect(volume);
        volume.connect(context.destination);
        tone.onended=()=>{tone.disconnect();volume.disconnect();};
        tone.start(start);
        tone.stop(start+.17);
      }
    } catch { /* Muted/unsupported audio does not change the completed job. */ }
  }

  close() {
    if(this.context&&this.context.state!=="closed")void this.context.close().catch(()=>{});
    this.context=null;
    this.pending.clear();
    this.finished.clear();
  }
}
