import React from "react";
import { AbsoluteFill, Img, interpolate, staticFile, useCurrentFrame, useVideoConfig, Easing } from "remotion";
import { Caption } from "../components/Caption";
import zoom from "../../public/data/zoom/zoom.json";
import { C, mono, sans } from "../theme";

/** 'Powers of ten': one continuous log zoom through real crops of the page (zoom_assets.py). Each finer level fades in
 *  once its resolution beats the level above; at letter scale the model's ink map crossfades into the raw CT. */
type Level = { level: number; cy: number; cx: number; w: number; box: [number, number, number, number]; ct_box: [number, number, number, number] | null };
const LV = (zoom as any).levels as Level[];
const UM = 2.399;
const NICE = [20000, 10000, 5000, 2000, 1000, 500, 200, 100, 50];   // um

export const Zoom: React.FC<{ from: number; to: number; captions: [number, number, string][] }> = ({ from, to, captions }) => {
  const f = useCurrentFrame();
  const { durationInFrames, width } = useVideoConfig();
  // zoom parameter: 0 = level `from`, 1 = level `to`, eased
  const t = interpolate(f, [15, durationInFrames - 30], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp",
    easing: Easing.inOut(Easing.cubic) });
  const lw = (i: number) => Math.log(LV[i].w);
  const logW = lw(from) + (lw(to) - lw(from)) * t;
  const viewW = Math.exp(logW);
  // view centre glides between level centres, keyed to the log-width
  const seg = (() => {
    for (let i = from; i < to; i++) if (logW <= lw(i) && logW >= lw(i + 1)) return i;
    return Math.min(to - 1, Math.max(from, from));
  })();
  const u = (lw(seg) - logW) / (lw(seg) - lw(seg + 1) || 1);
  const ease = (x: number) => x * x * (3 - 2 * x);
  const cy = LV[seg].cy + (LV[seg + 1].cy - LV[seg].cy) * ease(Math.min(1, Math.max(0, u)));
  const cx = LV[seg].cx + (LV[seg + 1].cx - LV[seg].cx) * ease(Math.min(1, Math.max(0, u)));
  const s = width / viewW;                                         // screen px per page px
  const viewH = viewW * 9 / 16;
  const place = (box: [number, number, number, number]) => {
    const [bt, bl, bh, bw] = box;
    return { left: (bl - (cx - viewW / 2)) * s, top: (bt - (cy - viewH / 2)) * s, width: bw * s, height: bh * s };
  };
  // a level is shown once the view is narrower than ~1.6x its own width (its pixels are then at least as fine)
  const shown = (i: number) => interpolate(viewW, [LV[i].w * 1.3, LV[i].w * 1.9], [1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const ctOn = interpolate(viewW, [950, 1500], [1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  // scale bar
  const umAcross = viewW * UM;
  const bar = NICE.find((n) => n <= umAcross * 0.22) ?? 50;
  const barPx = (bar / UM) * s;
  const label = bar >= 1000 ? `${bar / 1000} mm` : `${bar} µm`;
  const gold = "sepia(1) saturate(3.4) hue-rotate(-12deg) brightness(.95)";
  return (
    <AbsoluteFill style={{ background: C.bg, overflow: "hidden" }}>
      {LV.map((L, i) => i <= 3 && (
        <Img key={`ink${i}`} src={staticFile(`data/zoom/ink_L${i}.png`)}
          style={{ position: "absolute", ...place(L.box), opacity: i === 0 ? 1 : shown(i), filter: gold }} />
      ))}
      {LV.map((L, i) => i >= 3 && (
        <Img key={`ct${i}`} src={staticFile(`data/zoom/ct_L${i}.png`)}
          style={{ position: "absolute", ...place(L.ct_box!), opacity: i === 3 ? ctOn : shown(i) * ctOn,
            filter: "contrast(1.2) brightness(1.1)" }} />
      ))}
      <div style={{ position: "absolute", left: 80, bottom: 200, display: "flex", flexDirection: "column", gap: 8 }}>
        <div style={{ width: barPx, height: 4, background: C.text, boxShadow: "0 0 10px rgba(0,0,0,.8)" }} />
        <div style={{ fontFamily: mono, fontSize: 24, color: C.text, textShadow: "0 2px 10px #000" }}>{label}</div>
      </div>
      <div style={{ position: "absolute", right: 80, top: 60, fontFamily: sans, fontSize: 18, letterSpacing: 4, color: C.dim,
        textTransform: "uppercase", textAlign: "right" }}>
        {ctOn > 0.5 ? "Raw CT on the traced surface" : "Ink, as the model reads it"}<br />
        <span style={{ fontFamily: mono, letterSpacing: 0, textTransform: "none" }}>field of view {(umAcross / 1000).toFixed(umAcross < 3000 ? 2 : 1)} mm</span>
      </div>
      {captions.map(([a, b, text], i) => <Caption key={i} from={a} to={b}>{text}</Caption>)}
    </AbsoluteFill>
  );
};
