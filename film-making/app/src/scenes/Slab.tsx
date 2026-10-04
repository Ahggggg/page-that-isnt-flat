import React from "react";
import { AbsoluteFill, Img, interpolate, staticFile, useCurrentFrame, Easing } from "remotion";
import { SceneCaptions } from "../components/SceneCaptions";
import page from "../../public/data/page.json";
import band from "../../public/data/band.json";
import { C, mono, sans } from "../theme";

/** 'The slab': move the model's 62-layer window off the traced surface and watch the letters fade.
 *  Every map shown is a separate model run at that offset (the band run; the film crossfades between neighbouring runs,
 *  disclosed in the credits). Two curves, the same numbers as the interactive page: gold, the band-wide agreement with
 *  the organizers' published map; cyan, how well each run separates the organizers' labelled ink from papyrus (AUC on
 *  the labelled part of the band). */
const um = 2.399;
const W = 1000, LEFT = 210, EXAG = 8;
const XS_H = Math.round((109 * W) / band.win * EXAG);

export const Slab: React.FC = () => {
  const f = useCurrentFrame();
  const scores = (page as any).scores as Record<string, { r_s8: number }>;
  const arms = ((page as any).arms as string[]).filter((a) => a !== "R0").map(Number).sort((a, b) => a - b);
  const kc = interpolate(f, [40, 170, 330, 420], [0, Math.min(...arms), Math.max(...arms), 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.inOut(Easing.sin) });
  const k = arms.reduce((best, a) => (Math.abs(a - kc) < Math.abs(best - kc) ? a : best), arms[0]);
  // smooth dissolve: crossfade between the two real runs that bracket the current offset (disclosed in the credits)
  const lo = [...arms].reverse().find((a) => a <= kc) ?? arms[0];
  const hi = arms.find((a) => a >= kc) ?? arms[arms.length - 1];
  const wHi = hi === lo ? 0 : (kc - lo) / (hi - lo);
  const stripH = Math.round((2 * band.half * W) / band.win);
  const planeY = (p: number) => (p / 109) * XS_H;
  // curve geometry
  const cw = 560, ch = 360, cl = 60, ct = 70;
  const cx = (kk: number) => cl + ((kk * um + 60) / 120) * (cw - cl - 20);
  const cy = (r: number) => ct + (1 - Math.max(0, r)) * (ch - ct - 40);
  const pts = arms.filter((a) => scores[String(a)]).map((a) => `${cx(a).toFixed(1)},${cy(scores[String(a)].r_s8).toFixed(1)}`);
  const rLo = scores[String(lo)]?.r_s8 ?? 0, rHi = scores[String(hi)]?.r_s8 ?? 0;
  const r = rLo + (rHi - rLo) * wHi;
  const auc = ((page as any).labels?.auc ?? {}) as Record<string, number>;
  const aPts = arms.filter((a) => auc[String(a)] != null).map((a) => `${cx(a).toFixed(1)},${cy(auc[String(a)]).toFixed(1)}`);
  const aNow = (auc[String(lo)] ?? 0) + ((auc[String(hi)] ?? 0) - (auc[String(lo)] ?? 0)) * wHi;
  return (
    <AbsoluteFill style={{ background: C.bg }}>
      <div style={{ position: "absolute", left: LEFT, top: 40, fontFamily: sans, fontSize: 20, letterSpacing: 4, color: C.dim, textTransform: "uppercase" }}>
        Reading window moved by
        <span style={{ fontFamily: mono, color: C.page, letterSpacing: 0, marginLeft: 14, fontSize: 28, textTransform: "none" }}>
          {Math.abs(kc) < 0.5 ? "0 µm" : `${kc > 0 ? "+" : "−"}${Math.abs(kc * um).toFixed(0)} µm`}</span>
      </div>
      <div style={{ position: "absolute", left: LEFT, top: 80, width: W, height: stripH, overflow: "hidden", borderRadius: 4 }}>
        <Img src={staticFile(`data/arms/arm_${lo}.jpg`)} style={{ position: "absolute", width: W, height: stripH }} />
        <Img src={staticFile(`data/arms/arm_${hi}.jpg`)} style={{ position: "absolute", width: W, height: stripH, opacity: wHi }} />
      </div>
      <div style={{ position: "absolute", left: LEFT, top: 80 + stripH + 24, width: W, height: XS_H, overflow: "hidden", borderRadius: 4 }}>
        <Img src={staticFile(`data/xs_band_${String(band.line).padStart(5, "0")}.png`)} style={{ width: W, height: XS_H, filter: "contrast(1.25) brightness(1.15)" }} />
        <div style={{ position: "absolute", left: 0, right: 0, top: planeY(54), borderTop: "2px dashed rgba(98,210,227,.55)" }} />
        <div style={{ position: "absolute", left: 0, right: 0, top: planeY(23 + kc), height: planeY(62), background: "rgba(98,210,227,.15)",
          borderTop: `2px solid ${C.page}`, borderBottom: `2px solid ${C.page}` }} />
      </div>
      <div style={{ position: "absolute", left: LEFT, top: 80 + stripH + 24 + XS_H + 8, fontFamily: mono, fontSize: 15, color: C.dim }}>
        CT along the middle of the line · depth stretched ×{EXAG} · shaded: the 62 layers the model reads
      </div>
      <svg style={{ position: "absolute", left: LEFT + W + 40, top: 80 }} width={cw} height={ch}>
        <text x={cl} y={14} fill={C.ink} fontFamily={sans} fontSize={15} letterSpacing={2}>AGREEMENT WITH PUBLISHED MAP (r)</text>
        <text x={cl} y={34} fill={C.page} fontFamily={sans} fontSize={15} letterSpacing={2}>LABELLED LETTERS VS PAPYRUS (AUC)</text>
        <text x={cl} y={52} fill={C.dim} fontFamily={sans} fontSize={13}>each point a separate run · AUC on the labelled part of the band</text>
        <line x1={cl} x2={cw - 20} y1={cy(0)} y2={cy(0)} stroke="#2a313b" />
        <line x1={cx(0)} x2={cx(0)} y1={ct} y2={cy(0)} stroke="#2a313b" strokeDasharray="4 4" />
        <polyline points={pts.join(" ")} fill="none" stroke={C.ink} strokeWidth={3} />
        {aPts.length > 0 && <polyline points={aPts.join(" ")} fill="none" stroke={C.page} strokeWidth={3} />}
        <circle cx={cx(kc)} cy={cy(r)} r={8} fill={C.ink} />
        {aPts.length > 0 && <circle cx={cx(kc)} cy={cy(aNow)} r={8} fill={C.page} />}
        <text x={cl} y={ch - 14} fill={C.dim} fontFamily={mono} fontSize={13}>−55 µm</text>
        <text x={cx(0) - 4} y={ch - 14} fill={C.dim} fontFamily={mono} fontSize={13}>0</text>
        <text x={cw - 70} y={ch - 14} fill={C.dim} fontFamily={mono} fontSize={13}>+55 µm</text>
      </svg>
      {/* fixed readout under the chart, so the numbers never cross a curve */}
      <div style={{ position: "absolute", left: LEFT + W + 40 + cl, top: 80 + ch + 18, fontFamily: mono, fontSize: 26, lineHeight: 1.5 }}>
        <div style={{ color: C.ink }}>r&nbsp;&nbsp;&nbsp;= {r.toFixed(2)}</div>
        {aPts.length > 0 && <div style={{ color: C.page }}>AUC = {aNow.toFixed(2)}</div>}
      </div>
      <SceneCaptions scene="slab" />
    </AbsoluteFill>
  );
};
