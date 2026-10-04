import React from "react";
import { AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import page from "../../public/data/page.json";
import pages from "../../public/data/pages.json";
import { C, mono, sans, serif } from "../theme";

/** Takeaway card: the practical numbers. w035 band: the band run (data/film_numbers.py). Other pages: label AUC on 6 mm
 *  crops of eight labelled regions and four full labelled regions (data/pages.py). */
type Pg = { auc: Record<string, number>; drop38: number[]; call: string | null };
export const Takeaway: React.FC = () => {
  const f = useCurrentFrame();
  const { fps } = useVideoConfig();
  const a = (page as any).analysis;
  const inWin = (100 * (1 - a.sheet_depth.frac_outside_window)).toFixed(1);
  const r8 = Math.min(a.P1.per_arm["-8"].r_s8, a.P1.per_arm["8"].r_s8).toFixed(2);
  const crops = Object.values((pages as any).crops as Record<string, Pg>);
  const cropsIn = crops.filter((c) => Math.max(...c.drop38) <= 0.05).length;
  const w44 = ((pages as any).full.w044 as Pg).drop38[0].toFixed(3);
  const big = spring({ frame: f - 5, fps, config: { damping: 200 } });
  const line = (i: number) => interpolate(f, [50 + i * 40, 70 + i * 40], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const out = interpolate(f, [330, 360], [1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const lines = [
    `On ${cropsIn} of ${crops.length} labelled page samples, the model still tells the letters from papyrus nearly as well (within 0.05 AUC) with its window 38 µm off the trace.`,
    `Not on every page: w044 loses ${w44} AUC at −38 µm, for a reason we could not pin down. No page held to ±55 µm.`,
    `The map’s finer detail changes sooner: it matches the published map (r ≥ ${r8}) within about ±20 µm.`,
    `The sheet wanders, but the model’s 150 µm slab still holds it on ${inWin}% of this page.`,
    "On a page traced as well as this one, if ink goes missing, the mesh is rarely the reason.",
  ];
  const last = lines.length - 1;
  return (
    <AbsoluteFill style={{ background: C.bg, padding: "0 210px", justifyContent: "center", opacity: out }}>
      <div style={{ fontFamily: sans, fontSize: 22, letterSpacing: 5, color: C.page, textTransform: "uppercase", opacity: big }}>How close is close enough? Usually about</div>
      <div style={{ fontFamily: mono, fontSize: 150, color: C.ink, lineHeight: 1.05, opacity: big, transform: `translateY(${(1 - big) * 20}px)` }}>±40 µm</div>
      <div style={{ marginTop: 26, display: "grid", gap: 18, maxWidth: 1450 }}>
        {lines.map((t, i) => (
          <div key={i} style={{ fontFamily: i === last ? serif : sans, fontWeight: 300, fontSize: i === last ? 40 : 29, lineHeight: 1.3,
            color: i === 1 ? "#e9a090" : C.text, opacity: line(i), transform: `translateY(${(1 - line(i)) * 14}px)` }}>{t}</div>
        ))}
      </div>
    </AbsoluteFill>
  );
};
