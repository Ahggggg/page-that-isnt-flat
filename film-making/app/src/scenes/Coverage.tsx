import React from "react";
import { AbsoluteFill, Img, interpolate, staticFile, useCurrentFrame, spring, useVideoConfig } from "remotion";
import { SceneCaptions } from "../components/SceneCaptions";
import page from "../../public/data/page.json";
import { C, mono, sans } from "../theme";

/** The whole band, CT only: where the papyrus sits relative to the traced surface. No ink model involved. */
export const Coverage: React.FC = () => {
  const f = useCurrentFrame();
  const { fps } = useVideoConfig();
  const img = interpolate(f, [0, 30], [0, 1], { extrapolateRight: "clamp" });
  const cov = (page as any).coverage;
  const facts: [number, string][] = [[cov.gt24um, "more than 24 µm off the traced surface"], [cov.gt48um, "more than 48 µm off"],
    [cov.outside, "outside the layers the model reads"]];
  return (
    <AbsoluteFill style={{ background: C.bg }}>
      <div style={{ position: "absolute", left: 210, top: 48, fontFamily: sans, fontSize: 20, letterSpacing: 4, color: C.dim, textTransform: "uppercase" }}>
        A band of the page 17 × 49 mm · CT only, no ink model</div>
      <div style={{ position: "absolute", left: 210, top: 84, width: 1500, opacity: img }}>
        <Img src={staticFile("data/depth_band.png")} style={{ width: 1500, height: 527, borderRadius: 4 }} />
        <div style={{ display: "flex", gap: 28, marginTop: 10, fontFamily: mono, fontSize: 16, color: C.dim }}>
          <span><span style={{ display: "inline-block", width: 12, height: 12, background: "#1446a0", marginRight: 8 }} />papyrus above the traced surface</span>
          <span><span style={{ display: "inline-block", width: 12, height: 12, background: "#eb8c32", marginRight: 8 }} />below it</span>
        </div>
      </div>
      <div style={{ position: "absolute", left: 210, top: 680, display: "flex", gap: 90 }}>
        {facts.map(([v, k], i) => {
          const s = spring({ frame: f - 50 - i * 18, fps, config: { damping: 200 } });
          return (
            <div key={k} style={{ opacity: s, transform: `translateY(${(1 - s) * 16}px)` }}>
              <div style={{ fontFamily: mono, fontSize: 64, color: i === 2 ? C.ink : C.text }}>{(v * 100 * Math.min(1, s * 1.02)).toFixed(1)}%</div>
              <div style={{ fontFamily: sans, fontSize: 22, color: C.dim, maxWidth: 380 }}>{k}</div>
            </div>
          );
        })}
      </div>
      <SceneCaptions scene="coverage" />
    </AbsoluteFill>
  );
};
