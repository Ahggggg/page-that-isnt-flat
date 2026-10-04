import React from "react";
import { AbsoluteFill, Img, interpolate, staticFile, useCurrentFrame, Easing } from "remotion";
import { SceneCaptions } from "../components/SceneCaptions";
import { C, mono, sans } from "../theme";

/** 'Flip the sheet': the same 62 layers fed back-to-front. Both maps are real model runs (k = 0 forward, k = 0 reversed). */
export const Flip: React.FC<{ tag: string; pageW: number; pageH: number; rF: number; rR: number }> = ({ tag, pageW, pageH, rF, rR }) => {
  const f = useCurrentFrame();
  const flip = interpolate(f, [70, 130], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.inOut(Easing.cubic) });
  const showR = flip > 0.5;
  const left = (1920 - pageW) / 2, top = 190;
  return (
    <AbsoluteFill style={{ background: C.bg }}>
      <div style={{ position: "absolute", left, top: top - 64, fontFamily: sans, fontSize: 20, letterSpacing: 4, color: C.dim, textTransform: "uppercase" }}>
        {showR ? "Same layers, back to front" : "Front of the sheet first (as trained)"}
        <span style={{ fontFamily: mono, letterSpacing: 0, textTransform: "none", marginLeft: 18, color: showR ? C.page : C.ink }}>
          agreement with published map r = {(showR ? rR : rF).toFixed(2)}</span>
      </div>
      <div style={{ position: "absolute", left, top, width: pageW, height: pageH, perspective: 1600 }}>
        <div style={{ width: "100%", height: "100%", transform: `rotateX(${flip * 180}deg)`, transformStyle: "preserve-3d" }}>
          <Img src={staticFile(`data/slab/${tag}_+0.png`)} style={{ position: "absolute", width: pageW, height: pageH, backfaceVisibility: "hidden", borderRadius: 6 }} />
          <Img src={staticFile(`data/slab/${tag}_R0.png`)} style={{ position: "absolute", width: pageW, height: pageH, backfaceVisibility: "hidden",
            transform: "rotateX(180deg)", borderRadius: 6 }} />
        </div>
      </div>
      <SceneCaptions scene="flip" />
    </AbsoluteFill>
  );
};
