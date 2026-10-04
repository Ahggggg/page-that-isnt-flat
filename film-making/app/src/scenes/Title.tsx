import React from "react";
import { AbsoluteFill, Img, interpolate, staticFile, useCurrentFrame } from "remotion";
import { C, serif, sans } from "../theme";

/** Cold open: the page as the model reads it, deep in shadow, then the title. */
export const Title: React.FC = () => {
  const f = useCurrentFrame();
  const bgO = interpolate(f, [0, 60, 200, 240], [0, 0.55, 0.55, 0.25], { extrapolateRight: "clamp" });
  const scale = interpolate(f, [0, 240], [1.25, 1.12]);
  const t1 = interpolate(f, [40, 80], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const t2 = interpolate(f, [90, 130], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const out = interpolate(f, [215, 240], [1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  return (
    <AbsoluteFill style={{ background: C.bg }}>
      <AbsoluteFill style={{ opacity: bgO, transform: `scale(${scale})`, filter: "blur(1.5px)" }}>
        <Img src={staticFile("data/ink_8k.png")} style={{ width: "100%", height: "100%", objectFit: "cover", mixBlendMode: "screen",
          filter: "sepia(1) saturate(3) hue-rotate(-12deg) brightness(.9)" }} />
      </AbsoluteFill>
      <AbsoluteFill style={{ background: "radial-gradient(ellipse at center, rgba(6,8,12,.15) 0%, rgba(6,8,12,.92) 70%)" }} />
      <AbsoluteFill style={{ justifyContent: "center", alignItems: "center", opacity: out }}>
        <div style={{ fontFamily: serif, fontWeight: 500, fontSize: 132, color: C.text, letterSpacing: 2, opacity: t1,
          transform: `translateY(${(1 - t1) * 20}px)` }}>The Page That Isn’t Flat</div>
        <div style={{ fontFamily: sans, fontWeight: 300, fontSize: 30, color: C.dim, letterSpacing: 6, marginTop: 28, opacity: t2,
          textTransform: "uppercase" }}>How an ink model reads a crumpled Herculaneum page</div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
