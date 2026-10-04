import React from "react";
import { getInputProps, interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import { C, sans } from "../theme";

// Render with --props='{"captions":false}' for the scroll page, which draws the same text as call-outs.
const SHOW = (getInputProps() as { captions?: boolean }).captions !== false;

/** Lower-third caption: slides up and fades in at `from`, fades out at `to` (frames, scene-relative). */
export const Caption: React.FC<{ from: number; to: number; children: React.ReactNode; kicker?: string }> = ({ from, to, children, kicker }) => {
  const f = useCurrentFrame();
  const { fps } = useVideoConfig();
  if (!SHOW) return null;
  const enter = spring({ frame: f - from, fps, config: { damping: 200 } });
  const exit = interpolate(f, [to - 12, to], [1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const o = Math.min(enter, exit);
  return (
    <>
    <div style={{ position: "absolute", left: 0, right: 0, bottom: 0, height: 300, opacity: o * 0.9, pointerEvents: "none",
      background: "linear-gradient(to bottom, rgba(6,8,12,0) 0%, rgba(6,8,12,.72) 45%, rgba(6,8,12,.9) 100%)" }} />
    <div style={{ position: "absolute", left: 210, right: 210, bottom: 56, opacity: o, transform: `translateY(${(1 - enter) * 24}px)` }}>
      {kicker && <div style={{ fontFamily: sans, fontSize: 20, letterSpacing: 4, textTransform: "uppercase", color: C.page, marginBottom: 10 }}>{kicker}</div>}
      <div style={{ fontFamily: sans, fontWeight: 300, fontSize: 38, lineHeight: 1.3, color: C.text, maxWidth: 1300, textShadow: "0 2px 18px rgba(0,0,0,.8)" }}>{children}</div>
    </div>
    </>
  );
};
