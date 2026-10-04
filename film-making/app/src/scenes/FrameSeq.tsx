import React from "react";
import { AbsoluteFill, Img, interpolate, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { Caption } from "../components/Caption";
import { C, mono } from "../theme";

/** A Blender shot played as an image sequence (public/frames/<shot>/f_0001.png ...), with optional captions,
 *  an on-screen note (e.g. stated exaggeration), and fades in and out. */
export const FrameSeq: React.FC<{ shot: string; frames: number; captions?: [number, number, string][]; note?: string; fade?: number }> =
  ({ shot, frames, captions = [], note, fade = 12 }) => {
    const f = useCurrentFrame();
    const { durationInFrames } = useVideoConfig();
    const n = Math.min(frames, Math.max(1, f + 1));
    const o = interpolate(f, [0, fade, durationInFrames - fade, durationInFrames], [0, 1, 1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
    return (
      <AbsoluteFill style={{ background: C.bg, opacity: o }}>
        <Img src={staticFile(`frames/${shot}/f_${String(n).padStart(4, "0")}.png`)} style={{ width: "100%", height: "100%" }} />
        {note && <div style={{ position: "absolute", right: 60, top: 50, fontFamily: mono, fontSize: 18, color: C.dim }}>{note}</div>}
        {captions.map(([a, b, t], i) => <Caption key={i} from={a} to={b}>{t}</Caption>)}
      </AbsoluteFill>
    );
  };
