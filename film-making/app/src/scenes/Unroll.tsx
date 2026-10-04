import React from "react";
import { AbsoluteFill, Img, staticFile, useCurrentFrame } from "remotion";
import { Caption } from "../components/Caption";
import { C } from "../theme";

/** Blender render of the traced sheet unrolling (blender/unroll.py), played as an image sequence. */
export const Unroll: React.FC<{ frames: number }> = ({ frames }) => {
  const f = useCurrentFrame();
  const n = Math.min(frames, f + 1);
  return (
    <AbsoluteFill style={{ background: C.bg }}>
      <Img src={staticFile(`unroll/f_${String(n).padStart(4, "0")}.png`)} style={{ width: "100%", height: "100%" }} />
      <Caption from={10} to={120}>One traced sheet of PHerc. 139, in its true shape inside the scroll.</Caption>
      <Caption from={130} to={frames}>Unrolled, it becomes the page the ink model reads.</Caption>
    </AbsoluteFill>
  );
};
