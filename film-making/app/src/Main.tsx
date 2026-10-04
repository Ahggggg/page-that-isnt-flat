import React from "react";
import { AbsoluteFill, Series } from "remotion";
import { Title } from "./scenes/Title";
import { FrameSeq } from "./scenes/FrameSeq";
import { Zoom } from "./scenes/Zoom";
import { Cut } from "./scenes/Cut";
import { Slab } from "./scenes/Slab";
import { Flip } from "./scenes/Flip";
import { Coverage } from "./scenes/Coverage";
import { Takeaway } from "./scenes/Takeaway";
import { Credits } from "./scenes/Credits";
import { C } from "./theme";
import beats from "./beats.json";
import { tuplesOf } from "./components/SceneCaptions";

// Scene order and lengths in frames (30 fps) live in beats.json, with the on-screen text.
export const SCENES = Object.fromEntries(beats.scenes.map((s) => [s.id, s.frames])) as Record<string, number>;
export const MAIN_FRAMES = Object.values(SCENES).reduce((a, b) => a + b, 0);

export const Main: React.FC = () => (
  <AbsoluteFill style={{ background: C.bg }}>
    <Series>
      <Series.Sequence durationInFrames={SCENES.title}><Title /></Series.Sequence>
      <Series.Sequence durationInFrames={SCENES.scroll}>
        <FrameSeq shot="scroll" frames={360} captions={tuplesOf("scroll")} />
      </Series.Sequence>
      <Series.Sequence durationInFrames={SCENES.unroll}>
        <FrameSeq shot="unroll" frames={300} captions={tuplesOf("unroll")} />
      </Series.Sequence>
      <Series.Sequence durationInFrames={SCENES.zoomA}>
        <Zoom from={0} to={2} captions={tuplesOf("zoomA")} />
      </Series.Sequence>
      <Series.Sequence durationInFrames={SCENES.relief}>
        <FrameSeq shot="relief" frames={360} note="height ×6 · smoothed ~70 µm" captions={tuplesOf("relief")} />
      </Series.Sequence>
      <Series.Sequence durationInFrames={SCENES.zoomB}>
        <Zoom from={2} to={4} captions={tuplesOf("zoomB")} />
      </Series.Sequence>
      <Series.Sequence durationInFrames={SCENES.cut}><Cut /></Series.Sequence>
      <Series.Sequence durationInFrames={SCENES.slab}><Slab /></Series.Sequence>
      <Series.Sequence durationInFrames={SCENES.flip}><Flip tag="band" pageW={1500} pageH={527} rF={0.9996} rR={-0.02} /></Series.Sequence>
      <Series.Sequence durationInFrames={SCENES.coverage}><Coverage /></Series.Sequence>
      <Series.Sequence durationInFrames={SCENES.takeaway}><Takeaway /></Series.Sequence>
      <Series.Sequence durationInFrames={SCENES.credits}><Credits /></Series.Sequence>
    </Series>
  </AbsoluteFill>
);
