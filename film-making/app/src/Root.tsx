import React from "react";
import { Composition } from "remotion";
import { Title } from "./scenes/Title";
import { Cut } from "./scenes/Cut";
import { Slab } from "./scenes/Slab";
import { Main, MAIN_FRAMES } from "./Main";
import { Flip } from "./scenes/Flip";
import { Credits } from "./scenes/Credits";
import { Coverage } from "./scenes/Coverage";
import { Takeaway } from "./scenes/Takeaway";
import { Zoom } from "./scenes/Zoom";
import { FPS } from "./theme";

export const RemotionRoot: React.FC = () => (
  <>
    <Composition id="Title" component={Title} durationInFrames={240} fps={FPS} width={1920} height={1080} />
    <Composition id="Cut" component={Cut} durationInFrames={360} fps={FPS} width={1920} height={1080} />
  <Composition id="Slab" component={Slab} durationInFrames={450} fps={FPS} width={1920} height={1080} />
  <Composition id="Main" component={Main} durationInFrames={MAIN_FRAMES} fps={FPS} width={1920} height={1080} />
  <Composition id="FlipPoc" component={Flip} durationInFrames={300} fps={FPS} width={1920} height={1080}
    defaultProps={{ tag: "poc", pageW: 640, pageH: 640, rF: 0.996, rR: -0.10 }} />
  <Composition id="FlipBand" component={Flip} durationInFrames={300} fps={FPS} width={1920} height={1080}
    defaultProps={{ tag: "band", pageW: 1500, pageH: 527, rF: 0.9996, rR: -0.02 }} />
  <Composition id="Coverage" component={Coverage} durationInFrames={300} fps={FPS} width={1920} height={1080} />
  <Composition id="Takeaway" component={Takeaway} durationInFrames={360} fps={FPS} width={1920} height={1080} />
  <Composition id="ZoomA" component={Zoom} durationInFrames={300} fps={FPS} width={1920} height={1080}
    defaultProps={{ from: 0, to: 2, captions: [[20, 140, "The page we analysed: 5 cm of PHerc. 139."], [150, 290, "Closer: one line of Greek, as the model reads it."]] }} />
  <Composition id="ZoomB" component={Zoom} durationInFrames={300} fps={FPS} width={1920} height={1080}
    defaultProps={{ from: 2, to: 4, captions: [[20, 140, "One letter, about 2 mm tall."], [150, 290, "Beneath the letter, the papyrus itself: fibres laid in two directions, at the scan’s 2.4 µm resolution."]] }} />
  <Composition id="Credits" component={Credits} durationInFrames={300} fps={FPS} width={1920} height={1080} />
  </>
);
