import React from "react";
import { AbsoluteFill, Img, interpolate, staticFile, useCurrentFrame, Easing } from "remotion";
import { SceneCaptions } from "../components/SceneCaptions";
import { C, mono, sans } from "../theme";

/** 'The cut': a light line sweeps down a patch of the page; below, the CT cross-section along that line shows where
 *  the papyrus really is relative to the traced (virtual) page. Real data: the organizers' surface geometry and CT. */
import band from "../../public/data/band.json";
const ROWS: number[] = band.rows;            // cross-section rows (page coordinates, 16 px apart) about the text line
const PANEL = band.win;                      // 4096 px = 9.8 mm along the line
const PH = 2 * band.half;                    // rows of page shown in the ink strip
const W = 1500;                              // displayed width of both views
const EXAG = 10;                             // vertical exaggeration of the cross-section, stated on screen
const XS_H = Math.round((109 * W) / PANEL * EXAG);
const XS_TRUE = (109 * W) / PANEL;           // true-scale height of the cross-section

export const Cut: React.FC = () => {
  const f = useCurrentFrame();
  const sweep = interpolate(f, [30, 330], [0.06, 0.94], { extrapolateLeft: "clamp", extrapolateRight: "clamp",
    easing: Easing.inOut(Easing.cubic) });
  const row = ROWS[Math.min(ROWS.length - 1, Math.round(sweep * (ROWS.length - 1)))];
  const appear = interpolate(f, [0, 25], [0, 1], { extrapolateRight: "clamp" });
  const xsIn = interpolate(f, [40, 80], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const lineIn = interpolate(f, [110, 140], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const bandIn = interpolate(f, [230, 270], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const pageH = 270;
  const dispH = (PH * W) / PANEL;                          // displayed height of the whole strip
  const yInPage = ((row - (band.line - band.half)) / PH) * dispH;
  const pageTop = Math.min(0, Math.max(pageH - dispH, pageH / 2 - yInPage));
  const left = (1920 - W) / 2;
  // reveal: the cross-section appears at true scale, then is stretched to x10 depth, label counting up
  const ex = interpolate(f, [60, 120], [1, EXAG], { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.inOut(Easing.cubic) });
  const xsH = XS_TRUE * ex;
  const xsTop = 420 + (XS_H - xsH) / 2;
  const planeY = (p: number) => (p / 109) * xsH;
  return (
    <AbsoluteFill style={{ background: C.bg, opacity: appear }}>
      {/* page patch (ink as the organizers' model predicts it) with the moving cut line */}
      <div style={{ position: "absolute", left, top: 84, width: W, height: pageH, overflow: "hidden", borderRadius: 6 }}>
        <Img src={staticFile("data/ink_band.png")} style={{ position: "absolute", top: pageTop, width: W, height: dispH,
          filter: "sepia(1) saturate(3.5) hue-rotate(-12deg) brightness(.95)" }} />
        <div style={{ position: "absolute", left: 0, right: 0, top: pageTop + yInPage - 1, height: 3, background: C.page,
          boxShadow: `0 0 18px 4px ${C.page}` }} />
      </div>
      <div style={{ position: "absolute", left, top: 48, fontFamily: sans, fontSize: 20, letterSpacing: 4, color: C.dim,
        textTransform: "uppercase" }}>The page, as the ink model reads it · 9.8 mm of one line</div>
      {/* cross-section */}
      <div style={{ position: "absolute", left, top: xsTop, width: W, height: xsH, opacity: xsIn, overflow: "hidden", borderRadius: 4 }}>
        <Img src={staticFile(`data/xs_band_${String(row).padStart(5, "0")}.png`)}
          style={{ width: W, height: xsH, imageRendering: "auto", filter: "contrast(1.25) brightness(1.15)" }} />
        <div style={{ position: "absolute", left: 0, right: 0, top: planeY(23), height: planeY(85) - planeY(23),
          background: "rgba(98,210,227,.13)", borderTop: `1px solid rgba(98,210,227,.6)`, borderBottom: `1px solid rgba(98,210,227,.6)`,
          opacity: bandIn }} />
        <div style={{ position: "absolute", left: 0, right: 0, top: planeY(54), borderTop: `3px dashed ${C.page}`, opacity: lineIn,
          filter: `drop-shadow(0 0 6px ${C.page})` }} />
      </div>
      <div style={{ position: "absolute", left, top: 420 + XS_H + 12, width: W, display: "flex", justifyContent: "space-between",
        fontFamily: mono, fontSize: 18, color: C.dim, opacity: xsIn }}>
        <span>CT cross-section along the cut · {ex < 1.05 ? "true scale" : `depth stretched ×${ex.toFixed(ex < 9.95 ? 1 : 0)}`}</span>
        <span style={{ opacity: lineIn, color: C.page }}>- - - traced surface (the “flat” page)</span>
        <span style={{ opacity: bandIn, color: C.page }}>▮ model reads ±31 layers ≈ 150 µm</span>
      </div>
      <SceneCaptions scene="cut" />
    </AbsoluteFill>
  );
};
