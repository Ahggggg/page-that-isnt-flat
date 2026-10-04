import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { C, serif, sans, mono } from "../theme";

export const Credits: React.FC = () => {
  const f = useCurrentFrame();
  const o = interpolate(f, [0, 30, 270, 300], [0, 1, 1, 0], { extrapolateRight: "clamp" });
  const line = (t: string, s = 26, c = C.text, fam = sans) => <div style={{ fontFamily: fam, fontSize: s, color: c, marginTop: 14 }}>{t}</div>;
  return (
    <AbsoluteFill style={{ background: C.bg, justifyContent: "center", alignItems: "center", opacity: o, textAlign: "center" }}>
      <div style={{ fontFamily: serif, fontSize: 72, color: C.text }}>The Page That Isn’t Flat</div>
      {line("Jack Gilman · Andrew Gilman", 28, C.dim)}
      <div style={{ height: 40 }} />
      {line("Data: PHerc. 139, segment w035 — CT scan, traced surface, surface volume and ink prediction", 22, C.dim)}
      {line("© Vesuvius Challenge, open data under CC BY-NC 4.0 · ink model: scrollprize/ink_canonical_2um (MIT)", 22, C.dim)}
      {line("Ink labels used for scoring (not shown): Vesuvius Challenge, published 21 September 2026", 20, C.dim)}
      {line("Every scan and ink map is computed from that data. No letters were drawn or enhanced.", 22, C.ink)}
      {line("Illustrative: the unrolling motion, crossfades between neighbouring model runs, depth stretching (stated on screen), colour and lighting.", 18, C.dim)}
      <div style={{ height: 30 }} />
      {line("Prior work: window centring by flummoxjr, kadenpool and AndreasHad04 · transferred-mesh depth search by TAUIL Abd Elilah", 18, C.dim)}
      {line("Code and reproduction: github.com/Ahggggg (link in submission)", 20, C.dim, mono)}
    </AbsoluteFill>
  );
};
