import React from "react";
import beats from "../beats.json";
import { Caption } from "./Caption";

/** The film's on-screen text lives in beats.json, shared with the scroll page (which shows it as call-outs). */
export type Beat = { from: number; to: number; text: string; kicker?: string };
export const beatsOf = (scene: string): Beat[] => (beats.captions as Record<string, Beat[]>)[scene] ?? [];
export const tuplesOf = (scene: string): [number, number, string][] => beatsOf(scene).map((b) => [b.from, b.to, b.text]);

export const SceneCaptions: React.FC<{ scene: string }> = ({ scene }) => (
  <>{beatsOf(scene).map((b, i) => <Caption key={i} from={b.from} to={b.to} kicker={b.kicker}>{b.text}</Caption>)}</>
);
