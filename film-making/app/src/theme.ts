import { loadFont as loadSerif } from "@remotion/google-fonts/CormorantGaramond";
import { loadFont as loadSans } from "@remotion/google-fonts/Inter";
import { loadFont as loadMono } from "@remotion/google-fonts/JetBrainsMono";

export const serif = loadSerif("normal", { weights: ["500", "600"], subsets: ["latin"] }).fontFamily;
export const sans = loadSans("normal", { weights: ["300", "400", "500", "600"], subsets: ["latin"] }).fontFamily;
export const mono = loadMono("normal", { weights: ["400", "500"], subsets: ["latin"] }).fontFamily;

export const C = {
  bg: "#06080c",
  ink: "#f2b65a",        // ink as the model sees it: warm gold
  page: "#62d2e3",       // the virtual page / the model's reading band: cool cyan
  papyrus: "#b08a62",
  text: "#ece6da",
  dim: "#8a8f99",
};
export const FPS = 30;
