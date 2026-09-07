import { useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Float, Points, PointMaterial } from "@react-three/drei";
import { motion } from "framer-motion";
import * as THREE from "three";
import { ArrowRight, BookOpen, Sparkles } from "lucide-react";
import "./intro-gate.css";

export function IntroGate({ onComplete }: { onComplete: () => void }) {
  const completed = useRef(false);
  const finish = () => { if (!completed.current) { completed.current = true; onComplete(); } };
  return <motion.section className="intro-gate" exit={{ opacity: 0, scale: 1.04, filter: "blur(7px)" }} transition={{ duration: .65, ease: [0.22, 1, 0.36, 1] }}>
    <div className="intro-art"/><Canvas className="intro-canvas" dpr={[1, 1.5]} camera={{ position: [0, 0, 5], fov: 48 }} aria-label="Animated knowledge network"><Scene/></Canvas>
    <motion.div className="intro-brand" initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: .55 }}><div><BookOpen size={20}/></div><span>Adaptive Enterprise RAG</span></motion.div>
    <div className="intro-center"><motion.div initial={{ opacity: 0, scale: .7 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: .7, delay: .22, ease: [0.22, 1, 0.36, 1] }} className="intro-pulse"><Sparkles size={31}/></motion.div><motion.p initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: .5, duration: .55 }}>ADAPTIVE KNOWLEDGE INTELLIGENCE</motion.p><motion.h1 initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: .63, duration: .7, ease: [0.22, 1, 0.36, 1] }}>Adaptive<br/><em>Enterprise RAG</em></motion.h1><motion.span initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.05 }}>Connecting questions to verified knowledge.</motion.span></div>
    <motion.button className="intro-skip" onClick={finish} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.2 }}>Enter workspace <ArrowRight size={15}/></motion.button><div className="intro-progress"><motion.i initial={{ scaleX: 0 }} animate={{ scaleX: 1 }} transition={{ delay: .35, duration: 3.5, ease: "linear" }}/></div>
  </motion.section>;
}
function Scene() { return <><color attach="background" args={["#031426"]}/><ambientLight intensity={.35}/><Float speed={1.5} rotationIntensity={.35} floatIntensity={.5}><KnowledgeMesh/></Float><Particles/></> }
function KnowledgeMesh() { const mesh = useRef<THREE.Group>(null!); useFrame((_, delta) => { mesh.current.rotation.y += delta * .13; mesh.current.rotation.x = Math.sin(performance.now() * .00025) * .12; }); return <group ref={mesh}><mesh><icosahedronGeometry args={[1.25, 2]}/><meshStandardMaterial color="#0a9a91" emissive="#075d64" emissiveIntensity={.8} wireframe transparent opacity={.62}/></mesh><mesh scale={.62}><icosahedronGeometry args={[1.25, 2]}/><meshBasicMaterial color="#8ff5e8" wireframe transparent opacity={.32}/></mesh></group> }
function Particles() { const points = useMemo(() => { const values = new Float32Array(480 * 3); for (let i = 0; i < values.length; i += 3) { const radius = 2 + Math.random() * 3; const theta = Math.random() * Math.PI * 2; const phi = Math.acos(2 * Math.random() - 1); values[i] = radius * Math.sin(phi) * Math.cos(theta); values[i + 1] = radius * Math.sin(phi) * Math.sin(theta); values[i + 2] = radius * Math.cos(phi) - 1; } return values; }, []); const ref = useRef<THREE.Points>(null!); useFrame((_, delta) => { ref.current.rotation.y += delta * .025; }); return <Points ref={ref} positions={points} stride={3} frustumCulled={false}><PointMaterial transparent color="#76ded4" size={.018} sizeAttenuation depthWrite={false} opacity={.82}/></Points> }
