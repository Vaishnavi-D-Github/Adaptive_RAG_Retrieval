import { motion } from "framer-motion";
import type { ReactNode } from "react";
import "./adaptive-canvas.css";
import { ArrowUpRight, BrainCircuit, Database, FileCheck2, Orbit, Search, Sparkles } from "lucide-react";
import type { User } from "../types";

const prompts = [
  "Summarize the employee leave policy",
  "What are the key findings from the latest audit?",
  "Compare the current policy with the previous policy",
  "How do I submit an expense claim?"
];

export function AdaptiveCanvas({ user, onPrompt }: { user: User; onPrompt: (prompt: string) => void }) {
  return <div className="adaptive-canvas">
    <motion.div className="canvas-intro" initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: .55 }}>
      <span className="eyebrow">ADAPTIVE RETRIEVAL ENGINE · ONLINE</span>
      <h1>Explore your<br/><em>knowledge network.</em></h1>
      <p>Workspace ready for {user.full_name}. Ask one question. The system chooses and verifies the evidence depth before it answers.</p>
      <div className="canvas-prompts">{prompts.map((prompt, index) => <motion.button key={prompt} initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: .17 + index * .08 }} onClick={() => onPrompt(prompt)}><span>0{index + 1}</span>{prompt}<ArrowUpRight size={15}/></motion.button>)}</div>
    </motion.div>
    <motion.div className="network-stage" initial={{ opacity: 0, scale: .92 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: .7, delay: .12 }}>
      <div className="stage-grid"/>
      <motion.div className="orbit orbit-one" animate={{ rotate: 360 }} transition={{ duration: 20, repeat: Infinity, ease: "linear" }}/>
      <motion.div className="orbit orbit-two" animate={{ rotate: -360 }} transition={{ duration: 28, repeat: Infinity, ease: "linear" }}/>
      <motion.div className="signal signal-one" animate={{ opacity: [.25, 1, .25], scale: [.8, 1.2, .8] }} transition={{ duration: 2.8, repeat: Infinity }}/>
      <motion.div className="signal signal-two" animate={{ opacity: [1, .2, 1], scale: [1, .75, 1] }} transition={{ duration: 3.4, repeat: Infinity }}/>
      <motion.div className="core-node" animate={{ boxShadow: ["0 0 0 8px #38d6bf10", "0 0 0 21px #38d6bf08", "0 0 0 8px #38d6bf10"] }} transition={{ duration: 2.5, repeat: Infinity }}><BrainCircuit size={31}/><b>ADAPT</b></motion.div>
      <Node className="node-search" icon={<Search size={18}/>} label="Analyze"/><Node className="node-db" icon={<Database size={18}/>} label="Retrieve"/><Node className="node-verify" icon={<FileCheck2 size={18}/>} label="Verify"/><Node className="node-generate" icon={<Sparkles size={18}/>} label="Generate"/>
      <svg className="network-lines" viewBox="0 0 500 500" aria-hidden="true"><path d="M250 250 L115 145 M250 250 L387 158 M250 250 L128 362 M250 250 L381 352"/><path className="pulse-path" d="M250 250 L115 145 M250 250 L387 158 M250 250 L128 362 M250 250 L381 352"/></svg>
      <div className="stage-status"><span><i/> Adaptive engine ready</span><b><Orbit size={15}/> 4 intelligent stages</b></div>
    </motion.div>
  </div>;
}
function Node({ className, icon, label }: { className: string; icon: ReactNode; label: string }) { return <motion.div className={`network-node ${className}`} animate={{ y: [0, -5, 0] }} transition={{ duration: 3.5, repeat: Infinity, ease: "easeInOut", delay: className.includes("db") ? .5 : 0 }}><span>{icon}</span><b>{label}</b></motion.div> }
