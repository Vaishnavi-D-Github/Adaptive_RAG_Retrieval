import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Activity, BarChart3, BookOpen, Bot, ChevronLeft, FileText, History, LogOut, Menu, MessageSquarePlus, Settings, ShieldCheck, UserRound } from "lucide-react";
import { api } from "./services/api";
import type { User } from "./types";
import { ChatPage } from "./components/ChatPage";
import { DocumentsPage } from "./components/DocumentsPage";
import { HistoryPage } from "./components/HistoryPage";
import { AuthPage } from "./components/AuthPage";
import { IntroGate } from "./components/IntroGate";
import "./components/cinematic-theme.css";

type Page = "chat" | "history" | "documents" | "analytics" | "settings";
const items: { id: Page; label: string; icon: typeof History; hr?: boolean }[] = [
  { id: "chat", label: "New chat", icon: MessageSquarePlus }, { id: "history", label: "History", icon: History }, { id: "documents", label: "Documents", icon: FileText, hr: true }, { id: "analytics", label: "RAG insights", icon: BarChart3 }, { id: "settings", label: "Settings", icon: Settings }
];
export default function App() {
  const [user, setUser] = useState<User | null>(null); const [ready, setReady] = useState(false); const [introComplete, setIntroComplete] = useState(false); const [page, setPage] = useState<Page>("chat"); const [collapsed, setCollapsed] = useState(false); const [mobileOpen, setMobileOpen] = useState(false); const [auth, setAuth] = useState<"login" | "signup">("login");
  useEffect(() => { api.me().then(({ user }) => setUser(user)).catch(() => null).finally(() => setReady(true)); }, []);
  if (!ready) return <div className="app-loader"><Bot size={26} /><span>Preparing your workspace</span></div>;
  if (!user) return <AnimatePresence mode="wait">{!introComplete ? <IntroGate key="intro" onComplete={() => setIntroComplete(true)} /> : <motion.div key="auth" initial={{ opacity: 0, scale: 1.02 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: .7, ease: [0.22, 1, 0.36, 1] }}><AuthPage mode={auth} onMode={setAuth} onAuthenticated={setUser} /></motion.div>}</AnimatePresence>;
  const navigate = (target: Page) => { setPage(target); setMobileOpen(false); };
  const logout = async () => { await api.logout().catch(() => null); setUser(null); setPage("chat"); };
  return <div className="app-shell">
    <aside className={`sidebar ${collapsed ? "collapsed" : ""} ${mobileOpen ? "mobile-open" : ""}`}>
      <div className="sidebar-brand"><Brand compact={collapsed} /><button className="icon-button collapse" onClick={() => setCollapsed(!collapsed)} aria-label="Collapse sidebar"><ChevronLeft size={18} /></button></div>
      <button className="new-chat" onClick={() => navigate("chat")}><MessageSquarePlus size={18} /><span>New chat</span></button>
      <nav aria-label="Application"><p className="nav-label">WORKSPACE</p>{items.filter(x => !x.hr || user.role === "hr").map(({ id, label, icon: Icon }) => <button key={id} className={`nav-item ${page === id ? "active" : ""}`} onClick={() => navigate(id)}><Icon size={18}/><span>{label}</span></button>)}</nav>
      <div className="sidebar-footer"><div className="secure-note"><ShieldCheck size={16}/><span>Secure workspace</span></div><button className="profile-card" onClick={() => navigate("settings")}><Avatar user={user}/><span><b>{user.full_name}</b><small>{user.role === "hr" ? "HR administrator" : "Employee"}</small></span></button></div>
    </aside>
    <main className="main"><header className="header"><button className="icon-button mobile-menu" onClick={() => setMobileOpen(true)} aria-label="Open navigation"><Menu/></button><div className="header-title"><span className="eyebrow">ENTERPRISE KNOWLEDGE INTELLIGENCE</span><strong>{page === "chat" ? "Knowledge workspace" : items.find(x => x.id === page)?.label}</strong></div><div className="profile-menu"><span className="presence"/><Avatar user={user}/><div><b>{user.full_name}</b><small>{user.role}</small></div><button className="logout" onClick={logout} aria-label="Log out"><LogOut size={17}/></button></div></header>
      <AnimatePresence mode="wait"><motion.section key={page} className="page" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -5 }} transition={{ duration: .2 }}>
        {page === "chat" && <ChatPage user={user}/>} {page === "history" && <HistoryPage onOpen={(entry) => { navigate("chat"); }} />} {page === "documents" && user.role === "hr" && <DocumentsPage/>} {page === "analytics" && <AnalyticsPlaceholder/>} {page === "settings" && <SettingsPage user={user}/>}</motion.section></AnimatePresence>
    </main><div className={`scrim ${mobileOpen ? "visible" : ""}`} onClick={() => setMobileOpen(false)}/>
  </div>;
}
function Brand({ compact }: { compact: boolean }) { return <div className="brand"><div className="brand-mark"><BookOpen size={19}/><span className="node n1"/><span className="node n2"/></div>{!compact && <span><b>Adaptive RAG</b><small>Enterprise Intelligence</small></span>}</div> }
function Avatar({ user }: { user: User }) { return <div className="avatar" aria-label={user.full_name}>{user.full_name.split(" ").map(n => n[0]).slice(0, 2).join("").toUpperCase()}</div> }
function AnalyticsPlaceholder() { return <div className="empty-state"><Activity size={28}/><h2>RAG insights</h2><p>Analytics will be available as your authenticated query history grows.</p><span>Metrics are calculated only from real history records.</span></div> }
function SettingsPage({ user }: { user: User }) { return <div className="settings-page"><div><span className="eyebrow">ACCOUNT</span><h1>Settings</h1><p>Manage the preferences this workspace can support locally.</p></div><div className="settings-card"><UserRound/><div><b>{user.full_name}</b><p>{user.email} · {user.role === "hr" ? "HR" : "Employee"}</p></div></div><div className="settings-card"><Settings/><div><b>Retrieval experience</b><p>Choose a retrieval mode for each query in the composer. Technical diagnostics remain available per completed response.</p></div></div></div> }
