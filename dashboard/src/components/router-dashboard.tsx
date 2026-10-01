import { useEffect, useMemo, useState } from "react";
import {
  Activity, AlertTriangle, Box, ChevronLeft, ChevronRight, CircleDollarSign, Clock3,
  Code2, Download, Gauge, GitBranch, Layers, Menu, Moon, Package,
  Play, Route, Server, Sun, X, Zap,
} from "lucide-react";
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Line, LineChart, Pie, PieChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { Button } from "@/components/ui/button";
import {
  Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger,
} from "@/components/ui/dialog";
import { Switch } from "@/components/ui/switch";
import { cn } from "@/lib/utils";
import {
  useSummary, useModels, useProviders, useCategories,
  useVolume, useEvents, useCostOverTime, useModes, useTiers,
  useEventDetail,
} from "@/hooks/use-router-data";
import type {
  RecentEvent, SummaryStats, ModelDistribution, ProviderDistribution,
  CategoryDistribution, VolumePoint, CostPoint, ModeDistribution,
  TierDistribution,
} from "@/lib/api";

// ─── Chart palette (uses the CSS custom properties already in the theme) ─────

const CHART_COLORS = [
  "var(--chart-1)", "var(--chart-2)", "var(--chart-3)",
  "var(--chart-4)", "var(--chart-5)",
];

const TIER_BADGE: Record<string, string> = {
  standard: "bg-chart-2/10 text-chart-2 border-chart-2/20",
  advanced: "bg-chart-4/10 text-chart-4 border-chart-4/20",
};

const MODE_LABELS: Record<string, string> = {
  skill_based: "Skill-Based",
  quality_leaning: "Quality-Leaning",
  mixed: "Mixed",
  cost_sensitive: "Cost-Sensitive",
  cost_efficient: "Cost-Efficient",
};

// ─── Navigation items ───────────────────────────────────────────────────────

const navItems = [
  { label: "Overview", icon: Gauge, target: "overview" },
  { label: "Requests", icon: Activity, target: "request-volume" },
  { label: "Models", icon: GitBranch, target: "model-distribution" },
  { label: "Cost", icon: CircleDollarSign, target: "cost-section" },
  { label: "Runtime logs", icon: Code2, target: "event-stream" },
];

// ─── Reusable metric card ───────────────────────────────────────────────────

function MetricCard({ label, value, detail, trend, icon: Icon }: {
  label: string; value: string; detail: string; trend?: string;
  icon: typeof Activity;
}) {
  return (
    <article className="min-w-0 rounded-lg border bg-card p-4 shadow-sm">
      <div className="flex items-center justify-between gap-2 text-muted-foreground">
        <p className="truncate text-xs font-semibold uppercase">{label}</p>
        <Icon className="size-4 shrink-0" aria-hidden="true" />
      </div>
      <div className="mt-3 flex flex-wrap items-end gap-2">
        <strong className="font-mono text-2xl font-semibold text-card-foreground">{value}</strong>
        {trend && <span className="mb-1 rounded bg-chart-2/10 px-1.5 py-0.5 text-xs font-semibold text-chart-2">{trend}</span>}
      </div>
      <p className="mt-1.5 text-xs text-muted-foreground">{detail}</p>
    </article>
  );
}

// ─── Tooltip component for charts ───────────────────────────────────────────

function ChartTip({ active, payload, label }: {
  active?: boolean; payload?: Array<{ value?: number; name?: string }>; label?: string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-md border bg-popover px-3 py-2 shadow-lg">
      <p className="text-xs text-muted-foreground">{label}</p>
      {payload.map((entry, idx) => (
        <p key={idx} className="mt-1 font-mono text-sm font-semibold text-popover-foreground">
          {entry.value ?? 0} {entry.name || "requests"}
        </p>
      ))}
    </div>
  );
}

// ─── Event row for the live feed ────────────────────────────────────────────

function EventRow({ event, compact = false }: { event: RecentEvent; compact?: boolean }) {
  const time = new Date(event.timestamp * 1000).toLocaleTimeString([], { hour12: false });
  const tierClass = TIER_BADGE[event.selected_tier] ?? "border-border text-muted-foreground";
  return (
    <div className={cn(
      "grid items-center gap-3 border-b py-3 last:border-0",
      compact
        ? "grid-cols-[72px_78px_1fr] sm:grid-cols-[76px_80px_110px_1fr_68px]"
        : "grid-cols-[70px_1fr_62px] md:grid-cols-[76px_82px_110px_1fr_68px]"
    )}>
      <span className="font-mono text-xs text-muted-foreground">{time}</span>
      <span className={cn("hidden w-fit rounded-full border px-2 py-1 text-[11px] font-semibold capitalize md:inline-flex", tierClass)}>
        {event.selected_tier}
      </span>
      <span className="hidden truncate font-mono text-xs text-foreground md:block">
        {event.selected_key}
      </span>
      <div className="min-w-0">
        <span className={cn("mr-2 inline-flex rounded-full border px-2 py-0.5 text-[10px] font-semibold capitalize md:hidden", tierClass)}>
          {event.selected_tier}
        </span>
        <span className="text-sm text-foreground">{event.query_preview}</span>
      </div>
      <span className="text-right font-mono text-xs text-muted-foreground">
        {event.latency_ms != null ? `${Math.round(event.latency_ms)}ms` : "—"}
      </span>
    </div>
  );
}

// ─── Not-connected state ────────────────────────────────────────────────────

function NotConnected() {
  return (
    <div className="flex flex-col items-center justify-center rounded-xl border border-dashed bg-card py-20 text-center">
      <div className="flex size-14 items-center justify-center rounded-full bg-destructive/10">
        <AlertTriangle className="size-7 text-destructive" />
      </div>
      <h2 className="mt-5 text-lg font-bold text-card-foreground">Router not connected</h2>
      <p className="mt-2 max-w-sm text-sm text-muted-foreground">
        The coding-router Python process is not running or the dashboard server on{" "}
        <code className="rounded bg-muted px-1 py-0.5 font-mono text-xs">127.0.0.1:3000</code>{" "}
        is unreachable. Start the router to see live metrics.
      </p>
      <div className="mt-6 rounded-lg border bg-muted/50 px-4 py-3 font-mono text-xs text-muted-foreground">
        <span className="text-chart-2">$</span> coding-router "your query here"
      </div>
    </div>
  );
}

// ─── Sidebar ────────────────────────────────────────────────────────────────

function Sidebar({ collapsed, onCollapse, mobileOpen, onMobileClose, connected }: {
  collapsed: boolean; onCollapse: () => void; mobileOpen: boolean; onMobileClose: () => void;
  connected: boolean;
}) {
  const goTo = (target: string) => {
    document.getElementById(target)?.scrollIntoView({ behavior: "smooth", block: "start" });
    onMobileClose();
  };
  return (
    <>
      {mobileOpen && <div className="fixed inset-0 z-40 bg-foreground/20 backdrop-blur-sm lg:hidden" onClick={onMobileClose} aria-hidden="true" />}
      <aside className={cn("fixed inset-y-0 left-0 z-50 flex border-r bg-sidebar transition-[width,transform] duration-200 lg:translate-x-0", collapsed ? "w-[72px]" : "w-60", mobileOpen ? "translate-x-0" : "-translate-x-full")}>
        <div className="flex min-w-0 flex-1 flex-col">
          <div className="flex h-16 items-center border-b px-4">
            <div className="flex size-9 shrink-0 items-center justify-center rounded-md bg-primary text-primary-foreground"><Route className="size-5" /></div>
            {!collapsed && <div className="ml-3 min-w-0"><p className="truncate text-sm font-bold text-sidebar-foreground">CodeRoute</p><p className="font-mono text-[10px] text-muted-foreground">coding router</p></div>}
            <Button variant="ghost" size="icon" className="ml-auto lg:hidden" onClick={onMobileClose} aria-label="Close navigation"><X /></Button>
          </div>
          <nav className="flex-1 px-3 py-5" aria-label="Dashboard sections">
            {!collapsed && <p className="mb-2 px-2 text-[10px] font-bold uppercase text-muted-foreground">Runtime</p>}
            <div className="space-y-1">
              {navItems.map(({ label, icon: Icon, target }, index) => (
                <Button key={target} variant="ghost" onClick={() => goTo(target)} className={cn("w-full", collapsed ? "px-0" : "justify-start px-2", index === 0 && "bg-sidebar-accent text-sidebar-accent-foreground")} title={collapsed ? label : undefined}>
                  <Icon className="size-4" />{!collapsed && <span>{label}</span>}
                </Button>
              ))}
            </div>
          </nav>
          <div className="border-t p-4">
            <div className={cn("flex items-center", collapsed ? "justify-center" : "gap-2")}>
              <span className="relative flex size-2">
                {connected ? (
                  <>
                    <span className="absolute inline-flex size-full animate-ping rounded-full bg-chart-2 opacity-60" />
                    <span className="relative inline-flex size-2 rounded-full bg-chart-2" />
                  </>
                ) : (
                  <span className="relative inline-flex size-2 rounded-full bg-destructive" />
                )}
              </span>
              {!collapsed && (
                <div>
                  <p className={cn("text-[10px] font-bold uppercase", connected ? "text-chart-2" : "text-destructive")}>
                    {connected ? "Runtime connected" : "Disconnected"}
                  </p>
                  <p className="mt-0.5 font-mono text-[10px] text-muted-foreground">127.0.0.1:3000</p>
                </div>
              )}
            </div>
          </div>
        </div>
        <Button variant="outline" size="icon" onClick={onCollapse} className="absolute -right-4 top-5 hidden size-8 rounded-full bg-card lg:inline-flex" aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}>
          {collapsed ? <ChevronRight /> : <ChevronLeft />}
        </Button>
      </aside>
    </>
  );
}

// ─── Theme toggle ───────────────────────────────────────────────────────────

function ThemeControl({ dark, onChange, className }: { dark: boolean; onChange: (checked: boolean) => void; className?: string }) {
  return <div className={cn("flex h-9 items-center gap-2 rounded-md border bg-card px-2.5", className)}><Sun className="size-3.5 text-muted-foreground" /><Switch checked={dark} onCheckedChange={onChange} aria-label="Toggle dark mode" /><Moon className="size-3.5 text-muted-foreground" /></div>;
}

// ─── Logs dialog ────────────────────────────────────────────────────────────

function LogsDialog({ events, filter, setFilter }: { events: RecentEvent[]; filter: string; setFilter: (v: string) => void }) {
  const filters = ["all", "standard", "advanced"];
  const shown = filter === "all" ? events : events.filter((e) => e.selected_tier === filter);
  return (
    <Dialog>
      <DialogTrigger asChild><Button variant="link" className="hidden px-2 sm:inline-flex">View detailed logs</Button></DialogTrigger>
      <DialogContent className="max-h-[85vh] max-w-4xl overflow-hidden">
        <DialogHeader>
          <DialogTitle>Runtime event log</DialogTitle>
          <DialogDescription>Recent routing decisions reported by the active Python router process.</DialogDescription>
        </DialogHeader>
        <div className="flex gap-2 border-b pb-3">
          {filters.map((v) => (
            <Button key={v} size="sm" variant={filter === v ? "default" : "outline"} onClick={() => setFilter(v)} className="capitalize">{v}</Button>
          ))}
        </div>
        <div className="max-h-[55vh] overflow-y-auto pr-2">
          {shown.map((event) => <EventRow key={event.id} event={event} compact />)}
        </div>
      </DialogContent>
    </Dialog>
  );
}

// ─── Helpers ────────────────────────────────────────────────────────────────

function formatUsd(value: number): string {
  if (value < 0.01) return `$${value.toFixed(4)}`;
  return `$${value.toFixed(2)}`;
}

function formatLatency(ms: number): string {
  if (ms >= 1000) return `${(ms / 1000).toFixed(1)}s`;
  return `${Math.round(ms)}ms`;
}

// ─── Main dashboard ─────────────────────────────────────────────────────────

export function RouterDashboard() {
  const [dark, setDark] = useState(false);
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [filter, setFilter] = useState("all");

  // ─── Live data from the Python backend ──────────────────────
  const { data: summary } = useSummary();
  const { data: models } = useModels();
  const { data: providers } = useProviders();
  const { data: categories } = useCategories();
  const { data: volume } = useVolume();
  const { data: events } = useEvents();
  const { data: costData } = useCostOverTime();
  const { data: modes } = useModes();
  const { data: tiers } = useTiers();

  const connected = summary !== null && summary !== undefined;

  // ─── Theme persistence ──────────────────────────────────────
  useEffect(() => {
    const saved = window.localStorage.getItem("coderoute-theme");
    const wantsDark = saved ? saved === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
    setDark(wantsDark);
    document.documentElement.classList.toggle("dark", wantsDark);
  }, []);

  const toggleTheme = (checked: boolean) => {
    setDark(checked);
    document.documentElement.classList.toggle("dark", checked);
    window.localStorage.setItem("coderoute-theme", checked ? "dark" : "light");
  };

  // ─── Derived chart data ─────────────────────────────────────

  const volumeChart = useMemo(() => {
    if (!volume?.length) return [];
    return volume.map((p) => {
      const d = new Date(p.bucket_start * 1000);
      return { label: d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: false }), requests: p.count };
    });
  }, [volume]);

  const modelPieData = useMemo(() => {
    if (!models?.length) return [];
    return models.map((m, i) => ({
      name: m.model,
      value: m.count,
      color: CHART_COLORS[i % CHART_COLORS.length],
    }));
  }, [models]);

  const providerBarData = useMemo(() => {
    if (!providers?.length) return [];
    return providers.map((p, i) => ({
      name: p.provider,
      requests: p.count,
      fill: CHART_COLORS[i % CHART_COLORS.length],
    }));
  }, [providers]);

  const categoryPieData = useMemo(() => {
    if (!categories?.length) return [];
    return categories.map((c, i) => ({
      name: c.category.replace(/_/g, " "),
      value: c.count,
      color: CHART_COLORS[i % CHART_COLORS.length],
    }));
  }, [categories]);

  const costChart = useMemo(() => {
    if (!costData?.length) return [];
    return costData.map((c) => {
      const d = new Date(c.timestamp * 1000);
      return {
        label: d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: false }),
        cost: c.cumulative_cost ?? 0,
      };
    });
  }, [costData]);

  const modeBarData = useMemo(() => {
    if (!modes?.length) return [];
    return modes.map((m, i) => ({
      name: MODE_LABELS[m.mode] ?? m.mode,
      count: m.count,
      fill: CHART_COLORS[i % CHART_COLORS.length],
    }));
  }, [modes]);

  const tierDonutData = useMemo(() => {
    if (!tiers?.length) return [];
    return tiers.map((t, i) => ({
      name: t.tier,
      value: t.count,
      color: CHART_COLORS[i % CHART_COLORS.length],
    }));
  }, [tiers]);

  const errorRate = summary ? (summary.error_count / Math.max(summary.total_requests, 1) * 100) : 0;
  const avgCostPerReq = summary ? (summary.total_spend / Math.max(summary.total_requests, 1)) : 0;

  // ─── CSV export ─────────────────────────────────────────────
  const exportReport = () => {
    if (!events?.length) return;
    const header = "timestamp,query_preview,category,routing_mode,selected_key,selected_service,selected_tier,latency_ms,estimated_cost_usd,had_error";
    const rows = events.map((e) =>
      [e.timestamp, `"${e.query_preview.replaceAll('"', '""')}"`, e.category, e.routing_mode, e.selected_key, e.selected_service, e.selected_tier, e.latency_ms ?? "", e.estimated_cost_usd ?? "", e.had_error].join(",")
    );
    const url = URL.createObjectURL(new Blob([[header, ...rows].join("\n")], { type: "text/csv" }));
    const anchor = document.createElement("a");
    anchor.href = url; anchor.download = "coderoute-runtime-report.csv"; anchor.click(); URL.revokeObjectURL(url);
  };

  // ─── Render ─────────────────────────────────────────────────

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Sidebar
        collapsed={collapsed}
        onCollapse={() => setCollapsed((v) => !v)}
        mobileOpen={mobileOpen}
        onMobileClose={() => setMobileOpen(false)}
        connected={connected}
      />
      <main className={cn("min-h-screen transition-[margin] duration-200", collapsed ? "lg:ml-[72px]" : "lg:ml-60")}>
        {/* Mobile header */}
        <header className="sticky top-0 z-30 flex h-16 items-center border-b bg-background/90 px-4 backdrop-blur-md sm:px-6 lg:hidden">
          <Button variant="ghost" size="icon" onClick={() => setMobileOpen(true)} aria-label="Open navigation"><Menu /></Button>
          <div className="ml-3"><p className="text-sm font-bold">CodeRoute</p><p className="font-mono text-[10px] text-muted-foreground">runtime dashboard</p></div>
          <ThemeControl dark={dark} onChange={toggleTheme} className="ml-auto" />
        </header>

        <div className="mx-auto max-w-[1600px] p-4 sm:p-6 lg:p-8" id="overview">
          {/* Page header */}
          <div className="mb-7 flex flex-col justify-between gap-4 sm:flex-row sm:items-center">
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-2xl font-bold sm:text-3xl">Router Overview</h1>
                <span className="rounded-full border bg-card px-2 py-0.5 font-mono text-[10px] text-muted-foreground">LOCAL</span>
              </div>
              <p className="mt-1 text-sm text-muted-foreground">Live routing decisions and performance from your Python runtime.</p>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <div className={cn("flex h-9 items-center gap-2 rounded-md border bg-card px-3 text-xs font-semibold", connected ? "" : "border-destructive/30")}>
                <span className={cn("size-2 rounded-full", connected ? "bg-chart-2" : "bg-destructive")} />
                {connected ? "SYSTEM OPERATIONAL" : "NOT CONNECTED"}
              </div>
              <ThemeControl dark={dark} onChange={toggleTheme} className="hidden lg:flex" />
              <Button variant="outline" onClick={exportReport} disabled={!events?.length}><Download />Export report</Button>
            </div>
          </div>

          {/* Not connected state */}
          {!connected && <NotConnected />}

          {/* Connected state — all dashboard sections */}
          {connected && (
            <>
              {/* ── Summary cards ──────────────────────────────────────── */}
              <section className="grid grid-cols-2 gap-3 lg:grid-cols-3 xl:grid-cols-6" aria-label="Runtime summary">
                <MetricCard
                  label="Routed requests"
                  value={String(summary.total_requests)}
                  detail={`${summary.requests_today} today`}
                  icon={Activity}
                />
                <MetricCard
                  label="Active models"
                  value={String(summary.models_active)}
                  detail={`${summary.providers} provider${summary.providers !== 1 ? "s" : ""}`}
                  icon={Server}
                />
                <MetricCard
                  label="Est. spend"
                  value={formatUsd(summary.total_spend)}
                  detail={`${formatUsd(avgCostPerReq)} / request`}
                  icon={CircleDollarSign}
                />
                <MetricCard
                  label="Avg latency"
                  value={formatLatency(summary.avg_latency)}
                  detail="across all requests"
                  icon={Clock3}
                />
                <MetricCard
                  label="Errors"
                  value={String(summary.error_count)}
                  detail={`${errorRate.toFixed(1)}% error rate`}
                  trend={summary.error_count === 0 ? "✓ clean" : undefined}
                  icon={AlertTriangle}
                />
                <MetricCard
                  label="Providers"
                  value={String(summary.providers)}
                  detail={`${summary.models_active} models active`}
                  icon={Layers}
                />
              </section>

              {/* ── Request volume + Model distribution ────────────────── */}
              <section className="mt-4 grid gap-4 xl:grid-cols-[minmax(0,1.8fr)_minmax(320px,0.9fr)]">
                <article id="request-volume" className="rounded-lg border bg-card p-4 shadow-sm sm:p-5">
                  <div className="mb-5 flex items-center justify-between gap-3">
                    <div><h2 className="text-sm font-bold">Request volume</h2><p className="mt-1 text-xs text-muted-foreground">Routed requests over the last 24 hours</p></div>
                  </div>
                  <div className="h-64 w-full">
                    {volumeChart.length > 0 ? (
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={volumeChart} margin={{ top: 8, right: 8, left: -24, bottom: 0 }}>
                          <CartesianGrid vertical={false} stroke="var(--border)" strokeDasharray="3 3" />
                          <XAxis dataKey="label" axisLine={false} tickLine={false} tick={{ fill: "var(--muted-foreground)", fontSize: 11 }} />
                          <YAxis axisLine={false} tickLine={false} tick={{ fill: "var(--muted-foreground)", fontSize: 11 }} />
                          <Tooltip content={<ChartTip />} cursor={{ stroke: "var(--border)" }} />
                          <Line type="monotone" dataKey="requests" stroke="var(--chart-1)" strokeWidth={2.5} dot={false} activeDot={{ r: 4, fill: "var(--chart-1)", stroke: "var(--card)", strokeWidth: 2 }} />
                        </LineChart>
                      </ResponsiveContainer>
                    ) : (
                      <div className="flex h-full items-center justify-center text-xs text-muted-foreground">No volume data yet</div>
                    )}
                  </div>
                </article>

                <article id="model-distribution" className="rounded-lg border bg-card p-4 shadow-sm sm:p-5">
                  <div><h2 className="text-sm font-bold">Model distribution</h2><p className="mt-1 text-xs text-muted-foreground">Routing decisions by model</p></div>
                  {modelPieData.length > 0 ? (
                    <>
                      <div className="mt-2 h-44">
                        <ResponsiveContainer width="100%" height="100%">
                          <PieChart>
                            <Pie data={modelPieData} dataKey="value" nameKey="name" innerRadius={48} outerRadius={70} paddingAngle={2}>
                              {modelPieData.map((item) => <Cell key={item.name} fill={item.color} stroke="var(--card)" strokeWidth={2} />)}
                            </Pie>
                            <Tooltip content={<ChartTip />} />
                          </PieChart>
                        </ResponsiveContainer>
                      </div>
                      <div className="space-y-2">
                        {modelPieData.map((item) => (
                          <div key={item.name} className="flex items-center text-xs">
                            <span className="mr-2 size-2 rounded-full" style={{ backgroundColor: item.color }} />
                            <span className="truncate font-mono text-muted-foreground">{item.name}</span>
                            <span className="ml-auto font-mono font-semibold">{item.value}</span>
                          </div>
                        ))}
                      </div>
                    </>
                  ) : (
                    <div className="mt-4 flex h-44 items-center justify-center text-xs text-muted-foreground">No model data yet</div>
                  )}
                </article>
              </section>

              {/* ── Provider traffic + Category breakdown ──────────────── */}
              <section className="mt-4 grid gap-4 xl:grid-cols-2">
                <article className="rounded-lg border bg-card p-4 shadow-sm sm:p-5">
                  <div><h2 className="text-sm font-bold">Traffic by provider</h2><p className="mt-1 text-xs text-muted-foreground">Request count per cloud / local provider</p></div>
                  <div className="mt-4 h-56">
                    {providerBarData.length > 0 ? (
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={providerBarData} layout="vertical" margin={{ left: 10, right: 15 }}>
                          <XAxis type="number" hide />
                          <YAxis dataKey="name" type="category" axisLine={false} tickLine={false} width={120} tick={{ fill: "var(--muted-foreground)", fontSize: 11 }} />
                          <Tooltip content={<ChartTip />} cursor={{ fill: "var(--accent)" }} />
                          <Bar dataKey="requests" radius={[0, 4, 4, 0]} barSize={14}>
                            {providerBarData.map((item) => <Cell key={item.name} fill={item.fill} />)}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    ) : (
                      <div className="flex h-full items-center justify-center text-xs text-muted-foreground">No provider data yet</div>
                    )}
                  </div>
                </article>

                <article className="rounded-lg border bg-card p-4 shadow-sm sm:p-5">
                  <div><h2 className="text-sm font-bold">Category breakdown</h2><p className="mt-1 text-xs text-muted-foreground">Classifier category distribution</p></div>
                  {categoryPieData.length > 0 ? (
                    <>
                      <div className="mt-2 h-44">
                        <ResponsiveContainer width="100%" height="100%">
                          <PieChart>
                            <Pie data={categoryPieData} dataKey="value" nameKey="name" innerRadius={48} outerRadius={70} paddingAngle={2}>
                              {categoryPieData.map((item) => <Cell key={item.name} fill={item.color} stroke="var(--card)" strokeWidth={2} />)}
                            </Pie>
                            <Tooltip content={<ChartTip />} />
                          </PieChart>
                        </ResponsiveContainer>
                      </div>
                      <div className="space-y-2">
                        {categoryPieData.map((item) => (
                          <div key={item.name} className="flex items-center text-xs">
                            <span className="mr-2 size-2 rounded-full" style={{ backgroundColor: item.color }} />
                            <span className="truncate font-mono capitalize text-muted-foreground">{item.name}</span>
                            <span className="ml-auto font-mono font-semibold">{item.value}</span>
                          </div>
                        ))}
                      </div>
                    </>
                  ) : (
                    <div className="mt-4 flex h-44 items-center justify-center text-xs text-muted-foreground">No category data yet</div>
                  )}
                </article>
              </section>

              {/* ── Cost over time + Routing modes + Tier distribution ─── */}
              <section id="cost-section" className="mt-4 grid gap-4 xl:grid-cols-[minmax(0,1.8fr)_minmax(200px,0.5fr)_minmax(200px,0.5fr)]">
                <article className="rounded-lg border bg-card p-4 shadow-sm sm:p-5">
                  <div><h2 className="text-sm font-bold">Cumulative cost</h2><p className="mt-1 text-xs text-muted-foreground">Estimated USD spend over time (24h)</p></div>
                  <div className="mt-4 h-56">
                    {costChart.length > 0 ? (
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={costChart} margin={{ top: 8, right: 8, left: -10, bottom: 0 }}>
                          <CartesianGrid vertical={false} stroke="var(--border)" strokeDasharray="3 3" />
                          <XAxis dataKey="label" axisLine={false} tickLine={false} tick={{ fill: "var(--muted-foreground)", fontSize: 11 }} />
                          <YAxis axisLine={false} tickLine={false} tick={{ fill: "var(--muted-foreground)", fontSize: 11 }} tickFormatter={(v: number) => `$${v.toFixed(2)}`} />
                          <Tooltip
                            content={({ active, payload, label }) => {
                              if (!active || !payload?.length) return null;
                              return (
                                <div className="rounded-md border bg-popover px-3 py-2 shadow-lg">
                                  <p className="text-xs text-muted-foreground">{label}</p>
                                  <p className="mt-1 font-mono text-sm font-semibold text-popover-foreground">${(payload[0]?.value as number ?? 0).toFixed(4)}</p>
                                </div>
                              );
                            }}
                            cursor={{ stroke: "var(--border)" }}
                          />
                          <defs>
                            <linearGradient id="costGrad" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="var(--chart-3)" stopOpacity={0.3} />
                              <stop offset="95%" stopColor="var(--chart-3)" stopOpacity={0.02} />
                            </linearGradient>
                          </defs>
                          <Area type="monotone" dataKey="cost" stroke="var(--chart-3)" strokeWidth={2} fill="url(#costGrad)" dot={false} />
                        </AreaChart>
                      </ResponsiveContainer>
                    ) : (
                      <div className="flex h-full items-center justify-center text-xs text-muted-foreground">No cost data yet</div>
                    )}
                  </div>
                </article>

                <article className="rounded-lg border bg-card p-4 shadow-sm sm:p-5">
                  <div><h2 className="text-sm font-bold">Routing modes</h2><p className="mt-1 text-xs text-muted-foreground">Requests by routing policy</p></div>
                  <div className="mt-4 h-56">
                    {modeBarData.length > 0 ? (
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={modeBarData} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                          <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fill: "var(--muted-foreground)", fontSize: 9 }} angle={-45} textAnchor="end" height={60} />
                          <YAxis axisLine={false} tickLine={false} tick={{ fill: "var(--muted-foreground)", fontSize: 11 }} />
                          <Tooltip content={<ChartTip />} cursor={{ fill: "var(--accent)" }} />
                          <Bar dataKey="count" radius={[4, 4, 0, 0]} barSize={20}>
                            {modeBarData.map((item) => <Cell key={item.name} fill={item.fill} />)}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    ) : (
                      <div className="flex h-full items-center justify-center text-xs text-muted-foreground">No mode data yet</div>
                    )}
                  </div>
                </article>

                <article className="rounded-lg border bg-card p-4 shadow-sm sm:p-5">
                  <div><h2 className="text-sm font-bold">Tier split</h2><p className="mt-1 text-xs text-muted-foreground">Advanced vs standard</p></div>
                  {tierDonutData.length > 0 ? (
                    <>
                      <div className="mt-2 h-36">
                        <ResponsiveContainer width="100%" height="100%">
                          <PieChart>
                            <Pie data={tierDonutData} dataKey="value" nameKey="name" innerRadius={36} outerRadius={56} paddingAngle={3}>
                              {tierDonutData.map((item) => <Cell key={item.name} fill={item.color} stroke="var(--card)" strokeWidth={2} />)}
                            </Pie>
                            <Tooltip content={<ChartTip />} />
                          </PieChart>
                        </ResponsiveContainer>
                      </div>
                      <div className="space-y-2">
                        {tierDonutData.map((item) => (
                          <div key={item.name} className="flex items-center text-xs">
                            <span className="mr-2 size-2 rounded-full" style={{ backgroundColor: item.color }} />
                            <span className="font-mono capitalize text-muted-foreground">{item.name}</span>
                            <span className="ml-auto font-mono font-semibold">{item.value}</span>
                          </div>
                        ))}
                      </div>
                    </>
                  ) : (
                    <div className="mt-4 flex h-36 items-center justify-center text-xs text-muted-foreground">No tier data yet</div>
                  )}
                </article>
              </section>

              {/* ── Event stream ───────────────────────────────────────── */}
              <section className="mt-4" id="event-stream">
                <article className="min-w-0 rounded-lg border bg-card p-4 shadow-sm sm:p-5">
                  <div className="flex items-center justify-between gap-3">
                    <div className="flex items-center gap-2">
                      <div><h2 className="text-sm font-bold">Routing events</h2><p className="mt-1 text-xs text-muted-foreground">Recent routing decisions from the active process</p></div>
                      <span className="rounded border border-chart-2/20 bg-chart-2/10 px-2 py-1 text-[10px] font-bold text-chart-2">LIVE</span>
                    </div>
                    <div className="flex items-center gap-1">
                      {events && <LogsDialog events={events} filter={filter} setFilter={setFilter} />}
                    </div>
                  </div>
                  <div className="mt-3 overflow-hidden">
                    {events && events.length > 0 ? (
                      events.slice(0, 8).map((event) => <EventRow event={event} key={event.id} />)
                    ) : (
                      <div className="py-8 text-center text-xs text-muted-foreground">No events recorded yet — route a query to see data here</div>
                    )}
                  </div>
                </article>
              </section>

              {/* ── Footer ─────────────────────────────────────────────── */}
              <footer className="mt-5 flex flex-col gap-2 border-t py-4 text-xs text-muted-foreground sm:flex-row sm:items-center sm:justify-between">
                <span className="flex items-center gap-2"><Package className="size-3.5" />coding-router · Python</span>
                <span className="font-mono">127.0.0.1:3000 · polling every 5s</span>
              </footer>
            </>
          )}
        </div>
      </main>
    </div>
  );
}