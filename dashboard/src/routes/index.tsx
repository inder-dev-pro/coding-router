import { createFileRoute } from "@tanstack/react-router";
import { RouterDashboard } from "@/components/router-dashboard";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "CodeRoute — Python Router Dashboard" },
      { name: "description", content: "Monitor routing decisions, latency, cost, and runtime events from the CodeRoute Python library." },
      { property: "og:title", content: "CodeRoute — Python Router Dashboard" },
      { property: "og:description", content: "Local observability for your Python coding router." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: RouterDashboard,
});
