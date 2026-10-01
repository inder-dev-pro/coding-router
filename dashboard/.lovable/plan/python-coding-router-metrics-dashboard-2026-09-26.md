# Python Coding Router Metrics Dashboard

## Goal
Build a single-screen frontend dashboard for a Python coding-router library, closely following the supplied reference while using original interface code and demo metrics. The dashboard should feel like the local observability UI that opens when a developer runs the router, not a standalone SaaS product. There will be no login, backend, or prompt-entry workflow.

## Dashboard structure
- Fixed desktop sidebar with the Python library identity, Overview navigation, runtime status, package version, and a working collapse/expand control.
- Compact mobile header and slide-out navigation so the full dashboard remains usable on smaller screens.
- Header with “Router Overview,” a short local-runtime monitoring description, system status, light/dark toggle, and Export Report action.
- Six summary metrics: routed requests, active consumers, estimated spend, savings, average latency, and routes active.
- Request-volume line chart with selectable time ranges and hover details.
- Route-distribution donut chart with a readable legend and totals.
- Traffic-by-destination horizontal comparison chart.
- Recent routing event stream showing timestamp, complexity label, selected route/model, shortened prompt summary, and latency, framed as telemetry emitted by the running Python package.

## Working interactions
- Light and dark themes, respecting the device preference initially and remembering the user’s choice.
- Collapsible sidebar that remains recoverable in its compact state.
- Time-range selector that updates chart data and summary context.
- Chart hover tooltips and accessible legends.
- Live event control to pause/resume simulated incoming routing events; metrics and charts update consistently when new events arrive.
- Export Report downloads the currently displayed metrics and event rows as a CSV file.
- “View detailed logs” opens a focused dialog with the full event list and basic complexity filtering.
- Responsive layouts for desktop, tablet, and mobile with legible type and no clipped content.

## Visual direction
- Follow the reference’s restrained monitoring-console composition: pale neutral workspace, white surfaces, thin borders, subtle shadows, and compact information density.
- Use a balanced set of cyan, teal, amber, and red accents for routing status and chart categories rather than a one-color theme.
- Use a highly legible geometric sans-serif, normal letter spacing, clear hierarchy, and slightly larger minimum body text than the screenshot.
- Dark mode uses charcoal surfaces with softened contrast instead of saturated navy.
- Motion stays minimal: chart transitions, status pulse, sidebar movement, and new-row entry, all disabled when reduced motion is preferred.

## Technical details
- Implement reusable React components for navigation, metric cards, charts, event rows, filters, dialogs, and theme control.
- Use the project’s existing Tailwind design tokens, shadcn controls, Lucide icons, and Recharts package.
- Keep all metrics and event simulation in deterministic frontend data/state so the dashboard works immediately without Cloud or external services; structure the data boundary so the Python library can later supply the same metric and event shapes.
- Avoid account, workspace, billing, or hosted-product language; use local runtime, package, router, consumer, and destination terminology throughout.
- Add dashboard-specific page metadata and replace the current placeholder page.
- Verify the finished screen in light and dark themes at desktop and mobile sizes, including CSV export, sidebar, filtering, pause/resume, and chart interactions.
