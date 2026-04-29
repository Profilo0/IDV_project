"""
interaction_log.py
==================
Drop-in module for dashboard_v3.py that adds behavioral preference profiling.

Usage in dashboard_v3.py:
    from interaction_log import add_logging_to_app, INTERACTION_STORE_IDS

Paste the three layout additions and register_callbacks() call as directed below.

Theory grounding:
  Stated preferences  → weight sliders set explicitly by the user
  Revealed preferences → what the user actually explores (time, clicks, salary range)
  
  Combination approach from:
    Riedl et al. (2012) "Serendipity in Recommender Systems"
    Slovic et al. (1977) "Behavioral Decision Theory" — preference reversal under
    different elicitation methods; stated ≠ revealed is well-documented.
    Oral et al. (2023) — note that no reviewed tool captures revealed preferences.
"""

import json
import time
import math
from datetime import datetime
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

import dash
from dash import dcc, html, Input, Output, State, callback_context, ALL, clientside_callback

# ─── COLOUR PALETTE (matches dashboard_v3) ────────────────────────────────────
C = dict(
    bg="#F7F5F0", surf="#FFFFFF", panel="#F0EDE8", border="#D6D1C8",
    pad="#B85C28", hel="#1E6FAD", gold="#7A5F0A",
    good="#166534", warn="#B91C1C", text="#1C1917", muted="#57534E",
)

LAYOUT = dict(
    paper_bgcolor=C["bg"], plot_bgcolor=C["surf"],
    font=dict(family="IBM Plex Sans, sans-serif", color=C["text"], size=12),
    margin=dict(l=60, r=40, t=48, b=55),
    xaxis=dict(gridcolor=C["border"], zeroline=False, linecolor=C["border"]),
    yaxis=dict(gridcolor=C["border"], zeroline=False, linecolor=C["border"]),
    hoverlabel=dict(bgcolor=C["surf"], bordercolor=C["border"], font=dict(size=12)),
)

def LO(**kw):
    base = {k:v for k,v in LAYOUT.items() if k not in kw}
    return {**base, **kw}

# ─── TAB → CRITERION MAPPING ─────────────────────────────────────────────────
# Maps each dashboard tab to the decision criterion it most closely probes.
# Used to infer revealed preferences from tab dwell time.
TAB_TO_CRITERION = {
    "salary":   "salary",    # income comparison → salary matters
    "expenses": "col",       # personalised cost basket → cost of living matters
    "prices":   "col",       # market prices → cost of living
    "wages":    "salary",    # wage distribution → salary/career
    "city":     "qol",       # urban quality indicators → quality of life
    "whatif":   "salary",    # offer comparison → salary is the frame
    "decision": None,        # the choice stage itself — no single criterion
    "datasets": None,        # data exploration — ambiguous
}

# All criterion IDs (must match dashboard_v3.py CRITERIA + "salary")
ALL_CRITERIA = [
    "salary", "qol", "safe", "air", "trans",
    "hous", "col", "emp", "edu"
]

CRITERION_LABELS = {
    "salary": "💰 Salary",
    "qol":    "😊 Quality of life",
    "safe":   "🛡 Safety",
    "air":    "🌿 Air quality",
    "trans":  "🚌 Public transport",
    "hous":   "🏠 Housing affordability",
    "col":    "🛒 Cost of living",
    "emp":    "💼 Employment",
    "edu":    "🎓 Education",
}

# ─── STORE IDS (add these to your app.layout) ─────────────────────────────────
INTERACTION_STORE_IDS = [
    "log-tab-events",        # list of {tab, timestamp, session_elapsed_s}
    "log-salary-events",     # list of {value, timestamp, session_elapsed_s}
    "log-session-start",     # float: session start Unix timestamp
    "log-tab-dwell",         # dict {tab: total_seconds}
    "log-last-tab",          # str: name of the currently active tab
    "log-tab-entry-time",    # float: Unix timestamp when current tab was entered
]

# ─── LAYOUT COMPONENTS ────────────────────────────────────────────────────────
def log_stores():
    """Return a list of dcc.Store components. Add to app.layout."""
    return [
        dcc.Store(id="log-tab-events",    data=[]),
        dcc.Store(id="log-salary-events", data=[]),
        dcc.Store(id="log-session-start", data=None),
        dcc.Store(id="log-tab-dwell",     data={t:0 for t in TAB_TO_CRITERION}),
        dcc.Store(id="log-last-tab",      data="salary"),
        dcc.Store(id="log-tab-entry-time",data=None),
        dcc.Interval(id="log-ticker", interval=5000, n_intervals=0),  # 5s heartbeat
    ]

def log_panel():
    """
    Return the Behavioral Profile UI panel.
    Insert this as the content of a new 'profile' tab pane in app.layout.
    """
    S_note = {"color":C["muted"],"fontSize":"12px","lineHeight":"1.7",
              "padding":"10px 16px","background":C["panel"],
              "border":f"1px solid {C['border']}","borderRadius":"6px","marginBottom":"14px"}
    return html.Div([
        html.Div(
            "Your revealed preferences are inferred from where you spend time "
            "in the dashboard. This complements the explicit weights you set in "
            "the My Decision tab. Research shows stated and revealed preferences "
            "often diverge (Slovic et al., 1977) — the comparison can help you "
            "reflect on what you actually care about.",
            style=S_note),

        html.Div([
            # Left: radar chart comparing stated vs revealed
            html.Div([
                dcc.Graph(id="log-fig-radar", config={"displayModeBar":False}),
            ], style={"flex":"1","minWidth":"300px"}),

            # Right: raw stats + export
            html.Div([
                html.Div("Session statistics", style={
                    "color":C["muted"],"fontSize":"10px","letterSpacing":".1em",
                    "textTransform":"uppercase","fontWeight":"600","marginBottom":"12px"}),
                html.Div(id="log-stats-panel"),
                html.Hr(style={"borderColor":C["border"],"margin":"16px 0"}),
                html.Div("Salary exploration range", style={
                    "color":C["muted"],"fontSize":"10px","letterSpacing":".1em",
                    "textTransform":"uppercase","fontWeight":"600","marginBottom":"8px"}),
                dcc.Graph(id="log-fig-salary-hist",
                          config={"displayModeBar":False},
                          style={"height":"160px"}),
                html.Hr(style={"borderColor":C["border"],"margin":"16px 0"}),
                html.Button("💾 Export session log (JSON)",
                    id="log-export-btn", n_clicks=0,
                    style={"background":C["hel"],"color":"white","border":"none",
                           "borderRadius":"6px","padding":"8px 18px","cursor":"pointer",
                           "fontFamily":"inherit","fontSize":"13px","fontWeight":"600"}),
                dcc.Download(id="log-download"),
                html.Div(id="log-export-status",
                         style={"color":C["muted"],"fontSize":"11px","marginTop":"8px"}),
            ], style={"flex":"1","minWidth":"280px"}),
        ], style={"display":"flex","gap":"24px","flexWrap":"wrap","alignItems":"flex-start"}),

        html.Hr(style={"borderColor":C["border"],"margin":"20px 0"}),

        # Dwell-time bar chart
        html.Div("Time spent per tab (seconds)", style={
            "color":C["muted"],"fontSize":"10px","letterSpacing":".1em",
            "textTransform":"uppercase","fontWeight":"600","marginBottom":"8px"}),
        dcc.Graph(id="log-fig-dwell", config={"displayModeBar":False}),
    ], id="pane-profile", style={"display":"none","padding":"20px 28px"})


# ─── REGISTER CALLBACKS ───────────────────────────────────────────────────────
def register_callbacks(app):
    """
    Call this after defining your app.layout.
    Registers all interaction-logging callbacks.
    Requires that log_stores() components are in the layout.
    """

    # ── 1. Initialise session start time (client-side, one-shot) ─────────────
    clientside_callback(
        """
        function(n) {
            if (n === null || n === undefined) {
                return Date.now() / 1000;
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output("log-session-start", "data"),
        Input("log-ticker", "n_intervals"),
        State("log-session-start", "data"),
        prevent_initial_call=False,
    )

    # ── 2. Record tab navigation events ──────────────────────────────────────
    # Tab IDs that exist in the dashboard
    TAB_IDS = list(TAB_TO_CRITERION.keys()) + ["profile"]

    # We observe n_clicks on every tab button; the callback_context tells us which fired.
    @app.callback(
        Output("log-tab-events",    "data"),
        Output("log-tab-dwell",     "data"),
        Output("log-last-tab",      "data"),
        Output("log-tab-entry-time","data"),
        *[Input(f"tab-{t}","n_clicks") for t in TAB_IDS
          if True],  # all tabs including profile
        State("log-tab-events",    "data"),
        State("log-tab-dwell",     "data"),
        State("log-last-tab",      "data"),
        State("log-tab-entry-time","data"),
        State("log-session-start", "data"),
        prevent_initial_call=True,
    )
    def record_tab_event(*args):
        n_tabs  = len(TAB_IDS)
        clicks  = list(args[:n_tabs])
        events  = args[n_tabs]     or []
        dwell   = args[n_tabs+1]   or {t:0 for t in TAB_TO_CRITERION}
        last    = args[n_tabs+2]   or "salary"
        entry_t = args[n_tabs+3]
        sess_t  = args[n_tabs+4]

        ctx  = callback_context
        if not ctx.triggered:
            return events, dwell, last, entry_t

        now     = time.time()
        new_tab = ctx.triggered[0]["prop_id"].split("-",1)[1].split(".")[0]

        # Accumulate dwell time for the tab we are leaving
        if entry_t is not None and last in dwell:
            elapsed = now - float(entry_t)
            dwell   = dict(dwell)
            dwell[last] = round(dwell.get(last, 0) + elapsed, 1)

        sess_elapsed = round(now - float(sess_t), 1) if sess_t else 0
        events = list(events) + [{
            "tab":              new_tab,
            "timestamp":        datetime.utcnow().isoformat(),
            "session_elapsed_s":sess_elapsed,
        }]

        return events, dwell, new_tab, now

    # ── 3. Record salary slider events (sampled, not every px move) ──────────
    @app.callback(
        Output("log-salary-events","data"),
        Input("sal-slider","value"),
        State("log-salary-events","data"),
        State("log-session-start", "data"),
        prevent_initial_call=True,
    )
    def record_salary(val, events, sess_t):
        events = list(events or [])
        sess_elapsed = round(time.time() - float(sess_t), 1) if sess_t else 0
        events.append({
            "value":            val,
            "timestamp":        datetime.utcnow().isoformat(),
            "session_elapsed_s":sess_elapsed,
        })
        # Keep max 500 events to avoid Store bloat
        return events[-500:]

    # ── 4. Heartbeat — update dwell for currently active tab ─────────────────
    @app.callback(
        Output("log-tab-dwell","data",   allow_duplicate=True),
        Input("log-ticker","n_intervals"),
        State("log-tab-dwell",     "data"),
        State("log-last-tab",      "data"),
        State("log-tab-entry-time","data"),
        prevent_initial_call=True,
    )
    def heartbeat_dwell(_, dwell, last_tab, entry_t):
        if entry_t is None or last_tab is None:
            return dwell or {}
        now  = time.time()
        elapsed = now - float(entry_t)
        dwell = dict(dwell or {})
        # Only credit the last 5s (interval), not the full elapsed since entry
        # This prevents over-counting if the user leaves the browser idle
        dwell[last_tab] = round(dwell.get(last_tab, 0) + 5, 1)
        return dwell

    # ── 5. Compute revealed preference weights from dwell times ───────────────
    def compute_revealed(dwell):
        """
        Convert dwell-time dict to criterion scores (0–5 scale) by:
          1. Map tab dwell → criterion contribution (using TAB_TO_CRITERION)
          2. Aggregate (sum) across tabs that map to the same criterion
          3. Normalise to 0–5 scale matching the stated weight sliders
        Returns dict {criterion_id: float 0–5}
        """
        raw = {cid: 0.0 for cid in ALL_CRITERIA}
        for tab, seconds in (dwell or {}).items():
            crit = TAB_TO_CRITERION.get(tab)
            if crit and crit in raw:
                raw[crit] += seconds

        total = sum(raw.values())
        if total == 0:
            return {cid: 0.0 for cid in ALL_CRITERIA}

        # Normalise to 0–5
        max_val = max(raw.values(), default=1)
        return {cid: round(min(5.0, v / max_val * 5), 2)
                for cid, v in raw.items()}

    # ── 6. Radar chart: stated vs revealed ────────────────────────────────────
    @app.callback(
        Output("log-fig-radar","figure"),
        Input("log-tab-dwell","data"),
        *[Input(f"wt-{cid}","value") for cid in ALL_CRITERIA],
        prevent_initial_call=False,
    )
    def update_radar(dwell, *stated_vals):
        stated   = dict(zip(ALL_CRITERIA, [v or 0 for v in stated_vals]))
        revealed = compute_revealed(dwell)
        labels   = [CRITERION_LABELS.get(c, c) for c in ALL_CRITERIA]
        st_vals  = [stated.get(c,   0) for c in ALL_CRITERIA]
        rv_vals  = [revealed.get(c, 0) for c in ALL_CRITERIA]

        # Close the polygon
        labels  += [labels[0]]
        st_vals += [st_vals[0]]
        rv_vals += [rv_vals[0]]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=st_vals, theta=labels, fill="toself",
            name="Stated (sliders)",
            line=dict(color=C["hel"], width=2),
            fillcolor="rgba(30,111,173,0.12)",
            hovertemplate="<b>%{theta}</b><br>Stated weight: %{r:.1f}/5<extra></extra>"))
        fig.add_trace(go.Scatterpolar(
            r=rv_vals, theta=labels, fill="toself",
            name="Revealed (behaviour)",
            line=dict(color=C["pad"], width=2, dash="dash"),
            fillcolor="rgba(184,92,40,0.10)",
            hovertemplate="<b>%{theta}</b><br>Revealed score: %{r:.1f}/5<extra></extra>"))

        fig.update_layout(
            paper_bgcolor=C["bg"],
            polar=dict(
                bgcolor=C["surf"],
                radialaxis=dict(visible=True, range=[0,5],
                                tickfont=dict(size=9, color=C["muted"]),
                                gridcolor=C["border"]),
                angularaxis=dict(tickfont=dict(size=10, color=C["text"]),
                                 gridcolor=C["border"]),
            ),
            legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center",
                        font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
            hoverlabel=dict(bgcolor=C["surf"], bordercolor=C["border"],
                            font=dict(size=12)),
            title=dict(text="Stated vs Revealed Preferences",
                       font=dict(size=13, color=C["text"]), x=0.5),
            height=360, margin=dict(l=50, r=50, t=55, b=60),
        )
        return fig

    # ── 7. Salary histogram ───────────────────────────────────────────────────
    @app.callback(
        Output("log-fig-salary-hist","figure"),
        Input("log-salary-events","data"),
        prevent_initial_call=False,
    )
    def update_salary_hist(events):
        events = events or []
        if not events:
            fig = go.Figure()
            fig.update_layout(**LO(height=160,
                title=dict(text="No salary interactions yet",
                           font=dict(size=11, color=C["muted"]))))
            return fig
        vals = [e["value"] for e in events if "value" in e]
        if not vals:
            return go.Figure().update_layout(**LO(height=160))

        # Histogram of explored values
        fig = go.Figure(go.Histogram(
            x=vals, nbinsx=20,
            marker_color=C["hel"], opacity=0.75,
            hovertemplate="Range: %{x}<br>Interactions: %{y}<extra></extra>"))

        # Mark median explored value
        med = sorted(vals)[len(vals)//2]
        fig.add_vline(x=med, line_color=C["gold"], line_dash="dot",
            annotation_text=f"Median explored: €{med:,.0f}",
            annotation_font=dict(size=9, color=C["gold"]),
            annotation_position="top right")

        fig.update_layout(**LO(height=160, showlegend=False,
            title=dict(text="Salary values you explored",
                       font=dict(size=11, color=C["text"])),
            xaxis_title="Gross salary (€)", yaxis_title="Times",
            xaxis=dict(tickprefix="€", gridcolor=C["border"],
                       zeroline=False, linecolor=C["border"]),
            yaxis=dict(gridcolor=C["border"], zeroline=False,
                       linecolor=C["border"]),
            margin=dict(l=40, r=20, t=35, b=40)))
        return fig

    # ── 8. Dwell-time bar chart ───────────────────────────────────────────────
    @app.callback(
        Output("log-fig-dwell","figure"),
        Input("log-tab-dwell","data"),
        prevent_initial_call=False,
    )
    def update_dwell(dwell):
        dwell = dwell or {}
        tabs   = [t for t in TAB_TO_CRITERION if t in dwell]
        dwells = [round(dwell.get(t, 0), 1) for t in tabs]
        total  = sum(dwells)
        pcts   = [round(d/total*100, 1) if total else 0 for d in dwells]
        crit   = [TAB_TO_CRITERION.get(t) for t in tabs]
        cols   = [C["hel"] if c in ("salary","emp","edu") else
                  C["pad"] if c in ("col","hous") else
                  C["good"] for c in crit]
        labels = [f"{d}s ({p}%)" for d,p in zip(dwells, pcts)]

        fig = go.Figure(go.Bar(
            x=tabs, y=dwells, marker_color=cols,
            text=labels, textposition="outside",
            textfont=dict(size=10, color=C["text"]),
            hovertemplate="<b>%{x}</b><br>Time: %{y}s<br>"
                          "Maps to: %{customdata}<extra></extra>",
            customdata=[CRITERION_LABELS.get(c,"—") for c in crit]))

        fig.update_layout(**LO(height=220, showlegend=False,
            title=dict(text="Time spent per tab (5-second heartbeat resolution)",
                       font=dict(size=11, color=C["text"])),
            xaxis_title="Tab", yaxis_title="Seconds",
            xaxis=dict(gridcolor=C["border"], zeroline=False, linecolor=C["border"]),
            yaxis=dict(gridcolor=C["border"], zeroline=False, linecolor=C["border"]),
            margin=dict(l=50, r=20, t=40, b=45)))
        return fig

    # ── 9. Session statistics panel ───────────────────────────────────────────
    @app.callback(
        Output("log-stats-panel","children"),
        Input("log-tab-events",    "data"),
        Input("log-salary-events", "data"),
        Input("log-tab-dwell",     "data"),
        prevent_initial_call=False,
    )
    def update_stats(tab_evts, sal_evts, dwell):
        tab_evts = tab_evts or []; sal_evts = sal_evts or []; dwell = dwell or {}
        total_s  = sum(dwell.values())
        most_tab = max(dwell, key=dwell.get) if dwell else "—"
        sal_vals = [e["value"] for e in sal_evts if "value" in e]
        sal_range= f"€{min(sal_vals):,.0f} – €{max(sal_vals):,.0f}" if len(sal_vals)>1 else "—"
        S_row = {"display":"flex","justifyContent":"space-between",
                 "padding":"6px 0","borderBottom":f"1px solid {C['border']}",
                 "fontSize":"12px"}
        rows = [
            ("Tab navigations",    str(len(tab_evts))),
            ("Salary adjustments", str(len(sal_evts))),
            ("Total session time", f"{int(total_s)}s"),
            ("Most explored tab",  most_tab),
            ("Salary range explored", sal_range),
        ]
        return [html.Div([
            html.Span(k, style={"color":C["muted"]}),
            html.Span(v, style={"color":C["text"],"fontWeight":"600"}),
        ], style=S_row) for k,v in rows]

    # ── 10. Export session log ────────────────────────────────────────────────
    @app.callback(
        Output("log-download","data"),
        Output("log-export-status","children"),
        Input("log-export-btn","n_clicks"),
        State("log-tab-events",    "data"),
        State("log-salary-events", "data"),
        State("log-tab-dwell",     "data"),
        *[State(f"wt-{cid}","value") for cid in ALL_CRITERIA],
        prevent_initial_call=True,
    )
    def export_log(n, tab_evts, sal_evts, dwell, *stated_vals):
        stated   = dict(zip(ALL_CRITERIA, [v or 0 for v in stated_vals]))
        revealed = compute_revealed(dwell)
        payload  = {
            "export_timestamp": datetime.utcnow().isoformat(),
            "session_summary": {
                "tab_navigations":    len(tab_evts or []),
                "salary_adjustments": len(sal_evts or []),
                "total_dwell_s":      sum((dwell or {}).values()),
                "dwell_per_tab":      dwell,
            },
            "preferences": {
                "stated":   stated,
                "revealed": revealed,
                "divergence": {
                    cid: round(abs(stated.get(cid,0) - revealed.get(cid,0)), 2)
                    for cid in ALL_CRITERIA
                },
            },
            "raw_events": {
                "tab_events":    tab_evts or [],
                "salary_events": sal_evts or [],
            },
        }
        filename = f"session_log_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        return (dcc.send_string(json.dumps(payload, indent=2), filename),
                f"✓ Exported {filename}")

    print("  ✓ Interaction logging callbacks registered")


# ─── HOW TO INTEGRATE INTO dashboard_v3.py ────────────────────────────────────
INTEGRATION_GUIDE = """
INTEGRATION STEPS FOR dashboard_v3.py
======================================

1. At the top of dashboard_v3.py, add:
   from interaction_log import log_stores, log_panel, register_callbacks

2. In app.layout, add log_stores() to the list of children:
   dcc.Store(id="active-tab", data="salary"),
   dcc.Store(id="expense-totals", data={"pad":0,"hel":0}),
   *log_stores(),                            ← ADD THIS LINE

3. Add a new tab button to TABS_DEF:
   TABS_DEF = [
       ...existing tabs...,
       ("profile", "📈 My Profile"),         ← ADD THIS ENTRY
   ]

4. Add the pane to app.layout (just before the footer):
   log_panel(),                              ← ADD THIS LINE

5. After app.layout = html.Div([...]):
   register_callbacks(app)                   ← ADD THIS LINE

6. The switch_tabs callback in dashboard_v3.py already handles pane
   visibility dynamically — the "profile" tab will be picked up
   automatically because it uses the same tab-{id} / pane-{id} pattern.

That is all. No other changes to dashboard_v3.py required.
"""

if __name__ == "__main__":
    print(INTEGRATION_GUIDE)
