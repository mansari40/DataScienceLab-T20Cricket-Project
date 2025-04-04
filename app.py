import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
from matplotlib.lines import Line2D

@st.cache_data#data load once
def ld(path="IPL_2018_2024.xlsx"):
    df = pd.read_excel(path)
    cols = [
        "bat", "batruns", "out", "dismissal",
        "wagonX", "wagonY", "wagonZone", "line", "length",
        "control", "bowl_style", "year", "bat_hand"
    ]
    df = df[[c for c in cols if c in df.columns]].drop_duplicates()
    
    spin = {"OB", "LB", "LBG", "SLA", "LWS", "RWS"}
    pace = {"RF", "RFM", "RMF", "RM", "LF", "LFM", "LMF", "LM"}
    def classify_bowler(s):
        if s in spin:
            return "Spin"
        elif s in pace:
            return "Pace"
        else:
            return "Unknown"
    
    if "bowl_style" in df.columns:
        df["bowler_type"] = df["bowl_style"].apply(classify_bowler)
    else:
        df["bowler_type"] = "Unknown"
    
    return df

def shift_coords(df):
    centerX = df["wagonX"].median()
    centerY = df["wagonY"].median()
    df["plotX"] = df["wagonX"] - centerX
    df["plotY"] = df["wagonY"] - centerY
    return df


#all/zoneshot
def sd(df):
    zone_ct = df.groupby(["line", "length", "wagonZone"]).size().reset_index(name="ShotsInZone")
    total_ct = df.groupby(["line", "length"]).size().reset_index(name="AllShots")
    merged = pd.merge(zone_ct, total_ct, on=["line", "length"], how="left")
    def calc_sd(row):
        if row["ShotsInZone"] == 0:
            return 1.0
        return row["AllShots"] / row["ShotsInZone"]
    merged["shot_difficulty"] = merged.apply(calc_sd, axis=1)
    df = pd.merge(
        df,
        merged[["line", "length", "wagonZone", "shot_difficulty"]],
        on=["line", "length", "wagonZone"],
        how="left"
    )
    df["shot_difficulty"] = df["shot_difficulty"].fillna(1.0)
    return df

def wagonGen(df, ax):
    bdf = df[df["batruns"].isin([4,6])].copy()
    if bdf.empty:
        ax.set_title("Normal Wagon Wheel - No Boundaries")
        ax.add_patch(Circle((0,0),200,color="green",alpha=0.2))
        ax.set_xlim(-220,220)
        ax.set_ylim(-220,220)
        ax.set_aspect("equal", "box")
        ax.axis("off")
        return
    ax.set_title("General Boundary Wagon Wheel")
    ax.add_patch(Circle((0,0),200,color="green",alpha=0.2))
    bdf["angle_rad"] = np.arctan2(bdf["plotY"], bdf["plotX"])
    bdf["fixedX"] = 200 * np.cos(bdf["angle_rad"])
    bdf["fixedY"] = 200 * np.sin(bdf["angle_rad"])
    for _, row in bdf.iterrows():
        color = "green" if row["batruns"] == 4 else "purple"
        lw = 1.0 if row["batruns"] == 4 else 1.5
        ax.plot([0, row["fixedX"]], [0, row["fixedY"]], color=color, linewidth=lw, alpha=0.8)
    ax.set_xlim(-220,220)
    ax.set_ylim(-220,220)
    ax.set_aspect("equal", "box")
    ax.axis("off")
    legend_elements = [
        Line2D([0], [0], color="green", lw=2, label="4 runs"),
        Line2D([0], [0], color="purple", lw=2, label="6 runs")
    ]
    ax.legend(handles=legend_elements, loc="upper right")

def wagonSD(df, ax):
    bdf = df[df["batruns"].isin([4,6])].copy()
    if bdf.empty:
        ax.set_title("Intelligent Wagon Wheel - No Boundaries")
        ax.add_patch(Circle((0,0),200,color="green",alpha=0.2))
        ax.set_xlim(-220,220)
        ax.set_ylim(-220,220)
        ax.set_aspect("equal", "box")
        ax.axis("off")
        return
    ax.set_title("Intelligent Wagon Wheel")
    ax.add_patch(Circle((0,0),200,color="green",alpha=0.2))
    bdf["runs_pos"] = bdf["batruns"].clip(lower=0)
    bdf["magnitude_raw"] = bdf["runs_pos"] * bdf["shot_difficulty"]
    bdf["angle_rad"] = np.arctan2(bdf["plotY"], bdf["plotX"])
    max_mag = bdf["magnitude_raw"].max()
    scale_factor = 200 / max_mag if max_mag > 0 else 1
    bdf["scaled_mag"] = bdf["magnitude_raw"] * scale_factor
    bdf["intX"] = bdf["scaled_mag"] * np.cos(bdf["angle_rad"])
    bdf["intY"] = bdf["scaled_mag"] * np.sin(bdf["angle_rad"])
    for _, row in bdf.iterrows():
        color = "green" if row["batruns"] == 4 else "purple"
        lw = 1.0 if row["batruns"] == 4 else 1.5
        ax.plot([0, row["intX"]], [0, row["intY"]], color=color, linewidth=lw, alpha=0.8)
    ax.set_xlim(-220,220)
    ax.set_ylim(-220,220)
    ax.set_aspect("equal", "box")
    ax.axis("off")
    legend_elements = [
        Line2D([0], [0], color="green", lw=2, label="4 runs"),
        Line2D([0], [0], color="purple", lw=2, label="6 runs")
    ]
    ax.legend(handles=legend_elements, loc="upper right")

def wagonZoneWheel(df, ax):
    df = df[df["wagonZone"] != 0].copy()
    if df.empty:
        ax.set_title("Wagon Zone Wheel - No Valid Zones")
        return
    def zone_stats(sub):
        balls = len(sub)
        runs = sub["batruns"].sum()
        outs = sub["dismissal"].notna().sum()
        sr = 100 * runs / balls if balls > 0 else 0
        ave = runs / outs if outs > 0 else float("inf")
        return pd.Series({"Balls": balls, "Runs": runs, "SR": sr, "Ave": ave})
    
    zone_df = df.groupby("wagonZone").apply(zone_stats).reset_index()
    zone_df["SR"] = zone_df["SR"].round(1)
    zone_df["Ave"] = zone_df["Ave"].apply(lambda x: round(x,1) if math.isfinite(x) else float("inf"))
    total_runs = zone_df["Runs"].sum()
    zone_df["pct_runs"] = 100 * zone_df["Runs"] / total_runs if total_runs > 0 else 0
    max_zone_runs = zone_df["Runs"].max() if not zone_df.empty else 1
    scf = 0.6
    outer_r = 0.8

    for i in range(1, 9):
        row = zone_df[zone_df["wagonZone"] == i]
        if row.empty:
            sr, ave, balls, runs, pctruns = 0, 0, 0, 0, 0
        else:
            sr = row["SR"].values[0]
            ave = row["Ave"].values[0]
            balls = row["Balls"].values[0]
            runs = row["Runs"].values[0]
            pctruns = row["pct_runs"].values[0]

        ang = {
            1: (45, 90),
            2: (0, 45),
            3: (315, 360),
            4: (270, 315),
            5: (225, 270),
            6: (180, 225),
            7: (135, 180),
            8: (90, 135)
        }
        start_angle, end_angle = ang.get(i, (0, 0))
        wedge_green = Wedge((0, 0), outer_r, start_angle, end_angle,
                              facecolor="limegreen", alpha=0.5, edgecolor="gray")
        ax.add_patch(wedge_green)
        frac = (runs / max_zone_runs) * scf if max_zone_runs > 0 else 0
        wedge_blue = Wedge((0, 0), frac, start_angle, end_angle,
                             facecolor="blue", alpha=0.5)
        ax.add_patch(wedge_blue)
        
        mid_angle = (start_angle + end_angle) / 2
        mid_rad = math.radians(mid_angle)
        text_r = 0.55
        text_x = text_r * math.cos(mid_rad)
        text_y = text_r * math.sin(mid_rad)
        ave_str = "inf" if ave == float("inf") else f"{ave}"
        zone_text = (f"Zone {i}\n{sr} SR\n{balls}b\n{pctruns:.1f}% runs")
        ax.text(text_x, text_y, zone_text, ha="center", va="center", fontsize=8, color="black")

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("auto")
    ax.axis("off")
    ax.set_title("Wagon Zone Wheel")

def tab(df, df_global):
    if "bowl_style" not in df.columns:
        return pd.DataFrame()
    def style_metrics(g):
        b = len(g)
        r = g["batruns"].sum()
        d = g["dismissal"].notna().sum()
        boundary_ct = g[g["batruns"].isin([4,6])].shape[0]
        dot_ct = g[g["batruns"]==0].shape[0]
        sr_val = 100 * r / b if b > 0 else 0
        dismiss_pct = d
        boundary_pct = 100 * boundary_ct / b if b > 0 else 0
        dot_pct = 100 * dot_ct / b if b > 0 else 0
        ave_val = r / d if d > 0 else float("inf")
        return pd.Series({
            "Balls": b,
            "Runs": r,
            "SR": sr_val,
            "Dismissals": dismiss_pct,
            "Boundary%": boundary_pct,
            "Dot%": dot_pct,
            "Ave": ave_val,
            "Dismissals_count": d
        })
    local_df = df.groupby("bowl_style").apply(style_metrics).reset_index()
    gf = df_global.groupby("bowl_style").agg(
        total_runs=("batruns","sum"),
        total_balls=("batruns","count")
    ).reset_index()
    gf["global_SR"] = 100 * gf["total_runs"] / gf["total_balls"]
    merged = pd.merge(local_df, gf[["bowl_style","global_SR"]], on="bowl_style", how="left")
    merged["ExpRuns"] = merged["Balls"] * (merged["global_SR"] / 100)
    merged["Impact/100b"] = merged.apply(
        lambda r: round(((r["Runs"] - 20 * r["Dismissals_count"]) / r["Balls"]) * 100, 2)
                  if r["Balls"] > 0 else 0, axis=1
    )
    bowlABB = {
        "LB": "Left Arm Fast",
        "LBG": "Left Arm Fast (Gun)",
        "LF": "Left Fast",
        "LFM": "Left Fast Medium",
        "LM": "Left Medium",
        "LMF": "Left Medium Fast",
        "LWS": "Left Wrist Spin",
        "OB": "Off Break",
        "RF": "Right Fast",
        "RFM": "Right Fast Medium",
        "RMF": "Right Medium Fast",
        "RM": "Right Medium",
        "RM/OB/LB": "Right Medium/Off Break/Left Arm",
        "SLA": "Slow Leg Spin"
    }
    merged["Bowl Style"] = merged["bowl_style"].apply(lambda x: bowlABB.get(x, x))
    merged.drop(columns=["bowl_style"], inplace=True)
    for c in ["SR", "Dismissals", "Boundary%", "Dot%", "Ave", "ExpRuns"]:
        merged[c] = merged[c].apply(lambda x: round(x, 2) if isinstance(x, (int, float)) and math.isfinite(x) else "inf")
    merged["Balls"] = merged["Balls"].astype(int)
    merged["Runs"] = merged["Runs"].astype(int)
    merged = merged[[
        "Bowl Style", "Balls", "Runs", "SR", "Dismissals",
        "Boundary%", "Dot%", "Impact/100b"
    ]]
    return merged

def main():
    st.set_page_config(page_title="T20: SPORTS", layout="wide")
    st.title("T20 Cricket Sports\nData Science Lab project by Sai Arun | Mustafa | Mitali")

    df = ld()
    df_global = df.copy()

    batters = sorted(df["bat"].dropna().unique())
    selected_batter = st.sidebar.selectbox("Select Batter", batters)
    if "year" in df.columns:
        minY = int(df["year"].min())
        maxY = int(df["year"].max())
        year_range = st.sidebar.slider("Year range", minY, maxY, (minY, maxY))
    else:
        year_range = None
    bowler_type = st.sidebar.radio("Bowler Type", ["All", "Spin", "Pace"])
    control = st.sidebar.checkbox("Only controlled shots?", value=True)

    sub = df.copy()
    sub = sub[sub["bat"] == selected_batter]
    if year_range and "year" in sub.columns:
        sub = sub[(sub["year"] >= year_range[0]) & (sub["year"] <= year_range[1])]
    if bowler_type != "All":
        sub = sub[sub["bowler_type"] == bowler_type]
    if control and "control" in sub.columns:
        sub = sub[sub["control"] == 1]

    if sub.empty:
        st.warning("No data found.")
        return

    sub = shift_coords(sub)
    sub = sd(sub)

    hand = "RHB/LHB?"
    if "bat_hand" in sub.columns and sub["bat_hand"].dropna().size > 0:
        hand = sub["bat_hand"].dropna().iloc[0]

    total_runs = sub["batruns"].sum()
    balls_faced = len(sub)
    outs = sub["dismissal"].notna().sum()
    sr_val = 100 * total_runs / balls_faced if balls_faced > 0 else 0
    bound_ct = sub[sub["batruns"].isin([4,6])].shape[0]
    bound_pct = 100 * bound_ct / balls_faced if balls_faced > 0 else 0
    dot_ct = sub[sub["batruns"] == 0].shape[0]
    dot_pct = 100 * dot_ct / balls_faced if balls_faced > 0 else 0
    ave_val = total_runs / outs if outs > 0 else float("inf")

    st.markdown(f"""
    <div style='background-color:#1e1e1e; color:white; padding:20px; border-radius:5px; margin-bottom:20px;'>
      <h2>{selected_batter} | {hand}</h2>
      <p>
        Runs: {total_runs} | 
        Balls: {balls_faced} | 
        SR: {sr_val:.2f} | 
        Boundary%: {bound_pct:.2f}% | 
        Dot%: {dot_pct:.2f}% | 
        Ave: {"∞" if ave_val==float("inf") else f"{ave_val:.2f}"}
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Boundary Wagon Wheel: General vs. Intelligent")
    fig, (ax_left, ax_right) = plt.subplots(ncols=2, figsize=(14, 6))
    wagonGen(sub, ax_left)
    wagonSD(sub, ax_right)
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("Wagon Zone Wheel")
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    wagonZoneWheel(sub, ax2)
    fig2.tight_layout()
    st.pyplot(fig2)

    st.subheader("Specifications w.r.t Bowlstyle")
    sty_table = tab(sub, df_global)
    if not sty_table.empty:
        st.dataframe(sty_table)
        st.markdown(r"""
        **Explanations**:
        - **Shot Difficulty**: Shots in all zones / shots in each specific zone.
        - **Genral Wagon Wheel**: All boundaries.
        - **Intelligent Wagon Wheel**: Boundary w.r.t line length combination = (runs × sd).
        - **Wagon Zone Wheel**: Zone based specifications.
        - **Strike Rate**: Total runs / Balls x 100.
        - **Impact/100balls**: Imapct of player on team.
        """)
    else:
        st.write("No bowling style data found.")

if __name__=="__main__":
    main()
