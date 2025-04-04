import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Rectangle
from matplotlib.lines import Line2D
import seaborn as sns  # Added for your visualizations

@st.cache_data  # Data load once
def ld(path="IPL_2018_2024_copy.xlsx"):
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

# All/zone shot difficulty
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
        - **General Wagon Wheel**: All boundaries.
        - **Intelligent Wagon Wheel**: Boundary w.r.t line length combination = (runs × sd).
        - **Wagon Zone Wheel**: Zone based specifications.
        - **Strike Rate**: Total runs / Balls x 100.
        - **Impact/100balls**: Impact of player on team.
        """)
    else:
        st.write("No bowling style data found.")

    # Your Visualization 1: Percentage of Outs by Bowl Length
    st.subheader(f"Percentage of Outs by Bowl Length for {selected_batter}")
    length_summary = (
        sub
        .assign(out_flag=sub['out'].astype(bool))
        .groupby('length')
        .agg(
            total_balls=('out_flag', 'size'),
            total_outs=('out_flag', 'sum')
        )
        .reset_index()
    )
    total_outs = length_summary['total_outs'].sum()
    if total_outs > 0:
        length_summary['out_percentage'] = 100 * length_summary['total_outs'] / total_outs
    else:
        length_summary['out_percentage'] = 0
    length_summary = length_summary.sort_values('out_percentage', ascending=False)
    
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3 = sns.barplot(
        x='length', y='out_percentage',
        data=length_summary,
        order=length_summary['length'],
        palette='Set2'
    )
    ax3.bar_label(ax3.containers[0], fmt='%.1f%%', label_type='edge', padding=3, fontsize=10)
    plt.title(f'Percentage of Outs by Bowl Length for {selected_batter}', fontsize=14, fontweight='bold')
    plt.xlabel('Bowl Length', fontsize=12)
    plt.ylabel('Percentage of Total Outs (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, length_summary['out_percentage'].max() + 15 if total_outs > 0 else 100)
    plt.tight_layout()
    st.pyplot(fig3)

    # Your Visualization 2: Percentage of Outs by Bowl Line
    st.subheader(f"Percentage of Outs by Bowl Line for {selected_batter}")
    line_summary = sub[sub['out'] == 1]['line'].value_counts().reset_index()
    line_summary.columns = ['line', 'outs']
    total_outs = line_summary['outs'].sum()
    if total_outs > 0:
        line_summary['percentage'] = 100 * line_summary['outs'] / total_outs
    else:
        line_summary['percentage'] = 0
    line_summary = line_summary.sort_values('percentage', ascending=False)
    
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    ax4 = sns.barplot(x='line', y='percentage', data=line_summary, order=line_summary['line'], palette='Set1')
    for p in ax4.patches:
        ax4.text(p.get_x() + p.get_width() / 2, p.get_height(), f'{p.get_height():.1f}%', ha='center', va='bottom')
    plt.title(f'Percentage of Outs by Bowl Line for {selected_batter}')
    plt.xlabel('Bowl Line')
    plt.ylabel('Percentage of Outs (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig4)

    # Your Visualization 3: Percentage of Outs by Bowl Line and Length (Heatmap)
    st.subheader(f"Percentage of Outs by Bowl Line and Length for {selected_batter}")
    # Standardize the line column
    sub['line'] = sub['line'].str.upper().str.replace(' ', '_')
    pitch_data = sub[sub['out'] == 1].groupby(['line', 'length']).size().reset_index(name='out_count')
    
    # Map line and length to grid positions
    line_map = {'WIDE_DOWN_LEG': 0, 'DOWN_LEG': 1, 'ON_THE_STUMPS': 2, 'OUTSIDE_OFFSTUMP': 3, 'WIDE_OUTSIDE_OFFSTUMP': 4}
    length_map = {'YORKER': 0, 'FULL_TOSS': 1, 'FULL': 2, 'GOOD_LENGTH': 3, 'SHORT_OF_A_GOOD_LENGTH': 4, 'SHORT': 5}
    pitch_data['line_idx'] = pitch_data['line'].map(line_map).astype(float)
    pitch_data['length_idx'] = pitch_data['length'].map(length_map).astype(float)
    pitch_data = pitch_data.dropna(subset=['line_idx', 'length_idx'])
    
    total_outs = pitch_data['out_count'].sum()
    if total_outs > 0:
        pitch_data['out_percentage'] = (pitch_data['out_count'] / total_outs) * 100
    else:
        pitch_data['out_percentage'] = 0
    
    # Create a complete grid
    all_lines = list(line_map.values())
    all_lengths = list(length_map.values())
    all_combinations = pd.DataFrame(
        [(length_idx, line_idx) for length_idx in all_lengths for line_idx in all_lines],
        columns=['length_idx', 'line_idx']
    )
    pitch_data = all_combinations.merge(pitch_data, on=['length_idx', 'line_idx'], how='left').fillna({'out_percentage': 0})
    
    # Pivot for heatmap
    pivot_outs_percent = pitch_data.pivot(index='length_idx', columns='line_idx', values='out_percentage')
    
    # Plot the heatmap
    fig5, ax5 = plt.subplots(figsize=(10, 12))
    ax5.add_patch(Rectangle((-1, -1), 6, 1, fill=True, color='white', alpha=0.7))
    ax5.add_patch(Rectangle((-1, 5.5), 6, 1, fill=True, color='white', alpha=0.7))
    ax5.plot([2.5, 2.5], [-1, 6], color='black', linestyle='--')
    
    if total_outs > 0:
        sns.heatmap(pivot_outs_percent, annot=True, cmap='YlOrRd', fmt='.1f', cbar_kws={'label': 'Outs (%)'},
                    ax=ax5, linewidths=1, linecolor='gray', alpha=1.0)
    else:
        sns.heatmap(pivot_outs_percent, annot=True, cmap='YlOrRd', fmt='.1f', cbar_kws={'label': 'Outs (%)'},
                    ax=ax5, linewidths=1, linecolor='gray', alpha=1.0)
        plt.text(2.5, 3, f'No dismissals for {selected_batter}', ha='center', va='center', fontsize=12, color='black')
    
    plt.title(f'Percentage of Outs by Bowl Line and Length for {selected_batter}', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Bowl Line (Left to Right)', fontsize=12)
    plt.ylabel('Bowl Length (Top to Bottom)', fontsize=12)
    plt.xticks([0.5, 1.5, 2.5, 3.5, 4.5], ['Wide Down Leg', 'Down Leg', 'On Stumps', 'Outside Off', 'Wide Outside Off'], rotation=45, ha='right')
    plt.yticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], ['Yorker', 'Full Toss', 'Full', 'Good Length', 'Short of Good', 'Short'])
    ax5.set_xlim(-0.3, 5)
    ax5.set_ylim(-0.1, 6)
    plt.subplots_adjust(top=0.9)
    st.pyplot(fig5)

if __name__=="__main__":
    main()
