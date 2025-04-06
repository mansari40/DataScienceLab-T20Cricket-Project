import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Rectangle
from matplotlib.lines import Line2D
import seaborn as sns  # Added for your visualizations

@st.cache_data  # Data load once
def ld(path="IPL_2018_2024.xlsx"):
    df = pd.read_excel(path)
    cols = [
        "bat", "batruns", "out", "dismissal",
        "wagonX", "wagonY", "wagonZone", "line", "length",
        "control", "bowl_style", "year", "bat_hand"
    ]
    df = df[[c for c in cols if c in df.columns]].drop_duplicates()
    
    # Convert 'year' to integer if it exists
    if "year" in df.columns:
        df["year"] = df["year"].astype(int)
    
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
        return bdf
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
    return bdf

def wagonSD(df, ax):
    bdf = df[df["batruns"].isin([4,6])].copy()
    if bdf.empty:
        ax.set_title("Intelligent Wagon Wheel - No Boundaries")
        ax.add_patch(Circle((0,0),200,color="green",alpha=0.2))
        ax.set_xlim(-220,220)
        ax.set_ylim(-220,220)
        ax.set_aspect("equal", "box")
        ax.axis("off")
        return bdf
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
    return bdf

def wagonZoneWheel(df, ax):
    df = df[df["wagonZone"] != 0].copy()
    if df.empty:
        ax.set_title("Wagon Zone Wheel - No Valid Zones")
        return pd.DataFrame()
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
    return zone_df  # Return zone_df for insights

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

    # Create tabs for organizing visualizations
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Wagon Wheels", "Dismissal Analysis", "Bowling Style Stats",
        "Scoring Patterns Over Time", "Control vs. Non-Control Shots",
        "Pressure Situations Analysis", "Bowler Type Matchup"
    ])

    # Tab 1: Wagon Wheels
    with tab1:
        # Boundary Wagon Wheel: General vs. Intelligent with Insights
        st.subheader("Boundary Wagon Wheel: General vs. Intelligent")
        fig, (ax_left, ax_right) = plt.subplots(ncols=2, figsize=(14, 6))
        bdf = wagonGen(sub, ax_left)
        bdf = wagonSD(sub, ax_right)
        fig.tight_layout()
        st.pyplot(fig)

        # Insights for Boundary Wagon Wheel
        if not bdf.empty:
            # Count 4s and 6s
            fours = len(bdf[bdf["batruns"] == 4])
            sixes = len(bdf[bdf["batruns"] == 6])
            total_boundaries = fours + sixes
            
            # Most common zone for boundaries
            zone_counts = bdf["wagonZone"].value_counts()
            most_common_zone = zone_counts.idxmax() if not zone_counts.empty else "N/A"
            most_common_zone_count = zone_counts.max() if not zone_counts.empty else 0
            zone_percentage = (most_common_zone_count / total_boundaries * 100) if total_boundaries > 0 else 0

            # Analyze off-side vs. leg-side (zones 1-4 are off-side, 5-8 are leg-side)
            off_side_zones = [1, 2, 3, 4]
            leg_side_zones = [5, 6, 7, 8]
            off_side_boundaries = len(bdf[bdf["wagonZone"].isin(off_side_zones)])
            leg_side_boundaries = len(bdf[bdf["wagonZone"].isin(leg_side_zones)])
            off_side_pct = (off_side_boundaries / total_boundaries * 100) if total_boundaries > 0 else 0
            leg_side_pct = (leg_side_boundaries / total_boundaries * 100) if total_boundaries > 0 else 0
            side_preference = "off-side" if off_side_boundaries > leg_side_boundaries else "leg-side"

            # Shot difficulty analysis (from intelligent wagon wheel)
            bdf["magnitude_raw"] = bdf["batruns"] * bdf["shot_difficulty"]
            zone_difficulty = bdf.groupby("wagonZone")["magnitude_raw"].mean().sort_values(ascending=False)
            hardest_zone = zone_difficulty.idxmax() if not zone_difficulty.empty else "N/A"
            hardest_zone_value = zone_difficulty.max() if not zone_difficulty.empty else 0

            st.markdown(f"""
            **Insights**:
            - **Boundary Distribution**: {selected_batter} hit {fours} fours and {sixes} sixes, totaling {total_boundaries} boundaries.
            - **Favorite Zone**: The most common zone for boundaries is Zone {most_common_zone}, accounting for {zone_percentage:.1f}% of boundaries.
            - **Side Preference**: {selected_batter} favors the {side_preference}, with {off_side_pct:.1f}% of boundaries on the off-side and {leg_side_pct:.1f}% on the leg-side.
            - **Shot Difficulty**: The intelligent wagon wheel highlights Zone {hardest_zone} as the area where {selected_batter} excels in difficult shots (average weighted score: {hardest_zone_value:.1f}).
            - **Takeaway**: {selected_batter} could focus on diversifying shot selection to reduce predictability, while bowlers might target less dominant zones to restrict boundary scoring.
            """)
        else:
            st.markdown("**Insights**: No boundaries recorded for this selection.")

        # Wagon Zone Wheel with Insights
        st.subheader("Wagon Zone Wheel")
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        zone_df = wagonZoneWheel(sub, ax2)  # Modified to return zone_df
        fig2.tight_layout()
        st.pyplot(fig2)

        # Insights for Wagon Zone Wheel
        if not zone_df.empty:
            # Top scoring zone (highest runs and strike rate)
            top_zone_runs = zone_df.loc[zone_df["Runs"].idxmax()]
            top_zone_sr = zone_df.loc[zone_df["SR"].idxmax()]
            top_zone_runs_id = top_zone_runs["wagonZone"]
            top_zone_runs_value = top_zone_runs["Runs"]
            top_zone_runs_pct = top_zone_runs["pct_runs"]
            top_zone_sr_id = top_zone_sr["wagonZone"]
            top_zone_sr_value = top_zone_sr["SR"]

            # Weak zone (lowest strike rate)
            weak_zone = zone_df.loc[zone_df["SR"].idxmin()]
            weak_zone_id = weak_zone["wagonZone"]
            weak_zone_sr = weak_zone["SR"]
            weak_zone_balls = weak_zone["Balls"]

            # Calculate dismissals per zone
            dismissals_per_zone = sub[sub["dismissal"].notna()].groupby("wagonZone").size().reset_index(name="Dismissals")
            zone_df = zone_df.merge(dismissals_per_zone, on="wagonZone", how="left").fillna({"Dismissals": 0})
            high_dismissal_zone = zone_df.loc[zone_df["Dismissals"].idxmax()] if not zone_df["Dismissals"].eq(0).all() else None
            high_dismissal_zone_id = high_dismissal_zone["wagonZone"] if high_dismissal_zone is not None else "N/A"
            high_dismissal_count = high_dismissal_zone["Dismissals"] if high_dismissal_zone is not None else 0

            st.markdown(f"""
            **Insights**:
            - **Top Scoring Zone**: Zone {top_zone_runs_id} is {selected_batter}'s most productive, contributing {top_zone_runs_value} runs ({top_zone_runs_pct:.1f}% of total runs).
            - **Highest Strike Rate**: Zone {top_zone_sr_id} has the highest strike rate at {top_zone_sr_value:.1f}, indicating aggressive scoring.
            - **Weak Zone**: Zone {weak_zone_id} is the least effective, with a strike rate of {weak_zone_sr:.1f} over {weak_zone_balls} balls.
            - **Dismissal Risk**: Zone {high_dismissal_zone_id} has the highest dismissal count ({high_dismissal_count} dismissals), making it a potential vulnerability.
            - **Takeaway**: {selected_batter} might target Zone {weak_zone_id} to improve scoring, while bowlers could exploit Zone {high_dismissal_zone_id} to increase dismissal chances.
            """)
        else:
            st.markdown("**Insights**: No valid zone data available for this selection.")

    # Tab 2: Dismissal Analysis
    with tab2:
        # Visualization 1: Percentage of Outs by Bowl Length
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
        # Add percentage labels on all bars
        for i, bar in enumerate(ax3.patches):
            height = bar.get_height()
            if height < 5:  # Adjust this threshold as needed
                ax3.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 2,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=10
                )
            else:
                ax3.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 2,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=10
                )
        plt.title(f'Percentage of Outs by Bowl Length for {selected_batter}', fontsize=14, fontweight='bold')
        plt.xlabel('Bowl Length', fontsize=12)
        plt.ylabel('Percentage of Total Outs (%)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, length_summary['out_percentage'].max() + 15 if total_outs > 0 else 100)
        plt.tight_layout()
        st.pyplot(fig3)

        # Insights for Visualization 1
        if total_outs > 0:
            max_length = length_summary.iloc[0]['length']
            max_pct = length_summary.iloc[0]['out_percentage']
            max_balls = length_summary.iloc[0]['total_balls']
            min_length = length_summary.iloc[-1]['length']
            min_pct = length_summary.iloc[-1]['out_percentage']
            min_balls = length_summary.iloc[-1]['total_balls']
            st.markdown(f"""
            **Insights**:
            - **Most Dangerous Length**: {max_length} deliveries dismiss {selected_batter} most often ({max_pct:.1f}% of outs, {int(max_balls)} balls faced), making it the best length to bowl.
            - **Safest Length**: {min_length} deliveries are safest for the batter, with only {min_pct:.1f}% of dismissals ({int(min_balls)} balls faced).
            - **Takeaway**: Bowlers should prioritize {max_length} deliveries to maximize dismissal chances, while {selected_batter} might focus on improving technique against this length.
            """)
        else:
            st.markdown(f"""
            **Insights**: No dismissals recorded for {selected_batter} under the current filters.  
            Try adjusting the filters to see dismissal patterns.
            """)

        # Visualization 2: Percentage of Outs by Bowl Line
        st.subheader(f"Percentage of Outs by Bowl Line for {selected_batter}")
        # Standardize the 'line' column
        if 'line' in sub.columns:
            sub['line'] = sub['line'].str.upper().str.replace(' ', '_')
            line_summary = sub[sub['out'] == 1]['line'].value_counts().reset_index()
            line_summary.columns = ['line', 'outs']
            total_outs = line_summary['outs'].sum()
            total_balls_per_line = sub.groupby('line').size().reset_index(name='total_balls')
            if total_outs > 0:
                line_summary['percentage'] = 100 * line_summary['outs'] / total_outs
            else:
                line_summary['percentage'] = 0
            line_summary = line_summary.sort_values('percentage', ascending=False)
            line_summary = line_summary.merge(total_balls_per_line, on='line', how='left').fillna({'total_balls': 0})
            
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

            # Insights for Visualization 2
            if total_outs > 0:
                max_line = line_summary.iloc[0]['line']
                max_pct = line_summary.iloc[0]['percentage']
                max_balls = int(line_summary.iloc[0]['total_balls'])
                min_line = line_summary.iloc[-1]['line']
                min_pct = line_summary.iloc[-1]['percentage']
                min_balls = int(line_summary.iloc[-1]['total_balls'])
                st.markdown(f"""
                **Insights**:
                - **Most Dangerous Line**: {max_line} deliveries are most likely to dismiss {selected_batter} ({max_pct:.1f}% of outs, {max_balls} balls faced), making it the best line to target.
                - **Safest Line**: {min_line} deliveries are least effective, with only {min_pct:.1f}% of dismissals ({min_balls} balls faced).
                - **Takeaway**: Bowlers should focus on the {max_line} line to increase dismissal chances, while {selected_batter} might work on defending or attacking this line more effectively.
                """)
            else:
                st.markdown(f"""
                **Insights**: No dismissals recorded for {selected_batter} under the current filters.  
                Try adjusting the filters to see dismissal patterns.
                """)
        else:
            st.markdown("**Error**: 'line' column not found in the dataset. This visualization cannot be generated.")

        # Visualization 3: Percentage of Outs by Bowl Line and Length (Heatmap)
        st.subheader(f"Percentage of Outs by Bowl Line and Length for {selected_batter}")
        # Check if required columns exist
        if 'line' in sub.columns and 'length' in sub.columns:
            # Standardize the line column
            sub['line'] = sub['line'].str.upper().str.replace(' ', '_')
            
            # Define mappings
            line_map = {'WIDE_DOWN_LEG': 0, 'DOWN_LEG': 1, 'ON_THE_STUMPS': 2, 'OUTSIDE_OFFSTUMP': 3, 'WIDE_OUTSIDE_OFFSTUMP': 4}
            length_map = {'YORKER': 0, 'FULL_TOSS': 1, 'FULL': 2, 'GOOD_LENGTH': 3, 'SHORT_OF_A_GOOD_LENGTH': 4, 'SHORT': 5}

            # Filter out rows with unmapped values
            sub_filtered = sub[sub['line'].isin(line_map.keys()) & sub['length'].isin(length_map.keys())]
            if sub_filtered.empty:
                st.markdown("**Warning**: No data available after filtering for valid 'line' and 'length' values.")
            else:
                pitch_data = sub_filtered[sub_filtered['out'] == 1].groupby(['line', 'length']).size().reset_index(name='out_count')
                total_balls_per_combination = sub_filtered.groupby(['line', 'length']).size().reset_index(name='total_balls')
                
                # Map line and length to grid positions
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
                pitch_data = all_combinations.merge(pitch_data, on=['length_idx', 'line_idx'], how='left').fillna({'out_percentage': 0, 'out_count': 0})
                
                # Drop existing 'line' and 'length' columns to avoid duplicates
                if 'line' in pitch_data.columns:
                    pitch_data = pitch_data.drop(columns=['line'])
                if 'length' in pitch_data.columns:
                    pitch_data = pitch_data.drop(columns=['length'])
                
                # Add back the line and length columns
                mapping_df = pd.DataFrame(
                    [(line_map[line], length_map[length], line, length) 
                     for line in line_map for length in length_map],
                    columns=['line_idx', 'length_idx', 'line', 'length']
                )
                pitch_data = pitch_data.merge(mapping_df, on=['length_idx', 'line_idx'], how='left')
                
                # Merge with total_balls_per_combination
                pitch_data = pitch_data.merge(total_balls_per_combination, on=['line', 'length'], how='left').fillna({'total_balls': 0})
                
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

                # Insights for Visualization 3
                if total_outs > 0:
                    # Find the line-length combination with the highest percentage
                    max_idx = pitch_data['out_percentage'].idxmax()
                    max_line = pitch_data.loc[max_idx, 'line']
                    max_length = pitch_data.loc[max_idx, 'length']
                    max_pct = pitch_data.loc[max_idx, 'out_percentage']
                    max_balls = int(pitch_data.loc[max_idx, 'total_balls'])
                    # Find the combination with the lowest percentage (non-zero)
                    non_zero = pitch_data[pitch_data['out_percentage'] > 0]
                    if not non_zero.empty:
                        min_idx = non_zero['out_percentage'].idxmin()
                        min_line = non_zero.loc[min_idx, 'line']
                        min_length = non_zero.loc[min_idx, 'length']
                        min_pct = non_zero.loc[min_idx, 'out_percentage']
                        min_balls = int(non_zero.loc[min_idx, 'total_balls'])
                        min_text = f"- **Safest Combination**: The {min_line} {min_length} combination is least effective, with only {min_pct:.1f}% of dismissals ({min_balls} balls faced)."
                    else:
                        min_text = "- **Safest Combination**: No other combinations resulted in dismissals."
                    st.markdown(f"""
                    **Insights**:
                    - **Most Dangerous Combination**: The {max_line} {max_length} combination is most effective, accounting for {max_pct:.1f}% of {selected_batter}'s dismissals ({max_balls} balls faced).
                    {min_text}
                    - **Takeaway**: Bowlers should target the {max_line} {max_length} combination to maximize dismissal chances, while {selected_batter} should be cautious against this combination.
                    """)
                else:
                    st.markdown(f"""
                    **Insights**: No dismissals recorded for {selected_batter} under the current filters.  
                    Try adjusting the filters to see dismissal patterns.
                    """)
        else:
            st.markdown("**Error**: 'line' or 'length' column not found in the dataset. This visualization cannot be generated.")

    # Tab 3: Bowling Style Stats
    with tab3:
        st.subheader("Specifications w.r.t Bowlstyle")
        sty_table = tab(sub, df_global)
        if not sty_table.empty:
            st.dataframe(sty_table)
            # Insights for Bowling Style Stats
            if not sty_table.empty:
                # Highest Strike Rate (most comfortable bowling style)
                max_sr_row = sty_table.loc[sty_table['SR'].idxmax()]
                max_sr_style = max_sr_row['Bowl Style']
                max_sr_value = max_sr_row['SR']
                max_sr_balls = max_sr_row['Balls']

                # Lowest Strike Rate (least comfortable, non-zero runs)
                non_zero_runs = sty_table[sty_table['Runs'] > 0]
                if not non_zero_runs.empty:
                    min_sr_row = non_zero_runs.loc[non_zero_runs['SR'].idxmin()]
                    min_sr_style = min_sr_row['Bowl Style']
                    min_sr_value = min_sr_row['SR']
                    min_sr_balls = min_sr_row['Balls']
                    min_sr_text = f"- **Least Comfortable Style**: {selected_batter} struggles against {min_sr_style}, with a strike rate of {min_sr_value:.1f} over {min_sr_balls} balls."
                else:
                    min_sr_text = "- **Least Comfortable Style**: No bowling styles resulted in runs scored."

                # Highest Dismissal Count (most dangerous bowling style)
                max_dismiss_row = sty_table.loc[sty_table['Dismissals'].idxmax()]
                max_dismiss_style = max_dismiss_row['Bowl Style']
                max_dismiss_count = max_dismiss_row['Dismissals']
                max_dismiss_balls = max_dismiss_row['Balls']

                # Highest Boundary Percentage (most aggressive against)
                max_boundary_row = sty_table.loc[sty_table['Boundary%'].idxmax()]
                max_boundary_style = max_boundary_row['Bowl Style']
                max_boundary_pct = max_boundary_row['Boundary%']
                max_boundary_balls = max_boundary_row['Balls']

                # Highest Dot Percentage (most restrictive bowling style)
                max_dot_row = sty_table.loc[sty_table['Dot%'].idxmax()]
                max_dot_style = max_dot_row['Bowl Style']
                max_dot_pct = max_dot_row['Dot%']
                max_dot_balls = max_dot_row['Balls']

                # Highest Impact per 100 Balls (most effective overall)
                max_impact_row = sty_table.loc[sty_table['Impact/100b'].idxmax()]
                max_impact_style = max_impact_row['Bowl Style']
                max_impact_value = max_impact_row['Impact/100b']
                max_impact_balls = max_impact_row['Balls']

                st.markdown(f"""
                **Insights**:
                - **Most Comfortable Style**: {selected_batter} scores freely against {max_sr_style}, with a strike rate of {max_sr_value:.1f} over {max_sr_balls} balls.
                {min_sr_text}
                - **Most Dangerous Style**: {max_dismiss_style} has dismissed {selected_batter} {max_dismiss_count} times over {max_dismiss_balls} balls, making it the most effective.
                - **Most Aggressive Against**: {selected_batter} hits boundaries most frequently against {max_boundary_style}, with a boundary percentage of {max_boundary_pct:.1f}% over {max_boundary_balls} balls.
                - **Most Restrictive Style**: {max_dot_style} restricts {selected_batter} the most, with a dot ball percentage of {max_dot_pct:.1f}% over {max_dot_balls} balls.
                - **Highest Impact Style**: {max_impact_style} has the highest impact against {selected_batter}, with an impact of {max_impact_value:.1f} per 100 balls over {max_impact_balls} balls.
                - **Takeaway**: {selected_batter} should work on improving scoring against {max_dot_style} to reduce dot balls and be cautious against {max_dismiss_style} to avoid dismissals. Bowlers should use {max_dismiss_style} to target {selected_batter}'s weaknesses.
                """)
        else:
            st.write("No bowling style data found.")

    # Tab 4: Scoring Patterns Over Time
    with tab4:
        st.subheader(f"Scoring Patterns Over Time for {selected_batter}")
        if 'year' in sub.columns:
            # Calculate yearly metrics
            yearly_stats = sub.groupby('year').agg(
                total_runs=('batruns', 'sum'),
                total_balls=('batruns', 'count'),
                dismissals=('dismissal', lambda x: x.notna().sum()),
                boundaries=('batruns', lambda x: x.isin([4, 6]).sum()),
                dots=('batruns', lambda x: (x == 0).sum())
            ).reset_index()
            yearly_stats['strike_rate'] = 100 * yearly_stats['total_runs'] / yearly_stats['total_balls']
            yearly_stats['boundary_pct'] = 100 * yearly_stats['boundaries'] / yearly_stats['total_balls']
            yearly_stats['dot_pct'] = 100 * yearly_stats['dots'] / yearly_stats['total_balls']

            # Ensure years are sorted and unique for x-axis
            years = sorted(yearly_stats['year'].unique())

            # Line Chart: Strike Rate and Runs Over Time
            fig6, ax6 = plt.subplots(figsize=(10, 6))
            ax6.plot(yearly_stats['year'], yearly_stats['strike_rate'], marker='o', label='Strike Rate', color='blue')
            ax6.set_xlabel('Year')
            ax6.set_ylabel('Strike Rate', color='blue')
            ax6.tick_params(axis='y', labelcolor='blue')
            ax6_2 = ax6.twinx()
            ax6_2.plot(yearly_stats['year'], yearly_stats['total_runs'], marker='o', label='Total Runs', color='orange')
            ax6_2.set_ylabel('Total Runs', color='orange')
            ax6_2.tick_params(axis='y', labelcolor='orange')
            # Set x-axis ticks to whole years
            ax6.set_xticks(years)
            ax6.set_xticklabels(years, rotation=45)
            fig6.legend(loc='upper left')
            plt.title(f'Strike Rate and Runs Over Time for {selected_batter}')
            plt.tight_layout()
            st.pyplot(fig6)

            # Bar Chart: Dismissals Per Year
            fig7, ax7 = plt.subplots(figsize=(8, 5))
            sns.barplot(x='year', y='dismissals', data=yearly_stats, palette='Set2')
            # Set x-axis ticks to whole years
            ax7.set_xticks(range(len(years)))
            ax7.set_xticklabels(years, rotation=45)
            plt.title(f'Dismissals Per Year for {selected_batter}')
            plt.xlabel('Year')
            plt.ylabel('Number of Dismissals')
            plt.tight_layout()
            st.pyplot(fig7)

            # Insights
            if not yearly_stats.empty:
                peak_sr_year = yearly_stats.loc[yearly_stats['strike_rate'].idxmax(), 'year']
                peak_sr_value = yearly_stats['strike_rate'].max()
                peak_runs_year = yearly_stats.loc[yearly_stats['total_runs'].idxmax(), 'year']
                peak_runs_value = yearly_stats['total_runs'].max()
                max_dismissals_year = yearly_stats.loc[yearly_stats['dismissals'].idxmax(), 'year']
                max_dismissals_count = yearly_stats['dismissals'].max()
                st.markdown(f"""
                **Insights**:
                - **Peak Strike Rate**: {selected_batter} achieved their highest strike rate of {peak_sr_value:.1f} in {peak_sr_year}.
                - **Peak Scoring Year**: The most runs ({peak_runs_value}) were scored in {peak_runs_year}.
                - **Highest Dismissals**: {selected_batter} was dismissed {max_dismissals_count} times in {max_dismissals_year}, indicating a challenging year.
                - **Takeaway**: {selected_batter} should aim to replicate their {peak_sr_year} form, while bowlers can exploit recent trends (e.g., higher dismissals in {max_dismissals_year}) to target weaknesses.
                """)
            else:
                st.markdown("**Insights**: No yearly data available for this selection.")
        else:
            st.markdown("**Error**: 'year' column not found in the dataset. This tab cannot be generated.")

    # Tab 5: Control vs. Non-Control Shots
    with tab5:
        st.subheader(f"Control vs. Non-Control Shots for {selected_batter}")
        if 'control' in sub.columns:
            # Split data into controlled and non-controlled shots
            controlled = sub[sub['control'] == 1]
            non_controlled = sub[sub['control'] == 0]

            # Calculate metrics for each group
            control_stats = pd.DataFrame({
                'Type': ['Controlled', 'Non-Controlled'],
                'Balls': [len(controlled), len(non_controlled)],
                'Runs': [controlled['batruns'].sum(), non_controlled['batruns'].sum()],
                'Dismissals': [controlled['dismissal'].notna().sum(), non_controlled['dismissal'].notna().sum()],
                'Boundaries': [controlled['batruns'].isin([4, 6]).sum(), non_controlled['batruns'].isin([4, 6]).sum()],
                'Dots': [(controlled['batruns'] == 0).sum(), (non_controlled['batruns'] == 0).sum()]
            })
            control_stats['Strike Rate'] = 100 * control_stats['Runs'] / control_stats['Balls']
            control_stats['Boundary%'] = 100 * control_stats['Boundaries'] / control_stats['Balls']
            control_stats['Dismissal Rate'] = 100 * control_stats['Dismissals'] / control_stats['Balls']

            # Pie Chart: Proportion of Balls Faced
            fig8, ax8 = plt.subplots(figsize=(6, 6))
            ax8.pie(control_stats['Balls'], labels=control_stats['Type'], autopct='%1.1f%%', colors=['#66c2a5', '#fc8d62'])
            plt.title(f'Proportion of Balls Faced: Controlled vs. Non-Controlled for {selected_batter}')
            plt.tight_layout()
            st.pyplot(fig8)

            # Bar Chart: Compare Metrics
            fig9, ax9 = plt.subplots(figsize=(10, 6))
            metrics = ['Strike Rate', 'Boundary%', 'Dismissal Rate']
            x = np.arange(len(metrics))
            width = 0.35
            ax9.bar(x - width/2, control_stats[metrics].iloc[0], width, label='Controlled', color='#66c2a5')
            ax9.bar(x + width/2, control_stats[metrics].iloc[1], width, label='Non-Controlled', color='#fc8d62')
            ax9.set_xticks(x)
            ax9.set_xticklabels(metrics)
            ax9.set_ylabel('Percentage')
            ax9.set_title(f'Performance Metrics: Controlled vs. Non-Controlled Shots for {selected_batter}')
            ax9.legend()
            plt.tight_layout()
            st.pyplot(fig9)

            # Insights
            if control_stats['Balls'].sum() > 0:
                controlled_runs_pct = 100 * control_stats['Runs'].iloc[0] / control_stats['Runs'].sum()
                non_controlled_dismissal_zone = non_controlled[non_controlled['dismissal'].notna()]['wagonZone'].mode().iloc[0] if not non_controlled[non_controlled['dismissal'].notna()].empty else "N/A"
                st.markdown(f"""
                **Insights**:
                - **Run Contribution**: {controlled_runs_pct:.1f}% of {selected_batter}'s runs come from controlled shots, with a strike rate of {control_stats['Strike Rate'].iloc[0]:.1f}, compared to {control_stats['Strike Rate'].iloc[1]:.1f} on non-controlled shots.
                - **Dismissal Risk**: Non-controlled shots have a dismissal rate of {control_stats['Dismissal Rate'].iloc[1]:.1f}%, compared to {control_stats['Dismissal Rate'].iloc[0]:.1f}% for controlled shots.
                - **Non-Controlled Dismissal Zone**: Most dismissals on non-controlled shots occur in Zone {non_controlled_dismissal_zone}.
                - **Takeaway**: {selected_batter} should focus on improving shot control to reduce dismissals, while bowlers can target Zone {non_controlled_dismissal_zone} to induce edges.
                """)
            else:
                st.markdown("**Insights**: No data available for control analysis.")
        else:
            st.markdown("**Error**: 'control' column not found in the dataset. This tab cannot be generated.")

    # Tab 6: Pressure Situations Analysis
    with tab6:
        st.subheader(f"Pressure Situations Analysis for {selected_batter}")
        # Identify sequences of consecutive dot balls
        sub_sorted = sub.sort_index()  # Ensure the data is sorted by index (ball order)
        sub_sorted['dot'] = (sub_sorted['batruns'] == 0).astype(int)
        sub_sorted['dot_sequence'] = (sub_sorted['dot'].diff() != 0).cumsum()
        dot_sequences = sub_sorted[sub_sorted['dot'] == 1].groupby('dot_sequence').size().reset_index(name='sequence_length')
        dot_sequences = dot_sequences[dot_sequences['sequence_length'] >= 2]

        # Calculate dismissal rate and strike rate after dot sequences
        pressure_balls = []
        for seq_id in dot_sequences['dot_sequence']:
            seq_end_idx = sub_sorted[sub_sorted['dot_sequence'] == seq_id].index[-1]
            next_ball_idx = seq_end_idx + 1
            if next_ball_idx in sub_sorted.index:
                pressure_balls.append(next_ball_idx)
        pressure_df = sub_sorted.loc[pressure_balls] if pressure_balls else pd.DataFrame()

        # Metrics after pressure situations
        pressure_dismissals = pressure_df['dismissal'].notna().sum() if not pressure_df.empty else 0
        pressure_dismissal_rate = 100 * pressure_dismissals / len(pressure_balls) if pressure_balls else 0
        pressure_sr = 100 * pressure_df['batruns'].sum() / len(pressure_df) if not pressure_df.empty else 0

        # Bar Chart: Frequency of Dot Ball Sequences
        fig10, ax10 = plt.subplots(figsize=(8, 5))
        sequence_counts = dot_sequences['sequence_length'].value_counts().sort_index()
        sns.barplot(x=sequence_counts.index, y=sequence_counts.values, palette='Set2')
        plt.title(f'Frequency of Consecutive Dot Ball Sequences for {selected_batter}')
        plt.xlabel('Number of Consecutive Dot Balls')
        plt.ylabel('Frequency')
        plt.tight_layout()
        st.pyplot(fig10)

        # Insights
        if not dot_sequences.empty:
            max_sequence_length = dot_sequences['sequence_length'].max()
            max_sequence_count = dot_sequences['sequence_length'].value_counts().max()
            st.markdown(f"""
            **Insights**:
            - **Pressure Frequency**: {selected_batter} faced {len(dot_sequences)} sequences of 2+ consecutive dot balls, with the longest being {max_sequence_length} dot balls.
            - **Dismissal After Pressure**: The dismissal rate after a dot ball sequence is {pressure_dismissal_rate:.1f}%.
            - **Strike Rate After Pressure**: {selected_batter}'s strike rate after a dot ball sequence is {pressure_sr:.1f}, compared to their overall strike rate of {sr_val:.1f}.
            - **Takeaway**: {selected_batter} should work on rotating the strike to avoid pressure, while bowlers can build pressure with dot balls to increase dismissal chances.
            """)
        else:
            st.markdown("**Insights**: No sequences of 2+ consecutive dot balls found for this selection.")

    # Tab 7: Bowler Type Matchup (Spin vs. Pace)
    with tab7:
        st.subheader(f"Bowler Type Matchup: Spin vs. Pace for {selected_batter}")
        # Split data by bowler type
        spin_data = sub[sub['bowler_type'] == 'Spin']
        pace_data = sub[sub['bowler_type'] == 'Pace']

        # Calculate metrics for each group
        matchup_stats = pd.DataFrame({
            'Bowler Type': ['Spin', 'Pace'],
            'Balls': [len(spin_data), len(pace_data)],
            'Runs': [spin_data['batruns'].sum(), pace_data['batruns'].sum()],
            'Dismissals': [spin_data['dismissal'].notna().sum(), pace_data['dismissal'].notna().sum()],
            'Boundaries': [spin_data['batruns'].isin([4, 6]).sum(), pace_data['batruns'].isin([4, 6]).sum()]
        })
        matchup_stats['Strike Rate'] = 100 * matchup_stats['Runs'] / matchup_stats['Balls']
        matchup_stats['Boundary%'] = 100 * matchup_stats['Boundaries'] / matchup_stats['Balls']

        # Bar Chart: Compare Metrics
        fig11, ax11 = plt.subplots(figsize=(8, 5))
        metrics = ['Strike Rate', 'Boundary%']
        x = np.arange(len(metrics))
        width = 0.35
        ax11.bar(x - width/2, matchup_stats[metrics].iloc[0], width, label='Spin', color='#8da0cb')
        ax11.bar(x + width/2, matchup_stats[metrics].iloc[1], width, label='Pace', color='#fc8d62')
        ax11.set_xticks(x)
        ax11.set_xticklabels(metrics)
        ax11.set_ylabel('Percentage')
        ax11.set_title(f'Strike Rate and Boundary% Against Spin vs. Pace for {selected_batter}')
        ax11.legend()
        plt.tight_layout()
        st.pyplot(fig11)

        # Heatmap: Runs by Zone for Spin vs. Pace
        if 'wagonZone' in sub.columns:
            spin_zone_runs = spin_data.groupby('wagonZone')['batruns'].sum().reindex(range(1, 9), fill_value=0)
            pace_zone_runs = pace_data.groupby('wagonZone')['batruns'].sum().reindex(range(1, 9), fill_value=0)
            total_spin_runs = spin_zone_runs.sum()
            total_pace_runs = pace_zone_runs.sum()
            spin_zone_pct = 100 * spin_zone_runs / total_spin_runs if total_spin_runs > 0 else pd.Series(0, index=spin_zone_runs.index)
            pace_zone_pct = 100 * pace_zone_runs / total_pace_runs if total_pace_runs > 0 else pd.Series(0, index=pace_zone_runs.index)

            fig12, (ax12_1, ax12_2) = plt.subplots(ncols=2, figsize=(12, 5))
            sns.heatmap(spin_zone_pct.values.reshape(1, -1), annot=True, cmap='YlGnBu', fmt='.1f', ax=ax12_1,
                        xticklabels=range(1, 9), yticklabels=['Spin'], cbar_kws={'label': 'Run %'})
            ax12_1.set_title('Run Distribution by Zone Against Spin')
            sns.heatmap(pace_zone_pct.values.reshape(1, -1), annot=True, cmap='YlGnBu', fmt='.1f', ax=ax12_2,
                        xticklabels=range(1, 9), yticklabels=['Pace'], cbar_kws={'label': 'Run %'})
            ax12_2.set_title('Run Distribution by Zone Against Pace')
            plt.tight_layout()
            st.pyplot(fig12)

            # Insights
            if matchup_stats['Balls'].sum() > 0:
                preferred_type = 'Pace' if matchup_stats['Strike Rate'].iloc[1] > matchup_stats['Strike Rate'].iloc[0] else 'Spin'
                spin_fav_zone = spin_zone_pct.idxmax()
                pace_fav_zone = pace_zone_pct.idxmax()
                spin_fav_zone_pct = spin_zone_pct.max()
                pace_fav_zone_pct = pace_zone_pct.max()
                st.markdown(f"""
                **Insights**:
                - **Preference**: {selected_batter} has a strike rate of {matchup_stats['Strike Rate'].iloc[1]:.1f} against Pace and {matchup_stats['Strike Rate'].iloc[0]:.1f} against Spin, indicating a preference for {preferred_type}.
                - **Scoring Against Spin**: Zone {spin_fav_zone} is the most productive against spin, contributing {spin_fav_zone_pct:.1f}% of runs.
                - **Scoring Against Pace**: Zone {pace_fav_zone} dominates against pace, with {pace_fav_zone_pct:.1f}% of runs.
                - **Takeaway**: {selected_batter} should work on diversifying scoring against {'Spin' if preferred_type == 'Pace' else 'Pace'}, while bowlers can use {preferred_type.lower()} to target weaker zones.
                """)
            else:
                st.markdown("**Insights**: No data available for bowler type matchup.")
        else:
            st.markdown("**Error**: 'wagonZone' column not found in the dataset. Zone-based analysis cannot be generated.")

if __name__=="__main__":
    main()
