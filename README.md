# 🏏  T20-Cricket IPL Analysis (2018–2024)

An interactive Streamlit-based data science project analyzing batter performance and dismissal patterns in the Indian Premier League (IPL) T20 competition from 2018 to 2024. Built by Sai Arun, Mustafa, and Mitali as a Data Science Lab project, it leverages detailed ball-by-ball data—including line, length, control, wagon wheel coordinates, dismissal types, and more—to provide actionable insights for cricket enthusiasts, analysts, and teams.

# Features
Boundary Scoring Visualization:
General Wagon Wheel: Displays boundary shots (4s and 6s) with fixed lengths.
Intelligent Wagon Wheel: Adjusts shot length based on runs and shot difficulty.
Wagon Zone Wheel: Maps scoring zones (1-8) with strike rates, runs, and dismissal risks.
Shot Difficulty Analysis: Calculates and visualizes shot difficulty by line, length, and zone.


# Dismissal Trends:
Bar charts for outs by line and length.
Heatmap showing dismissal percentages across line-length combinations.
Bowling Style Breakdown: Compares batter performance against specific bowling styles (e.g., RFM, OB) with metrics like strike rate, boundary%, and impact per 100 balls.
Temporal Analysis: Tracks scoring patterns, strike rates, and dismissals over years.
Control vs. Non-Control Shots: Analyzes performance differences between controlled and uncontrolled shots.
Pressure Situations: Examines batter response after consecutive dot balls.
Bowler Type Matchup: Contrasts performance against spin vs. pace with zone-wise run distribution.
Strategic Insights: Each tab provides detailed takeaways for batters (e.g., areas to improve) and bowlers (e.g., zones to target).

# Dataset
Source: IPL_2018_2024.xlsx (ball-by-ball data from IPL matches, 2018–2024).
Key Variables:
bat: Batter name
batruns: Runs per ball
out/dismissal: Out status and type
wagonX/wagonY/wagonZone: Shot coordinates and zones
line/length: Delivery trajectory
control: Shot control flag
bowl_style/bowler_type: Bowling style and type (Spin/Pace)
year: Match year
bat_hand: Batter handedness

# Project Structure
Tabs:
Introduction: Overview of cricket, T20, and IPL.
Data Analysis: 7 sub-tabs with visualizations and insights.
Help: Explanation of variables.
Sidebar Filters: Select batter, year range, bowler type (All/Spin/Pace), and control shots.

# Requirements
Python 3.8+
Libraries: streamlit, pandas, numpy, matplotlib, seaborn.

# Usage
Select a batter and adjust filters in the sidebar.
Explore the "Data Analysis" tab for visualizations and insights.
Use the "Help" tab to understand the data variables.

# Contributors
Sai Arun
Mustafa
Mitali
