import base64
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Student Performance Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "Student_Performance_rf_model.pkl"
DATASET_PATH = APP_DIR / "student_exam_data_new.csv"
IMAGE_PATH = APP_DIR / "image.png"
HISTORY_COLUMNS = [
    "Timestamp",
    "Study Hours",
    "Previous Exam Score",
    "Prediction",
    "Pass Probability",
    "Confidence",
    "Risk Level",
    "Performance Band",
]


def encode_image(image_path: Path) -> str:
    if not image_path.exists():
        return ""
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def inject_global_styles() -> None:
    background_data = encode_image(IMAGE_PATH)
    background_css = ""
    if background_data:
        background_css = f"""
        .stApp::before {{
            content: "";
            position: fixed;
            inset: 0;
            background:
                linear-gradient(rgba(3, 7, 18, 0.82), rgba(8, 15, 32, 0.9)),
                url("data:image/png;base64,{background_data}") center/cover no-repeat;
            z-index: -2;
            opacity: 0.45;
        }}
        """

    st.markdown(
        f"""
        <style>
        :root {{
            --bg-primary: #020817;
            --bg-secondary: rgba(15, 23, 42, 0.84);
            --bg-card: rgba(255, 255, 255, 0.08);
            --bg-strong: rgba(15, 23, 42, 0.92);
            --text-main: #f8fafc;
            --text-soft: #cbd5e1;
            --text-muted: #94a3b8;
            --accent: #38bdf8;
            --accent-2: #22c55e;
            --accent-3: #f59e0b;
            --danger: #fb7185;
            --border: rgba(255, 255, 255, 0.15);
            --shadow: 0 24px 48px rgba(0, 0, 0, 0.3);
        }}

        .stApp {{
            background:
                radial-gradient(circle at top left, rgba(56, 189, 248, 0.16), transparent 28%),
                radial-gradient(circle at top right, rgba(34, 197, 94, 0.12), transparent 24%),
                linear-gradient(135deg, #020617 0%, #081121 58%, #111827 100%);
            color: var(--text-main);
        }}

        {background_css}

        [data-testid="stHeader"] {{
            background: transparent;
        }}

        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, rgba(2, 6, 23, 0.96), rgba(15, 23, 42, 0.97));
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }}

        section[data-testid="stSidebar"] * {{
            color: #e2e8f0 !important;
        }}

        .hero-card,
        .glass-card,
        .metric-card,
        .feature-card,
        .prediction-card,
        .edu-card,
        .footer-card,
        .highlight-card {{
            background: var(--bg-card);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--border);
            border-radius: 24px;
            box-shadow: var(--shadow);
            padding: 1.4rem;
            animation: fadeUp 0.8s ease both;
            transition: transform 0.28s ease, box-shadow 0.28s ease, border-color 0.28s ease;
        }}

        .hero-card:hover,
        .glass-card:hover,
        .metric-card:hover,
        .feature-card:hover,
        .prediction-card:hover,
        .edu-card:hover,
        .highlight-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 28px 54px rgba(0, 0, 0, 0.38);
            border-color: rgba(56, 189, 248, 0.34);
        }}

        .hero-title {{
            font-size: 3rem;
            font-weight: 900;
            line-height: 1.06;
            margin-bottom: 0.45rem;
            background: linear-gradient(90deg, #ffffff, #93c5fd, #86efac);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .hero-subtitle {{
            font-size: 1.02rem;
            color: var(--text-soft);
            line-height: 1.8;
        }}

        .tiny-label {{
            font-size: 0.8rem;
            color: #a5b4fc;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin-bottom: 0.35rem;
            font-weight: 700;
        }}

        .section-title {{
            font-size: 1.35rem;
            font-weight: 800;
            color: var(--text-main);
            margin-bottom: 0.65rem;
        }}

        .section-text {{
            color: var(--text-soft);
            line-height: 1.72;
            font-size: 0.97rem;
        }}

        .metric-value {{
            font-size: 2.2rem;
            font-weight: 900;
            color: #ffffff;
        }}

        .metric-caption {{
            color: #bfdbfe;
            font-size: 0.92rem;
            margin-top: 0.3rem;
        }}

        .status-pass {{
            color: #86efac;
            font-weight: 900;
        }}

        .status-fail {{
            color: #fda4af;
            font-weight: 900;
        }}

        .badge {{
            display: inline-block;
            padding: 0.36rem 0.82rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 800;
            margin-right: 0.42rem;
            margin-bottom: 0.42rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}

        .badge-blue {{ background: rgba(56, 189, 248, 0.16); color: #bae6fd; }}
        .badge-green {{ background: rgba(34, 197, 94, 0.16); color: #bbf7d0; }}
        .badge-amber {{ background: rgba(245, 158, 11, 0.16); color: #fde68a; }}
        .badge-pink {{ background: rgba(244, 114, 182, 0.16); color: #fbcfe8; }}

        .highlight-card {{
            min-height: 132px;
            position: relative;
            overflow: hidden;
        }}

        .highlight-card::before {{
            content: "";
            position: absolute;
            inset: 0;
            pointer-events: none;
        }}

        .study-glow::before {{
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.16), rgba(59, 130, 246, 0.05));
        }}

        .score-glow::before {{
            background: linear-gradient(135deg, rgba(34, 197, 94, 0.16), rgba(16, 185, 129, 0.05));
        }}

        .predict-glow::before {{
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.16), rgba(244, 114, 182, 0.05));
        }}

        .highlight-title,
        .highlight-text {{
            position: relative;
            z-index: 1;
        }}

        .highlight-title {{
            font-size: 1.05rem;
            font-weight: 800;
            color: #f8fafc;
            margin-bottom: 0.35rem;
        }}

        .highlight-text {{
            color: #dbeafe;
            line-height: 1.62;
            font-size: 0.92rem;
        }}

        div[data-testid="stMetric"] {{
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 18px;
            padding: 1rem;
            box-shadow: var(--shadow);
        }}

        .stButton button,
        .stDownloadButton button {{
            width: 100%;
            border-radius: 16px;
            border: 1px solid rgba(56, 189, 248, 0.28);
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.95), rgba(34, 197, 94, 0.95));
            color: white;
            font-weight: 800;
            transition: all 0.28s ease;
            box-shadow: 0 14px 26px rgba(34, 197, 94, 0.18);
        }}

        .stButton button:hover,
        .stDownloadButton button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 18px 32px rgba(56, 189, 248, 0.28);
        }}

        .predict-center-wrap {{
            max-width: 420px;
            margin: 0.4rem auto 0 auto;
        }}

        .stSelectbox label,
        .stNumberInput label,
        .stSlider label {{
            color: #e2e8f0 !important;
            font-weight: 700 !important;
        }}

        .created-by-big {{
            font-size: 1.28rem;
            font-weight: 900;
            background: linear-gradient(90deg, #ffffff, #93c5fd, #86efac);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .footer-note {{
            text-align: center;
            color: #cbd5e1;
            font-size: 1rem;
        }}

        @keyframes fadeUp {{
            from {{
                opacity: 0;
                transform: translateY(16px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        @media (max-width: 768px) {{
            .hero-title {{
                font-size: 2.15rem;
            }}
            .hero-subtitle {{
                font-size: 0.95rem;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_model():
    model = None
    error = None
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as exc:
        error = f"Model load failed: {exc}"
    return model, error


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    if DATASET_PATH.exists():
        return pd.read_csv(DATASET_PATH)
    return pd.DataFrame(columns=["Study Hours", "Previous Exam Score", "Pass/Fail"])


def init_session_state() -> None:
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = pd.DataFrame(columns=HISTORY_COLUMNS)
    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None
    if "session_started" not in st.session_state:
        st.session_state.session_started = datetime.now()


def get_probability(model, features: pd.DataFrame) -> float:
    if model is None:
        return 0.0
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        if len(probabilities) > 1:
            return float(probabilities[1])
        return float(probabilities[0])
    prediction = model.predict(features)[0]
    return float(prediction)


def get_risk_level(pass_probability: float) -> str:
    if pass_probability >= 0.8:
        return "Low Academic Risk"
    if pass_probability >= 0.6:
        return "Moderate Academic Risk"
    return "High Academic Risk"


def get_performance_band(study_hours: float, previous_score: float, pass_probability: float) -> str:
    if pass_probability >= 0.85 and previous_score >= 75 and study_hours >= 5:
        return "High Performer"
    if pass_probability >= 0.65:
        return "Steady Performer"
    return "Needs Support"


def build_recommendations(study_hours: float, previous_score: float, pass_probability: float) -> list[str]:
    recommendations = []

    if study_hours < 2:
        recommendations.append("Increase daily study time to at least 2 to 3 focused hours for stronger consistency.")
    elif study_hours < 5:
        recommendations.append("Your study rhythm is fair. Structured revision blocks can improve retention and test readiness.")
    else:
        recommendations.append("Your study commitment is a strength. Maintain that routine and add mock-test practice.")

    if previous_score < 50:
        recommendations.append("Previous performance is a warning signal. Focus first on core concepts and weak topics.")
    elif previous_score < 70:
        recommendations.append("You have a workable base. Target question-solving speed and revision discipline.")
    else:
        recommendations.append("Your previous score is a positive sign. Aim to preserve accuracy under exam pressure.")

    if pass_probability < 0.5:
        recommendations.append("Immediate academic support is recommended, including mentor review and a short recovery study plan.")
    elif pass_probability < 0.75:
        recommendations.append("You are in a sensitive range. Weekly progress tracking can materially improve your outcome.")
    else:
        recommendations.append("Current signals are favorable. Keep up consistency and avoid last-minute preparation gaps.")

    return recommendations


def predict_student(model, study_hours: float, previous_score: float) -> dict:
    if model is None:
        raise ValueError("Student_Performance_rf_model.pkl is missing or could not be loaded.")

    features = pd.DataFrame(
        [{"Study Hours": float(study_hours), "Previous Exam Score": float(previous_score)}]
    )

    raw_prediction = model.predict(features)[0]
    pass_probability = get_probability(model, features)
    passed = bool(raw_prediction == 1 or pass_probability >= 0.5)
    fail_probability = 1 - pass_probability
    confidence = max(pass_probability, fail_probability)
    risk_level = get_risk_level(pass_probability)
    performance_band = get_performance_band(study_hours, previous_score, pass_probability)
    recommendations = build_recommendations(study_hours, previous_score, pass_probability)

    return {
        "prediction_text": "PASS" if passed else "FAIL",
        "passed": passed,
        "pass_probability": round(pass_probability * 100, 2),
        "fail_probability": round(fail_probability * 100, 2),
        "confidence": round(confidence * 100, 2),
        "risk_level": risk_level,
        "performance_band": performance_band,
        "recommendations": recommendations,
    }


def add_history(study_hours: float, previous_score: float, result: dict) -> None:
    row = pd.DataFrame(
        [
            {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Study Hours": study_hours,
                "Previous Exam Score": previous_score,
                "Prediction": result["prediction_text"],
                "Pass Probability": result["pass_probability"],
                "Confidence": result["confidence"],
                "Risk Level": result["risk_level"],
                "Performance Band": result["performance_band"],
            }
        ]
    )
    st.session_state.prediction_history = pd.concat(
        [st.session_state.prediction_history, row], ignore_index=True
    )
    st.session_state.last_prediction = row.iloc[0].to_dict()


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def build_text_report(study_hours: float, previous_score: float, result: dict) -> bytes:
    lines = [
        "Student Performance Prediction Report",
        "=" * 36,
        f"Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Student Inputs",
        "-" * 15,
        f"Study Hours: {study_hours}",
        f"Previous Exam Score: {previous_score}",
        "",
        "Prediction Summary",
        "-" * 18,
        f"Decision: {result['prediction_text']}",
        f"Pass Probability: {result['pass_probability']}%",
        f"Fail Probability: {result['fail_probability']}%",
        f"Model Confidence: {result['confidence']}%",
        f"Risk Level: {result['risk_level']}",
        f"Performance Band: {result['performance_band']}",
        "",
        "Recommendations",
        "-" * 15,
    ]
    lines.extend([f"- {item}" for item in result["recommendations"]])
    return "\n".join(lines).encode("utf-8")


def get_plot_layout() -> dict:
    return {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": "#e2e8f0"},
        "margin": {"l": 20, "r": 20, "t": 55, "b": 25},
    }


def show_home(dataset: pd.DataFrame) -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="tiny-label">AI-Powered Academic Intelligence</div>
            <div class="hero-title">Student Performance Prediction System</div>
            <div class="hero-subtitle">
                A modern, portfolio-quality Streamlit experience for predicting student pass or fail outcomes,
                understanding academic risk, and exploring performance trends through elegant analytics and a
                polished glassmorphism interface.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="section-title">Smarter Student Evaluation</div>
                <div class="section-text">
                    Predict pass or fail probability using the trained Random Forest model with confidence
                    scoring, academic risk labeling, and targeted recommendations.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="section-title">Interactive Analytics</div>
                <div class="section-text">
                    Explore score distributions, study-hour trends, pass-rate visuals, and dynamic performance
                    patterns across both dataset records and live prediction history.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="section-title">Presentation-Ready Design</div>
                <div class="section-text">
                    Built with a dark academic-tech visual system, responsive layout, animated cards,
                    polished charts, and professional storytelling elements for strong portfolio impact.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    history = st.session_state.prediction_history
    total_rows = len(dataset)
    session_preds = len(history)
    dataset_pass_rate = (
        round(dataset["Pass/Fail"].astype(float).mean() * 100, 2) if not dataset.empty else 0.0
    )

    left, right = st.columns([1.2, 1])
    with left:
        st.markdown(
            """
            <div class="glass-card">
                <span class="badge badge-blue">Prediction Engine</span>
                <span class="badge badge-green">Live Dashboard</span>
                <span class="badge badge-amber">Academic Risk</span>
                <span class="badge badge-pink">Student Insights</span>
                <div class="section-text" style="margin-top: 0.85rem;">
                    This app transforms a simple binary classifier into a complete decision-support experience.
                    It is ideal for academic demos, ML portfolios, educational dashboards, or internal student
                    support prototypes where outcomes need to be fast, explainable, and visually impressive.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="tiny-label">Live Snapshot</div>
                <div class="metric-value">{session_preds}</div>
                <div class="metric-caption">Predictions completed in this session</div>
                <hr style="border: 0.5px solid rgba(255,255,255,0.08); margin: 1rem 0;">
                <div class="section-text">Dataset Rows: <strong>{total_rows}</strong></div>
                <div class="section-text">Dataset Pass Rate: <strong>{dataset_pass_rate}%</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def show_dashboard(dataset: pd.DataFrame) -> None:
    st.markdown("## Dashboard")
    history = st.session_state.prediction_history.copy()

    base_df = dataset.copy()
    if not base_df.empty:
        base_df["Prediction Label"] = np.where(base_df["Pass/Fail"] == 1, "PASS", "FAIL")

    if base_df.empty and history.empty:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title">No data available yet</div>
                <div class="section-text">
                    Add the CSV file or make live predictions to populate the dashboard with metrics and charts.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    total_students = len(base_df)
    pass_count = int(base_df["Pass/Fail"].sum()) if not base_df.empty else 0
    fail_count = total_students - pass_count if total_students else 0
    avg_hours = round(base_df["Study Hours"].mean(), 2) if not base_df.empty else 0.0
    avg_score = round(base_df["Previous Exam Score"].mean(), 2) if not base_df.empty else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Students", total_students)
    c2.metric("Pass", pass_count)
    c3.metric("Fail", fail_count)
    c4.metric("Avg Study Hours", avg_hours)
    c5.metric("Avg Prev Score", avg_score)

    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        if not base_df.empty:
            pie = px.pie(
                base_df,
                names="Prediction Label",
                title="Dataset Pass vs Fail Split",
                color="Prediction Label",
                color_discrete_map={"PASS": "#22c55e", "FAIL": "#fb7185"},
                hole=0.56,
            )
            pie.update_layout(**get_plot_layout())
            st.plotly_chart(pie, use_container_width=True)

    with row1_col2:
        if not base_df.empty:
            box = px.box(
                base_df,
                x="Prediction Label",
                y="Previous Exam Score",
                color="Prediction Label",
                title="Previous Score by Outcome",
                color_discrete_map={"PASS": "#38bdf8", "FAIL": "#f43f5e"},
            )
            box.update_layout(**get_plot_layout())
            st.plotly_chart(box, use_container_width=True)

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        if not base_df.empty:
            scatter = px.scatter(
                base_df,
                x="Study Hours",
                y="Previous Exam Score",
                color="Prediction Label",
                title="Study Hours vs Previous Score",
                color_discrete_map={"PASS": "#22c55e", "FAIL": "#fb7185"},
                opacity=0.8,
            )
            scatter.update_layout(**get_plot_layout())
            st.plotly_chart(scatter, use_container_width=True)

    with row2_col2:
        if not history.empty:
            history_chart = history.copy()
            history_chart["Sequence"] = np.arange(1, len(history_chart) + 1)
            line = px.line(
                history_chart,
                x="Sequence",
                y=["Pass Probability", "Confidence"],
                markers=True,
                title="Live Prediction Probability Trend",
                color_discrete_sequence=["#2dd4bf", "#60a5fa"],
            )
            line.update_layout(**get_plot_layout())
            st.plotly_chart(line, use_container_width=True)
        else:
            st.markdown(
                """
                <div class="glass-card">
                    <div class="section-title">Live history is empty</div>
                    <div class="section-text">
                        Run one or more student predictions to unlock session-based probability and confidence trends.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def show_prediction() -> None:
    st.markdown("## Student Prediction")
    model, error = load_model()
    if error:
        st.warning(error)

    top1, top2, top3 = st.columns(3)
    with top1:
        st.markdown(
            """
            <div class="highlight-card study-glow">
                <div class="highlight-title">Study Rhythm</div>
                <div class="highlight-text">
                    Daily study hours act as a direct discipline signal and heavily influence readiness.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top2:
        st.markdown(
            """
            <div class="highlight-card score-glow">
                <div class="highlight-title">Past Performance</div>
                <div class="highlight-text">
                    Previous exam score is a strong academic baseline that helps estimate current momentum.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top3:
        st.markdown(
            """
            <div class="highlight-card predict-glow">
                <div class="highlight-title">Prediction Output</div>
                <div class="highlight-text">
                    Generate pass or fail outcome, probability, confidence, academic risk, and action advice.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.form("student_prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            study_hours = st.slider("Daily Study Hours", min_value=0.0, max_value=15.0, value=4.5, step=0.5)
        with col2:
            previous_score = st.slider("Previous Exam Score", min_value=0.0, max_value=100.0, value=68.0, step=1.0)

        st.markdown(
            """
            <div class="glass-card" style="margin-top: 0.75rem; text-align: center;">
                <div class="section-title" style="margin-bottom: 0.35rem;">Ready to Evaluate This Student?</div>
                <div class="section-text" style="margin-bottom: 0.4rem;">
                    Submit the academic profile to generate a polished predictive summary and student support guidance.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="predict-center-wrap">', unsafe_allow_html=True)
        submitted = st.form_submit_button("Predict Student Performance")
        st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        with st.spinner("Analyzing academic profile and generating performance insights..."):
            try:
                result = predict_student(model, study_hours, previous_score)
                add_history(study_hours, previous_score, result)

                status_class = "status-pass" if result["passed"] else "status-fail"
                st.markdown(
                    f"""
                    <div class="prediction-card">
                        <div class="tiny-label">Prediction Outcome</div>
                        <div class="hero-title" style="font-size: 2rem; margin-bottom: 0.2rem;">
                            <span class="{status_class}">{result['prediction_text']}</span>
                        </div>
                        <div class="section-text">
                            Pass Probability <strong>{result['pass_probability']}%</strong> |
                            Fail Probability <strong>{result['fail_probability']}%</strong> |
                            Model Confidence <strong>{result['confidence']}%</strong> |
                            Risk <strong>{result['risk_level']}</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Pass Probability", f"{result['pass_probability']}%")
                m2.metric("Model Confidence", f"{result['confidence']}%")
                m3.metric("Risk Level", result["risk_level"])
                m4.metric("Performance Band", result["performance_band"])

                chart_col, rec_col = st.columns([1, 1.08])
                with chart_col:
                    gauge = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=result["pass_probability"],
                            number={"suffix": "%", "font": {"color": "#f8fafc"}},
                            gauge={
                                "axis": {"range": [0, 100], "tickcolor": "#cbd5e1"},
                                "bar": {"color": "#22c55e" if result["passed"] else "#fb7185"},
                                "bgcolor": "rgba(255,255,255,0.05)",
                                "steps": [
                                    {"range": [0, 40], "color": "rgba(244,63,94,0.28)"},
                                    {"range": [40, 70], "color": "rgba(245,158,11,0.22)"},
                                    {"range": [70, 100], "color": "rgba(34,197,94,0.22)"},
                                ],
                            },
                            title={"text": "Pass Probability", "font": {"color": "#e2e8f0"}},
                        )
                    )
                    gauge.update_layout(height=355, **get_plot_layout())
                    st.plotly_chart(gauge, use_container_width=True)

                with rec_col:
                    recommendation_markup = "".join(f"<li>{item}</li>" for item in result["recommendations"])
                    st.markdown(
                        f"""
                        <div class="glass-card">
                            <div class="section-title">Smart Recommendations</div>
                            <ul style="color:#e2e8f0; line-height:1.75;">{recommendation_markup}</ul>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                report_bytes = build_text_report(study_hours, previous_score, result)
                st.download_button(
                    "Download Prediction Report",
                    data=report_bytes,
                    file_name=f"student_prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                )
            except Exception as exc:
                st.error(f"Prediction could not be completed: {exc}")


def show_analytics(dataset: pd.DataFrame) -> None:
    st.markdown("## Analytics")
    history = st.session_state.prediction_history.copy()

    if dataset.empty and history.empty:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title">Analytics will appear once data is available</div>
                <div class="section-text">
                    This section automatically builds visual insights from the CSV dataset and your prediction history.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if not dataset.empty:
        d1, d2 = st.columns(2)
        with d1:
            hist_hours = px.histogram(
                dataset,
                x="Study Hours",
                nbins=20,
                color=dataset["Pass/Fail"].map({1: "PASS", 0: "FAIL"}),
                title="Study Hours Distribution",
                color_discrete_map={"PASS": "#22c55e", "FAIL": "#fb7185"},
            )
            hist_hours.update_layout(**get_plot_layout())
            st.plotly_chart(hist_hours, use_container_width=True)

        with d2:
            hist_score = px.histogram(
                dataset,
                x="Previous Exam Score",
                nbins=20,
                color=dataset["Pass/Fail"].map({1: "PASS", 0: "FAIL"}),
                title="Previous Score Distribution",
                color_discrete_map={"PASS": "#38bdf8", "FAIL": "#f43f5e"},
            )
            hist_score.update_layout(**get_plot_layout())
            st.plotly_chart(hist_score, use_container_width=True)

        dataset_corr = dataset[["Study Hours", "Previous Exam Score", "Pass/Fail"]].corr().round(2)
        heatmap = px.imshow(
            dataset_corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Tealgrn",
            title="Correlation Heatmap",
        )
        heatmap.update_layout(**get_plot_layout())
        st.plotly_chart(heatmap, use_container_width=True)

    if not history.empty:
        h1, h2 = st.columns(2)
        with h1:
            risk_counts = history["Risk Level"].value_counts().reset_index()
            risk_counts.columns = ["Risk Level", "Count"]
            risk_fig = px.bar(
                risk_counts,
                x="Risk Level",
                y="Count",
                color="Risk Level",
                title="Live Risk Level Distribution",
                color_discrete_sequence=["#38bdf8", "#f59e0b", "#fb7185"],
            )
            risk_fig.update_layout(**get_plot_layout())
            st.plotly_chart(risk_fig, use_container_width=True)

        with h2:
            band_counts = history["Performance Band"].value_counts().reset_index()
            band_counts.columns = ["Performance Band", "Count"]
            band_fig = px.pie(
                band_counts,
                names="Performance Band",
                values="Count",
                title="Live Performance Band Split",
                hole=0.52,
                color_discrete_sequence=["#22c55e", "#38bdf8", "#f59e0b", "#fb7185"],
            )
            band_fig.update_layout(**get_plot_layout())
            st.plotly_chart(band_fig, use_container_width=True)

        st.download_button(
            "Export Prediction History CSV",
            data=dataframe_to_csv_bytes(history),
            file_name="student_prediction_history.csv",
            mime="text/csv",
        )


def show_dataset(dataset: pd.DataFrame) -> None:
    st.markdown("## Dataset Info")
    schema_df = pd.DataFrame(
        [
            ["Study Hours", "Daily study duration used for model input", "Numeric"],
            ["Previous Exam Score", "Prior exam score from 0 to 100", "Numeric"],
            ["Pass/Fail", "Target class where 1 indicates pass and 0 indicates fail", "Binary"],
        ],
        columns=["Column", "Description", "Type"],
    )

    st.markdown(
        """
        <div class="glass-card">
            <div class="section-title">Dataset Schema</div>
            <div class="section-text">
                The current model is trained on two academic performance indicators: study effort and previous score.
                This keeps the prediction pipeline simple, explainable, and aligned with the available training data.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(schema_df, use_container_width=True, hide_index=True)

    if not dataset.empty:
        st.markdown("### Dataset Preview")
        st.dataframe(dataset.head(15), use_container_width=True, hide_index=True)

        summary = pd.DataFrame(
            {
                "Metric": ["Rows", "Average Study Hours", "Average Previous Score", "Pass Rate"],
                "Value": [
                    len(dataset),
                    round(dataset["Study Hours"].mean(), 2),
                    round(dataset["Previous Exam Score"].mean(), 2),
                    f"{round(dataset['Pass/Fail'].mean() * 100, 2)}%",
                ],
            }
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)

        st.download_button(
            "Export Dataset CSV",
            data=dataframe_to_csv_bytes(dataset),
            file_name="student_exam_data_export.csv",
            mime="text/csv",
        )


def show_education() -> None:
    st.markdown("## Student Education")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class="edu-card">
                <div class="section-title">What improves student outcomes?</div>
                <div class="section-text">
                    Consistent study habits, active recall, revision planning, and strong conceptual clarity tend
                    to produce more stable academic performance than occasional high-effort bursts.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="edu-card">
                <div class="section-title">Why previous score matters</div>
                <div class="section-text">
                    Previous performance is often a proxy for base preparation, confidence, and topic familiarity,
                    which is why it becomes a powerful signal inside a predictive model.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="glass-card">
            <div class="section-title">Practical improvement strategy</div>
            <div class="section-text">
                1. Split study time into focused blocks with short reviews.<br>
                2. Revisit weak topics before practicing advanced questions.<br>
                3. Track performance weekly instead of only before exams.<br>
                4. Use mock papers to improve speed and confidence.<br>
                5. Maintain routine, sleep quality, and revision balance for better long-term retention.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_about() -> None:
    model, error = load_model()
    dataset = load_dataset()
    st.markdown("## About")
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="section-title">Developer Section</div>
            <div class="section-text">
                This application upgrades a simple prediction script into a complete academic analytics platform
                with modular page architecture, modern UI styling, model caching, session-based history, downloads,
                and interactive charts for a high-quality ML portfolio presentation.
                <br><br>
                <strong>Tech Stack:</strong> Streamlit, Pandas, NumPy, Plotly, Joblib, Scikit-Learn
                <br>
                <strong>Model Status:</strong> {"Loaded" if model is not None else "Missing"}
                <br>
                <strong>Dataset Rows:</strong> {len(dataset)}
                <br>
                <strong>Session Started:</strong> {st.session_state.session_started.strftime('%Y-%m-%d %H:%M:%S')}
                <br>
                <strong>Created by:</strong> <span class="created-by-big">Tejas</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if error:
        st.info(error)


def render_sidebar(dataset: pd.DataFrame) -> str:
    st.sidebar.markdown(
        """
        <div style="padding:0.5rem 0 1rem 0;">
            <div style="font-size:1.45rem; font-weight:900; color:#f8fafc;">Academic Insight Suite</div>
            <div style="color:#94a3b8; margin-top:0.2rem;">Advanced student performance intelligence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio(
        "Navigate",
        [
            "🏠 Home",
            "📊 Dashboard",
            "🔍 Prediction",
            "📈 Analytics",
            "📚 Dataset Info",
            "💡 Student Education",
            "👨‍💻 About",
        ],
    )

    history = st.session_state.prediction_history
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Session Snapshot")
    st.sidebar.write(f"Dataset Rows: {len(dataset)}")
    st.sidebar.write(f"Predictions: {len(history)}")
    st.sidebar.write(
        f"Session Pass Rate: {round(history['Prediction'].eq('PASS').mean() * 100, 2) if len(history) else 0.0}%"
    )
    st.sidebar.write(f"Model File: {'Available' if MODEL_PATH.exists() else 'Missing'}")
    st.sidebar.write(f"Image File: {'Available' if IMAGE_PATH.exists() else 'Missing'}")
    return page


def render_footer() -> None:
    st.markdown(
        """
        <div class="footer-card" style="margin-top:1rem;">
            <div class="footer-note">
                Student Performance Prediction System | Academic Analytics Experience | Created by
                <span class="created-by-big">Tejas</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_global_styles()
    init_session_state()
    dataset = load_dataset()
    page = render_sidebar(dataset)

    if page == "🏠 Home":
        show_home(dataset)
    elif page == "📊 Dashboard":
        show_dashboard(dataset)
    elif page == "🔍 Prediction":
        show_prediction()
    elif page == "📈 Analytics":
        show_analytics(dataset)
    elif page == "📚 Dataset Info":
        show_dataset(dataset)
    elif page == "💡 Student Education":
        show_education()
    elif page == "👨‍💻 About":
        show_about()

    render_footer()


if __name__ == "__main__":
    main()
