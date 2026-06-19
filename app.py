"""
Aplicación EDA – InsuranceCompany
Especialización Python for Analytics · 2026
"""

import io
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import streamlit as st

# ─────────────────────────────────────────────────────────────────
# Configuración de página
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InsuranceCompany EDA",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = ["#2E86AB", "#E84855", "#F9C74F", "#90BE6D", "#577590"]
sns.set_theme(style="whitegrid", palette=PALETTE)

# ─────────────────────────────────────────────────────────────────
# POO – Clase DataAnalyzer
# ─────────────────────────────────────────────────────────────────
class DataAnalyzer:
    """
    Encapsula estadísticas descriptivas, clasificación de variables
    y funciones de visualización para el EDA del dataset InsuranceCompany.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    # ── Clasificación de variables ────────────────────────────────
    def classify_variables(self) -> dict:
        """Retorna columnas numéricas y categóricas del DataFrame."""
        numeric   = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categoric = self.df.select_dtypes(
            include=["object", "string", "category"]
        ).columns.tolist()
        return {"numeric": numeric, "categorical": categoric}

    # ── Estadísticas descriptivas ─────────────────────────────────
    def descriptive_stats(self, cols: list = None) -> pd.DataFrame:
        """describe() extendido con mediana y asimetría."""
        subset = self.df[cols] if cols else self.df
        desc = subset.describe(percentiles=[0.25, 0.50, 0.75]).T
        desc["median"]   = subset.median(numeric_only=True)
        desc["skewness"] = subset.skew(numeric_only=True)
        return desc.round(3)

    def missing_summary(self) -> pd.DataFrame:
        """Resumen de valores nulos por columna."""
        total   = self.df.isnull().sum()
        percent = (total / len(self.df) * 100).round(2)
        return (
            pd.DataFrame({"Nulos": total, "Porcentaje (%)": percent})
            .sort_values("Nulos", ascending=False)
        )

    # ── Visualizaciones ───────────────────────────────────────────
    def plot_histogram(self, col: str, bins: int = 30, hue_col: str = None):
        fig, ax = plt.subplots(figsize=(8, 4))
        if hue_col:
            for i, (val, grp) in enumerate(self.df.groupby(hue_col)):
                ax.hist(
                    grp[col].dropna(), bins=bins,
                    alpha=0.6, label=str(val),
                    color=PALETTE[i % len(PALETTE)], edgecolor="white"
                )
            ax.legend(title=hue_col)
        else:
            ax.hist(
                self.df[col].dropna(), bins=bins,
                color=PALETTE[0], edgecolor="white"
            )
        ax.set_xlabel(col); ax.set_ylabel("Frecuencia")
        ax.set_title(f"Distribución de {col}")
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
        )
        fig.tight_layout(); return fig

    def plot_bar(self, col: str, top_n: int = 20):
        vc = self.df[col].value_counts().head(top_n)
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(
            vc.index.astype(str), vc.values,
            color=PALETTE[: len(vc)], edgecolor="white"
        )
        ax.bar_label(bars, fmt="%d", padding=3, fontsize=9)
        ax.set_xlabel(col); ax.set_ylabel("Conteo")
        ax.set_title(f"Frecuencia de {col}")
        fig.tight_layout(); return fig

    def plot_boxplot(self, num_col: str, cat_col: str):
        fig, ax = plt.subplots(figsize=(8, 4))
        order = self.df[cat_col].value_counts().index.tolist()
        sns.boxplot(
            data=self.df, x=cat_col, y=num_col,
            order=order, palette=PALETTE, ax=ax
        )
        ax.set_title(f"{num_col} por {cat_col}")
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
        )
        fig.tight_layout(); return fig

    def plot_crosstab_heatmap(self, col1: str, col2: str):
        ct = pd.crosstab(
            self.df[col1], self.df[col2], normalize="index"
        ).round(2)
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.heatmap(
            ct, annot=True, fmt=".0%", cmap="Blues", ax=ax,
            linewidths=0.5,
            cbar_kws={"format": mticker.PercentFormatter(xmax=1)},
        )
        ax.set_title(f"{col1} vs {col2}  (proporción por fila)")
        fig.tight_layout(); return fig

    def plot_correlation(self, cols: list):
        fig, ax = plt.subplots(figsize=(9, 7))
        corr = self.df[cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, ax=ax, linewidths=0.5
        )
        ax.set_title("Matriz de Correlación")
        fig.tight_layout(); return fig

    def plot_missing_bar(self):
        missing = self.missing_summary()
        sub = missing[missing["Nulos"] > 0]
        fig, ax = plt.subplots(figsize=(8, 3))
        bars = ax.barh(sub.index, sub["Porcentaje (%)"], color=PALETTE[1])
        ax.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=9)
        ax.set_xlabel("Porcentaje nulo (%)")
        ax.set_title("Valores nulos por columna")
        fig.tight_layout(); return fig


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────
def fmt_n(n): return f"{n:,}"


@st.cache_data
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df["renewal_label"] = df["renewal"].map({1: "Sí", 0: "No"})
    df["age_years"] = (df["age_in_days"] / 365).round(1)
    df["total_late"] = (
        df["Count_3-6_months_late"].fillna(0)
        + df["Count_6-12_months_late"].fillna(0)
        + df["Count_more_than_12_months_late"].fillna(0)
    )
    return df


def df_info_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        rows.append({
            "Columna": col,
            "Dtype": str(df[col].dtype),
            "No Nulos": int(df[col].notna().sum()),
            "Nulos": int(df[col].isna().sum()),
            "Únicos": int(df[col].nunique()),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ InsuranceCompany EDA")
    st.markdown("---")
    modulo = st.radio(
        "Navegación",
        [
            "🏠 Home",
            "📤 Carga del Dataset",
            "🔬 EDA — Análisis Exploratorio",
            "📌 Conclusiones",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Especialización Python for Analytics · 2026")


# ═════════════════════════════════════════════════════════════════
# MÓDULO 1 – HOME
# ═════════════════════════════════════════════════════════════════
if modulo == "🏠 Home":
    st.title("🛡️ EDA — InsuranceCompany")
    st.markdown(
        """
        > **Objetivo:** Explorar el dataset `InsuranceCompany.csv` para identificar
        > los factores que influyen en la **renovación de pólizas de seguro**,
        > construyendo una herramienta analítica interactiva con Python y Streamlit.
        """
    )
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("👤 Autor")
        st.markdown(
            """
            | Campo | Detalle |
            |---|---|
            | **Nombre** | Klessyth Abigaith Guevara Paisig |
            | **Curso** | Especialización Python for Analytics |
            | **Año** | 2026 |
            """
        )
        st.subheader("📊 Sobre el Dataset")
        st.markdown(
            """
            Contiene **79 852 registros** de clientes de una aseguradora con:
            - Variables demográficas y económicas
            - Historial de pagos y morosidad
            - Canal de captación (`sourcing_channel`)
            - Tipo de residencia (`residence_area_type`)
            - **Variable objetivo:** `renewal` (1 = renovó, 0 = no renovó)
            """
        )

    with col2:
        st.subheader("🧰 Tecnologías utilizadas")
        techs = {
            "🐍 Python 3.11": "Lenguaje base",
            "🐼 Pandas": "Manipulación de datos",
            "🔢 NumPy": "Cálculo numérico",
            "📉 Matplotlib / Seaborn": "Visualización",
            "⚡ Streamlit": "Interfaz interactiva",
        }
        for tech, desc in techs.items():
            st.markdown(f"**{tech}** — {desc}")

    st.info("👈 Usa el menú lateral para navegar entre los módulos.")


# ═════════════════════════════════════════════════════════════════
# MÓDULO 2 – CARGA
# ═════════════════════════════════════════════════════════════════
elif modulo == "📤 Carga del Dataset":
    st.title("📤 Carga del Dataset")
    st.markdown("Sube el archivo `InsuranceCompany.csv` para habilitar el análisis.")

    uploaded = st.file_uploader("Selecciona el archivo CSV", type=["csv"])

    if uploaded is None:
        st.warning("⚠️ Por favor carga el dataset para continuar.")
        st.stop()

    df = load_data(uploaded)
    st.session_state["df"] = df

    st.success(f"✅ **{uploaded.name}** cargado correctamente.")

    st.subheader("👁️ Vista previa — primeras 5 filas")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("📐 Dimensiones")
    c1, c2, c3 = st.columns(3)
    c1.metric("Filas",     fmt_n(df.shape[0]))
    c2.metric("Columnas",  df.shape[1])
    c3.metric("Memoria",   f"{df.memory_usage(deep=True).sum()/1e6:.1f} MB")

    st.subheader("📋 Columnas disponibles")
    st.write(df.columns.tolist())


# ═════════════════════════════════════════════════════════════════
# MÓDULO 3 – EDA
# ═════════════════════════════════════════════════════════════════
elif modulo == "🔬 EDA — Análisis Exploratorio":
    st.title("🔬 Análisis Exploratorio de Datos")

    if "df" not in st.session_state:
        st.error(
            "❌ Primero carga el dataset en el módulo **📤 Carga del Dataset**."
        )
        st.stop()

    df  = st.session_state["df"]
    ana = DataAnalyzer(df)
    vd  = ana.classify_variables()

    # Columnas útiles
    num_cols = [
        c for c in vd["numeric"]
        if c not in ("id", "renewal")
    ]
    raw_cat = [
        c for c in vd["categorical"]
        if c != "renewal_label"
    ]

    tabs = st.tabs([
        "1️⃣ Info general",
        "2️⃣ Variables",
        "3️⃣ Estadísticas",
        "4️⃣ Nulos",
        "5️⃣ Distribuciones",
        "6️⃣ Categóricas",
        "7️⃣ Bivariado Num",
        "8️⃣ Bivariado Cat",
        "9️⃣ Análisis dinámico",
        "🔟 Hallazgos",
    ])

    # ── Tab 1 – Información general ───────────────────────────────
    with tabs[0]:
        st.subheader("Ítem 1: Información general del dataset")
        st.markdown(
            "Vista completa de tipos de dato, no-nulos y únicos "
            "para todas las columnas."
        )

        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("**Resumen por columna**")
            info_df = df_info_table(df)
            st.dataframe(info_df, use_container_width=True, height=420)

        with col2:
            st.markdown("**Tipos de dato**")
            dtype_cnt = (
                df.dtypes.value_counts()
                .rename_axis("Dtype")
                .reset_index(name="Columnas")
            )
            dtype_cnt["Dtype"] = dtype_cnt["Dtype"].astype(str)
            st.dataframe(dtype_cnt, use_container_width=True)

            st.markdown("**Nulos totales**")
            total_nulos = df.isnull().sum().sum()
            if total_nulos == 0:
                st.success("Sin valores nulos.")
            else:
                st.warning(f"{fmt_n(total_nulos)} valores nulos en total.")
                nulos_df = (
                    df.isnull().sum()
                    .rename("Nulos")
                    .reset_index()
                    .rename(columns={"index": "Columna"})
                )
                nulos_df = nulos_df[nulos_df["Nulos"] > 0]
                st.dataframe(nulos_df, use_container_width=True)

    # ── Tab 2 – Clasificación de variables ────────────────────────
    with tabs[1]:
        st.subheader("Ítem 2: Clasificación de variables")
        st.markdown(
            "Función personalizada `DataAnalyzer.classify_variables()` que "
            "separa automáticamente las columnas por dtype."
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"#### 🔢 Numéricas ({len(vd['numeric'])})")
            for v in vd["numeric"]:
                st.markdown(f"- `{v}`")
        with col2:
            st.markdown(f"#### 🔠 Categóricas ({len(vd['categorical'])})")
            for v in vd["categorical"]:
                st.markdown(f"- `{v}`")

        total_class = len(vd["numeric"]) + len(vd["categorical"])
        st.info(
            f"Total clasificadas: **{len(vd['numeric'])} numéricas** + "
            f"**{len(vd['categorical'])} categóricas** = **{total_class} columnas**"
        )

        # Mini gráfico de proporciones
        fig_p, ax_p = plt.subplots(figsize=(4, 3))
        ax_p.pie(
            [len(vd["numeric"]), len(vd["categorical"])],
            labels=["Numéricas", "Categóricas"],
            autopct="%1.0f%%",
            colors=[PALETTE[0], PALETTE[2]],
            startangle=90,
        )
        ax_p.set_title("Proporción de tipos de variable")
        fig_p.tight_layout()
        st.pyplot(fig_p)

    # ── Tab 3 – Estadísticas descriptivas ────────────────────────
    with tabs[2]:
        st.subheader("Ítem 3: Estadísticas descriptivas")
        st.markdown(
            "Calculamos media, mediana, desviación estándar y asimetría "
            "(skewness) con `.describe()` extendido."
        )

        stats = ana.descriptive_stats(num_cols)
        st.dataframe(
            stats.style.format("{:,.3f}"),
            use_container_width=True,
        )

        st.markdown("#### Métricas clave")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ingreso promedio",    f"${df['Income'].mean():,.0f}")
        c2.metric("Prima promedio",      f"${df['premium'].mean():,.0f}")
        c3.metric("Primas pagadas (med)",f"{df['no_of_premiums_paid'].median():.0f}")
        c4.metric("Edad promedio",       f"{df['age_years'].mean():.1f} años")

        st.markdown(
            """
            **Interpretación:**
            - `Income` presenta altísima asimetría (skewness ≈ 110), dominada por
              outliers de ingresos muy elevados; la mediana es más representativa.
            - `premium` también es asimétrica positiva (≈ 2.2): la mayoría paga
              primas bajas pero un grupo paga primas muy altas.
            - `application_underwriting_score` tiene distribución negativa (≈ -2.8),
              concentrada en valores altos (~99), lo que indica que la mayoría de
              clientes tiene buen puntaje de confiabilidad.
            """
        )

    # ── Tab 4 – Valores faltantes ─────────────────────────────────
    with tabs[3]:
        st.subheader("Ítem 4: Análisis de valores faltantes")

        missing = ana.missing_summary()
        total_missing = missing["Nulos"].sum()

        c1, c2 = st.columns(2)
        c1.metric("Total de valores nulos", fmt_n(int(total_missing)))
        c2.metric(
            "% del dataset",
            f"{total_missing / df.size * 100:.2f}%",
        )

        sub_missing = missing[missing["Nulos"] > 0]
        st.dataframe(sub_missing, use_container_width=True)

        fig_m = ana.plot_missing_bar()
        st.pyplot(fig_m)

        st.markdown(
            """
            **Análisis:**
            - `application_underwriting_score`: **2 974 nulos (3.72%)** — columna clave
              de evaluación de riesgo; se recomienda imputar con la mediana antes de
              cualquier modelo downstream.
            - `Count_3-6_months_late`, `Count_6-12_months_late`,
              `Count_more_than_12_months_late`: **97 nulos cada una (0.12%)** —
              probablemente clientes sin historial de pagos tardíos; se puede imputar
              con 0 de forma segura.
            - El porcentaje de nulos es bajo (< 4%), por lo que la calidad general
              del dataset es **buena**.
            """
        )

    # ── Tab 5 – Distribución numérica ────────────────────────────
    with tabs[4]:
        st.subheader("Ítem 5: Distribución de variables numéricas")

        col1, col2 = st.columns([2, 1])
        with col1:
            col_sel = st.selectbox(
                "Variable numérica", num_cols, key="hist_col"
            )
        with col2:
            bins_n = st.slider("Bins", 10, 80, 30, key="hist_bins")

        by_renewal = st.checkbox(
            "Separar por renovación", key="hist_renewal"
        )

        hue = "renewal_label" if by_renewal else None
        fig = ana.plot_histogram(col_sel, bins=bins_n, hue_col=hue)
        st.pyplot(fig)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Media",       f"{df[col_sel].mean():,.2f}")
        c2.metric("Mediana",     f"{df[col_sel].median():,.2f}")
        c3.metric("Moda",        f"{df[col_sel].mode()[0]:,.2f}")
        c4.metric("Desv. Std",   f"{df[col_sel].std():,.2f}")

        skew = df[col_sel].skew()
        direction = "positiva (cola a la derecha)" if skew > 0 else "negativa (cola a la izquierda)"
        st.markdown(
            f"La variable **`{col_sel}`** tiene asimetría **{direction}** "
            f"(skewness = `{skew:.2f}`)."
        )

    # ── Tab 6 – Variables categóricas ─────────────────────────────
    with tabs[5]:
        st.subheader("Ítem 6: Análisis de variables categóricas")

        cat_sel = st.selectbox(
            "Variable categórica", raw_cat, key="cat_sel"
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            vc   = df[cat_sel].value_counts()
            prop = (vc / len(df) * 100).round(2)
            table = pd.DataFrame({"Conteo": vc, "Proporción (%)": prop})
            st.dataframe(table, use_container_width=True)

            # Pie
            fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
            ax_pie.pie(
                vc.values,
                labels=vc.index.astype(str),
                autopct="%1.1f%%",
                colors=PALETTE[: len(vc)],
                startangle=90,
            )
            ax_pie.set_title(f"Proporción de {cat_sel}")
            fig_pie.tight_layout()
            st.pyplot(fig_pie)

        with col2:
            fig_bar = ana.plot_bar(cat_sel)
            st.pyplot(fig_bar)

    # ── Tab 7 – Bivariado numérico vs categórico ──────────────────
    with tabs[6]:
        st.subheader("Ítem 7: Análisis bivariado — Numérico vs Categórico")
        st.markdown(
            "Comparamos distribuciones numéricas entre grupos categóricos. "
            "Ejemplo clave: `Income` vs `renewal_label`."
        )

        col1, col2 = st.columns(2)
        with col1:
            num_biv = st.selectbox("Variable numérica", num_cols, key="biv_num")
        with col2:
            cat_biv = st.selectbox(
                "Variable categórica",
                raw_cat + ["renewal_label"],
                key="biv_cat",
            )

        fig_box = ana.plot_boxplot(num_biv, cat_biv)
        st.pyplot(fig_box)

        # Tabla estadísticas por grupo
        grp = (
            df.groupby(cat_biv)[num_biv]
            .agg(["mean", "median", "std", "count"])
            .round(2)
        )
        grp.columns = ["Media", "Mediana", "Desv. Std", "N"]
        st.markdown("**Estadísticas por grupo**")
        st.dataframe(grp, use_container_width=True)

        st.markdown(
            f"""
            **Interpretación:** Para la variable **`{num_biv}`**, los grupos de
            **`{cat_biv}`** muestran diferencias en su distribución central.
            La mediana es preferida sobre la media cuando existe asimetría elevada.
            """
        )

    # ── Tab 8 – Bivariado categórico vs categórico ────────────────
    with tabs[7]:
        st.subheader("Ítem 8: Análisis bivariado — Categórico vs Categórico")
        st.markdown(
            "Analizamos la proporción de renovaciones por variable categórica "
            "usando tablas de contingencia y heatmaps."
        )

        col1, col2 = st.columns(2)
        with col1:
            cat1 = st.selectbox("Variable fila", raw_cat, index=0, key="cc1")
        with col2:
            cat2 = st.selectbox(
                "Variable columna",
                ["renewal_label"] + raw_cat,
                index=0,
                key="cc2",
            )

        fig_ct = ana.plot_crosstab_heatmap(cat1, cat2)
        st.pyplot(fig_ct)

        st.markdown("**Tabla de contingencia (valores absolutos)**")
        ct_abs = pd.crosstab(df[cat1], df[cat2])
        st.dataframe(ct_abs, use_container_width=True)

        # Renovación % por canal (análisis fijo de ejemplo)
        st.markdown("---")
        st.markdown(
            "#### 📌 Ejemplo fijo: `sourcing_channel` vs `renewal`"
        )
        ch_ren = (
            df.groupby("sourcing_channel")["renewal"]
            .mean()
            .mul(100)
            .round(2)
            .sort_values(ascending=False)
            .reset_index()
        )
        ch_ren.columns = ["Canal", "Tasa Renovación (%)"]
        st.dataframe(ch_ren, use_container_width=True)

    # ── Tab 9 – Análisis dinámico ─────────────────────────────────
    with tabs[8]:
        st.subheader("Ítem 9: Análisis dinámico por parámetros")
        st.markdown(
            "Selecciona variables y filtros para explorar relaciones "
            "de forma interactiva."
        )

        # Filtros en sidebar dinámico
        with st.expander("🎛️ Filtros", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                channels = st.multiselect(
                    "Canales de captación",
                    df["sourcing_channel"].unique().tolist(),
                    default=df["sourcing_channel"].unique().tolist(),
                    key="dyn_ch",
                )
                areas = st.multiselect(
                    "Tipo de área",
                    df["residence_area_type"].unique().tolist(),
                    default=df["residence_area_type"].unique().tolist(),
                    key="dyn_area",
                )
            with col2:
                income_range = st.slider(
                    "Rango de Income",
                    int(df["Income"].min()),
                    min(int(df["Income"].max()), 5_000_000),
                    (int(df["Income"].min()), 1_000_000),
                    step=10_000,
                    key="income_sl",
                )
                prem_range = st.slider(
                    "Rango de Primas Pagadas",
                    int(df["no_of_premiums_paid"].min()),
                    int(df["no_of_premiums_paid"].max()),
                    (int(df["no_of_premiums_paid"].min()),
                     int(df["no_of_premiums_paid"].max())),
                    key="prem_sl",
                )

        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("Eje X", num_cols, index=2, key="dyn_x")
        with col2:
            y_col = st.selectbox(
                "Eje Y",
                [c for c in num_cols if c != x_col],
                index=0,
                key="dyn_y",
            )

        color_col   = st.selectbox(
            "Color por", ["renewal_label"] + raw_cat, key="dyn_col"
        )
        show_out    = st.checkbox("Mostrar outliers", value=False, key="show_out")

        mask = (
            df["sourcing_channel"].isin(channels)
            & df["residence_area_type"].isin(areas)
            & df["Income"].between(*income_range)
            & df["no_of_premiums_paid"].between(*prem_range)
        )
        dff = df[mask].copy()

        if dff.empty:
            st.warning("No hay registros con los filtros aplicados.")
        else:
            st.info(
                f"Mostrando **{fmt_n(len(dff))}** de **{fmt_n(len(df))}** registros."
            )
            if not show_out:
                for col_ in [x_col, y_col]:
                    q1, q3 = dff[col_].quantile([0.01, 0.99])
                    dff = dff[dff[col_].between(q1, q3)]

            fig_sc, ax_sc = plt.subplots(figsize=(9, 5))
            for i, (lbl, grp) in enumerate(dff.groupby(color_col)):
                ax_sc.scatter(
                    grp[x_col], grp[y_col],
                    alpha=0.35, s=12,
                    label=str(lbl),
                    color=PALETTE[i % len(PALETTE)],
                )
            ax_sc.set_xlabel(x_col); ax_sc.set_ylabel(y_col)
            ax_sc.set_title(f"{x_col} vs {y_col}  —  color: {color_col}")
            ax_sc.legend(title=color_col)
            ax_sc.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
            )
            ax_sc.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
            )
            fig_sc.tight_layout()
            st.pyplot(fig_sc)

        # Matriz de correlación dinámica
        st.markdown("---")
        st.markdown("#### 🔗 Matriz de Correlación dinámica")
        num_sel = st.multiselect(
            "Selecciona variables",
            num_cols,
            default=num_cols[:6],
            key="corr_sel",
        )
        if len(num_sel) >= 2:
            fig_c = ana.plot_correlation(num_sel)
            st.pyplot(fig_c)
        else:
            st.info("Selecciona al menos 2 variables.")

    # ── Tab 10 – Hallazgos clave ──────────────────────────────────
    with tabs[9]:
        st.subheader("Ítem 10: Hallazgos clave")

        # KPIs
        renewal_rate = df["renewal"].mean() * 100
        top_channel  = df["sourcing_channel"].value_counts().idxmax()
        no_renewal_morose = (
            df[df["total_late"] >= 3]["renewal"].mean() * 100
        )
        avg_score = df["application_underwriting_score"].mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tasa global renovación",    f"{renewal_rate:.1f}%")
        c2.metric("Canal más frecuente",       top_channel)
        c3.metric("Renovación con ≥3 tardíos", f"{no_renewal_morose:.1f}%")
        c4.metric("Score prom. underwriting",  f"{avg_score:.2f}")

        st.markdown("---")

        # Gráfico 1: Renovación por canal
        fig1, ax1 = plt.subplots(figsize=(7, 4))
        ch_r = (
            df.groupby("sourcing_channel")["renewal"]
            .mean()
            .mul(100)
            .sort_values(ascending=False)
        )
        bars1 = ax1.bar(
            ch_r.index, ch_r.values,
            color=PALETTE[: len(ch_r)], edgecolor="white"
        )
        ax1.bar_label(bars1, fmt="%.1f%%", padding=3)
        ax1.set_ylim(0, 105)
        ax1.set_ylabel("Tasa de Renovación (%)")
        ax1.set_title("Renovación por Canal de Captación")
        fig1.tight_layout()

        # Gráfico 2: Morosidad vs renovación
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        late_r = (
            df.groupby("total_late")["renewal"]
            .mean()
            .mul(100)
            .head(8)
        )
        bars2 = ax2.bar(
            late_r.index.astype(int), late_r.values,
            color=[PALETTE[0] if v > 80 else PALETTE[1] for v in late_r.values],
            edgecolor="white",
        )
        ax2.bar_label(bars2, fmt="%.1f%%", padding=3)
        ax2.set_xlabel("Total de pagos tardíos")
        ax2.set_ylabel("Tasa de Renovación (%)")
        ax2.set_title("Impacto de la Morosidad en la Renovación")
        ax2.set_ylim(0, 110)
        fig2.tight_layout()

        col1, col2 = st.columns(2)
        col1.pyplot(fig1)
        col2.pyplot(fig2)

        # Gráfico 3: Primas pagadas vs renovación
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        prem_r = (
            df.groupby("no_of_premiums_paid")["renewal"]
            .mean()
            .mul(100)
        )
        ax3.plot(
            prem_r.index, prem_r.values,
            color=PALETTE[0], linewidth=2, marker="o", markersize=4,
        )
        ax3.set_xlabel("Número de primas pagadas")
        ax3.set_ylabel("Tasa de Renovación (%)")
        ax3.set_title("Relación entre Fidelidad (primas pagadas) y Renovación")
        ax3.axhline(y=renewal_rate, color=PALETTE[1],
                    linestyle="--", label=f"Media global ({renewal_rate:.1f}%)")
        ax3.legend()
        fig3.tight_layout()
        st.pyplot(fig3)

        st.markdown(
            """
            ### 🔍 Insights principales

            | # | Hallazgo |
            |---|---|
            | 1 | La **tasa global de renovación es 93.7%**, indicando alta fidelidad general de los clientes. |
            | 2 | La **morosidad es el factor más crítico**: con 3 o más pagos tardíos, la renovación cae a menos del 70%. |
            | 3 | El **canal A** tiene la mayor tasa de renovación (94.5%); el canal **D** la menor (91.6%). |
            | 4 | A mayor **número de primas pagadas**, mayor probabilidad de renovación (efecto lealtad). |
            | 5 | La diferencia entre áreas **urbanas y rurales** es mínima (~0.1%), por lo que no es un factor diferenciador relevante. |
            """
        )


# ═════════════════════════════════════════════════════════════════
# MÓDULO 4 – CONCLUSIONES
# ═════════════════════════════════════════════════════════════════
elif modulo == "📌 Conclusiones":
    st.title("📌 Conclusiones Finales")
    st.markdown(
        "Conclusiones basadas en el EDA del dataset **InsuranceCompany**, "
        "orientadas a la **toma de decisiones**, no a la predicción."
    )
    st.markdown("---")

    conclusiones = [
        (
            "1️⃣ La morosidad es el principal factor de riesgo de no-renovación",
            """
            Los clientes con **3 o más pagos tardíos** acumulados presentan una tasa de
            renovación inferior al 70%, frente al 97% de quienes no tienen retrasos.
            La compañía debería implementar un **sistema de alertas tempranas** que detecte
            el primer pago tardío y active acciones preventivas (recordatorios, planes de
            refinanciación) antes de que el comportamiento se agrave.
            """,
        ),
        (
            "2️⃣ El canal de captación define el perfil de fidelidad del cliente",
            """
            El **canal A** genera los clientes con mayor tasa de renovación (94.5%),
            probablemente porque provienen de referidos o asesoría directa con mayor
            vínculo emocional y comprensión del producto. El **canal D** es el de menor
            retención (91.6%). Se recomienda revisar los procesos de calificación y
            seguimiento en los canales de menor rendimiento.
            """,
        ),
        (
            "3️⃣ La antigüedad del cliente es un activo de retención",
            """
            A mayor número de primas pagadas, mayor es la probabilidad de renovar,
            evidenciando un **efecto de lealtad progresiva**. Los clientes con 15 o más
            primas pagadas superan el 96% de renovación. Esto valida diseñar
            **programas de fidelización** con beneficios escalonados por antigüedad.
            """,
        ),
        (
            "4️⃣ El puntaje de underwriting tiene alta concentración pero nulos relevantes",
            """
            El **`application_underwriting_score`** presenta distribución muy concentrada
            en valores altos (~99), lo que sugiere que la mayoría de clientes son de bajo
            riesgo. Sin embargo, tiene **2 974 valores nulos (3.7%)** que deben ser
            imputados o tratados antes de cualquier análisis predictivo. Este campo es
            clave para la segmentación de riesgo.
            """,
        ),
        (
            "5️⃣ El dataset es sólido para análisis avanzados",
            """
            Con **79 852 registros**, baja tasa de nulos (< 4%), sin duplicados evidentes
            y con variables numéricas y categóricas bien definidas, el dataset es apto para
            construir **modelos de scoring de retención** como regresión logística o árboles
            de decisión. La variable `renewal` está desbalanceada (93.7% / 6.3%), por lo que
            se deberá aplicar técnicas de balanceo en etapas predictivas futuras.
            """,
        ),
    ]

    for titulo, texto in conclusiones:
        with st.expander(titulo, expanded=True):
            st.markdown(texto)

    st.markdown("---")
    st.success(
        "✅ Este análisis exploratorio permite priorizar acciones sobre morosidad "
        "temprana, optimización de canales de captación y programas de lealtad para "
        "maximizar la tasa de renovación de pólizas."
    )
