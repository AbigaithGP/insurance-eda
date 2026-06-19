# 🛡️ InsuranceCompany EDA — Streamlit App

Aplicación interactiva de **Análisis Exploratorio de Datos (EDA)** sobre el dataset `InsuranceCompany.csv`, desarrollada como proyecto final de la **Especialización Python for Analytics**.

---

## 📋 Descripción del proyecto

El objetivo es identificar los factores que influyen en la **renovación de pólizas de seguro** mediante análisis exploratorio, visualización y estadística descriptiva, sin construcción de modelos predictivos.

La herramienta está construida con **Python + Streamlit** y organizada en módulos navegables:

| Módulo | Contenido |
|--------|-----------|
| 🏠 Home | Presentación del proyecto y tecnologías |
| 📤 Carga del Dataset | Uploader, vista previa y dimensiones |
| 🔬 EDA | 10 ítems de análisis con tabs interactivos |
| 📌 Conclusiones | 5 conclusiones orientadas a la toma de decisiones |

---

## 🧰 Tecnologías utilizadas

- **Python 3.11**
- **Pandas** — manipulación de datos
- **NumPy** — cálculo numérico
- **Matplotlib / Seaborn** — visualización
- **Streamlit** — interfaz interactiva

---

## 🚀 Instrucciones de ejecución

```bash
# 1. Clona el repositorio
git clone https://github.com/tu-usuario/insurance-eda.git
cd insurance-eda

# 2. Instala dependencias
pip install -r requirements.txt

# 3. Ejecuta la aplicación
streamlit run app.py
```

Luego abre tu navegador en `http://localhost:8501`.

---

## 📂 Estructura del repositorio

```
insurance-eda/
├── app.py                  # Aplicación principal
├── requirements.txt        # Dependencias
├── InsuranceCompany.csv    # Dataset
└── README.md               # Este archivo
```

---

## 🔗 Links relevantes

- 📁 Repositorio GitHub: _[enlace aquí]_
- 🌐 App desplegada en Streamlit Cloud: _[enlace aquí]_

---

## 📊 Capturas de la app

> _(Añadir capturas de pantalla de cada módulo tras el despliegue)_

---

## 📌 Conclusiones principales

1. La **morosidad acumulada** es el factor más asociado a la no-renovación.
2. Los **canales A y B** generan los clientes con mayor tasa de renovación.
3. A mayor **número de primas pagadas**, mayor propensión a renovar.
4. La base de clientes es **heterogénea** en ingreso y prima.
5. El dataset está **limpio y completo**, listo para modelos avanzados.

---

*Especialización Python for Analytics · 2025*
