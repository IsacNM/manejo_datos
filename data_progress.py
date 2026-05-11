# Repositorio: https://github.com/IsacNM/manejo_datos
import pandas as pd
import numpy as np
from scipy import stats

# CARGA DEL DATASET 
URL = "https://raw.githubusercontent.com/IsacNM/manejo_datos/refs/heads/main/pipol_datos.csv"
df = pd.read_csv(URL, index_col=[0])

# Renombrar columnas al inglés para facilitar su manejo
df.columns = ["age", "sex", "income", "height", "city", "education", "children"]

print("=" * 55)
print("  ANÁLISIS DE INTEGRIDAD – pipol_datos.csv")
print("=" * 55)
print(f"\nDimensiones del dataset: {df.shape[0]} filas × {df.shape[1]} columnas")
print("\nTipos de dato:\n", df.dtypes)


# VALORES NULOS 
def reporte_nulos(df):
    nulos = df.isnull().sum()
    pct   = (nulos / len(df) * 100).round(2)
    reporte = pd.DataFrame({"Nulos": nulos, "Porcentaje (%)": pct})
    reporte = reporte[reporte["Nulos"] > 0]
    print("\nValores Nulos: ")
    print(reporte.to_string())

reporte_nulos(df)


# 3. VALORES ÚNICOS CATEGÓRICOS 
def reporte_categoricos(df, columnas):
    print("\ Valores Únicos por Columna Categórica: ")
    for col in columnas:
        unicos = set(df[col].dropna().unique())
        print(f"  {col}: {sorted(unicos)}")

reporte_categoricos(df, ["sex", "city", "education"])


# DETECCIÓN DE OUTLIERS EN EDAD (Z-SCORE) 
def detectar_outliers_zscore(df, columna, umbral=3.0):
    """Devuelve índices con z-score superior al umbral."""
    serie = df[columna].dropna()
    z = np.abs(stats.zscore(serie))
    idx_outliers = serie[z > umbral].index
    print(f"\n Outliers en '{columna}' (|z| > {umbral}): ")
    print(f"  Cantidad : {len(idx_outliers)}")
    print(f"  Valores  : {sorted(df.loc[idx_outliers, columna].unique())}")
    return idx_outliers

idx_age_outliers = detectar_outliers_zscore(df, "age")


# DETECCIÓN DE NEGATIVOS 
def detectar_negativos(df, columnas):
    """Informa cuántos negativos hay en cada columna."""
    print("\n Valores Negativos: ")
    for col in columnas:
        neg = (df[col] < 0).sum()
        if neg > 0:
            print(f"  {col}: {neg} registros negativos")

detectar_negativos(df, ["age", "income", "children"])


# ESTADÍSTICAS DESCRIPTIVAS RELEVANTES 
print("\nEstadísticas descriptivas: ")
print(df[["age", "income", "height", "children"]].describe().round(2))


# FUNCIONES DE CORRECCIÓN

# Reemplazar negativos por 0 
def eliminar_negativos(df, columnas):
    """Reemplaza valores negativos por 0 en las columnas indicadas."""
    df = df.copy()
    for col in columnas:
        negativos = (df[col] < 0).sum()
        df[col] = df[col].apply(lambda x: 0 if x < 0 else x)
        print(f"  [F1] '{col}': {negativos} negativos → reemplazados por 0")
    return df


# Tratar outliers de edad por mediana (Z-score) 
def tratar_outliers_edad(df, columna="age", umbral=3.0):
    """Reemplaza edades con |z| > umbral por la mediana de la columna."""
    df = df.copy()
    serie = df[columna].dropna()
    z = np.abs(stats.zscore(serie))
    idx_out = serie[z > umbral].index
    mediana = serie.median()
    df.loc[idx_out, columna] = mediana
    print(f"\n  [F2] '{columna}': {len(idx_out)} outliers → reemplazados por mediana ({mediana})")
    return df


# Corregir faltas ortográficas en educación 
MAPA_EDUCACION = {
    "mastre"      : "Master",
    "pHd"         : "PhD",
    "Bachelors"   : "Bachelor",
    "no education": "No Education",
}

def corregir_educacion(df, columna="education", mapa=MAPA_EDUCACION):
    """Normaliza los valores de la columna de educación según el mapa."""
    df = df.copy()
    cambios = 0
    for incorrecto, correcto in mapa.items():
        mask = df[columna] == incorrecto
        cambios += mask.sum()
        df.loc[mask, columna] = correcto
    print(f"\n  [F3] '{columna}': {cambios} registros corregidos")
    print(f"       Valores únicos tras corrección: {sorted(df[columna].dropna().unique())}")
    return df


# Rellenar nulos en columnas categóricas 
def rellenar_nulos_moda(df, columnas):
    """Rellena NaN de columnas categóricas con la moda."""
    df = df.copy()
    for col in columnas:
        nulos = df[col].isnull().sum()
        if nulos > 0:
            moda = df[col].mode()[0]
            df[col].fillna(moda, inplace=True)
            print(f"  [F4] '{col}': {nulos} NaN → rellenados con moda ('{moda}')")
    return df


# APLICACIÓN DE CORRECCIONES
print("\n\n" + "=" * 55)
print("  APLICANDO CORRECCIONES")
print("=" * 55)

df_clean = df.copy()
df_clean = eliminar_negativos(df_clean, ["income", "children"])
df_clean = tratar_outliers_edad(df_clean, "age")
df_clean = corregir_educacion(df_clean)
df_clean = rellenar_nulos_moda(df_clean, ["sex", "city"])

print("\n── Estado final tras limpieza ─────────────────────")
print(f"  Nulos restantes:\n{df_clean.isnull().sum()}")
print(f"\n  Estadísticas age, income, children:")
print(df_clean[["age", "income", "children"]].describe().round(2))

print("\n✔  Pipeline de limpieza completado.")
