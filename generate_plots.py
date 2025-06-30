import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# Cargo datos
print("Attempting to load full_experiment_results.csv...")
try:
    df = pd.read_csv("full_experiment_results.csv")
    print("CSV loaded successfully.")
    print("Shape of df:", df.shape)
    print("Head of df:\n", df.head())
    print("Columns in df:", df.columns.tolist())
    print("Data types of df columns:\n", df.dtypes)
except FileNotFoundError:
    print("ERROR: full_experiment_results.csv not found! Please run main.py to generate it.")
    exit()
except Exception as e:
    print(f"ERROR loading full_experiment_results.csv: {e}")
    exit()

# Filtrar sólo datos válidos de DOA promedio para el arreglo
print("\nFiltering df_array...")
df_array = df[(df['mic_pair'] == 'array_avg_adj_pairs') & df['doa_array_error_deg'].notna()].copy()
print("Shape of df_array:", df_array.shape)
print("Head of df_array:\n", df_array.head())

required_cols_for_conversion = ['actual_elevation_src_to_array_center_deg', 'actual_dist_src_to_array_center_m',
                                'mic_separation_m', 'num_mics_processed', 'rt60_target_s', 'fs_hz']
missing_cols = [col for col in required_cols_for_conversion if col not in df_array.columns]

if 'actual_elevation_src_to_array_center_deg' not in df_array.columns and 'elevation_angle_deg' in df_array.columns:
    print("Info: 'actual_elevation_src_to_array_center_deg' not found, renaming from 'elevation_angle_deg'.")
    df_array.rename(columns={'elevation_angle_deg': 'actual_elevation_src_to_array_center_deg'}, inplace=True)
    missing_cols = [col for col in required_cols_for_conversion if col not in df_array.columns]

if missing_cols:
    print(f"WARNING: The following required columns are missing from df_array: {missing_cols}")
    print("Available columns:", df_array.columns.tolist())

print("\nConverting columns to numeric...")
for col in required_cols_for_conversion:
    if col in df_array.columns:
        df_array.loc[:, col] = pd.to_numeric(df_array[col], errors='coerce')
        print(f"Converted {col} to numeric. NaN count: {df_array[col].isna().sum()}")
    else:
        print(f"Warning: Column {col} not found for numeric conversion.")

print("\nData types after conversion:\n", df_array.dtypes)
print("Head after conversion:\n", df_array.head())


# Función general para línea con promedio y error estándar
def plot_metric_vs_param(df_plot, param_col, title, xlabel):
    print(f"\nGenerating plot: {title}")
    if df_plot.empty:
        print(f"Skipping '{title}' because DataFrame is empty.")
        return
    if param_col not in df_plot.columns:
        print(f"Skipping '{title}' because column '{param_col}' is missing.")
        return
    if df_plot[param_col].isna().all():
        print(f"Skipping '{title}' because '{param_col}' is all NaNs.")
        return
    if 'doa_array_error_deg' not in df_plot.columns or df_plot['doa_array_error_deg'].isna().all():
        print(f"Skipping '{title}' because 'doa_array_error_deg' is missing or all NaNs.")
        return
    if 'tdoa_method_for_avg_doa' not in df_plot.columns or df_plot['tdoa_method_for_avg_doa'].isna().all():
        print(f"Warning: 'tdoa_method_for_avg_doa' missing or all NaN, plotting without hue.")
        sns.lineplot(
            data=df_plot,
            x=param_col,
            y='doa_array_error_deg',
            errorbar='sd',
            marker='o'
        )
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Error promedio DOA (grados)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        return

    plt.figure(figsize=(8,5))
    sns.lineplot(
        data=df_plot,
        x=param_col,
        y='doa_array_error_deg',
        hue='tdoa_method_for_avg_doa',
        errorbar='sd',
        marker='o'
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Error promedio DOA (grados)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# --- 1. Error vs Ángulo de elevación ---
plot_metric_vs_param(
    df_array,
    'actual_elevation_src_to_array_center_deg',
    "Error promedio DOA vs Ángulo de elevación",
    "Ángulo de elevación (grados)"
)

# --- 2. Error vs Distancia fuente - arreglo ---
plot_metric_vs_param(
    df_array,
    'actual_dist_src_to_array_center_m',
    "Error promedio DOA vs Distancia fuente-arreglo",
    "Distancia (m)"
)

# --- 3. Error vs Separación entre micrófonos ---
plot_metric_vs_param(
    df_array,
    'mic_separation_m',
    "Error promedio DOA vs Separación entre micrófonos",
    "Separación entre micrófonos (m)"
)

# --- 4. Error vs Cantidad de micrófonos ---
plot_metric_vs_param(
    df_array,
    'num_mics_processed',
    "Error promedio DOA vs Cantidad de micrófonos",
    "Cantidad de micrófonos"
)

# --- 5. Error vs Tiempo de reverberación RT60 ---
plot_metric_vs_param(
    df_array,
    'rt60_target_s',
    "Error promedio DOA vs Tiempo de reverberación RT60",
    "Tiempo RT60 (s)"
)

# --- 6. Error vs Frecuencia de muestreo ---
plot_metric_vs_param(
    df_array,
    'fs_hz',
    "Error promedio DOA vs Frecuencia de muestreo",
    "Frecuencia de muestreo (Hz)"
)

# --- 7. Error promedio DOA vs SNR para todas las frecuencias de muestreo ---
title_last_plot = "Error promedio DOA vs SNR para distintas frecuencias de muestreo"
print(f"\nGenerating plot: {title_last_plot}")
if df_array.empty:
    print(f"Skipping '{title_last_plot}' because df_array is empty.")
elif 'snr_db' not in df_array.columns or df_array['snr_db'].isna().all():
    print(f"Skipping '{title_last_plot}' because 'snr_db' is missing or all NaN.")
elif 'doa_array_error_deg' not in df_array.columns or df_array['doa_array_error_deg'].isna().all():
    print(f"Skipping '{title_last_plot}' because 'doa_array_error_deg' is missing or all NaN.")
elif 'fs_hz' not in df_array.columns or df_array['fs_hz'].isna().all():
    print(f"Skipping '{title_last_plot}' because 'fs_hz' is missing or all NaN.")
else:
    plt.figure(figsize=(8,5))
    sns.lineplot(
        data=df_array,
        x='snr_db',
        y='doa_array_error_deg',
        hue='fs_hz',
        errorbar='sd',
        marker='o',
        palette='viridis'
    )
    plt.title(title_last_plot)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Error promedio DOA (grados)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

print("\n--- Script generate_plots.py finished ---")
