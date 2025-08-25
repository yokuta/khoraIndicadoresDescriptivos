import pandas as pd
import streamlit as st
import io
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import zipfile
import tempfile
import os
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import streamlit as st
from sqlalchemy import create_engine
import psycopg2


# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Indicadores INE por Municipio",
    page_icon="🌆",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------- DATABASE CONNECTION --------------------
@st.cache_resource
def get_db_connection():
    """Create database connection"""
    try:
        db_url = st.secrets["postgres"]["db_url"]
        engine = create_engine(db_url)
        return engine
    except Exception as e:
        st.error(f"❌ Error conectando a la base de datos: {e}")
        st.stop()

# -------------------- GEOSPATIAL FUNCTIONS --------------------
def process_shapefile(uploaded_file):
    """Process uploaded shapefile (zip) and return GeoDataFrame"""
    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract the zip file
            with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Find the .shp file
            shp_file = None
            for file in os.listdir(temp_dir):
                if file.endswith('.shp'):
                    shp_file = os.path.join(temp_dir, file)
                    break
            
            if shp_file is None:
                st.error("❌ No se encontró archivo .shp en el ZIP")
                return None
            
            # Read the shapefile
            gdf = gpd.read_file(shp_file)
            return gdf
            
    except Exception as e:
        st.error(f"❌ Error procesando shapefile: {str(e)}")
        return None

def process_geojson(uploaded_file):
    """Process uploaded GeoJSON file and return GeoDataFrame"""
    try:
        gdf = gpd.read_file(uploaded_file)
        return gdf
    except Exception as e:
        st.error(f"❌ Error procesando GeoJSON: {str(e)}")
        return None

def display_geodata_info(gdf, filename):
    """Display information about the GeoDataFrame"""
    st.success(f"✅ Datos geoespaciales cargados: **{filename}**")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Geometrías", len(gdf))
    with col2:
        st.metric("Columnas", len(gdf.columns))
    with col3:
        st.metric("CRS", str(gdf.crs) if gdf.crs else "No definido")
    with col4:
        geom_types = gdf.geometry.geom_type.unique()
        st.metric("Tipo geometría", ", ".join(geom_types))
    
    # Show attribute table - remove ALL geometry-related columns
    st.subheader("📋 Tabla de Atributos")
    display_df = gdf.copy()
    
    # Remove all potential geometry columns
    geom_cols_to_remove = ['geometry', 'geom', 'geom_wkt']
    for col in geom_cols_to_remove:
        if col in display_df.columns:
            display_df = display_df.drop(columns=[col])
    
    st.dataframe(display_df.head(10), use_container_width=True)
    
    # Show bounds
    bounds = gdf.total_bounds
    st.subheader("🗺️ Extensión Geográfica")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Min X (Oeste):** {bounds[0]:.6f}")
        st.write(f"**Min Y (Sur):** {bounds[1]:.6f}")
    with col2:
        st.write(f"**Max X (Este):** {bounds[2]:.6f}")
        st.write(f"**Max Y (Norte):** {bounds[3]:.6f}")

def create_folium_map(gdf, map_title="Mapa"):
    """Create a Folium map from GeoDataFrame"""
    # Ensure CRS is WGS84 for web mapping
    if gdf.crs != 'EPSG:4326':
        gdf_web = gdf.to_crs('EPSG:4326')
    else:
        gdf_web = gdf.copy()
    
    # Calculate center
    bounds = gdf_web.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8,
        tiles='OpenStreetMap'
    )
    
    # Add GeoDataFrame to map
    folium.GeoJson(
        gdf_web.__geo_interface__,
        style_function=lambda feature: {
            'fillColor': 'blue',
            'color': 'black',
            'weight': 2,
            'fillOpacity': 0.3,
        },
        popup=folium.GeoJsonPopup(
            fields=[col for col in gdf_web.columns if col != 'geometry']
        )

    ).add_to(m)
    
    return m

def perform_spatial_clip(gdf_data, gdf_clip):
    """Perform spatial clipping operation and recalculate area"""
    try:
        # Ensure both GDFs have valid CRS
        if gdf_data.crs is None:
            gdf_data.set_crs("EPSG:25830", inplace=True)
        if gdf_clip.crs is None:
            gdf_clip.set_crs("EPSG:25830", inplace=True)
            
        # Reproject to match
        if gdf_data.crs != gdf_clip.crs:
            gdf_data = gdf_data.to_crs(gdf_clip.crs)

        # Perform clip
        clipped_gdf = gpd.clip(gdf_data, gdf_clip)

        if clipped_gdf.empty:
            return None

        # Recalculate area using WGS84 (like your working code)
        clipped_wgs84 = clipped_gdf.to_crs(epsg=4326)
        area_m2 = calculate_ellipsoidal_area(clipped_wgs84)
        clipped_gdf["area_m2"] = area_m2
        clipped_gdf["area_ha"] = [a / 10000 for a in area_m2]
        clipped_gdf["estal"] = clipped_gdf["area_ha"]  # Keep compatibility with your code

        return clipped_gdf

    except Exception as e:
        st.error(f"❌ Error en operación de recorte: {str(e)}")
        return None

def export_geodata(gdf, filename_base, format_type):
    """Export GeoDataFrame to different formats"""
    try:
        if format_type == "GeoJSON":
            geojson_str = gdf.to_json()
            return geojson_str, f"{filename_base}.geojson", "application/json"
        
        elif format_type == "Shapefile":
            # Create a temporary directory and zip file
            with tempfile.TemporaryDirectory() as temp_dir:
                shp_path = os.path.join(temp_dir, f"{filename_base}.shp")
                gdf.to_file(shp_path)
                
                # Create zip file
                zip_path = os.path.join(temp_dir, f"{filename_base}_shapefile.zip")
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for file in os.listdir(temp_dir):
                        if file.startswith(filename_base) and not file.endswith('.zip'):
                            zipf.write(os.path.join(temp_dir, file), file)
                
                # Read zip file as bytes
                with open(zip_path, 'rb') as f:
                    zip_data = f.read()
                
                return zip_data, f"{filename_base}_shapefile.zip", "application/zip"
        
        elif format_type == "CSV":
            # Convert to regular DataFrame (lose geometry)
            df = pd.DataFrame(gdf.drop(columns=['geometry']))
            csv_data = df.to_csv(index=False)
            return csv_data, f"{filename_base}.csv", "text/csv"
            
    except Exception as e:
        st.error(f"❌ Error exportando datos: {str(e)}")
        return None, None, None

def display_file_info(uploaded_file, df):
    """Display information about the uploaded file"""
    st.success(f"✅ Archivo cargado: **{uploaded_file.name}**")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Filas", len(df))
    with col2:
        st.metric("Columnas", len(df.columns))
    with col3:
        st.metric("Tamaño", f"{uploaded_file.size / 1024:.1f} KB")
    
    # Show basic info about the dataset
    st.subheader("📋 Información del Dataset")
    st.write("**Columnas:**")
    st.write(", ".join(df.columns.tolist()))
    
    st.write("**Primeras 5 filas:**")
    st.dataframe(df.head(), use_container_width=True)
    
    # Data types
    st.write("**Tipos de datos:**")
    dtype_df = pd.DataFrame({
        'Columna': df.dtypes.index,
        'Tipo': df.dtypes.values
    })
    st.dataframe(dtype_df, use_container_width=True, hide_index=True)
    
    # Basic statistics for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        st.write("**Estadísticas básicas (columnas numéricas):**")
        st.dataframe(df[numeric_cols].describe(), use_container_width=True)
    
    # Check for missing values
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        st.write("**Valores faltantes:**")
        missing_df = pd.DataFrame({
            'Columna': missing_data.index,
            'Valores faltantes': missing_data.values,
            'Porcentaje': (missing_data.values / len(df) * 100).round(2)
        })
        missing_df = missing_df[missing_df['Valores faltantes'] > 0]
        st.dataframe(missing_df, use_container_width=True, hide_index=True)

def process_uploaded_file(uploaded_file):
    """Process the uploaded file and return a DataFrame"""
    try:
        if uploaded_file.name.endswith('.csv'):
            # Try different encodings for CSV
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    uploaded_file.seek(0)  # Reset file pointer
                    df = pd.read_csv(uploaded_file, encoding='latin-1')
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding='cp1252')
        
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        
        elif uploaded_file.name.endswith('.json'):
            df = pd.read_json(uploaded_file)
        
        elif uploaded_file.name.endswith('.parquet'):
            df = pd.read_parquet(uploaded_file)
        
        else:
            st.error("❌ Formato de archivo no soportado. Use CSV, Excel, JSON o Parquet.")
            return None
            
        return df
        
    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: {str(e)}")
        return None

from pyproj import Geod

def calculate_ellipsoidal_area(gdf):
    """Calculate ellipsoidal area (like QGIS $area) in m² using WGS84"""
    geod = Geod(ellps="WGS84")

    areas = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            areas.append(0)
        else:
            if geom.geom_type == "Polygon":
                area, _ = geod.geometry_area_perimeter(geom)
            elif geom.geom_type == "MultiPolygon":
                area = sum(geod.geometry_area_perimeter(p)[0] for p in geom.geoms)
            else:
                area = 0
            areas.append(abs(area))  # Ensure positive

    return areas

# -------------------- LOAD DATASETS --------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_parquet("structured_population.parquet")
        df.columns = df.columns.astype(str)
        df_censo = pd.read_parquet("structured_censo.parquet")
        df_hog_2011 = pd.read_parquet("structured_censo2011_hogares.parquet")
        df_hog_2021 = pd.read_parquet("structured_censo2021_hogares.parquet")
        df_censo2011 = pd.read_parquet("structured_censo2011_viviendas.parquet")
        df_dgt = pd.read_parquet("dgt2023.parquet")
        df_dgt["municipio_completo"] = df_dgt["Código INE"].astype(str).str.zfill(5) + " " + df_dgt["Municipio"]
        return df, df_censo, df_hog_2011, df_hog_2021, df_censo2011, df_dgt
    except Exception as e:
        st.error(f"❌ No se pudieron cargar los archivos Parquet: {e}")
        return None, None, None, None, None, None

@st.cache_data
def load_internal_bases_all_codsiu(selected_muni):
    """Carga todos los CODSIU (1-20) para un municipio"""
    try:
        engine = get_db_connection()
        query = """
            SELECT * 
            FROM dev_codeine.siu_siose_with_municipalities
            WHERE municipality ILIKE %(municipality)s
              AND "CODSIU" BETWEEN 1 AND 20
        """
        with engine.connect() as conn:
            gdf_all = gpd.read_postgis(query, conn, geom_col="geom", params={
                "municipality": f"%{selected_muni}%"
            })
        return gdf_all
    except Exception as e:
        st.error(f"❌ Error cargando capas base desde PostgreSQL: {e}")
        return None



from pathlib import Path

@st.cache_data
def load_municipio_geojson_by_code(municipio, df):
    """Carga GeoJSON usando el código INE del municipio"""
    try:
        code_ine = df[df["municipio"] == municipio]["municipio"].astype(str).str.zfill(5).values[0]
    except IndexError:
        st.warning(f"No se encontró código INE para el municipio '{municipio}'")
        return None

    # Buscar el archivo que empieza por ese código
    folder = Path("geojson_municipios")
    matching_files = list(folder.glob(f"{code_ine}*.geojson"))

    if not matching_files:
        st.warning(f"⚠️ No se encontró un GeoJSON para el código INE {code_ine}")
        return None

    try:
        return gpd.read_file(matching_files[0])
    except Exception as e:
        st.warning(f"⚠️ Error leyendo GeoJSON de {municipio}: {e}")
        return None

# -------------------- MAIN APP --------------------
st.title("📊 Indicadores INE por Municipio")

# Add tabs for different functionalities
# Only one tab: Análisis INE
tab1 = st.container()


with tab1:
    st.markdown("---")
    
    # Load original data
    data_loaded = load_data()
    if all(d is not None for d in data_loaded):
        df, df_censo, df_hog_2011, df_hog_2021, df_censo2011, df_dgt = data_loaded
    else:
        st.error("❌ No se pudieron cargar los datos base del INE")
        st.stop()

    # Constants
    YEARS = ["2024", "2023", "2022", "2021"]
    age_65_plus = ["65_69", "70_74", "75_79", "80_84", "85_89", "90_94", "95_99", "100"]
    age_85_plus = ["85_89", "90_94", "95_99", "100"]
    ages_0_14 = ["0_4", "5_9", "10_14"]
    ages_15_64 = ["15_19", "20_24", "25_29", "30_34", "35_39", "40_44", "45_49", "50_54", "55_59", "60_64"]

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 🏨️ Selección de Municipio")
        municipalities = sorted(df["municipio"].dropna().unique(), key=str.lower)
        search_term = st.text_input("🔍 Buscar municipio:", placeholder="Escribe para buscar un municipio...")

        if search_term:
            filtered_municipalities = [m for m in municipalities if search_term.lower() in m.lower()]
            if filtered_municipalities:
                selected_muni = st.selectbox("Municipios encontrados:", filtered_municipalities, index=None)
            else:
                st.warning("❌ No se encontraron municipios que coincidan con tu búsqueda.")
                selected_muni = None
        else:
            selected_muni = st.selectbox("O selecciona directamente:", municipalities, index=None)

    with col2:
        if selected_muni:
            st.markdown("### ℹ️ Información Eje")
            st.info(f"**Municipio seleccionado:**\n{selected_muni}")
            try:
                total_pop_2024 = df[df["municipio"] == selected_muni]["total_total_total_2024"].values[0]
                st.metric("Población Total 2024", f"{total_pop_2024:,}" if total_pop_2024 else "No disponible")
            except:
                pass

    if selected_muni:
        st.markdown("---")
        pop_df = df[df["municipio"] == selected_muni]
        with st.spinner("🔍 Procesando capa geográfica del municipio..."):
            gdf_muni = load_municipio_geojson_by_code(selected_muni, df)
            municipio_area_ha = sum(calculate_ellipsoidal_area(gdf_muni.to_crs(4326))) / 10000
            st.write(f"🟫 Superficie del municipio: {municipio_area_ha:,.2f} ha")

            gdf_all_codsiu = load_internal_bases_all_codsiu(selected_muni)

            # Recorte general para todos los CODSIU
            if gdf_all_codsiu is not None:
                gdf_all_clipped = perform_spatial_clip(gdf_all_codsiu, gdf_muni)
                if gdf_all_clipped is not None and not gdf_all_clipped.empty:
                    gdf_all_clipped["CODSIU"] = gdf_all_clipped["CODSIU"].astype(int)
                    for cod in [9, 12, 14, 16, 18]:                        
                        subset = gdf_all_clipped[gdf_all_clipped["CODSIU"] == cod]
                        if not subset.empty:
                            estal_sum = subset["estal"].sum()
                            st.session_state[f"sup_cultivos_{cod:02d}"] = estal_sum
                        else:
                            st.session_state[f"sup_cultivos_{cod:02d}"] = None                                
            else:
                st.info("ℹ️ No se encontró geometría para este municipio o falló la carga.")

        if pop_df.empty:
            st.error("❌ No se encontraron datos para el municipio seleccionado.")
            st.stop()

        censo_df = df_censo[df_censo["Municipio de residencia"].str.contains(selected_muni, case=False, na=False)]

        # Vivienda/hogar histórico
        hog_2011 = df_hog_2011[df_hog_2011["municipio"].str.contains(selected_muni, case=False, na=False)]
        hog_2021 = df_hog_2021[df_hog_2021["municipio"].str.contains(selected_muni, case=False, na=False)]
        viv_2011 = df_censo2011[df_censo2011["Municipio de residencia"].str.contains(selected_muni, case=False, na=False)]

        try:
            n_hog_2011 = hog_2011["nHogares"].values[0]
            n_hog_2021 = hog_2021["nHogares"].values[0]
            var_hogares_pct = round((n_hog_2021 - n_hog_2011) / n_hog_2011 * 100, 2)
        except:
            var_hogares_pct = None

        try:
            n_viv_2011 = viv_2011["viviendasTotal"].values[0]
            n_viv_2021 = censo_df["viviendasT"].values[0]
            crecimiento_viviendas_pct = round((n_viv_2021 - n_viv_2011) / n_viv_2011 * 100, 2)
        except:
            crecimiento_viviendas_pct = None

        try:
            n_viv_vacias_2011 = viv_2011["viviendasVacias"].values[0]
            viv_vacia_pct_2011 = round(n_viv_vacias_2011 / n_viv_2011 * 100, 2)
        except:
            viv_vacia_pct_2011 = None

        # Vehículos DGT
        df_dgt["municipio_completo"] = df_dgt["Código INE"].astype(str).str.zfill(5) + " " + df_dgt["Municipio"]
        dgt_row = df_dgt[df_dgt["municipio_completo"].str.lower() == selected_muni.lower()]

        try:
            turismos = dgt_row["Parque Turismos"].values[0]
            motos = dgt_row["Parque Motocicletas"].values[0]
            total_veh = dgt_row["Parque Total"].values[0]
            pop_2024 = pop_df["total_total_total_2024"].values[0]

            veh_1000hab = round((turismos + motos) / pop_2024 * 1000, 2) if pop_2024 else None
            pct_turismos = round(turismos / total_veh * 100, 2) if total_veh else None
            pct_motos = round(motos / total_veh * 100, 2) if total_veh else None
        except:
            veh_1000hab = None
            pct_turismos = None
            pct_motos = None

        # Result table
        results = []

        # -------------------- CALCULATE POPULATION VARIATION FOR EACH YEAR --------------------
        pop_variation_dict = {}

        try:
            hist_df_raw = pd.read_parquet("population/poblacion_completa.parquet")
            hist_df_raw.rename(columns={hist_df_raw.columns[0]: "municipio"}, inplace=True)
            hist_row = hist_df_raw[hist_df_raw["municipio"].str.contains(selected_muni, case=False, na=False)]

            if not hist_row.empty:
                hist_row = hist_row.iloc[0]

                def clean_series(series):
                    return pd.to_numeric(series.replace(r"^\s*$", pd.NA, regex=True), errors="coerce")

                pop_t = clean_series(hist_row.filter(like="_t")).dropna()
                pop_years = [int(col.split("_")[0]) for col in pop_t.index]
                pop_series = pd.Series(pop_t.values, index=pop_years).sort_index()

                for year in YEARS:
                    y = int(year)
                    if y in pop_series.index and (y - 10) in pop_series.index:
                        base = pop_series[y - 10]
                        current = pop_series[y]
                        pct = round((current - base) / base * 100, 2) if base else None
                        pop_variation_dict[year] = pct
                    else:
                        pop_variation_dict[year] = None
                    

        except:
            for year in YEARS:
                pop_variation_dict[year] = None

        for year in YEARS:
            total = pop_df.get(f"total_total_total_{year}", pd.Series([0])).values[0]
            over_65 = pop_df[[f"total_{age}_total_{year}" for age in age_65_plus if f"total_{age}_total_{year}" in pop_df.columns]].sum(axis=1).values[0]
            over_85 = pop_df[[f"total_{age}_total_{year}" for age in age_85_plus if f"total_{age}_total_{year}" in pop_df.columns]].sum(axis=1).values[0]
            foreign = pop_df.get(f"total_total_EX_{year}", pd.Series([0])).values[0]
            pop_0_14 = pop_df[[f"total_{age}_total_{year}" for age in ages_0_14 if f"total_{age}_total_{year}" in pop_df.columns]].sum(axis=1).values[0]
            pop_15_64 = pop_df[[f"total_{age}_total_{year}" for age in ages_15_64 if f"total_{age}_total_{year}" in pop_df.columns]].sum(axis=1).values[0]

            row = {
                "Año": year,
                "Variación Poblacional Últimos 10 años (%)": pop_variation_dict.get(year),
                "D.22.a. Envejecimiento (%)": round(over_65 / total * 100, 2) if total else None,
                "D.22.b. Senectud (%)": round(over_85 / over_65 * 100, 2) if over_65 else None,
                "Población extranjera (%)": round(foreign / total * 100, 2) if total else None,
                "D.24.a. Dependencia total (%)": round((pop_0_14 + over_65) / pop_15_64 * 100, 2) if pop_15_64 else None,
                "D.24.b. Dependencia infantil (%)": round(pop_0_14 / pop_15_64 * 100, 2) if pop_15_64 else None,
                "D.24.c. Dependencia mayores (%)": round(over_65 / pop_15_64 * 100, 2) if pop_15_64 else None,
                "%Vivienda secundaria": None,
                "D.25 Viviendas por persona": None,
                "VARIACIÓN HOGARES 2011-2021 (%)": var_hogares_pct if year == "2021" else None,
                "CRECIMIENTO PARQUE VIVIENDAS 2011-2021 (%)": crecimiento_viviendas_pct if year == "2021" else None,
                "VIVIENDA VACÍA 2011 (%)": viv_vacia_pct_2011 if year == "2021" else None,
                "D.18.a. Vehículos domiciliados cada 1000 hab.": veh_1000hab if year == "2024" else None,
                "D.18.b. % Turismos": pct_turismos if year == "2023" else None,
                "D.18.c. % Motocicletas": pct_motos if year == "2023" else None
            }

            if (
                "sup_cultivos_09" in st.session_state and 
                st.session_state["sup_cultivos_09"] is not None and
                total
            ):
                verde_1000hab = round(st.session_state["sup_cultivos_09"] / (total / 1000), 2)
                row["SUPERFICIE VERDE (ha cada 1.000 hab)"] = verde_1000hab
            else:
                row["SUPERFICIE VERDE (ha cada 1.000 hab)"] = None

            # Indicador nuevo: Superficie cultivos código14 / superficie municipio
            try:
                if (
                    "sup_cultivos_14" in st.session_state and 
                    st.session_state["sup_cultivos_14"] is not None and
                    gdf_muni is not None and 
                    not gdf_muni.empty
                ):
                    muni_area_ha = sum(calculate_ellipsoidal_area(gdf_muni)) / 10000  # ha
                    if muni_area_ha > 0:
                        sup14_pct = round((st.session_state["sup_cultivos_14"] / muni_area_ha) * 100, 2)
                        row["% Cultivos Código 14 / Sup. Municipio"] = sup14_pct
                    else:
                        row["% Cultivos Código 14 / Sup. Municipio"] = None
                else:
                    row["% Cultivos Código 14 / Sup. Municipio"] = None
            except:
                row["% Cultivos Código 14 / Sup. Municipio"] = None

            # Indicadores para código 12
            try:
                if (
                    "sup_cultivos_12" in st.session_state and 
                    st.session_state["sup_cultivos_12"] is not None and
                    gdf_muni is not None and 
                    not gdf_muni.empty
                ):
                    # Indicador 1: solo la superficie
                    row["Superficie Cultivos (cod: 12) ha"] = round(st.session_state["sup_cultivos_12"], 2)
            
                    # Indicador 2: porcentaje sobre sup. municipal
                    muni_area_ha = sum(calculate_ellipsoidal_area(gdf_muni)) / 10000  # ha
                    if muni_area_ha > 0:
                        pct12 = round((st.session_state["sup_cultivos_12"] / muni_area_ha) * 100, 2)
                        row["% Cultivos Código 12 / Sup. Municipio"] = pct12
                    else:
                        row["% Cultivos Código 12 / Sup. Municipio"] = None
                else:
                    row["Superficie Cultivos (cod: 12) ha"] = None
                    row["% Cultivos Código 12 / Sup. Municipio"] = None
            except:
                row["Superficie Cultivos (cod: 12) ha"] = None
                row["% Cultivos Código 12 / Sup. Municipio"] = None
            
            if year == "2021":
                try:
                    v_total = censo_df["viviendasT"].values[0]
                    v_nop = censo_df["viviendasNoP"].values[0]
                    pop_2021 = pop_df["total_total_total_2021"].values[0]
                    row["%Vivienda secundaria"] = round((v_nop / v_total) * 100, 2)
                    row["D.25 Viviendas por persona"] = round((v_total / pop_2021) * 1000, 4)
                except:
                    pass
            # Indicador nuevo: Superficie cultivos código16 / superficie municipio
            try:
                if (
                    "sup_cultivos_16" in st.session_state and 
                    st.session_state["sup_cultivos_16"] is not None and
                    gdf_muni is not None and 
                    not gdf_muni.empty
                ):
                    muni_area_ha = sum(calculate_ellipsoidal_area(gdf_muni)) / 10000  # ha
                    if muni_area_ha > 0:
                        pct16 = round((st.session_state["sup_cultivos_16"] / muni_area_ha) * 100, 2)
                        row["% Cultivos Código 16 / Sup. Municipio"] = pct16
                    else:
                        row["% Cultivos Código 16 / Sup. Municipio"] = None
                else:
                    row["% Cultivos Código 16 / Sup. Municipio"] = None
            except:
                row["% Cultivos Código 16 / Sup. Municipio"] = None

            # Indicador nuevo: Superficie cultivos código16 / superficie municipio
            try:
                if (
                    "sup_cultivos_18" in st.session_state and 
                    st.session_state["sup_cultivos_18"] is not None and
                    gdf_muni is not None and 
                    not gdf_muni.empty
                ):
                    muni_area_ha = sum(calculate_ellipsoidal_area(gdf_muni)) / 10000  # ha
                    if muni_area_ha > 0:
                        pct16 = round((st.session_state["sup_cultivos_18"] / muni_area_ha) * 100, 2)
                        row["% Cultivos Código 18 / Sup. Municipio"] = pct16
                    else:
                        row["% Cultivos Código 18 / Sup. Municipio"] = None
                else:
                    row["% Cultivos Código 18 / Sup. Municipio"] = None
            except:
                row["% Cultivos Código 18 / Sup. Municipio"] = None

            results.append(row)

        results_df = pd.DataFrame(results)

        st.markdown(f"### 📈 Indicadores para **{selected_muni}**")

        # Wider table, narrow map, with spacing to push the map to the right
        col1, spacer, col2 = st.columns([3.5, 0.1, 0.8])
        
        with col1:
            if not results_df.empty:
                st.dataframe(results_df, use_container_width=True, hide_index=True)
            else:
                st.info("No hay datos disponibles para mostrar.")

        with col2:
            show_map = st.toggle("🗺️ Mostrar/Ocultar Mapa del Municipio", value=False)
      

        if selected_muni and gdf_muni is not None and show_map:
            st.markdown("### 🗺️ Mapa del Municipio con Recortes SIU")

            gdf_muni_4326 = gdf_muni.to_crs(4326)
            bounds = gdf_muni_4326.total_bounds
            center_lat = (bounds[1] + bounds[3]) / 2
            center_lon = (bounds[0] + bounds[2]) / 2

            m = folium.Map(location=[center_lat, center_lon], zoom_start=11)

            # Capa del municipio (rojo)
            folium.GeoJson(
                gdf_muni_4326.__geo_interface__,
                name="Municipio",
                style_function=lambda feature: {
                    'fillColor': 'red',
                    'color': 'red',
                    'weight': 2,
                    'fillOpacity': 0.1,
                },
                tooltip="Municipio"
            ).add_to(m)

            import matplotlib

            if gdf_all_clipped is not None and not gdf_all_clipped.empty:
                gdf_all_clipped = gdf_all_clipped.to_crs(4326)
                
                # Generar paleta de colores
                unique_codsiu = sorted(gdf_all_clipped["CODSIU"].unique())
                color_map = plt.get_cmap('tab20', len(unique_codsiu))  # 20 colores
                codsiu_to_color = {
                    cod: matplotlib.colors.rgb2hex(color_map(i)) for i, cod in enumerate(unique_codsiu)
                }
            
                for codsiu in unique_codsiu:
                    subset = gdf_all_clipped[gdf_all_clipped["CODSIU"] == codsiu]
                    folium.GeoJson(
                        subset.__geo_interface__,
                        name=f"CODSIU {codsiu}",
                        style_function=lambda feature, color=codsiu_to_color[codsiu]: {
                            'fillColor': color,
                            'color': color,
                            'weight': 1,
                            'fillOpacity': 0.4,
                        },
                        tooltip=folium.GeoJsonTooltip(
                            fields=["CODSIU", "descripcion", "municipality"],
                            aliases=["Código SIU:", "Descripción:", "Municipio:"],
                            localize=True,
                            sticky=True,
                            labels=True,
                            style="""
                                background-color: white;
                                border: 1px solid #ccc;
                                border-radius: 3px;
                                box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
                                font-size: 10px;
                                padding: 4px;
                            """
                        )

                    ).add_to(m)
            

            st_folium(m, width=1500, height=500, key="map_municipio_expandido")

        
        # -------------------- HISTORICAL POPULATION GRAPH --------------------
        try:
            hist_df_raw = pd.read_parquet("population/poblacion_completa.parquet")
            hist_df_raw.rename(columns={hist_df_raw.columns[0]: "municipio"}, inplace=True)
            hist_row = hist_df_raw[hist_df_raw["municipio"].str.contains(selected_muni, case=False, na=False)]
            if not hist_row.empty:
                hist_row = hist_row.iloc[0]

                def clean_series(series):
                    return pd.to_numeric(series.replace(r"^\s*$", pd.NA, regex=True), errors="coerce")

                pop_t = clean_series(hist_row.filter(like="_t")).dropna()
                pop_h = clean_series(hist_row.filter(like="_h")).dropna()
                pop_m = clean_series(hist_row.filter(like="_m")).dropna()

                def extract_years(series):
                    return [int(col.split("_")[0]) for col in series.index]

                years = extract_years(pop_t)

                hist_df = pd.DataFrame({
                    "Año": years,
                    "Total": pop_t.values,
                    "Hombres": pop_h.values if len(pop_h) else [None] * len(years),
                    "Mujeres": pop_m.values if len(pop_m) else [None] * len(years)
                }).sort_values("Año")

                st.markdown("### 📉 Evolución Histórica de la Población")
                st.line_chart(hist_df.set_index("Año"))

            else:
                st.warning("⚠️ No hay datos históricos disponibles para este municipio.")

        except Exception as e:
            st.error(f"❌ Error cargando datos históricos de población: {e}")

        st.markdown("---")
        col1, col2 = st.columns([1, 1])
        with col1:
            csv = results_df.to_csv(index=False)
            st.download_button("📥 Descargar CSV", csv, f"indicadores_{selected_muni.replace(' ', '_')}.csv", "text/csv")
        with col2:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                results_df.to_excel(writer, index=False, sheet_name="Indicadores")
            st.download_button("📊 Descargar Excel", buffer.getvalue(), f"indicadores_{selected_muni.replace(' ', '_')}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.markdown("---")
        st.info("👆 **Instrucciones:**\n1. Usa el cuadro de búsqueda para encontrar un municipio\n2. O selecciona directamente de la lista desplegable\n3. Los indicadores se mostrarán automáticamente")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Municipios", len(municipalities))
        with col2:
            st.metric("Años de Datos", len(YEARS))
        with col3:
            st.metric("Indicadores", "15")

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
        📊 Aplicación de Indicadores INE por Municipio<br>
        Datos del Instituto Nacional de Estadística (INE)
    </div>
    """, unsafe_allow_html=True)