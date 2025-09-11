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
from sqlalchemy import create_engine
import unicodedata, re, numpy as np
# --- DEBUG ARRANQUE: mostrar trazas en pantalla y logs ---
import streamlit as st
import traceback, sys, os
st.set_page_config(page_title="Indicadores Khôra", layout="wide")
import contextlib 

def main():
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


    def compute_d03b_municipal(selected_muni: str, gdf_muni: gpd.GeoDataFrame, gdf_all_codsiu: gpd.GeoDataFrame):
        """
        Calcula D.03b a nivel del término municipal seleccionado:
          - Numerador: área ha de (CODSIU=14 ∩ municipio)
          - Denominador: área ha de (clase de suelo urbana/urbanizable ∩ municipio)
        Devuelve dict con áreas y el indicador, más los GeoDataFrames recortados.
        """
        if gdf_muni is None or gdf_muni.empty:
            return None
        if gdf_all_codsiu is None or gdf_all_codsiu.empty:
            st.warning("No hay capa SIU/SIOSE cargada para el municipio.")
            return None

        crs_metric = "EPSG:25830"

        # Asegura CRS métrico
        for g in (gdf_muni, gdf_all_codsiu):
            if g.crs is None:
                g.set_crs(crs_metric, inplace=True)
            elif g.crs.to_string() != crs_metric:
                g.to_crs(crs_metric, inplace=True)

        # --- Numerador: CODSIU = 14, recortado al municipio ---
        siu14 = gdf_all_codsiu[gdf_all_codsiu["CODSIU"].astype(int) == 14].copy()
        if siu14.empty:
            area_cod14_ha = 0.0
            siu14_clip = siu14
        else:
            if siu14.crs.to_string() != crs_metric:
                siu14 = siu14.to_crs(crs_metric)
            # Intersección con el término municipal
            siu14_clip = gpd.overlay(siu14, gdf_muni, how="intersection", keep_geom_type=True)
            # Disolver para evitar solapes de piezas dentro del municipio
            if not siu14_clip.empty:
                siu14_clip = siu14_clip.dissolve().explode(index_parts=False)
            area_cod14_ha = siu14_clip.geometry.area.sum() / 10000.0 if not siu14_clip.empty else 0.0

            # --- Denominador: Clase de suelo urbana/urbanizable, recortada al municipio ---
            gdf_clase = load_clase_suelo_by_municipality_v2(selected_muni)
            if gdf_clase is None or gdf_clase.empty:
                st.warning("No hay clase de suelo para el municipio (consulta vacía).")
                return {
                    "area_cod14_ha": round(area_cod14_ha, 2),
                    "area_suelo_urb_ha": 0.0,
                    "d03b": 0.0,
                    "siu14_clip": siu14_clip,
                    "suelo_urb_clip": gpd.GeoDataFrame(geometry=[], crs=crs_metric),
                }

            # A CRS métrico
            if gdf_clase.crs is None or gdf_clase.crs.to_string() != crs_metric:
                gdf_clase = gdf_clase.to_crs(crs_metric)

            # Normalizador de texto
            def _norm_txt(s):
                if pd.isna(s): return ""
                import unicodedata, re
                s = str(s).strip()
                s = "".join(c for c in unicodedata.normalize("NFKD", s) if unicodedata.category(c) != "Mn")
                s = re.sub(r"\s+", " ", s.upper())
                return s

            gdf_clase["_clase_norm"] = gdf_clase["clasesuelo"].apply(_norm_txt)

            # 1) Intersección de TODAS las clases con el municipio (para debug y fallback)
            clip_all = gpd.overlay(gdf_clase, gdf_muni, how="intersection", keep_geom_type=True)
            area_all_ha = (clip_all.geometry.area.sum() / 10000.0) if not clip_all.empty else 0.0

            # 2) DENOMINADOR ROBUSTO: "todo salvo SUELO NO URBANIZABLE"
            mask_urb = clip_all["_clase_norm"] != "SUELO NO URBANIZABLE"
            suelo_urb_clip = clip_all[mask_urb].copy()

            # Si quieres EXCLUIR/INCLUIR SISTEMAS GENERALES explícitamente, descomenta:
            # include_sg = True
            # if not include_sg:
            #     suelo_urb_clip = suelo_urb_clip[suelo_urb_clip["_clase_norm"] != "SISTEMAS GENERALES"]

            # 3) Disolver para evitar dobles conteos
            if not suelo_urb_clip.empty:
                suelo_urb_clip = suelo_urb_clip.dissolve().explode(index_parts=False)

            area_suelo_urb_ha = suelo_urb_clip.geometry.area.sum() / 10000.0 if not suelo_urb_clip.empty else 0.0
            # Muestra qué clases están entrando/saliendo
            if area_suelo_urb_ha == 0.0 and area_all_ha > 0:
                st.info("El denominador sigue a 0; muestro las clases presentes en la intersección para ajustar reglas:")
                st.write(clip_all["_clase_norm"].value_counts())

            # --- Indicador ---
            d03b = 0.0 if area_suelo_urb_ha <= 0 else (area_cod14_ha / area_suelo_urb_ha) * 100.0
        return {
            "area_cod14_ha": round(area_cod14_ha, 2),
            "area_suelo_urb_ha": round(area_suelo_urb_ha, 2),
            "d03b": round(d03b, 2),
            "siu14_clip": siu14_clip,
            "suelo_urb_clip": suelo_urb_clip,
        }

    def compute_d04_municipal_by_muni(selected_muni: str, gdf_muni: gpd.GeoDataFrame):
        """
        D.04 (municipal): % de superficie municipal que es
        ('SUELO URBANIZABLE NO DELIMITADO O SECTORIZADO', 'SUELO NO URBANIZABLE')
        """
        if gdf_muni is None or gdf_muni.empty:
            return {"d04": None, "area_obj_ha": 0.0, "area_muni_ha": 0.0}

        # Área municipal (ha) con área elipsoidal (WGS84)
        muni_wgs = gdf_muni.to_crs(4326)
        area_muni_ha = sum(calculate_ellipsoidal_area(muni_wgs)) / 10000.0

        gdf_clase = load_clase_suelo_by_municipality_v2(selected_muni)
        if gdf_clase is None or gdf_clase.empty:
            return {"d04": 0.0 if area_muni_ha > 0 else None,
                    "area_obj_ha": 0.0,
                    "area_muni_ha": round(area_muni_ha, 2)}

        # Normalización de texto
        def _norm(s):
            if pd.isna(s): return ""
            s = "".join(c for c in unicodedata.normalize("NFKD", str(s)) if unicodedata.category(c) != "Mn")
            return re.sub(r"\s+", " ", s.upper().strip())

        objetivos = {
            "SUELO URBANIZABLE NO DELIMITADO O SECTORIZADO",
            "SUELO NO URBANIZABLE",
        }
        gdf_clase = gdf_clase.copy()
        gdf_clase["_clase_norm"] = gdf_clase["clasesuelo"].apply(_norm)
        gdf_obj = gdf_clase[gdf_clase["_clase_norm"].isin({_norm(v) for v in objetivos})].copy()
        if gdf_obj.empty:
            return {"d04": 0.0 if area_muni_ha > 0 else None,
                    "area_obj_ha": 0.0,
                    "area_muni_ha": round(area_muni_ha, 2)}

        # CRS métrico + reparar geometrías
        crs_metric = gdf_muni.crs or "EPSG:25830"
        if gdf_obj.crs is None or gdf_obj.crs != crs_metric:
            gdf_obj = gdf_obj.to_crs(crs_metric)
        gdf_m = gdf_muni if (gdf_muni.crs == crs_metric) else gdf_muni.to_crs(crs_metric)

        # 🔧 Normaliza el nombre de la columna de geometría a 'geometry' para evitar KeyError
        if gdf_obj.geometry.name != "geometry":
            gdf_obj = gdf_obj.set_geometry(gdf_obj.geometry.name).rename_geometry("geometry")
        if gdf_m.geometry.name != "geometry":
            gdf_m = gdf_m.set_geometry(gdf_m.geometry.name).rename_geometry("geometry")

        try:
            gdf_obj["geometry"] = gdf_obj.geometry.buffer(0)
            gdf_m["geometry"]   = gdf_m.geometry.buffer(0)
        except Exception:
            pass

        # Intersección + disolver (equivale a ST_Union(ST_Intersection(...)))
        inter = gpd.overlay(gdf_obj, gdf_m, how="intersection", keep_geom_type=True)
        if inter.empty:
            area_obj_ha = 0.0
        else:
            inter_dis = inter.dissolve().explode(index_parts=False)
            inter_wgs = inter_dis.to_crs(4326)
            area_obj_ha = sum(calculate_ellipsoidal_area(inter_wgs)) / 10000.0

        d04 = (area_obj_ha / area_muni_ha * 100.0) if area_muni_ha > 0 else None
        return {
            "d04": None if d04 is None else round(d04, 2),
            "area_obj_ha": round(area_obj_ha, 2),
            "area_muni_ha": round(area_muni_ha, 2),
        }



    from sqlalchemy import text

    def compute_d02a_municipal_postgis(selected_muni: str, gdf_muni: gpd.GeoDataFrame):
        """
        D.02.a (CORINE) municipal, ejecutado en PostGIS:
        % de superficie municipal ocupada por CORINE (códigos 111..142)
        """
        if gdf_muni is None or gdf_muni.empty:
            return {"d02a_pct": None, "area_corine_ha": 0.0, "area_muni_ha": 0.0}

        # 1) Disolver municipio y llevar a EPSG:25830
        crs_metric = "EPSG:25830"
        muni_25830 = gdf_muni.to_crs(crs_metric).dissolve().explode(index_parts=False)
        geom_union = muni_25830.unary_union
        if geom_union.is_empty:
            return {"d02a_pct": None, "area_corine_ha": 0.0, "area_muni_ha": 0.0}

        muni_wkt = geom_union.wkt  # WKT en 25830

        # 2) SQL en PostGIS (todo en BD)
        codes = ("111","112","121","122","123","124","131","132","133","141","142")
        sql = text("""
            WITH muni AS (
                SELECT ST_GeomFromText(:muni_wkt, 25830) AS geom
            ),
            corine_sel AS (
                SELECT geom
                FROM public.corine_d02
                WHERE "CODE_18" = ANY(:codes)
            ),
            inter AS (
                SELECT ST_Union(ST_Intersection(c.geom, m.geom)) AS geom
                FROM corine_sel c
                CROSS JOIN muni m
                WHERE ST_Intersects(c.geom, m.geom)
            )
            SELECT
                COALESCE(ST_Area(i.geom) / 10000.0, 0) AS area_corine_ha,
                ST_Area(m.geom) / 10000.0                 AS area_muni_ha,
                CASE
                    WHEN i.geom IS NULL OR ST_Area(m.geom) = 0 THEN 0
                    ELSE (ST_Area(i.geom) / ST_Area(m.geom)) * 100
                END AS d02a_pct
            FROM muni m
            LEFT JOIN inter i ON TRUE;
        """)

        try:
            engine = get_db_connection()
            with engine.connect() as conn:
                # NOTA: para pasar arrays a ANY() con SQLAlchemy text, mándalo como lista
                res = conn.execute(sql, {"muni_wkt": muni_wkt, "codes": list(codes)}).mappings().first()
            if not res:
                return {"d02a_pct": None, "area_corine_ha": 0.0, "area_muni_ha": 0.0}

            out = {
                "d02a_pct": round(float(res["d02a_pct"]), 2) if res["d02a_pct"] is not None else None,
                "area_corine_ha": round(float(res["area_corine_ha"]), 2) if res["area_corine_ha"] is not None else 0.0,
                "area_muni_ha": round(float(res["area_muni_ha"]), 2) if res["area_muni_ha"] is not None else 0.0,
            }
            return out
        except Exception as e:
            st.error(f"❌ Error calculando D.02.a (CORINE) en PostGIS: {e}")
            return {"d02a_pct": None, "area_corine_ha": 0.0, "area_muni_ha": 0.0}


    @st.cache_data
    def load_siu_recintos_by_municipality(selected_muni: str):
        """
        Carga CODSIU/descripcion/geom de dev_codeine.siu_recintos_municipalities
        filtrando por municipality ILIKE %<nombre>%.
        Devuelve GeoDataFrame con geometry estandarizada.
        """
        try:
            engine = get_db_connection()
            sql = """
                SELECT id, municipality, usosuelo, geom
                FROM dev_codeine.siu_recintos_municipalities
                WHERE municipality ILIKE %(m)s
            """
            with engine.connect() as conn:
                gdf = gpd.read_postgis(sql, conn, geom_col="geom", params={"m": f"%{selected_muni}%"})
            if not gdf.empty:
                # normaliza columna geometry
                if gdf.geometry.name != "geom":
                    gdf = gdf.set_geometry(gdf.geometry.name)
                gdf = gdf.rename_geometry("geometry")
            return gdf
        except Exception as e:
            st.error(f"❌ Error cargando SIU recintos: {e}")
            return None

   
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
            dgt_files = {
                "2021": "dgt2021.parquet",
                "2022": "dgt2022.parquet",
                "2023": "dgt2023.parquet",
                "2024": "dgt2024.parquet",
            }
            df_dgt_by_year = {}
            for y, path in dgt_files.items():
                try:
                    d = pd.read_parquet(path)
                    # clave de emparejamiento igual que usabas
                    d["municipio_completo"] = d["Código INE"].astype(str).str.zfill(5) + " " + d["Municipio"]
                    df_dgt_by_year[y] = d
                except Exception:
                    df_dgt_by_year[y] = None  # si falta el fichero, evita fallar

            return df, df_censo, df_hog_2011, df_hog_2021, df_censo2011, df_dgt_by_year
        
                
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



    # === PARO & CONTRATOS DESDE PARQUET GLOBAL ===

    def normalize_muni(name: str) -> str:
        """Quita código si viene '28079 Madrid', elimina tildes y pasa a MAYÚSCULAS."""
        if pd.isna(name):
            return None
        s = str(name).strip()
        m = re.match(r"^\s*\d+\s+(.+)$", s)
        if m:
            s = m.group(1)
        s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
        return " ".join(s.upper().split())

    @st.cache_data
    def load_sepe_parquet(path="sepe_global.parquet"):
        df = pd.read_parquet(path)

        # Unificar municipio y normalizar mes
        if "pMunicipio" in df.columns or "cMunicipio" in df.columns:
            df["muni"] = df.get("pMunicipio").fillna(df.get("cMunicipio"))
            df["muni_norm"] = df["muni"].apply(normalize_muni)
        else:
            raise ValueError("El parquet SEPE no tiene columnas de municipio (pMunicipio/cMunicipio).")

        mes_map = {
            "enero":1,"febrero":2,"marzo":3,"abril":4,"mayo":5,"junio":6,
            "julio":7,"agosto":8,"septiembre":9,"setiembre":9,"octubre":10,
            "noviembre":11,"diciembre":12
        }
        df["mes_norm"] = df["mes"].astype(str).str.strip().str.lower()
        df["mes_num"]  = df["mes_norm"].map(mes_map)
        df = df[df["mes_num"].notna()].copy()
        df["anio"] = df["anio"].astype(int)
        df["fecha"] = pd.to_datetime(dict(year=df["anio"], month=df["mes_num"].astype(int), day=1))

        # >>> NUEVO: preservar orden original para "primera coincidencia"
        df["row_order"] = df.reset_index().index

        return df


    @st.cache_data
    def load_clase_suelo_by_municipality_v2(selected_muni: str):
        """
        dev_codeine.clase_suelo_municipalities: id, municipality, geom, clasesuelo
        """
        try:
            engine = get_db_connection()
            sql = """
                SELECT id, municipality, geom, clasesuelo
                FROM dev_codeine.clase_suelo_municipalities
                WHERE municipality ILIKE %(m)s
            """
            with engine.connect() as conn:
                gdf = gpd.read_postgis(sql, conn, geom_col="geom", params={"m": f"%{selected_muni}%"})
            if not gdf.empty:
                # standardize geometry name to 'geometry'
                if gdf.geometry.name != "geom":
                    gdf = gdf.set_geometry(gdf.geometry.name)
                gdf = gdf.rename_geometry("geometry")
            return gdf
        except Exception as e:
            st.error(f"❌ Error cargando clase de suelo: {e}")
            return None


    @st.cache_data
    def load_siu_recintos_by_municipality(selected_muni: str):
        """
        dev_codeine.siu_recintos_municipalities: id, municipality, geom, usosuelo
        """
        try:
            engine = get_db_connection()
            sql = """
                SELECT id, municipality, geom, usosuelo
                FROM dev_codeine.siu_recintos_municipalities
                WHERE municipality ILIKE %(m)s
            """
            with engine.connect() as conn:
                gdf = gpd.read_postgis(sql, conn, geom_col="geom", params={"m": f"%{selected_muni}%"})
            if not gdf.empty:
                if gdf.geometry.name != "geom":
                    gdf = gdf.set_geometry(gdf.geometry.name)
                gdf = gdf.rename_geometry("geometry")
            return gdf
        except Exception as e:
            st.error(f"❌ Error cargando SIU recintos: {e}")
            return None

    def build_clase_siu_area_and_intersection(selected_muni: str, gdf_muni: gpd.GeoDataFrame):
        """
        Returns dict with:
        - 'tabla_clase'  : DataFrame (clasesuelo, Área (ha))
        - 'tabla_siu'    : DataFrame (usosuelo, Área (ha))
        - 'tabla_long'   : DataFrame (usosuelo, clasesuelo, Área (ha))
        - 'matriz_pivot' : DataFrame pivot (rows=usosuelo, cols=clasesuelo, values ha)
        All areas computed ellipsoidally using calculate_ellipsoidal_area on WGS84.
        """
        if gdf_muni is None or gdf_muni.empty:
            return {"tabla_clase": None, "tabla_siu": None, "tabla_long": None, "matriz_pivot": None}

        # Load tables
        gdf_clase = load_clase_suelo_by_municipality_v2(selected_muni)
        gdf_siu   = load_siu_recintos_by_municipality(selected_muni)

        if (gdf_clase is None or gdf_clase.empty) and (gdf_siu is None or gdf_siu.empty):
            return {"tabla_clase": None, "tabla_siu": None, "tabla_long": None, "matriz_pivot": None}

        # Clip to municipality
        gdf_clase_clip = perform_spatial_clip(gdf_clase, gdf_muni) if (gdf_clase is not None and not gdf_clase.empty) else None
        gdf_siu_clip   = perform_spatial_clip(gdf_siu,   gdf_muni) if (gdf_siu   is not None and not gdf_siu.empty)   else None

        # --- Tabla 1: área por clasesuelo ---
        tabla_clase = None
        if gdf_clase_clip is not None and not gdf_clase_clip.empty:
            gdfC = gdf_clase_clip.copy()
            gdfC["_cat"] = gdfC["clasesuelo"].fillna("SIN_CATEGORIA").astype(str)
            gdfC_dis = gdfC.dissolve(by="_cat", as_index=False)
            area_m2 = calculate_ellipsoidal_area(gdfC_dis.to_crs(4326))
            gdfC_dis["Área (ha)"] = [a/10000 for a in area_m2]
            tabla_clase = (gdfC_dis[["_cat","Área (ha)"]]
                        .rename(columns={"_cat":"clasesuelo"})
                        .sort_values("Área (ha)", ascending=False)
                        .reset_index(drop=True))

        # --- Tabla 2: área por usosuelo (SIU) ---
        tabla_siu = None
        if gdf_siu_clip is not None and not gdf_siu_clip.empty:
            gdfS = gdf_siu_clip.copy()
            gdfS["_uso"] = gdfS["usosuelo"].fillna("SIN_USO").astype(str)
            # dissolve by use (avoid overlaps double counting)
            gdfS_dis = gdfS.dissolve(by="_uso", as_index=False)
            area_m2 = calculate_ellipsoidal_area(gdfS_dis.to_crs(4326))
            gdfS_dis["Área (ha)"] = [a/10000 for a in area_m2]
            tabla_siu = (gdfS_dis[["_uso","Área (ha)"]]
                        .rename(columns={"_uso":"usosuelo"})
                        .sort_values("Área (ha)", ascending=False)
                        .reset_index(drop=True))

        # --- Tabla 3: intersección usosuelo × clasesuelo ---
        tabla_long = None
        matriz_pivot = None
        if (gdf_clase_clip is not None and not gdf_clase_clip.empty) and (gdf_siu_clip is not None and not gdf_siu_clip.empty):
            # Work in metric CRS for overlay robustness, then compute area ellipsoidally
            crs_metric = gdf_muni.crs or "EPSG:25830"
            A = gdf_siu_clip if (gdf_siu_clip.crs and gdf_siu_clip.crs.to_string() == crs_metric) else gdf_siu_clip.to_crs(crs_metric)
            B = gdf_clase_clip if (gdf_clase_clip.crs and gdf_clase_clip.crs.to_string() == crs_metric) else gdf_clase_clip.to_crs(crs_metric)

            # Keep only needed columns
            Ause = A[["usosuelo","geometry"]].copy()
            Bcla = B[["clasesuelo","geometry"]].copy()

            inter = gpd.overlay(Ause, Bcla, how="intersection", keep_geom_type=True)
            if not inter.empty:
                # dissolve by (usosuelo, clasesuelo) to avoid slivers being summed twice
                inter_dis = inter.dissolve(by=["usosuelo","clasesuelo"], as_index=False)
                inter_wgs = inter_dis.to_crs(4326)
                area_m2 = calculate_ellipsoidal_area(inter_wgs)
                inter_dis["Área (ha)"] = [a/10000 for a in area_m2]

                tabla_long = (inter_dis[["usosuelo","clasesuelo","Área (ha)"]]
                            .sort_values(["usosuelo","Área (ha)"], ascending=[True, False])
                            .reset_index(drop=True))

                # Pivot matrix (usosuelo x clasesuelo)
                try:
                    matriz_pivot = (tabla_long
                        .pivot_table(index="usosuelo", columns="clasesuelo", values="Área (ha)",
                                    aggfunc="sum", fill_value=0.0)
                        .sort_index())
                except Exception:
                    matriz_pivot = None

        return {
            "tabla_clase": tabla_clase,
            "tabla_siu": tabla_siu,
            "tabla_long": tabla_long,
            "matriz_pivot": matriz_pivot
        }


    @st.cache_data
    def load_sscc_by_municipality(selected_muni: str):
        """
        Carga secciones censales del municipio. Intenta dos nombres de tabla comunes:
        - dev_codeine."Censo_2021_SSCC"
        - dev_codeine.seccen

        Debe devolver columnas: cusec3, nmun, geom
        """
        engine = get_db_connection()
        # Opción 1: Censo_2021_SSCC
        sql_opts = [
            'SELECT "cusec3", "nmun", geom FROM dev_codeine."Censo_2021_SSCC" WHERE "nmun" ILIKE %(m)s',
            'SELECT "cusec3", "nmun", geom FROM dev_codeine.seccen WHERE "nmun" ILIKE %(m)s',
        ]
        for sql in sql_opts:
            try:
                with engine.connect() as conn:
                    gdf = gpd.read_postgis(sql, conn, geom_col="geom", params={"m": f"%{selected_muni}%"})
                if not gdf.empty:
                    return gdf
            except Exception:
                pass
        st.error("❌ No pude cargar las secciones censales (revisa el nombre de la tabla/permiso).")
        return None

    def subset_muni_first(df: pd.DataFrame, display_muni: str) -> pd.DataFrame:
        """
        Devuelve solo las filas del municipio (muni_norm) quedándose con
        el PRIMER valor crudo de 'muni' que aparece en el parquet (por row_order).
        Útil cuando 'SORIA' aparece como municipio y luego como provincia.
        """
        target = normalize_muni(display_muni)
        if "muni_norm" not in df.columns:
            return df.iloc[0:0].copy()

        # Filtra por el normalizado y orden original
        cand = df[df["muni_norm"] == target].copy()
        if "row_order" in cand.columns:
            cand = cand.sort_values("row_order")
        if cand.empty:
            return cand

        # Primer 'muni' crudo que aparece para ese muni_norm
        first_raw = cand["muni"].dropna().iloc[0]

        # Mantén solo ese 'muni' crudo
        filtered = df[(df["muni_norm"] == target) & (df["muni"] == first_raw)].copy()

        # Mensaje útil si hubo más de un crudo distinto
        other_raws = cand["muni"].dropna().unique().tolist()
        if len(set(other_raws)) > 1:
            st.caption(f"🔎 Coincidencias múltiples para **{display_muni}**: {other_raws}. "
                    f"Usando la primera: **{first_raw}**")

        return filtered


    def build_timeseries(df_sepe: pd.DataFrame, display_muni: str):
        target = normalize_muni(display_muni)
        df_muni = subset_muni_first(df_sepe, display_muni)
        # --- PARO ---
        if "tipo" in df_sepe.columns:
            df_paro = df_sepe[(df_sepe["tipo"]=="p") & (df_sepe["muni_norm"]==target)].copy()
        else:
            df_paro = df_sepe[df_sepe["muni_norm"]==target].filter(regex=r"^p|fecha").copy()
        ts_paro = pd.DataFrame()
        if "pTotal" in df_paro.columns:
            ts_paro = (df_paro.groupby("fecha", as_index=False)["pTotal"].sum()
                            .sort_values("fecha"))

        # --- CONTRATOS ---
        if "tipo" in df_sepe.columns:
            df_cont = df_sepe[(df_sepe["tipo"]=="c") & (df_sepe["muni_norm"]==target)].copy()
        else:
            df_cont = df_sepe[df_sepe["muni_norm"]==target].filter(regex=r"^c|fecha").copy()
        ts_cont = pd.DataFrame()
        if "cTotal" in df_cont.columns:      # <-- nombre correcto
            ts_cont = (df_cont.groupby("fecha", as_index=False)["cTotal"].sum()
                            .sort_values("fecha"))

        return ts_paro, ts_cont





    def _sector_percentages_from_row(row) -> pd.Series:
        tot = float(row.get("cTotal", 0) or 0)
        if tot <= 0 or pd.isna(tot):
            return pd.Series([None]*4, index=["agr","ind","con","ser"])
        return pd.Series([
            round(row.get("cSAgricultura", 0)/tot*100, 2),
            round(row.get("cSIndustria", 0)/tot*100, 2),
            round(row.get("cSConstruccion", 0)/tot*100, 2),
            round(row.get("cSServicios", 0)/tot*100, 2),
        ], index=["agr","ind","con","ser"])


    def sector_shares_by_year(df_sepe: pd.DataFrame, display_muni: str, debug: bool=False) -> pd.DataFrame:
        target = normalize_muni(display_muni)

        # 1) Filtra CONTRATOS del municipio y fija un único "muni" crudo
        dfc_all = subset_muni_first(df_sepe, display_muni)
        dfc = dfc_all[dfc_all.get("tipo") == "c"].copy()
        if dfc.empty:
            if debug: st.info("D.26: no hay filas para este municipio.")
            return pd.DataFrame(columns=[
                "anio","agr_avg","ind_avg","con_avg","ser_avg","agr_sep","ind_sep","con_sep","ser_sep"
            ])

        # 2) Asegura numéricos
        cols = ["cTotal","cSAgricultura","cSIndustria","cSConstruccion","cSServicios"]
        for c in cols:
            if c in dfc.columns:
                dfc[c] = pd.to_numeric(dfc[c], errors="coerce").fillna(0)
            else:
                dfc[c] = 0

        # 3) Quedarse con UNA fila por (anio, mes_num) usando el orden original si existe
        if "row_order" not in dfc.columns:
            dfc["row_order"] = dfc.reset_index().index

        # Ordena por año, mes y orden original; luego elimina duplicados por (anio, mes_num)
        dfc_clean = (dfc.sort_values(["anio","mes_num","row_order"])
                    .drop_duplicates(subset=["anio","mes_num"], keep="first")
                    .copy())

        # 4) Construir "monthly" DESDE LOS BRUTOS (ya depurados a 1 fila por mes)
        #    (groupby + sum no cambia los valores porque ya hay 1 fila por mes, pero lo dejamos por seguridad)
        monthly = (dfc_clean.groupby(["anio","mes_num"], as_index=False)[cols].sum())

        # 5) % mensuales por sector
        monthly["agr_pct"] = (monthly["cSAgricultura"] / monthly["cTotal"]).where(monthly["cTotal"]>0).mul(100)
        monthly["ind_pct"] = (monthly["cSIndustria"]   / monthly["cTotal"]).where(monthly["cTotal"]>0).mul(100)
        monthly["con_pct"] = (monthly["cSConstruccion"]/ monthly["cTotal"]).where(monthly["cTotal"]>0).mul(100)
        monthly["ser_pct"] = (monthly["cSServicios"]   / monthly["cTotal"]).where(monthly["cTotal"]>0).mul(100)

        # 6) MEDIA anual (promedio simple de los % mensuales)
        year_avg = (monthly.groupby("anio", as_index=False)[["agr_pct","ind_pct","con_pct","ser_pct"]]
                            .mean().round(2)
                            .rename(columns={
                                "agr_pct":"agr_avg","ind_pct":"ind_avg",
                                "con_pct":"con_avg","ser_pct":"ser_avg"
                            }))

        # 7) SEPTIEMBRE: % solo para mes 9
        sep = (monthly[monthly["mes_num"]==9][["anio","agr_pct","ind_pct","con_pct","ser_pct"]]
            .rename(columns={
                "agr_pct":"agr_sep","ind_pct":"ind_sep",
                "con_pct":"con_sep","ser_pct":"ser_sep"
            }).round(2))

        out = year_avg.merge(sep, on="anio", how="left")

        # 8) Debug opcional mostrando EXACTAMENTE los datos depurados
        if debug:
            st.subheader("🧪 D.26 DEBUG – % mensuales y medias anuales (desde brutos depurados)")
            st.caption("Brutos depurados a 1 fila por mes (dfc_clean):")
            cols_dbg = ["anio","mes_num","muni","archivo","sheet_name"] + cols
            cols_dbg = [c for c in cols_dbg if c in dfc_clean.columns]
            st.dataframe(dfc_clean.sort_values(["anio","mes_num"])[cols_dbg],
                        use_container_width=True)

            st.caption("Matriz mensual (tras depurar → 1 fila/mes y calcular %):")
            st.dataframe(monthly.sort_values(["anio","mes_num"])[
                ["anio","mes_num","cTotal","cSAgricultura","cSIndustria","cSConstruccion","cSServicios",
                "agr_pct","ind_pct","con_pct","ser_pct"]
            ], use_container_width=True)

            st.caption("Medias anuales de % + septiembre:")
            dbg = out.copy()
            dbg["sum_%_avg"] = dbg[["agr_avg","ind_avg","con_avg","ser_avg"]].sum(axis=1).round(2)
            st.dataframe(dbg.sort_values("anio"), use_container_width=True)

        return out


    def keep_one_per_month(df: pd.DataFrame, prefer: str = "first") -> pd.DataFrame:
        """
        Devuelve una única fila por (anio, mes_num).
        prefer = "first"  -> usa el orden original (row_order) si existe.
        prefer = "min"    -> usa la fila con menor cTotal (si existe cTotal).
        """
        key = ["anio", "mes_num"]
        tmp = df.copy()

        if prefer == "min" and "cTotal" in tmp.columns:
            tmp = (tmp.sort_values(key + ["cTotal"])
                    .drop_duplicates(subset=key, keep="first"))
        else:
            extra = ["row_order"] if "row_order" in tmp.columns else []
            tmp = (tmp.sort_values(key + extra)
                    .drop_duplicates(subset=key, keep="first"))
        return tmp


    def show_d26_debug(df_sepe: pd.DataFrame, display_muni: str):
        """
        Depuración de D.26 para un municipio:
        - Bruto mensual (filas originales)
        - Agregado por mes (sumas y %)
        - Agregado anual (sumas y %)
        - Septiembre (sumas y %)
        - Conteo de filas por mes (posibles duplicados)
        """
        

        # 1) filtra CONTRATOS y aplica "quédate con el primer municipio crudo"
        dfc_all = df_sepe[df_sepe.get("tipo") == "c"].copy()
        dfc = subset_muni_first(dfc_all, display_muni)   # <<--- AQUÍ se fuerza el primero
        if dfc.empty:
            st.warning("No hay registros de CONTRATOS para este municipio en el parquet.")
            return

        # 2) asegura tipos numéricos
        cols = ["cTotal","cSAgricultura","cSIndustria","cSConstruccion","cSServicios"]
        for c in cols:
            dfc[c] = pd.to_numeric(dfc[c], errors="coerce")
        dfc[cols] = dfc[cols].fillna(0)

        # 3) selector de año
        years = sorted(dfc["anio"].dropna().astype(int).unique())
        year_sel = st.selectbox("Año (depuración D.26)", years, index=len(years)-1)
        dfc_y = dfc[dfc["anio"] == year_sel].copy()
        dfc_y = keep_one_per_month(dfc_y, prefer="first")

        # --- (1) bruto mensual
        st.subheader("🔢 Bruto mensual (filas originales)")
        raw_cols = ["anio","mes_num","mes","fecha","muni","archivo","sheet_name"] + cols
        raw_cols = [c for c in raw_cols if c in dfc_y.columns]
        df_raw = dfc_y[raw_cols].copy()

        sec_cols = [c for c in ["cSAgricultura","cSIndustria","cSConstruccion","cSServicios"] if c in df_raw.columns]
        if sec_cols:
            df_raw["sum_sectores"] = df_raw[sec_cols].sum(axis=1)
            df_raw["gap_total_minus_sect"] = df_raw.get("cTotal", 0) - df_raw["sum_sectores"]
            if "cSAgricultura" in df_raw.columns and "cTotal" in df_raw.columns:
                df_raw["%Agr_fila"] = (df_raw["cSAgricultura"] / df_raw["cTotal"] * 100)\
                                        .replace([np.inf, -np.inf], np.nan).round(2)
        st.dataframe(df_raw.sort_values(["anio","mes_num"]), use_container_width=True)

        # --- (2) agregado por mes
        st.subheader("📦 Agregado por mes (sumas y %)")
        gb = (dfc_y.groupby(["anio","mes_num","mes"], as_index=False)[cols].sum())
        gb["sum_sectores"] = gb[["cSAgricultura","cSIndustria","cSConstruccion","cSServicios"]].sum(axis=1)
        gb["gap_total_minus_sect"] = gb["cTotal"] - gb["sum_sectores"]
        gb["%Agr"] = (gb["cSAgricultura"] / gb["cTotal"] * 100).where(gb["cTotal"] > 0).round(2)
        gb["%Ind"] = (gb["cSIndustria"]   / gb["cTotal"] * 100).where(gb["cTotal"] > 0).round(2)
        gb["%Con"] = (gb["cSConstruccion"]/ gb["cTotal"] * 100).where(gb["cTotal"] > 0).round(2)
        gb["%Ser"] = (gb["cSServicios"]   / gb["cTotal"] * 100).where(gb["cTotal"] > 0).round(2)
        st.dataframe(gb.sort_values("mes_num"), use_container_width=True)

        # --- (3) agregado anual
        st.subheader("🧮 Agregado ANUAL (sumas del año y %)")
        annual = dfc_y[cols].sum().to_frame("valor").T
        annual.insert(0, "anio", year_sel)
        annual["%Agr"] = (annual["cSAgricultura"] / annual["cTotal"] * 100).where(annual["cTotal"] > 0).round(2)
        annual["%Ind"] = (annual["cSIndustria"]   / annual["cTotal"] * 100).where(annual["cTotal"] > 0).round(2)
        annual["%Con"] = (annual["cSConstruccion"]/ annual["cTotal"] * 100).where(annual["cTotal"] > 0).round(2)
        annual["%Ser"] = (annual["cSServicios"]   / annual["cTotal"] * 100).where(annual["cTotal"] > 0).round(2)
        st.dataframe(annual, use_container_width=True)

        # --- (4) septiembre
        st.subheader("📌 SEPTIEMBRE (sumas de septiembre y %)")
        sep = dfc_y[dfc_y["mes_num"] == 9][cols].sum().to_frame("valor").T
        sep.insert(0, "anio", year_sel)
        sep["%Agr"] = (sep["cSAgricultura"] / sep["cTotal"] * 100).where(sep["cTotal"] > 0).round(2)
        sep["%Ind"] = (sep["cSIndustria"]   / sep["cTotal"] * 100).where(sep["cTotal"] > 0).round(2)
        sep["%Con"] = (sep["cSConstruccion"]/ sep["cTotal"] * 100).where(sep["cTotal"] > 0).round(2)
        sep["%Ser"] = (sep["cSServicios"]   / sep["cTotal"] * 100).where(sep["cTotal"] > 0).round(2)
        st.dataframe(sep, use_container_width=True)

        # --- (5) conteo filas por mes (ya sobre un único municipio crudo)
        st.caption("🧩 Conteo de filas por mes (para detectar duplicados o múltiples filas por mes)")
        cnt = dfc_y.groupby(["anio","mes_num"]).size().reset_index(name="n_filas")
        st.dataframe(cnt.sort_values("mes_num"), use_container_width=True)


    def compute_d28(df_sepe: pd.DataFrame, display_muni: str, year: int, month_num: int = 9):
        """
        Calcula D.28.* a partir del parquet SEPE para un municipio y año dados,
        tomando como referencia el mes 'month_num' (por defecto septiembre=9).

        Devuelve: dict con n_total, n_25_44, n_mujer y flags de disponibilidad.
        """
        # 1) Filtrado municipio (misma lógica que usas para evitar duplicados)
        dfp = subset_muni_first(df_sepe, display_muni)
        dfp = dfp[(dfp.get("tipo") == "p") & (dfp["anio"] == year) & (dfp["mes_num"] == month_num)].copy()
        if dfp.empty:
            return {"n_total": None, "n_25_44": None, "n_mujer": None}

        # 2) Asegura numéricos
        need_cols = ["pTotal","pH2544","pM2544","pM25","pM45"]
        for c in need_cols:
            if c in dfp.columns:
                dfp[c] = pd.to_numeric(dfp[c], errors="coerce").fillna(0)
            else:
                dfp[c] = 0

        # 3) Quédate con UNA fila por mes (por si acaso)
        dfp = keep_one_per_month(dfp, prefer="first")

        # 4) Cálculos
        n_total  = float(dfp["pTotal"].iloc[0]) if "pTotal" in dfp.columns and not dfp.empty else None
        n_25_44  = float(dfp["pH2544"].iloc[0] + dfp["pM2544"].iloc[0]) if not dfp.empty else None
        n_mujer  = float((dfp["pM25"].iloc[0] + dfp["pM2544"].iloc[0] + dfp["pM45"].iloc[0])) if not dfp.empty else None

        return {"n_total": n_total, "n_25_44": n_25_44, "n_mujer": n_mujer}

    def build_suc_adc_mask_from_clips(gdf_clase_clip: gpd.GeoDataFrame,
                                    gdf_siu_clip: gpd.GeoDataFrame,
                                    crs_metric: str = "EPSG:25830") -> gpd.GeoDataFrame | None:
        """
        Devuelve un GeoDataFrame con UNA o varias geometrías que representan (SUC ∪ ADC)
        usando las capas YA RECORTADAS AL MUNICIPIO (gdf_clase_clip y gdf_siu_clip).
        - SUC = filas de clase de suelo con 'SUELO URBANO'
        - ADC = usosuelo en {'DESARROLLO CONSOLIDADO','SUELO EDIFICADO',
                            'SUELO EN PROCESO DE EDIFICACION',
                            'SUELO URBANIZADO O EN PROCESO DE URBANIZACION'}
        """
        if (gdf_clase_clip is None or gdf_clase_clip.empty) and (gdf_siu_clip is None or gdf_siu_clip.empty):
            return None

        def _norm(s):
            if pd.isna(s): return ""
            s = "".join(c for c in unicodedata.normalize("NFKD", str(s)) if unicodedata.category(c) != "Mn")
            return re.sub(r"\s+", " ", s.upper().strip())

        ADC_SET = {
            "DESARROLLO CONSOLIDADO",
            "SUELO EDIFICADO",
            "SUELO EN PROCESO DE EDIFICACION",
            "SUELO URBANIZADO O EN PROCESO DE URBANIZACION",
        }
        adc_norms = {_norm(x) for x in ADC_SET}

        # Asegura CRS métrico
        parts = []
        if gdf_clase_clip is not None and not gdf_clase_clip.empty:
            C = gdf_clase_clip.copy()
            if C.crs is None: C.set_crs(crs_metric, inplace=True)
            if C.crs.to_string() != crs_metric: C = C.to_crs(crs_metric)
            if "clasesuelo" in C.columns:
                C["_clase_norm"] = C["clasesuelo"].apply(_norm)
                C_urb = C[C["_clase_norm"] == "SUELO URBANO"][["geometry"]].copy()
                if not C_urb.empty:
                    try: C_urb["geometry"] = C_urb.buffer(0)
                    except: pass
                    parts.append(C_urb)

        if gdf_siu_clip is not None and not gdf_siu_clip.empty:
            S = gdf_siu_clip.copy()
            if S.crs is None: S.set_crs(crs_metric, inplace=True)
            if S.crs.to_string() != crs_metric: S = S.to_crs(crs_metric)
            if "usosuelo" in S.columns:
                S["_uso_norm"] = S["usosuelo"].apply(_norm)
                S_adc = S[S["_uso_norm"].isin(adc_norms)][["geometry"]].copy()
                if not S_adc.empty:
                    try: S_adc["geometry"] = S_adc.buffer(0)
                    except: pass
                    parts.append(S_adc)

        if not parts:
            return None

        mask = pd.concat(parts, ignore_index=True)
        # Disolver para obtener una o pocas geometrías compactas (unión):
        mask = mask.dissolve().explode(index_parts=False)
        if mask.geometry.name != "geometry":
            mask = mask.set_geometry(mask.geometry.name).rename_geometry("geometry")
        return mask


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
            df, df_censo, df_hog_2011, df_hog_2021, df_censo2011, df_dgt_by_year = data_loaded

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
                st.markdown("### ℹ️ Información:")
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


                # ====== 🔼 SUBIR CAPA TRAS SELECCIONAR MUNICIPIO ======
                st.markdown("### 📤 Subir capa vectorial para este municipio")
                
                uploaded = st.file_uploader(
                    "Sube ZIP (shapefile), GeoJSON, GML, GPKG o CAT del Catastro",
                    type=["zip", "geojson", "gml", "gpkg", "cat"],
                    accept_multiple_files=False,
                    key="uploader_muni"
                )
                
                def _read_any_vector(fileobj):
                    """
                    Devuelve un GeoDataFrame desde:
                    - ZIP con shapefile (usa tu process_shapefile)
                    - GeoJSON / GML / GPKG guardando temporalmente y usando GeoPandas
                    - CAT: intenta como GML; si no, avisa
                    """
                    import tempfile, pathlib
                    suffix = pathlib.Path(fileobj.name).suffix.lower()
                
                    if suffix == ".zip":
                        # usa tu función existente
                        try:
                            return process_shapefile(fileobj)
                        except Exception as e:
                            st.error(f"❌ No se pudo leer el shapefile del ZIP: {e}")
                            return None
                
                    # Guardar temporalmente para gpd.read_file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(fileobj.getvalue())
                        tmp_path = tmp.name
                
                    try:
                        if suffix in (".geojson", ".gml", ".gpkg"):
                            return gpd.read_file(tmp_path)
                
                        if suffix == ".cat":
                            # Heurística: si es GML/XML, gpd puede abrirlo; si no, avisamos
                            with open(tmp_path, "rb") as f:
                                head = f.read(4096).lower()
                            looks_xml = (b"<?xml" in head[:200]) or (b"<gml" in head)
                            if looks_xml:
                                return gpd.read_file(tmp_path)
                            else:
                                st.warning("El .CAT no parece ser GML/XML legible. Ábrelo en QGIS y exporta a GPKG/SHP.")
                                return None
                
                        st.error("Formato no soportado. Usa ZIP/GeoJSON/GML/GPKG/CAT.")
                        return None
                    except Exception as e:
                        st.error(f"❌ Error leyendo la capa: {e}")
                        return None
                
                if uploaded is not None:
                    gdf_user = _read_any_vector(uploaded)
                    if gdf_user is not None and not gdf_user.empty:
                        # Info básica de la capa subida
                        display_geodata_info(gdf_user, uploaded.name)
                
                        # Recortar contra el municipio (reutiliza tu función perform_spatial_clip)
                        st.markdown("#### ✂️ Recortar la capa al término municipal")
                        do_clip = st.toggle("Recortar al municipio", value=True)
                        if do_clip:
                            clipped = perform_spatial_clip(gdf_user, gdf_muni)
                            if clipped is not None and not clipped.empty:
                                st.success(f"✅ Recorte hecho. Geometrías resultantes: {len(clipped)}")
                                gdf_to_show = clipped
                            else:
                                st.info("No hubo intersección con el municipio (o resultó vacío). Mostrando capa original.")
                                gdf_to_show = gdf_user
                        else:
                            gdf_to_show = gdf_user
                        
                        """
                        # Mapa rápido con Folium (reutiliza tu creador de mapas)
                        st.markdown("#### 🗺️ Vista rápida en mapa")
                        try:
                            m_user = create_folium_map(gdf_to_show, map_title=f"Capa: {uploaded.name}")
                            st_folium(m_user, width=1200, height=500, key="map_user_layer")
                        except Exception as e:
                            st.warning(f"No se pudo renderizar el mapa: {e}")
                        """
                        # Exportadores (reutiliza tu export_geodata)
                        st.markdown("#### ⬇️ Exportar resultados")
                        colx, coly, colz = st.columns(3)
                        with colx:
                            data, fname, mime = export_geodata(gdf_to_show, "capa_municipio", "GeoJSON")
                            if data:
                                st.download_button("Descargar GeoJSON", data, fname, mime)
                        with coly:
                            data, fname, mime = export_geodata(gdf_to_show, "capa_municipio", "Shapefile")
                            if data:
                                st.download_button("Descargar Shapefile (.zip)", data, fname, mime)
                        with colz:
                            data, fname, mime = export_geodata(gdf_to_show, "capa_municipio", "CSV")
                            if data:
                                st.download_button("Descargar CSV (atributos)", data, fname, mime)
                


                # 3) --- Bloque SIU + D03b: limpia el else y evita el else del try/except ---
                gdf_all_codsiu = load_internal_bases_all_codsiu(selected_muni)

                if gdf_all_codsiu is not None:
                    gdf_all_clipped = perform_spatial_clip(gdf_all_codsiu, gdf_muni)
                    if gdf_all_clipped is not None and not gdf_all_clipped.empty:
                        gdf_all_clipped["CODSIU"] = gdf_all_clipped["CODSIU"].astype(int)
                        for cod in [1, 2, 3, 9, 12, 14, 15, 16, 17, 18, 19]:
                            subset = gdf_all_clipped[gdf_all_clipped["CODSIU"] == cod]
                            st.session_state[f"sup_cultivos_{cod:02d}"] = subset["estal"].sum() if not subset.empty else None
                    else:
                        # si no hay recorte, limpia para no arrastrar valores de otro municipio
                        for cod in [1, 2, 3, 9, 12, 14, 15, 16, 17, 18, 19]:
                            st.session_state[f"sup_cultivos_{cod:02d}"] = None
                else:
                    st.info("ℹ️ No se encontró geometría SIU para este municipio o falló la carga.")
                    for cod in [1, 2, 3, 9, 12, 14, 15, 16, 17, 18, 19]:
                        st.session_state[f"sup_cultivos_{cod:02d}"] = None

                # --- Calcula D03b solo para la tabla ---
                d03b_value = None
                try:
                    if gdf_muni is not None and not gdf_muni.empty and gdf_all_codsiu is not None:
                        res03b = compute_d03b_municipal(selected_muni, gdf_muni, gdf_all_codsiu)
                        if res03b:
                            d03b_value = res03b["d03b"]
                except Exception:
                    d03b_value = None



                # --- Calcula D04 municipal una vez (como haces con D03b) ---
                d04_value = None
                try:
                    if gdf_muni is not None and not gdf_muni.empty:
                        res04 = compute_d04_municipal_by_muni(selected_muni, gdf_muni)
                        if res04:
                            d04_value = res04["d04"]
                except Exception as e:
                    st.error(f"Error calculando D.04: {e}")
                    import traceback; st.code(traceback.format_exc())
                    d04_value = None

                # --- Calcula D.02.a (CORINE) en PostGIS ---
                d02a_corine_value = None
                try:
                    if gdf_muni is not None and not gdf_muni.empty:
                        res02a = compute_d02a_municipal_postgis(selected_muni, gdf_muni)
                        if res02a:
                            d02a_corine_value = res02a["d02a_pct"]
                except Exception as e:
                    st.error(f"Error D.02.a CORINE: {e}")
                    d02a_corine_value = None


                
            # === SUC / ADC / SUCADCtotal (hectáreas) ===
            # Uses: load_clase_suelo_by_municipality_v2, load_siu_recintos_by_municipality,
            #       perform_spatial_clip, calculate_ellipsoidal_area

            # Normalizer (uppercase, remove accents, collapse spaces)
            import unicodedata as _u, re as _re
            def _norm_txt(_s):
                if pd.isna(_s): return ""
                _s = str(_s).strip()
                _s = "".join(c for c in _u.normalize("NFKD", _s) if _u.category(c) != "Mn")
                _s = _re.sub(r"\s+", " ", _s.upper())
                return _s

            # 1) Load & clip both layers
            gdf_clase = load_clase_suelo_by_municipality_v2(selected_muni)
            gdf_siu   = load_siu_recintos_by_municipality(selected_muni)

            gdf_clase_clip = perform_spatial_clip(gdf_clase, gdf_muni) if (gdf_clase is not None and not gdf_clase.empty) else None
            gdf_siu_clip   = perform_spatial_clip(gdf_siu,   gdf_muni) if (gdf_siu   is not None and not gdf_siu.empty)   else None

            # 2) SUC: SUELO URBANO (en clasesuelo)
            SUC_ha = 0.0
            if gdf_clase_clip is not None and not gdf_clase_clip.empty and "clasesuelo" in gdf_clase_clip.columns:
                gC = gdf_clase_clip.copy()
                gC["_clase_norm"] = gC["clasesuelo"].apply(_norm_txt)
                gC_urb = gC[gC["_clase_norm"] == "SUELO URBANO"].copy()
                if not gC_urb.empty:
                    gC_urb = gC_urb.dissolve().explode(index_parts=False)
                    area_m2 = calculate_ellipsoidal_area(gC_urb.to_crs(4326))
                    SUC_ha = sum(a/10000 for a in area_m2)

            # 3) ADC: suma de usos en usosuelo
            ADC_ha = 0.0
            ADC_SET = {
                "DESARROLLO CONSOLIDADO",
                "SUELO EDIFICADO",
                "SUELO EN PROCESO DE EDIFICACION",
                "SUELO URBANIZADO O EN PROCESO DE URBANIZACION",
            }
            if gdf_siu_clip is not None and not gdf_siu_clip.empty and "usosuelo" in gdf_siu_clip.columns:
                gS = gdf_siu_clip.copy()
                gS["_uso_norm"] = gS["usosuelo"].apply(_norm_txt)
                target_norms = {_norm_txt(x) for x in ADC_SET}
                gS_adc = gS[gS["_uso_norm"].isin(target_norms)].copy()
                if not gS_adc.empty:
                    gS_adc = gS_adc.dissolve().explode(index_parts=False)
                    area_m2 = calculate_ellipsoidal_area(gS_adc.to_crs(4326))
                    ADC_ha = sum(a/10000 for a in area_m2)

            # 4) Total
            SUCADCtotal_ha = SUC_ha*1000 + ADC_ha*1000
            
            
            sucadc_mask_gdf = build_suc_adc_mask_from_clips(gdf_clase_clip, gdf_siu_clip)
            st.session_state["sucadc_mask_gdf"] = sucadc_mask_gdf   
            # Store/print
            st.session_state["SUC_ha"] = round(SUC_ha, 2)
            st.session_state["ADC_ha"] = round(ADC_ha, 2)
            st.session_state["SUCADCtotal_ha"] = round(SUCADCtotal_ha, 2)
            

            # ====================== 🏗️ EDIFICIOS: recorte por Municipio y por SUC+ADC ======================
            st.markdown("### 🏗️ Edificios: recorte por **Municipio** y por **SUC+ADC**")

            edif_file = st.file_uploader(
                "Sube el GML/SHP/GPKG/GeoJSON de **edificios** (huellas de edificios)",
                type=["gml", "zip", "gpkg", "geojson"],
                accept_multiple_files=False,
                key="uploader_edificios"
            )

            if edif_file is not None and selected_muni and gdf_muni is not None and not gdf_muni.empty:
                gdf_edif = None
                try:
                    gdf_edif = _read_any_vector(edif_file)
                except Exception as e:
                    st.error(f"❌ No pude leer la capa de edificios: {e}")

                if gdf_edif is not None and not gdf_edif.empty:
                    display_geodata_info(gdf_edif, edif_file.name)
                    edif_muni = perform_spatial_clip(gdf_edif, gdf_muni)
                    if edif_muni is None or edif_muni.empty:
                        st.warning("⚠️ No hay intersección entre los edificios y el término municipal.")
                    else:
                        edif_muni = edif_muni.copy()
                        n_muni = len(edif_muni)
                        area_muni_ha = float(edif_muni.get("area_ha", pd.Series(dtype=float)).sum()) if "area_ha" in edif_muni.columns else None
                        st.success(f"✅ Edificios dentro del municipio: **{n_muni:,}**")
                        if area_muni_ha is not None:
                            st.caption(f"Área total de huellas recortadas (Municipio): **{area_muni_ha:,.2f} ha**")

                    # Recorte 2: Edificios ∩ (SUC ∪ ADC)
                    sucadc_mask_gdf = st.session_state.get("sucadc_mask_gdf", None)
                    edif_sucadc = None
                    if sucadc_mask_gdf is None or sucadc_mask_gdf.empty:
                        st.info("ℹ️ No hay máscara SUC+ADC disponible (revisa que existan capas de clase de suelo y SIU para el municipio).")
                    else:
                        # Reutiliza perform_spatial_clip (ajusta CRS y recalcula áreas)
                        # Si ya hiciste el recorte municipal, recortamos ese resultado contra SUC+ADC (ambos dentro del municipio)
                        base_to_clip = edif_muni if (edif_muni is not None and not edif_muni.empty) else gdf_edif
                        edif_sucadc = perform_spatial_clip(base_to_clip, sucadc_mask_gdf)
                        if edif_sucadc is None or edif_sucadc.empty:
                            st.warning("⚠️ No hay edificios dentro de la máscara SUC+ADC.")
                        else:
                            edif_sucadc = edif_sucadc.copy()
                            n_sucadc = len(edif_sucadc)
                            area_sucadc_ha = float(edif_sucadc.get("area_ha", pd.Series(dtype=float)).sum()) if "area_ha" in edif_sucadc.columns else None
                            st.success(f"✅ Edificios dentro de SUC+ADC: **{n_sucadc:,}**")
                            if area_sucadc_ha is not None:
                                st.caption(f"Área total de huellas recortadas (SUC+ADC): **{area_sucadc_ha:,.2f} ha**")
                            # ====== CÁLCULO: Σ (área_huella_m2 × numberOfFloorsAboveGround) en SUC+ADC ======
                            # 1) Localiza la columna de plantas (case-insensitive)
                            floors_col = next((c for c in edif_sucadc.columns
                                            if c.lower() == "numberoffloorsaboveground"), None)

                            if floors_col is None:
                                st.warning("⚠️ No se encontró el campo 'numberOfFloorsAboveGround' en la capa de edificios.")
                            else:
                                # 2) Asegura área en m² para cada geometría del recorte SUC+ADC
                                #    (perform_spatial_clip ya crea 'area_m2'; si faltase, la calculamos elipsoidalmente)
                                if "area_m2" not in edif_sucadc.columns:
                                    try:
                                        edif_sucadc["area_m2"] = calculate_ellipsoidal_area(edif_sucadc.to_crs(4326))
                                    except Exception:
                                        # fallback métrico si algo falla con elipsoidal
                                        crs_metric = "EPSG:25830"
                                        g_tmp = edif_sucadc if (edif_sucadc.crs and edif_sucadc.crs.to_string()==crs_metric) \
                                                else edif_sucadc.to_crs(crs_metric)
                                        edif_sucadc["area_m2"] = g_tmp.geometry.area

                                # 3) Normaliza plantas a numérico (negativos → 0)
                                edif_sucadc["_floors_num"] = pd.to_numeric(edif_sucadc[floors_col], errors="coerce").fillna(0)
                                edif_sucadc["_floors_num"] = edif_sucadc["_floors_num"].clip(lower=0)

                                # 4) Superficie construida aproximada (GFA proxy) por edificio y total
                                edif_sucadc["m2_construidos_aprox"] = edif_sucadc["area_m2"] * edif_sucadc["_floors_num"]
                                total_m2_construidos = float(edif_sucadc["m2_construidos_aprox"].sum())

                                # 5) Mostrar resultado
                                st.subheader("🏗️ Superficie construida aproximada dentro de SUC+ADC")
                                colA, colB = st.columns(2)
                                with colA:
                                    st.metric("Σ (área × plantas)", f"{total_m2_construidos:,.0f} m²")
                                with colB:
                                    st.caption(f"Edificios contados: {len(edif_sucadc):,} • Campo de plantas usado: **{floors_col}**")

                                # 6) (Opcional) Top 10 edificios por m² construidos aprox.
                                with st.expander("Ver top 10 edificios por m² construidos aprox."):
                                    cols_show = [floors_col, "area_m2", "m2_construidos_aprox"]
                                    cols_show = [c for c in cols_show if c in edif_sucadc.columns]
                                    st.dataframe(
                                        edif_sucadc[cols_show]
                                            .sort_values("m2_construidos_aprox", ascending=False)
                                            .head(10),
                                        use_container_width=True
                                    )




    

                
            st.markdown("### 🧮 SUC / ADC / Total (ha)")
            c1, c2, c3 = st.columns(3)
            c1.metric("SUC (ha) – SUELO URBANO", f"{st.session_state['SUC_ha']:,}")
            c2.metric("ADC (ha) – usos seleccionados", f"{st.session_state['ADC_ha']:,}")
            c3.metric("SUC + ADC (ha)", f"{st.session_state['SUCADCtotal_ha']:,}")
           
                
        
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

            # === SEPE para D.26 y D.28 (carga una sola vez) ===
            try:
                parquet_path_for_d26 = st.session_state.get("sepe_parquet_path", "sepe_global.parquet")
                df_sepe_all = load_sepe_parquet(parquet_path_for_d26)
                sector_year_df = sector_shares_by_year(df_sepe_all, selected_muni, debug=False)
            except Exception as e:
                st.warning(f"No se pudieron calcular D.26 desde SEPE: {e}")
                sector_year_df = pd.DataFrame()
                df_sepe_all = None



            for year in YEARS:
                total = pop_df.get(f"total_total_total_{year}", pd.Series([0])).values[0]
                over_65 = pop_df[[f"total_{age}_total_{year}" for age in age_65_plus if f"total_{age}_total_{year}" in pop_df.columns]].sum(axis=1).values[0]
                over_85 = pop_df[[f"total_{age}_total_{year}" for age in age_85_plus if f"total_{age}_total_{year}" in pop_df.columns]].sum(axis=1).values[0]
                foreign = pop_df.get(f"total_total_EX_{year}", pd.Series([0])).values[0]
                pop_0_14 = pop_df[[f"total_{age}_total_{year}" for age in ages_0_14 if f"total_{age}_total_{year}" in pop_df.columns]].sum(axis=1).values[0]
                pop_15_64 = pop_df[[f"total_{age}_total_{year}" for age in ages_15_64 if f"total_{age}_total_{year}" in pop_df.columns]].sum(axis=1).values[0]

                # --- Indicadores DGT por año ---
                veh_1000hab, pct_turismos, pct_motos = None, None, None
                antig_media = None 
                try:
                    df_dgt_year = df_dgt_by_year.get(year)
                    if df_dgt_year is not None:
                        dgt_row = df_dgt_year[df_dgt_year["municipio_completo"].str.lower() == selected_muni.lower()]
                        if not dgt_row.empty:
                            turismos = dgt_row["Parque Turismos"].values[0]
                            motos = dgt_row["Parque Motocicletas"].values[0]
                            total_veh = turismos + motos
                            total_total = dgt_row["Parque Total"].values[0]
                            pop_year = total
                            if pop_year and (turismos is not None) and (motos is not None):
                                veh_1000hab = round((turismos + motos) / pop_year * 1000, 2)
                            if total_veh:
                                pct_turismos = round(turismos / total_total * 100, 2)
                                pct_motos = round(motos / total_total * 100, 2)
                                        # Antigüedad media del parque (media simple) — D.18.d
                            cols_antig = [
                                "Antigüedad Media de Camiones",
                                "Antigüedad Media de Turismos",
                                "Antigüedad Media de Furgonetas",
                                "Antigüedad Media de Ciclomotores",
                                "Antigüedad Media de Motocicletas",
                            ]
                            vals = []
                            for c in cols_antig:
                                if c in dgt_row.columns:
                                    vals.append(pd.to_numeric(dgt_row[c].iloc[0], errors="coerce"))
                            # media de los disponibles (ignora NaN)
                            if len(vals):
                                m = np.nanmean(vals)
                                antig_media = round(float(m), 2) if not np.isnan(m) else None
                except Exception:
                    pass

                row = {
                    "Año": year,
                    "D.1 Variación Poblacional Últimos 10 años (%)": pop_variation_dict.get(year),
                    "D.18.a. Vehículos domiciliados cada 1000 hab.": veh_1000hab,
                    "D.18.b. % Turismos": pct_turismos,
                    "D.18.c. % Motocicletas": pct_motos,
                    "D.18.d. Antigüedad media del parque (años)": antig_media,
                    "D.22.a. Envejecimiento (%)": round(over_65 / total * 100, 2) if total else None,
                    "D.22.b. Senectud (%)": round(over_85 / over_65 * 100, 2) if over_65 else None,
                    "D.23 Población extranjera (%)": round(foreign / total * 100, 2) if total else None,
                    "D.24.a. Dependencia total (%)": round((pop_0_14 + over_65) / pop_15_64 * 100, 2) if pop_15_64 else None,
                    "D.24.b. Dependencia infantil (%)": round(pop_0_14 / pop_15_64 * 100, 2) if pop_15_64 else None,
                    "D.24.c. Dependencia mayores (%)": round(over_65 / pop_15_64 * 100, 2) if pop_15_64 else None,
                    "D.29 Viviendas por persona": None,
                    "D.32 Variación hogares 2011-2021 (%)": var_hogares_pct if year == "2021" else None,
                    "D.33 Crecimiento parque viviendas 2011-2021 (%)": crecimiento_viviendas_pct if year == "2021" else None,
                    "D.34 Vivienda secundaria (%)": None,
                    "D.35 Vivienda vacía 2011 (%)": viv_vacia_pct_2011 if year == "2021" else None,
                }
            
                row["D.02.a (CORINE) Superficie artificial (%)"] = d02a_corine_value
                # Indicador nuevo: Superficie cultivos código16 / superficie municipio
                try:
                    if (
                        "sup_cultivos_16" in st.session_state and 
                        st.session_state["sup_cultivos_16"] is not None and
                        gdf_muni is not None and 
                        not gdf_muni.empty
                    ):
   
                        muni_area_ha = sum(calculate_ellipsoidal_area(gdf_muni.to_crs(4326))) / 10000

                        if muni_area_ha > 0:
                            pct16 = round((st.session_state["sup_cultivos_16"] / muni_area_ha) * 100, 2)
                            row["D.02.b. Superficie de Cultivos (cod_16)"] = pct16
                        else:
                            row["D.02.b. Superficie de Cultivos (cod_16)"] = None
                    else:
                        row["D.02.b. Superficie de Cultivos (cod_16)"] = None
                except:
                    row["D.02.b. Superficie de Cultivos (cod_16)"] = None

                # Indicador nuevo: Superficie cultivos código19 / superficie municipio
                try:
                    if (
                        "sup_cultivos_19" in st.session_state and 
                        st.session_state["sup_cultivos_19"] is not None and
                        gdf_muni is not None and 
                        not gdf_muni.empty
                    ):
                        muni_area_ha = sum(calculate_ellipsoidal_area(gdf_muni.to_crs(4326))) / 10000

                        if muni_area_ha > 0:
                            pct19 = round((st.session_state["sup_cultivos_19"] / muni_area_ha) * 100, 2)
                            row["D.02.c. Superficie de zonas húmedas (%) (cod_19)"] = pct19
                        else:
                            row["D.02.c. Superficie de zonas húmedas (%) (cod_19)"] = None
                    else:
                        row["D.02.c. Superficie de zonas húmedas (%) (cod_19)"] = None
                except Exception:
                    row["D.02.c. Superficie de zonas húmedas (%) (cod_19)"] = None

                
                try:
                    if (
                        "sup_cultivos_17" in st.session_state and 
                        st.session_state["sup_cultivos_17"] is not None and
                        gdf_muni is not None and 
                        not gdf_muni.empty
                    ):
                        muni_area_ha = sum(calculate_ellipsoidal_area(gdf_muni.to_crs(4326))) / 10000

                        if muni_area_ha > 0:
                            pct17 = round((st.session_state["sup_cultivos_17"] / muni_area_ha) * 100, 2)
                            row["D.02.d. Superficie forestal (%) (cod_17)"] = pct17
                        else:
                            row["D.02.d. Superficie forestal (%) (cod_17)"] = None
                    else:
                        row["D.02.d. Superficie forestal (%) (cod_17)"] = None
                except Exception:
                    row["D.02.d. Superficie forestal (%) (cod_17)"] = None


                try:
                    if (
                        "sup_cultivos_15" in st.session_state and 
                        st.session_state["sup_cultivos_15"] is not None and
                        gdf_muni is not None and 
                        not gdf_muni.empty
                    ):
                        muni_area_ha = sum(calculate_ellipsoidal_area(gdf_muni.to_crs(4326))) / 10000

                        if muni_area_ha > 0:
                            pct15 = round((st.session_state["sup_cultivos_15"] / muni_area_ha) * 100, 2)
                            row["D.02.e. Superficie minería (%) (cod_15)"] = pct15
                        else:
                            row["D.02.e. Superficie minería (%) (cod_15)"] = None
                    else:
                        row["D.02.e. Superficie minería (%) (cod_15)"] = None
                except Exception:
                    row["D.02.e. Superficie minería (%) (cod_15)"] = None

                if (
                    "sup_cultivos_09" in st.session_state and 
                    st.session_state["sup_cultivos_09"] is not None and
                    total
                ):
                    verde_1000hab = round(st.session_state["sup_cultivos_09"] / (total / 1000), 2)
                    row["D.5. Superficie verde (ha cada 1.000 hab) (cod_09)"] = verde_1000hab
                else:
                    row["D.5. Superficie verde (ha cada 1.000 hab) (cod_09)"] = None

                # Indicador nuevo: Superficie cultivos código14 / superficie municipio
                try:
                    if (
                        "sup_cultivos_14" in st.session_state and 
                        st.session_state["sup_cultivos_14"] is not None and
                        gdf_muni is not None and 
                        not gdf_muni.empty
                    ):
                        muni_area_ha = sum(calculate_ellipsoidal_area(gdf_muni.to_crs(4326))) / 10000
                        if muni_area_ha > 0:
                            sup14_pct = round((st.session_state["sup_cultivos_14"] / muni_area_ha) * 100, 2)
                            row["D.03.a. Superficie municipal de explotaciones agrarias y forestales (cod_14)"] = sup14_pct
                        else:
                            row["D.03.a. Superficie municipal de explotaciones agrarias y forestales (cod_14)"] = None
                    else:
                        row["D.03.a. Superficie municipal de explotaciones agrarias y forestales (cod_14)"] = None
                except:
                    row["D.03.a. Superficie municipal de explotaciones agrarias y forestales (cod_14)"] = None

                # Indicadores para código 12
                try:
                    if (
                        "sup_cultivos_12" in st.session_state and 
                        st.session_state["sup_cultivos_12"] is not None and
                        gdf_muni is not None and 
                        not gdf_muni.empty
                    ):
                        # Indicador 1: solo la superficie
                        row["D.17.a. Superficie infraestructura de transporte (ha)(cod: 12)"] = round(st.session_state["sup_cultivos_12"], 2)
                
                        # Indicador 2: porcentaje sobre sup. municipalmunicipio_area_ha

                        
                        muni_area_ha = sum(calculate_ellipsoidal_area(gdf_muni.to_crs(4326))) / 10000
                        pct12 = round((st.session_state["sup_cultivos_12"] / muni_area_ha) * 100, 2)
                        row["D.17.b. Superficie infraestructura de transporte (%)(cod: 12)"] = pct12
                    else:
                        row["D.17.a. Superficie infraestructura de transporte (ha)(cod: 12)"] = None
                        row["D.17.b. Superficie infraestructura de transporte (%)(cod: 12)"] = None
                except:
                    row["D.17.a. Superficie infraestructura de transporte (ha)(cod: 12)"] = None
                    row["D.17.b. Superficie infraestructura de transporte (%)(cod: 12)"] = None
                
                if year == "2021":
                    try:
                        v_total = censo_df["viviendasT"].values[0]
                        v_nop = censo_df["viviendasNoP"].values[0]
                        pop_2021 = pop_df["total_total_total_2021"].values[0]
                        row["D.34 Vivienda secundaria (%)"] = round((v_nop / v_total) * 100, 2)
                        row["D.29 Viviendas por persona"] = round((v_total / pop_2021) * 1000, 4)       
                    except:
                        pass
                
               
                row["D.03.b. % COD14 sobre Suelo Urb./Urbanizable"] = d03b_value
                row["D.04. % SNU + SUD no delimitado sobre municipio"] = d04_value

                # D.06 municipal = población INE del año / (SUC + ADC) [ha]
                sucadc_ha = st.session_state.get("SUCADCtotal_ha")  # set earlier when you computed SUC/ADC
                row["D.06 Densidad sobre SUC+ADC (hab/ha)"] = (
                    round(total / sucadc_ha, 2) if (total and sucadc_ha and sucadc_ha > 0) else None
                )

                # Indicador nuevo: Superficie cultivos código19 / superficie municipio
                try:
                    if (
                        "sup_cultivos_03" in st.session_state and 
                        "sup_cultivos_02" in st.session_state and 
                        "sup_cultivos_01" in st.session_state and 
                        st.session_state["sup_cultivos_03"] is not None and
                        st.session_state["sup_cultivos_02"] is not None and
                        st.session_state["sup_cultivos_01"] is not None and
                        gdf_muni is not None and 
                        not gdf_muni.empty
                    ):
                        muni_area_ha = sum(calculate_ellipsoidal_area(gdf_muni.to_crs(4326))) / 10000

                        if muni_area_ha > 0:
                            pct3 = round(st.session_state["sup_cultivos_03"], 2)
                            pct2 = round(st.session_state["sup_cultivos_02"], 2)
                            pct1 = round(st.session_state["sup_cultivos_01"], 2)
                            row["D.07. Suelo urbano discontinuo (%)"] = pct3*100/(pct3+pct1+pct2)
                        else:
                            row["D.07. Suelo urbano discontinuo (%)"] = None
                    else:
                        row["D.07. Suelo urbano discontinuo (%)"] = None
                except Exception:
                    row["D.07. Suelo urbano discontinuo (%)"] = None


                # D.08.d: Densidad de vivienda (viv/ha) = viviendas / (SUC+ADC)
                try:
                    sucadc_ha = st.session_state.get("SUCADCtotal_ha")
                    viv_tot_2021 = None
                    if censo_df is not None and not censo_df.empty and "viviendasT" in censo_df.columns:
                        vv = pd.to_numeric(censo_df["viviendasT"].iloc[0], errors="coerce")
                        viv_tot_2021 = None if pd.isna(vv) else float(vv)

                    if sucadc_ha and sucadc_ha > 0 and viv_tot_2021:
                        # Usamos el valor de viviendas 2021 para todos los años (última referencia disponible)
                        row["D.08. Densidad de vivienda (viv/ha)"] = round(viv_tot_2021 / sucadc_ha, 2)
                    else:
                        row["D.08. Densidad de vivienda (viv/ha)"] = None
                except Exception:
                    row["D.08, Densidad de vivienda (viv/ha)"] = None



                # --- D.26: % trabajadores por sector (media anual SEPE Contratos) ---
                vals = sector_year_df[sector_year_df["anio"] == int(year)]
                if not vals.empty:
                    row["D.26.a. Trabajadores en sector agricultura (%)"]  = vals.iloc[0]["agr_avg"]
                    row["D.26.b. Trabajadores en sector industria (%)"]    = vals.iloc[0]["ind_avg"]
                    row["D.26.c. Trabajadores en sector construcción (%)"] = vals.iloc[0]["con_avg"]
                    row["D.26.d. Trabajadores en sector servicios (%)"]    = vals.iloc[0]["ser_avg"]
                else:
                    row["D.26.a. Trabajadores en sector agricultura (%)"]  = None
                    row["D.26.b. Trabajadores en sector industria (%)"]    = None
                    row["D.26.c. Trabajadores en sector construcción (%)"] = None
                    row["D.26.d. Trabajadores en sector servicios (%)"]    = None

                # --- D.28 (SEPTIEMBRE) ---
                # Si SOLO quieres 2024, deja la condición del año. Si lo quieres para todos los años, quita el 'and int(year) == 2024'.
                d28a = d28b = d28c = None
                if df_sepe_all is not None:  # and int(year) == 2024:
                    d28 = compute_d28(df_sepe_all, selected_muni, int(year), month_num=9)
                    n_total  = d28.get("n_total")
                    n_25_44  = d28.get("n_25_44")
                    n_mujer  = d28.get("n_mujer")

                    if n_total and pop_15_64:  # evita divisiones por 0/None
                        d28a = round(n_total / pop_15_64 * 100, 2)
                    if n_total:
                        d28b = round(n_25_44 / n_total * 100, 2) if n_25_44 is not None else None
                        d28c = round(n_mujer / n_total * 100, 2)   if n_mujer is not None else None

                row["D.28.a. Porcentaje de parados total (%)"] = d28a
                row["D.28.b. Porcentaje de parados entre 25 y 44 años (%)"] = d28b
                row["D.28.c. Porcentaje de paro femenino (%)"] = d28c




                results.append(row)

            results_df = pd.DataFrame(results)
        

            # Ordenar columnas por el número después de "D."
            def get_d_number(col):
                match = re.search(r"D\.(\d+)", col)
                if match:
                    return int(match.group(1))
                return 9999  # columnas que no tengan D.x al final

            ordered_cols = sorted(results_df.columns, key=get_d_number)

            # Reordenar DataFrame
            results_df = results_df[["Año"] + [c for c in ordered_cols if c != "Año"]]


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
        
            # Comparativa septiembre (foto del año) – solo para información
            if selected_muni and 'sector_year_df' in locals() and not sector_year_df.empty:
                comp_sep = (sector_year_df[["anio","agr_sep","ind_sep","con_sep","ser_sep"]]
                            .rename(columns={
                                "anio":"Año",
                                "agr_sep":"Sept. Agricultura (%)",
                                "ind_sep":"Sept. Industria (%)",
                                "con_sep":"Sept. Construcción (%)",
                                "ser_sep":"Sept. Servicios (%)"
                            })
                            .sort_values("Año"))
                st.caption("📌 Comparativa de **septiembre** por año (referencia, la tabla principal usa la **media anual**):")
                st.dataframe(comp_sep, use_container_width=True, hide_index=True)

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

                    from matplotlib import cm, colors
                    cmap = cm.get_cmap('tab20', len(unique_codsiu))
                    codsiu_to_color = {cod: colors.to_hex(cmap(i)) for i, cod in enumerate(unique_codsiu)}
                
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


                # ===================== MAPA 2: Intersecciones de Clase de Suelo =====================
                st.markdown("### 🗺️ Intersecciones: Clase de Suelo en el municipio")
                
                if selected_muni and gdf_muni is not None and not gdf_muni.empty:
                    gdf_clase = load_clase_suelo_by_municipality_v2(selected_muni)
                    
                    if gdf_clase is None or gdf_clase.empty:
                        st.info("ℹ️ No se encontraron registros de clase de suelo para este municipio.")
                    else:
                        # 1) Intersección/recorte con el término municipal (usa tu helper)
                        gdf_clase_clip = perform_spatial_clip(gdf_clase, gdf_muni)
                
                        if gdf_clase_clip is None or gdf_clase_clip.empty:
                            st.warning("⚠️ La capa de clase de suelo no intersecta con el municipio (resultado vacío).")
                        else:
                            # 2) Detectar una columna de categoría para colorear el mapa
                            #    Ajusta la prioridad a tus nombres reales si ya los sabes.
                            candidate_cols = [
                                "clase_suelo", "clase", "clas_suelo", "Clase", "tipo", "category", "Tipo", "uso"
                            ]
                            category_col = next((c for c in candidate_cols if c in gdf_clase_clip.columns), None)
                
                            # 3) Pasar a WGS84 para webmapping
                            gdf_muni_4326  = gdf_muni.to_crs(4326)
                            gdf_clase_4326 = gdf_clase_clip.to_crs(4326)
                
                            # 4) Construir mapa
                            bounds = gdf_muni_4326.total_bounds
                            center_lat = (bounds[1] + bounds[3]) / 2
                            center_lon = (bounds[0] + bounds[2]) / 2
                            m2 = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="OpenStreetMap")
                
                            # Capa del municipio en borde
                            folium.GeoJson(
                                gdf_muni_4326.__geo_interface__,
                                name="Municipio",
                                style_function=lambda f: {"color": "black", "weight": 2, "fillOpacity": 0.0}
                            ).add_to(m2)
                
                            # 5) Coloreado por categoría (si existe) o color único
                            import matplotlib
                            import matplotlib.pyplot as plt
                
                            if category_col:
                                unique_vals = [str(v) for v in gdf_clase_4326[category_col].fillna("SIN_CATEGORIA").unique()]
                                from matplotlib import cm, colors
                                cmap = cm.get_cmap("tab20", len(unique_vals))
                                lut = {val: colors.to_hex(cmap(i)) for i, val in enumerate(sorted(unique_vals))}
                                
                                def styler(feat):
                                    val = str(feat["properties"].get(category_col, "SIN_CATEGORIA"))
                                    return {"color": lut.get(val, "#555555"), "fillColor": lut.get(val, "#555555"),
                                            "weight": 1, "fillOpacity": 0.5}
                
                                gj = folium.GeoJson(
                                    gdf_clase_4326.__geo_interface__,
                                    name="Clase de suelo (intersecciones)",
                                    style_function=styler,
                                    tooltip=folium.GeoJsonTooltip(
                                        fields=[category_col] + [c for c in gdf_clase_4326.columns if c not in ("geometry","geom")][:5],
                                        aliases=[category_col] + ["Atributo 1", "Atributo 2", "Atributo 3", "Atributo 4", "Atributo 5"],
                                        sticky=True
                                    ),
                                ).add_to(m2)
                
                                # Leyenda simple
                                legend_html = """
                                <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999; 
                                            background: white; padding: 8px 10px; border: 1px solid #ccc; 
                                            border-radius: 6px; font-size: 12px; max-height: 220px; overflow:auto;">
                                    <b>Clase de suelo</b><br>
                                """
                                for val in sorted(unique_vals):
                                    legend_html += f'<div style="margin:3px 0;"><span style="display:inline-block;width:12px;height:12px;background:{lut[val]};margin-right:6px;border:1px solid #999;"></span>{val}</div>'
                                legend_html += "</div>"
                                m2.get_root().html.add_child(folium.Element(legend_html))
                
                            else:
                                # Sin categoría detectable: pintar uniforme
                                folium.GeoJson(
                                    gdf_clase_4326.__geo_interface__,
                                    name="Clase de suelo (intersecciones)",
                                    style_function=lambda f: {"color": "#377eb8", "fillColor": "#377eb8",
                                                              "weight": 1, "fillOpacity": 0.5},
                                    tooltip=folium.GeoJsonTooltip(
                                        fields=[c for c in gdf_clase_4326.columns if c not in ("geometry","geom")][:6],
                                        aliases=["Campo 1","Campo 2","Campo 3","Campo 4","Campo 5","Campo 6"],
                                        sticky=True
                                    ),
                                ).add_to(m2)
                
                            folium.LayerControl(collapsed=False).add_to(m2)
                            st_folium(m2, width=1500, height=520, key="map_clase_suelo")            
                else:
                    st.info("Selecciona un municipio para ver la clase de suelo.")

            
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

    st.markdown("## 📈 Evolución temporal SEPE (paro y contratos)")
    parquet_path = st.text_input("Ruta parquet SEPE", value="sepe_global.parquet")
    st.session_state["sepe_parquet_path"] = parquet_path

    try:
        
        df_sepe = load_sepe_parquet(parquet_path)
        if selected_muni:
            df_sepe = subset_muni_first(df_sepe, selected_muni)
        if selected_muni:
            ts_paro, ts_cont = build_timeseries(df_sepe, selected_muni)

            colA, colB = st.columns(2)
            with colA:
                st.subheader("Paro total")
                if not ts_paro.empty:
                    st.line_chart(ts_paro.set_index("fecha"))
                else:
                    st.info("No hay datos de paro para este municipio en el parquet.")

            with colB:
                st.subheader("Contratos totales")
                if not ts_cont.empty:
                    st.line_chart(ts_cont.set_index("fecha"))
                else:
                    st.info("No hay datos de contratos para este municipio en el parquet.")

            with st.expander("Ver desgloses (si existen en el parquet)"):
                muni_norm_target = normalize_muni(selected_muni)

                posibles_p = ["pSAgricultura","pSIndustria","pSConstruccion","pSServicios","pSSinEmpleo",
                            "pH25","pH2544","pH45","pM25","pM2544","pM45"]
                cols_presentes_p = [c for c in posibles_p if c in df_sepe.columns]
                if cols_presentes_p:
                    dfp = (df_sepe[(df_sepe.get("tipo")=="p") & (df_sepe["muni_norm"]==muni_norm_target)]
                        [["fecha"]+cols_presentes_p]
                        .groupby("fecha", as_index=False).sum().sort_values("fecha"))
                    st.write("Paro - desgloses")
                    st.dataframe(dfp.tail(12), use_container_width=True)
                else:
                    st.caption("No se encontraron columnas de desglose de paro.")

                posibles_c = ["cSAgricultura","cSIndustria","cSConstruccion","cSServicios",
                            "cHInicIndefinido","cHTemporal","cHConvertIndefinido",
                            "cMInicIndefinido","cMTemporal","cMConvertIndefinido"]
                cols_presentes_c = [c for c in posibles_c if c in df_sepe.columns]
                if cols_presentes_c:
                    dfc = (df_sepe[(df_sepe.get("tipo")=="c") & (df_sepe["muni_norm"]==muni_norm_target)]
                        [["fecha"]+cols_presentes_c]
                        .groupby("fecha", as_index=False).sum().sort_values("fecha"))
                    st.write("Contratos - desgloses")
                    st.dataframe(dfc.tail(12), use_container_width=True)
                else:
                    st.caption("No se encontraron columnas de desglose de contratos.")

    except Exception as e:
        st.error(f"❌ Error con el parquet SEPE: {e}")

    with st.expander("🔎 Depuración D.26 — contratos por sector (valores brutos y %)", expanded=False):
        try:
            # Usa la misma ruta que el text input de abajo, o fija aquí la ruta
            parquet_path_for_d26 = st.session_state.get("sepe_parquet_path", "sepe_global.parquet")
            df_sepe_all = load_sepe_parquet(parquet_path_for_d26)
            show_d26_debug(df_sepe_all, selected_muni)
        except Exception as e:
            st.error(f"No se pudo generar la depuración D.26: {e}")

    

    st.markdown("### 📋 Áreas (ha) y 🔀 Intersección **usosuelo × clasesuelo**")

    if selected_muni and gdf_muni is not None and not gdf_muni.empty:
        out = build_clase_siu_area_and_intersection(selected_muni, gdf_muni)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Clase de Suelo (ha)")
            if out["tabla_clase"] is None or out["tabla_clase"].empty:
                st.info("Sin datos de clase de suelo (o sin intersección).")
            else:
                st.dataframe(out["tabla_clase"], use_container_width=True, hide_index=True)
                st.caption(f"Total: **{out['tabla_clase']['Área (ha)'].sum():,.2f} ha**")

        with col2:
            st.subheader("SIU – usosuelo (ha)")
            if out["tabla_siu"] is None or out["tabla_siu"].empty:
                st.info("Sin datos de SIU (o sin intersección).")
            else:
                st.dataframe(out["tabla_siu"], use_container_width=True, hide_index=True)
                st.caption(f"Total: **{out['tabla_siu']['Área (ha)'].sum():,.2f} ha**")

        st.subheader("Intersección usosuelo × clasesuelo (ha)")
        if out["tabla_long"] is None or out["tabla_long"].empty:
            st.info("No hay intersección entre SIU y Clase de suelo dentro del municipio.")
        else:
            st.dataframe(out["tabla_long"], use_container_width=True, hide_index=True)
            if out["matriz_pivot"] is not None:
                st.caption("Matriz (ha): usosuelo × clasesuelo")
                st.dataframe(out["matriz_pivot"].style.format("{:,.2f}"), use_container_width=True)
    else:
        st.info("Selecciona un municipio para calcular áreas e intersecciones.")

    # === ETL Catastro (local) – Integración con Streamlit ===
    
    import comando  # <-- tu script ETL
    def _solo_nombre_muni(s: str) -> str:
        """
        Quita cualquier número/código inicial y separadores comunes.
        '08019 Barcelona' -> 'Barcelona'
        '08019 - Barcelona' -> 'Barcelona'
        """
        if s is None:
            return ""
        s = str(s).strip()
        # quita números iniciales + separadores (- – — . · : y espacios)
        s = re.sub(r"^\s*\d+\s*[-–—:.·]*\s*", "", s)
        return s.strip()

    st.markdown("## 🧱 ETL Catastro (archivos locales)")

    # 1) Mostrar el municipio ya elegido en tu UI (usas selected_muni de tu app)
    selected_muni_clean = _solo_nombre_muni(selected_muni) if selected_muni else None

    if not selected_muni_clean:
        st.info("Selecciona un municipio arriba para poder ejecutar el ETL.")
    else:
        st.write(f"Municipio seleccionado: **{selected_muni_clean}**")

        # 2) Entradas de rutas (Streamlit no tiene 'selector de carpeta', así que usamos text_input)
        col_a, col_b = st.columns(2)
        with col_a:
            cat_dir = st.text_input(
                "Carpeta con ficheros CAT/CAT.gz",
                value=r"C:\ruta\a\mi\carpeta\CAT",
                placeholder=r"C:\Users\...\Desktop\CAT"
            )
        with col_b:
            out_dir = st.text_input(
                "Carpeta de salida",
                value=r"C:\ruta\a\mi\carpeta\ETL_Salida",
                placeholder=r"C:\Users\...\Desktop\ETL_Salida"
            )

        # 3) Botón para ejecutar
        run = st.button("▶️ Ejecutar ETL con estas rutas", disabled=not selected_muni_clean)

        # 4) Ejecutar y capturar logs de print() de tu script
        if run:
            # Validaciones mínimas
            if not os.path.isdir(cat_dir):
                st.error("❌ La carpeta de entrada CAT no existe.")
            elif not out_dir:
                st.error("❌ Indica una carpeta de salida.")
            else:
                os.makedirs(out_dir, exist_ok=True)

                # Captura de stdout/stderr para mostrar en la UI
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    with st.status("Ejecutando ETL…", expanded=True) as status:
                        try:
                            st.write("📦 Carpeta CAT:", cat_dir)
                            st.write("📤 Carpeta salida:", out_dir)
                            st.write("🏙️ Municipio:", selected_muni_clean)

                            # usa el municipio LIMPIO
                            comando.run_etl(cat_dir, selected_muni_clean, out_dir)

                            status.update(label="✅ ETL finalizado", state="complete")
                        except Exception as e:
                            status.update(label="❌ ETL con errores", state="error")
                            st.exception(e)

                # Mostrar el log completo
                st.subheader("📝 Log de ejecución")
                st.text(buf.getvalue())

                # 5) Si todo fue bien, ofrecer descargas de los CSV generados
                esperados = [
                    "reg11_finca.csv",
                    "reg13_uc.csv",
                    "reg14_construccion.csv",
                    "reg15_inmueble.csv",
                    "reg16_reparto.csv",
                    "reg17_cultivo.csv",
                    "resumen_parcela.csv",
                ]
                generados = [f for f in esperados if os.path.isfile(os.path.join(out_dir, f))]
                if generados:
                    st.subheader("⬇️ Archivos generados")
                    for fname in generados:
                        fpath = os.path.join(out_dir, fname)
                        with open(fpath, "rb") as fh:
                            st.download_button(
                                f"Descargar {fname}",
                                data=fh.read(),
                                file_name=fname,
                                mime="text/csv",
                            )
                else:
                    st.info("No se encontraron CSV esperados en la carpeta de salida.")


    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; font-size: 0.8em;'>
            📊 Aplicación de Indicadores INE por Municipio<br>
            Datos del Instituto Nacional de Estadística (INE)
        </div>
        """, unsafe_allow_html=True)
    
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Mostrar en la UI y forzar que salga en logs
        st.error("Error al iniciar la app:")
        st.exception(e)
        traceback.print_exc(file=sys.stderr)
        raise