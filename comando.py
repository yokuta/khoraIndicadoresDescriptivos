# etl_cat.py
# -*- coding: utf-8 -*-

from pathlib import Path
import gzip
import io
import re
import csv
import sys
import unicodedata
import contextlib 
# =========================
# Utilidades de parsing
# =========================

def _slice(line: str, pos1: int, length: int) -> str:
    """Corte 1-based inclusive; posiciones del PDF empiezan en 1."""
    i0 = pos1 - 1
    i1 = i0 + length
    if i0 < 0 or i0 >= len(line):
        return ""
    return line[i0:i1]

def _to_int(s: str, default=0) -> int:
    s = (s or "").strip()
    return int(s) if s.isdigit() else default

def _to_float_scaled_int(s: str, scale_pow10=0, default=0.0) -> float:
    """Para numéricos con decimales implícitos (no se usa aquí, pero útil)."""
    s = (s or "").strip()
    if not s.isdigit():
        return default
    num = int(s)
    return num / (10 ** scale_pow10)

def _norm_txt(s: str) -> str:
    return (s or "").rstrip()

def _norm_key(s: str) -> str:
    """Normaliza para comparación: minúsculas, sin tildes, colapsa espacios."""
    s = (s or "").strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s)
    return s

USO_15 = {
    "A": "Almacén/Estacionamiento",
    "V": "Residencial",
    "I": "Industrial",
    "O": "Oficinas",
    "C": "Comercial",
    "K": "Deportivo",
    "T": "Espectáculos",
    "G": "Ocio y Hostelería",
    "Y": "Sanidad y Beneficencia",
    "E": "Cultural",
    "R": "Religioso",
    "M": "Urbanización/Jardinería/Suelo sin edificar",
    "P": "Edificio singular",
    "B": "Almacén agrario",
    "J": "Industrial agrario",
    "Z": "Agrario",
}

# Algunos códigos 14 (destino) frecuentes (amplía si quieres)
DESTINO_14 = {
    "V": "Vivienda",
    "AAL": "Almacén",
    "KPS": "Deportivo (pista)",
    "KDP": "Deportivo",
    "YPO": "Porche (100%)",
    "YJD": "Jardín (100%)",
    # Añade más si los vas viendo en tus CAT
}

# =========================
# Parseadores por tipo
# =========================

def parse_11(line: str) -> dict:
    # Registro de Finca (11) — posiciones según PDF (revisión 2022)
    return {
        "tipo": "11",
        "delegacion": _slice(line, 24, 2),
        "cod_muni_dgc": _slice(line, 26, 3),
        "parcela": _slice(line, 31, 14),
        "prov_cod_ine": _slice(line, 51, 2),
        "prov_nom": _norm_txt(_slice(line, 53, 25)),
        "muni_cod_dgc": _slice(line, 78, 3),
        "muni_cod_ine": _slice(line, 81, 3),
        "muni_nom": _norm_txt(_slice(line, 84, 40)),
        "paraje_nom": _norm_txt(_slice(line, 266, 30)),
        "sup_parcela_m2": _to_int(_slice(line, 296, 10)),
        "sup_construida_total_m2": _to_int(_slice(line, 306, 7)),
        "sup_sobre_rasante_m2": _to_int(_slice(line, 313, 7)),
        "sup_bajo_rasante_m2": _to_int(_slice(line, 320, 7)),
        "sup_cubierta_m2": _to_int(_slice(line, 327, 7)),
        "coord_x": _slice(line, 334, 9),
        "coord_y": _slice(line, 343, 10),
        "srs": _norm_txt(_slice(line, 667, 10)),
    }

def parse_13(line: str) -> dict:
    # Registro de Unidad Constructiva (13)
    return {
        "tipo": "13",
        "delegacion": _slice(line, 24, 2),
        "cod_muni_dgc": _slice(line, 26, 3),
        "clase_uc": _slice(line, 29, 2),  # UR/RU/BI
        "parcela": _slice(line, 31, 14),
        "uc": _slice(line, 45, 4),
        "prov_nom": _norm_txt(_slice(line, 53, 25)),
        "muni_nom": _norm_txt(_slice(line, 84, 40)),
        "anio_construccion": _slice(line, 296, 4),
        "indicador_anio": _slice(line, 300, 1),  # E/+/-
        "huella_m2": _to_int(_slice(line, 301, 7)),
        "long_fachada_cm": _to_int(_slice(line, 308, 5)),
        "uc_matriz": _norm_txt(_slice(line, 410, 4)),  # ver histórico
    }

def parse_14(line: str) -> dict:
    # Registro de Construcción/Local (14)
    return {
        "tipo": "14",
        "delegacion": _slice(line, 24, 2),
        "cod_muni_dgc": _slice(line, 26, 3),
        "parcela": _slice(line, 31, 14),
        "elem_orden": _slice(line, 45, 4),
        "cargo": _slice(line, 51, 4),       # Nº de cargo (vacío si elemento común)
        "uc": _slice(line, 55, 4),
        "bloque": _slice(line, 59, 4),
        "esc": _slice(line, 63, 2),
        "planta": _slice(line, 65, 3),
        "puerta": _slice(line, 68, 3),
        "destino_cod": _slice(line, 71, 3).strip(),
        "reforma_tipo": _slice(line, 74, 1),
        "reforma_anio": _slice(line, 75, 4),
        "antig_efectiva": _slice(line, 79, 4),
        "local_interior": _slice(line, 83, 1),
        "sup_local_m2": _to_int(_slice(line, 84, 7)),
        "sup_porches_m2": _to_int(_slice(line, 91, 7)),
        "sup_otras_plantas_m2": _to_int(_slice(line, 98, 7)),
    }

def parse_15(line: str) -> dict:
    # Registro de Bien Inmueble (15)
    rc_parc = _slice(line, 31, 14)
    cargo = _slice(line, 45, 4)
    c1 = _slice(line, 49, 1)
    c2 = _slice(line, 50, 1)
    rc20 = f"{rc_parc}{cargo}{c1}{c2}".strip()
    uso_letra = _slice(line, 428, 1)
    return {
        "tipo": "15",
        "delegacion": _slice(line, 24, 2),
        "cod_muni_dgc": _slice(line, 26, 3),
        "clase_bi": _slice(line, 29, 2),  # UR/RU/BI
        "parcela": rc_parc,
        "cargo": cargo,
        "rc20": rc20,
        "prov_nom": _norm_txt(_slice(line, 95, 25)),
        "muni_nom": _norm_txt(_slice(line, 126, 40)),
        "bloque": _slice(line, 246, 4),
        "esc": _slice(line, 250, 2),
        "planta": _slice(line, 252, 3),
        "puerta": _slice(line, 255, 3),
        "uso15_letra": uso_letra,
        "uso15_desc": USO_15.get(uso_letra, ""),
        "sup_elem_m2": _to_int(_slice(line, 442, 10)),
        "sup_suelo_m2": _to_int(_slice(line, 452, 10)),
        "coef_prop_1e6": _to_int(_slice(line, 462, 9)),  # 3 enteros + 6 decimales
    }

def parse_16(line: str) -> dict:
    # Registro de reparto de elementos comunes (16)
    # El bloque repetitivo de 15 repartos requiere expansión; aquí sacamos cabecera + texto bruto.
    return {
        "tipo": "16",
        "delegacion": _slice(line, 24, 2),
        "cod_muni_dgc": _slice(line, 26, 3),
        "parcela": _slice(line, 31, 14),
        "elem_orden": _slice(line, 45, 4),
        "subparc_calif": _slice(line, 49, 2),
        "segmento": _slice(line, 51, 4),
        "bloque_repartos": _slice(line, 55, 885),  # si necesitas, expándelo
    }

def parse_17(line: str) -> dict:
    # Registro de Cultivos (17)
    return {
        "tipo": "17",
        "delegacion": _slice(line, 24, 2),
        "cod_muni_dgc": _slice(line, 26, 3),
        "clase_suelo": _slice(line, 29, 2),  # UR/RU
        "parcela": _slice(line, 31, 14),
        "subparcela": _slice(line, 45, 4).strip(),
        "cargo_imputado": _slice(line, 51, 4),
        "tipo_subparc": _slice(line, 55, 1),  # T/A/D
        "sup_subparc_m2": _to_int(_slice(line, 56, 10)),
        "calif_catastral": _slice(line, 66, 2),
        "denom_cultivo": _norm_txt(_slice(line, 68, 40)),
        "intensidad": _slice(line, 108, 2),
        "mod_rep": _slice(line, 127, 3),
    }

# =========================
# Lectura de ficheros CAT
# =========================

def open_cat_file(path: Path):
    """Devuelve un iterable de líneas (str) tanto para .CAT como .CAT.gz."""
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "rb") as fh:
            data = fh.read()
        # Los CAT son texto cp1252 normalmente
        text = data.decode("cp1252", errors="ignore")
        return io.StringIO(text)
    else:
        # .CAT normal
        return open(path, "r", encoding="cp1252", errors="ignore")

def scan_parcels_for_municipio(cat_dir: Path, municipio_objetivo: str):
    """
    Lee todos los .CAT / .CAT.gz del directorio y devuelve:
    - set de referencias 'parcela' (14 chars) cuyo municipio (reg.11) coincide.
    - diccionario mapping 'parcela' -> registro 11 completo (último visto).
    """
    target = _norm_key(municipio_objetivo)
    parcelas = set()
    reg11_by_parcela = {}
    files = sorted([p for p in cat_dir.rglob("*") if p.is_file() and p.suffix.lower() in (".cat", ".gz")])
    for path in files:
        with open_cat_file(path) as fh:
            for line in fh:
                if len(line) < 10:
                    continue
                t = line[0:2]
                if t == "11":
                    r11 = parse_11(line)
                    muni = _norm_key(r11.get("muni_nom", ""))
                    if muni == target:
                        parc = r11["parcela"]
                        if parc:
                            parcelas.add(parc)
                            # Conserva el último 11 visto para esa parcela
                            reg11_by_parcela[parc] = r11 | {"_archivo": path.name}
    return parcelas, reg11_by_parcela

def extract_all_for_parcels(cat_dir: Path, parcelas_objetivo: set[str]):
    """
    Extrae 11/13/14/15/16/17 SOLO para las parcelas dadas.
    Devuelve listas de dicts.
    """
    out11, out13, out14, out15, out16, out17 = [], [], [], [], [], []
    files = sorted([p for p in cat_dir.rglob("*") if p.is_file() and p.suffix.lower() in (".cat", ".gz")])

    for path in files:
        with open_cat_file(path) as fh:
            linea = 0
            for line in fh:
                linea += 1
                if len(line) < 10:
                    continue
                t = line[0:2]

                if t == "11":
                    r = parse_11(line)
                    if r["parcela"] in parcelas_objetivo:
                        r["_archivo"] = path.name
                        r["_linea"] = linea
                        out11.append(r)

                elif t == "13":
                    r = parse_13(line)
                    if r["parcela"] in parcelas_objetivo:
                        r["_archivo"] = path.name
                        r["_linea"] = linea
                        out13.append(r)

                elif t == "14":
                    r = parse_14(line)
                    if r["parcela"] in parcelas_objetivo:
                        # Traducción de destino si la tenemos
                        r["destino_desc"] = DESTINO_14.get(r["destino_cod"], "")
                        r["_archivo"] = path.name
                        r["_linea"] = linea
                        out14.append(r)

                elif t == "15":
                    r = parse_15(line)
                    if r["parcela"] in parcelas_objetivo:
                        r["_archivo"] = path.name
                        r["_linea"] = linea
                        out15.append(r)

                elif t == "16":
                    r = parse_16(line)
                    if r["parcela"] in parcelas_objetivo:
                        r["_archivo"] = path.name
                        r["_linea"] = linea
                        out16.append(r)

                elif t == "17":
                    r = parse_17(line)
                    if r["parcela"] in parcelas_objetivo:
                        r["_archivo"] = path.name
                        r["_linea"] = linea
                        out17.append(r)

    return out11, out13, out14, out15, out16, out17

def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    # Orden de columnas: claves ordenadas con _archivo/_linea al final
    keys = sorted({k for r in rows for k in r.keys() if k not in {"_archivo", "_linea"}})
    keys += [k for k in ["_archivo", "_linea"] if any(k in r for r in rows)]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

def make_summary(out_dir: Path, municipio: str, r11, r13, r14, r15, r16, r17):
    """Crea un resumen útil por parcela y por cargo (con algunas métricas)."""
    # Por parcela → sumas básicas de 14 y 15
    # Suma 14 sup_local_m2 por parcela
    from collections import defaultdict
    sum14 = defaultdict(int)
    for r in r14:
        sum14[r["parcela"]] += (r.get("sup_local_m2") or 0)
    # Suma 15 sup_elem_m2 por parcela
    sum15 = defaultdict(int)
    for r in r15:
        sum15[r["parcela"]] += (r.get("sup_elem_m2") or 0)
    # Cuenta usos 15 por parcela
    usos_count = defaultdict(lambda: defaultdict(int))
    for r in r15:
        u = (r.get("uso15_letra") or "").strip().upper()
        if u:
            usos_count[r["parcela"]][u] += 1

    rows = []
    # Toma un 11 como “maestro” de parcela
    r11_by_parc = {}
    for rr in r11:
        r11_by_parc[rr["parcela"]] = rr

    for parcela, base in r11_by_parc.items():
        fila = {
            "municipio": municipio,
            "parcela": parcela,
            "provincia": base.get("prov_nom",""),
            "muni_nom": base.get("muni_nom",""),
            "paraje_nom": base.get("paraje_nom",""),
            "sup_parcela_m2": base.get("sup_parcela_m2", 0),
            "sup_construida_total_m2": base.get("sup_construida_total_m2", 0),
            "sum_sup14_m2": sum14.get(parcela, 0),
            "sum_sup15_m2": sum15.get(parcela, 0),
            "srs": base.get("srs",""),
            "coord_x": base.get("coord_x",""),
            "coord_y": base.get("coord_y",""),
        }
        # Añade conteos de usos (15)
        for u, desc in USO_15.items():
            fila[f"n_inmuebles_uso_{u}"] = usos_count[parcela].get(u, 0)
        rows.append(fila)

    write_csv(out_dir / "resumen_parcela.csv", rows)

# =========================
# Main
# =========================

def run_etl(cat_dir: str, municipio: str, out_dir: str):
    cat_dir = Path(cat_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"📦 Carpeta CAT: {cat_dir}")
    print(f"🏙️  Municipio objetivo: {municipio}")
    print("🔎 Buscando parcelas del municipio (reg.11)...")

    parcelas, reg11_idx = scan_parcels_for_municipio(cat_dir, municipio)
    if not parcelas:
        print("❌ No se encontraron parcelas de ese municipio en los ficheros .CAT/.gz")
        return

    print(f"✅ Parcelas encontradas: {len(parcelas)}")
    print("📥 Extrayendo 11/13/14/15/16/17 de esas parcelas...")

    r11, r13, r14, r15, r16, r17 = extract_all_for_parcels(cat_dir, parcelas)

    # Exportar
    write_csv(out_dir / "reg11_finca.csv", r11)
    write_csv(out_dir / "reg13_uc.csv", r13)
    write_csv(out_dir / "reg14_construccion.csv", r14)
    write_csv(out_dir / "reg15_inmueble.csv", r15)
    write_csv(out_dir / "reg16_reparto.csv", r16)
    write_csv(out_dir / "reg17_cultivo.csv", r17)

    # Resumen útil por parcela
    make_summary(out_dir, municipio, r11, r13, r14, r15, r16, r17)

    print("✅ Exportado:")
    print(f" - {out_dir/'reg11_finca.csv'}  ({len(r11)} filas)")
    print(f" - {out_dir/'reg13_uc.csv'}     ({len(r13)} filas)")
    print(f" - {out_dir/'reg14_construccion.csv'} ({len(r14)} filas)")
    print(f" - {out_dir/'reg15_inmueble.csv'}     ({len(r15)} filas)")
    print(f" - {out_dir/'reg16_reparto.csv'}      ({len(r16)} filas)")
    print(f" - {out_dir/'reg17_cultivo.csv'}      ({len(r17)} filas)")
    print(f" - {out_dir/'resumen_parcela.csv'}    (1 por parcela)")

if __name__ == "__main__":
    # Ejemplo de uso:
    # python etl_cat.py "C:\Users\khora\Desktop\CAT" "Puertollano" "C:\Users\khora\Desktop\ETL_Puertollano"
    if len(sys.argv) < 4:
        print("Uso:")
        print("  python etl_cat.py <carpeta_CAT> <municipio> <carpeta_salida>")
        sys.exit(1)
    run_etl(sys.argv[1], sys.argv[2], sys.argv[3])