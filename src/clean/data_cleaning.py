import pandas as pd
import os
from pathlib import Path

# ── Resolve paths relative to project root (works from any cwd) ──────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DATA_RAW = _PROJECT_ROOT / "data" / "raw"
_DATA_CLEAN = _PROJECT_ROOT / "data" / "clean"

df = pd.read_csv(_DATA_RAW / "correos_combined_filled.csv")
# df_spam = pd.read_csv(_DATA_RAW / "correos_spam_10.csv")

# Crear carpeta de salida
os.makedirs(_DATA_CLEAN, exist_ok=True)

# ── Columnas esperadas con valores por defecto ────────────────────────────
_FILL_COLS = [
    "subject",
    "body",
    "attachments",
    "from_name",
    "from_email",
    "to_name",
    "to_email",
]

for col in _FILL_COLS:
    if col not in df.columns:
        df[col] = ""
    else:
        df[col] = df[col].fillna("")

# for col in _FILL_COLS:
#    if col not in df_spam.columns:
#        df_spam[col] = ""
#    else:
#        df_spam[col] = df_spam[col].fillna("")


# ── Crear un .txt por cada email ──────────────────────────────────────────
def write_emails(dataframe: pd.DataFrame, output_dir: Path) -> None:
    for _, row in dataframe.iterrows():
        content = f"""De: {row["from_name"]} <{row["from_email"]}>
Para: {row["to_name"]} <{row["to_email"]}>
Fecha: {row["date"]}
Adjuntos: {row["attachments"]}
 
Asunto: {row["subject"]}
Cuerpo:
{row["body"]}
"""
        filename = f"{row['id']}.txt"
        filepath = output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)


write_emails(df, _DATA_CLEAN)
# write_emails(df_spam, _DATA_CLEAN)

print(f"✅ Emails guardados en carpeta: {_DATA_CLEAN}")
