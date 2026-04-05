import re
from pathlib import Path
from fnmatch import fnmatch


PATRONES = [
    (r"AKIA[0-9A-Z]{16}", "AWS Access Key ID"),
    (r"aws_secret_access_key\s*=\s*[A-Za-z0-9/+=]{40}", "AWS Secret Access Key"),
    (r"ghp_[A-Za-z0-9]{36,}", "GitHub PAT"),
    (r"AIza[0-9A-Za-z\-_]{35}", "Google API Key"),
    (r"(?i)(AccountKey|SharedAccessSignature|sas_token)\s*=\s*[A-Za-z0-9%+/=_.\-]{20,}", "Azure key/SAS"),
    (r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*['\"][^'\"\s]{12,}['\"]", "Generic secret literal"),
]


EXTENSIONES = {".py", ".json", ".yml", ".yaml", ".toml", ".md", ".txt", ".env", ".cfg", ".ini", ".dockerignore", ".gitignore", ".sh"}
NOMBRES_DIRECTOS = {".env", ".env.local", ".env.dev", ".env.prod"}


def resultado(path, pattern_name, ok, msg):
    return {"dataset": str(path), "check": f"secrets::{pattern_name}", "ok": ok, "message": msg}


def ignorar(rel_path: str, ignore_globs: list):
    return any(fnmatch(rel_path, g) for g in ignore_globs)


def escaneo_secretos(repo_root: Path, ignore_globs: list):
    resultados_lst = []

    for p in repo_root.rglob("*"):
        if p.is_dir():
            continue

        rel = p.relative_to(repo_root).as_posix()

        if ignorar(rel, ignore_globs):
            continue

        es_nombre_directo = p.name.lower() in NOMBRES_DIRECTOS
        if p.suffix.lower() not in EXTENSIONES and not es_nombre_directo:
            continue

        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        for pattern, label in PATRONES:
            if re.search(pattern, text):
                resultados_lst.append(resultado(rel, label, False, f"Posible secreto en: {rel} ({label})"))

    if not any(not r["ok"] for r in resultados_lst):
        resultados_lst.append({"dataset": "repo", "check": "secrets::global", "ok": True, "message": "No obvious secrets found"})

    return resultados_lst