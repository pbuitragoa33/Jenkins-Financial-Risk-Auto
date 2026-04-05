# Chequeos basicos de archivos y estructura del proyecto

from pathlib import Path


def resultado(name, ok, msg):

    return {"dataset": "repo", "check": name, "ok": ok, "message": msg}


def check_rutas_requeridas(root: Path, required_paths: list):

    resultados_lst = []

    for rel in required_paths:

        p = root / rel
        ok = p.exists()
        resultados_lst.append(resultado(f"required::{rel}", ok, f"{rel} {'Existe' if ok else 'Falta'}"))

    return resultados_lst


def check_dockerfile_basico(dockerfile: Path):

    nombre = f"dockerfile::basic::{dockerfile.as_posix()}"

    if not dockerfile.exists():
        return [resultado(nombre, False, f"Dockerfile no encontrado en {dockerfile.as_posix()}")]

    texto = dockerfile.read_text(encoding="utf-8", errors="ignore")

    msgs = []
    ok = True

    if "FROM" not in texto:

        ok = False
        msgs.append("No se encontro linea FROM")

    if "FROM" in texto and ":latest" in texto:
        
        ok = False
        msgs.append("Evita usar :latest en la imagen")

    if "pip install -r requirements.txt" not in texto and "poetry install" not in texto:
        msgs.append("No se encontro comando de instalacion de dependencias comun")

    return [resultado(nombre, ok, "; ".join(msgs) if msgs else "Chequeo Dockerfile basico OK")]