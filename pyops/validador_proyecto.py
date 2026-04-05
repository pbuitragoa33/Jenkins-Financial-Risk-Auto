import sys
import argparse
from pathlib import Path

try:
    from .chequeo_archivos import check_rutas_requeridas, check_dockerfile_basico
    from .scanner_secretos import escaneo_secretos
except ImportError:
    # Permite ejecutar el script directamente: python pyops/validador_proyecto.py
    from chequeo_archivos import check_rutas_requeridas, check_dockerfile_basico
    from scanner_secretos import escaneo_secretos


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Validador local de estructura y secretos del proyecto"
    )
    modo = parser.add_mutually_exclusive_group()
    modo.add_argument(
        "-s",
        "--silent",
        action="store_true",
        help="Salida minima: solo resumen final",
    )
    modo.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Salida detallada: muestra todos los checks",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    repo_raiz = Path(__file__).resolve().parent.parent

    resultados = []

    rutas_requeridas = [
        "pipeline_MLOps",
        "pipeline_MLOps/src",
        "pipeline_MLOps/src/cargar_datos.py",
        "pipeline_MLOps/src/comprension_eda.ipynb",
        "pipeline_MLOps/src/heuristic_model.py",
        "pipeline_MLOps/src/model_training_evaluation.py",
        "pipeline_MLOps/src/model_deploy.py",
        "pipeline_MLOps/src/model_interface.py",
        "Jenkinsfile",
        "requirements.txt",
    ]
    resultados += check_rutas_requeridas(repo_raiz, rutas_requeridas)
    resultados += check_dockerfile_basico(repo_raiz / "pipeline_MLOps" / "Dockerfile")

    ignore_globs = [
        ".git/**", ".venv/**", "venv/**", "**/__pycache__/**",
        "**/*.pkl", "**/*.db", "models/**", "monitoring.db"
    ]
    resultados += escaneo_secretos(repo_raiz, ignore_globs)

    fails = [r for r in resultados if not r["ok"]]
    ok_count = len(resultados) - len(fails)

    if not args.silent:
        print("=== Resultado de validacion del proyecto ===")
        print(f"Checks: {len(resultados)} | OK: {ok_count} | FAIL: {len(fails)}")

    if args.verbose and not args.silent:
        for r in resultados:
            estado = "OK" if r["ok"] else "FAIL"
            print(f"[{estado}] {r['check']} -> {r['message']}")
    elif not args.silent:
        for r in fails:
            print(f"[FAIL] {r['check']} -> {r['message']}")

    if fails:
        print("Validacion finalizada con errores.")
    else:
        print("Validacion finalizada correctamente.")

    return 1 if fails else 0



if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))