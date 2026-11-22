"""Paso 2 — Carga, entrenamiento y publicación en Azure Custom Vision (API REST v3.3).

Variables en .env:
    CV_TRAINING_ENDPOINT        p. ej. https://eastus.api.cognitive.microsoft.com/
    CV_TRAINING_KEY             clave del recurso de entrenamiento (AnimalIdentificationJGR)
    CV_PROJECT_ID               GUID del proyecto (lo imprime `crear-proyecto`)
    CV_PREDICTION_RESOURCE_ID   ID ARM del recurso de predicción (para publicar):
        az cognitiveservices account show -n AnimalIdentificationJGR-Prediction \
           -g ComputerVisionImages --query id -o tsv

Flujo de experimentos (cada paso crea una iteración comparable):
    python 02_custom_vision.py crear-proyecto --nombre DanosVehiculares
    python 02_custom_vision.py subir --max-por-clase 150      # subconjunto pequeño
    python 02_custom_vision.py entrenar --tipo Regular         # Iteración 1 (baseline)
    python 02_custom_vision.py subir                           # resto de imágenes
    python 02_custom_vision.py entrenar --tipo Regular         # Iteración 2 (todos los datos)
    python 02_custom_vision.py entrenar --tipo Advanced --horas 1 --force   # Iteración 3
    python 02_custom_vision.py estado
    python 02_custom_vision.py publicar --iteracion <ID> --nombre iter3-advanced
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import requests

from src import config

UPLOAD_LOG = config.DATA_DIR / "subidas_custom_vision.json"
CV_RESULTS = config.RESULTS_DIR / "custom_vision"


def base() -> str:
    ep = (config.get("CV_TRAINING_ENDPOINT") or "").rstrip("/")
    if not ep:
        sys.exit("Falta CV_TRAINING_ENDPOINT en .env")
    return f"{ep}/customvision/v3.3/training"


def headers(json_body: bool = True) -> dict:
    key = config.get("CV_TRAINING_KEY")
    if not key:
        sys.exit("Falta CV_TRAINING_KEY en .env")
    h = {"Training-key": key}
    if json_body:
        h["Content-Type"] = "application/json"
    return h


def call(method: str, path: str, **kw) -> requests.Response:
    for attempt in range(8):
        r = requests.request(method, base() + path, timeout=120, **kw)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(min(2 ** attempt, 30))
            continue
        if r.status_code >= 400:
            sys.exit(f"{method} {path} -> HTTP {r.status_code}: {r.text[:500]}")
        return r
    sys.exit(f"{method} {path}: demasiados reintentos")


def project_id() -> str:
    pid = config.get("CV_PROJECT_ID")
    if not pid:
        sys.exit("Falta CV_PROJECT_ID en .env (ejecuta primero `crear-proyecto`)")
    return pid


# ----------------------------------------------------------------------------- comandos

def cmd_crear_proyecto(args):
    domains = call("GET", "/domains", headers=headers()).json()
    dom = next(d for d in domains if d["type"] == "Classification" and d["name"] == args.dominio)
    params = {"name": args.nombre, "domainId": dom["id"], "classificationType": "Multiclass",
              "description": "Peritaje automático: vehículo con daño vs sin daño (Actividad 6 VPC)"}
    proj = call("POST", "/projects", headers=headers(), params=params).json()
    print(f"Proyecto creado: {proj['name']}  dominio={args.dominio}")
    print(f"Agrega a .env ->  CV_PROJECT_ID={proj['id']}")


def ensure_tags(pid: str) -> dict:
    existing = {t["name"]: t["id"] for t in call("GET", f"/projects/{pid}/tags", headers=headers()).json()}
    for tag in config.TAGS:
        if tag not in existing:
            t = call("POST", f"/projects/{pid}/tags", headers=headers(), params={"name": tag}).json()
            existing[tag] = t["id"]
            print(f"Tag creado: {tag}")
    return existing


def cmd_subir(args):
    pid = project_id()
    tags = ensure_tags(pid)
    log = json.loads(UPLOAD_LOG.read_text(encoding="utf-8")) if UPLOAD_LOG.exists() else {}
    done = set(log.get(pid, []))
    rng = random.Random(42)
    for tag in config.TAGS:
        files = sorted((config.DATA_DIR / "train" / tag).glob("*.jpg"))
        rng.shuffle(files)  # orden aleatorio fijo -> el subconjunto es estratificado y reproducible
        if args.max_por_clase:
            files = files[: args.max_por_clase]
        pending = [f for f in files if f.name not in done]
        print(f"{tag}: {len(pending)} imágenes por subir ({len(files) - len(pending)} ya subidas)")
        batch, size = [], 0
        for i, f in enumerate(pending):
            data = f.read_bytes()
            batch.append(f)
            size += len(data)
            if len(batch) == 64 or size > 20e6 or i == len(pending) - 1:
                multipart = [("file", (b.name, b.read_bytes(), "application/octet-stream")) for b in batch]
                r = call("POST", f"/projects/{pid}/images", headers=headers(json_body=False),
                         params={"tagIds": tags[tag]}, files=multipart).json()
                bad = [im for im in r.get("images", []) if im["status"] not in ("OK", "OKDuplicate")]
                for im in bad:
                    print(f"   ! {im.get('sourceUrl', '')} -> {im['status']}")
                done.update(b.name for b in batch)
                log[pid] = sorted(done)
                UPLOAD_LOG.write_text(json.dumps(log, indent=1), encoding="utf-8")
                print(f"   subidas {min(i + 1, len(pending))}/{len(pending)}")
                batch, size = [], 0
    print(f"Total en el proyecto según log: {len(done)} imágenes")


def save_performance(pid: str, it: dict):
    CV_RESULTS.mkdir(parents=True, exist_ok=True)
    perf = call("GET", f"/projects/{pid}/iterations/{it['id']}/performance",
                headers=headers(), params={"threshold": 0.5}).json()
    out = {"iteracion": it, "performance_threshold_0.5": perf}
    path = CV_RESULTS / f"{it['name'].replace(' ', '_')}.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Precision={perf['precision']:.3f}  Recall={perf['recall']:.3f}  AP={perf['averagePrecision']:.3f}")
    for t in perf["perTagPerformance"]:
        print(f"    {t['name']:<10} P={t['precision']:.3f} R={t['recall']:.3f} AP={t['averagePrecision']:.3f}")
    print(f"  Guardado en {path}")


def cmd_entrenar(args):
    pid = project_id()
    params = {"trainingType": args.tipo}
    if args.tipo == "Advanced":
        params["reservedBudgetInHours"] = args.horas
    if args.force:
        params["forceTrain"] = "true"
    it = call("POST", f"/projects/{pid}/train", headers=headers(), params=params).json()
    print(f"Entrenando {it['name']} ({args.tipo}) id={it['id']}")
    t0 = time.time()
    while it["status"] == "Training":
        time.sleep(15)
        it = call("GET", f"/projects/{pid}/iterations/{it['id']}", headers=headers()).json()
        print(f"  {int(time.time() - t0)} s  estado={it['status']}", flush=True)
    if it["status"] != "Completed":
        sys.exit(f"Entrenamiento terminó con estado {it['status']}")
    save_performance(pid, it)


def cmd_estado(args):
    pid = project_id()
    its = call("GET", f"/projects/{pid}/iterations", headers=headers()).json()
    for it in sorted(its, key=lambda x: x["created"]):
        print(f"{it['name']:<14} id={it['id']} estado={it['status']:<10} tipo={it.get('trainingType')} "
              f"publicada={it.get('publishName') or '-'}")
        if it["status"] == "Completed" and args.guardar:
            save_performance(pid, it)


def cmd_publicar(args):
    pid = project_id()
    res = config.get("CV_PREDICTION_RESOURCE_ID")
    if not res:
        sys.exit("Falta CV_PREDICTION_RESOURCE_ID en .env (ver docstring)")
    call("POST", f"/projects/{pid}/iterations/{args.iteracion}/publish", headers=headers(),
         params={"publishName": args.nombre, "predictionId": res})
    print(f"Iteración {args.iteracion} publicada como '{args.nombre}'.")
    print(f"Prediction URL (archivo): <CV_PREDICTION_ENDPOINT>/customvision/v3.0/Prediction/{pid}"
          f"/classify/iterations/{args.nombre}/image")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("crear-proyecto")
    p.add_argument("--nombre", default="DanosVehiculares")
    p.add_argument("--dominio", default="General [A2]")
    p.set_defaults(func=cmd_crear_proyecto)
    p = sub.add_parser("subir")
    p.add_argument("--max-por-clase", type=int, default=0, help="0 = todas")
    p.set_defaults(func=cmd_subir)
    p = sub.add_parser("entrenar")
    p.add_argument("--tipo", choices=["Regular", "Advanced"], default="Regular")
    p.add_argument("--horas", type=int, default=1)
    p.add_argument("--force", action="store_true", help="reentrenar aunque no cambien los datos")
    p.set_defaults(func=cmd_entrenar)
    p = sub.add_parser("estado")
    p.add_argument("--guardar", action="store_true", help="guardar métricas de cada iteración")
    p.set_defaults(func=cmd_estado)
    p = sub.add_parser("publicar")
    p.add_argument("--iteracion", required=True)
    p.add_argument("--nombre", required=True)
    p.set_defaults(func=cmd_publicar)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
