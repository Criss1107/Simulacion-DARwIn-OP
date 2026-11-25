"""Controlador Supervisor (controllers/env/env.py): entrena y evalua un agente.

Edita las constantes de abajo, recarga el mundo (File -> Reload World) y dale Play
(velocidad rapida). Resultados en  <proyecto>/resultados/ :
    <NOMBRE>_s<SEMILLA>_train.csv   un episodio por fila (entrenamiento)
    <NOMBRE>_s<SEMILLA>_eval.csv    un episodio por fila (evaluacion)
    <NOMBRE>_s<SEMILLA>.pt          checkpoint
    <NOMBRE>_s<SEMILLA>_hist.json   metricas por bloque de updates
Cada corrida sobrescribe los archivos de su mismo nombre.
"""
import json
import os
import time
from dataclasses import replace

from carrera_env import CarreraEnv
from registro import RegistroEpisodios
import pva_ac as P

# configuracion
MODO = "evaluar"          #"humo" (30 updates) - "entrenar" - "evaluar" (carga el .pt y evalua)
AGENTE = "pva"         # "pva" - "a2c" - "aleatorio" - "heuristico" - "heuristico_directo"
SEMILLA = 0
NOMBRE = "pva"      # nombre del experimento en los archivos (p. ej. "pva_sin_aborto")
CAMBIOS_CFG = {}       #p. ej. {"K": 1}, {"lam_s": 0.0}, {"H": 4}
FLAGS_PVA = {}         #p. ej. {"use_abort": False}, {"use_abs": False}
N_EVAL = 30            #episodios de evaluacion (mismas pistas para todos los agentes)
N_OBST = 3             # variante del entorno para la evaluacion
META_DIST = 3.0        # variante del entorno para la evaluacion
Y_INICIO = 0.4         #variante: desplazamiento lateral de la salida (m)
META_LAT = 0.5         # variante: la meta cae a +-META_LAT m del eje
ANCHO = 0.6            #variante: los obstaculos caen a +-ANCHO m del eje
MAX_PASOS = 250        #variante: limite de pasos por episodio
SOBRESCRIBIR = False   #True solo si quieres repetir un experimento que ya tiene .pt


if AGENTE in ("aleatorio", "heuristico", "heuristico_directo"):
    NOMBRE = AGENTE               #ñas referencias siempre se guardan con su propio nombre
if MODO == "humo":
    NOMBRE = NOMBRE + "_humo"     # las pruebas cortas nunca pisan resultados reales

RAIZ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
RES = os.path.join(RAIZ, "resultados")
base = os.path.join(RES, f"{NOMBRE}_s{SEMILLA}")
estandar = (N_OBST, META_DIST, Y_INICIO, META_LAT, ANCHO, MAX_PASOS) == (3, 3.0, 0.0, 0.5, 0.6, 250)
ruta_eval = base + ("_eval.csv" if estandar else
                    f"_eval_obs{N_OBST}_d{META_DIST:g}_y{Y_INICIO:g}_l{META_LAT:g}_a{ANCHO:g}_t{MAX_PASOS}.csv")

LAM_ABS = CAMBIOS_CFG.pop("lam_abs", 0.5)
cfg = replace(P.Cfg(seed=SEMILLA), **CAMBIOS_CFG)
n_upd = 30 if MODO == "humo" else cfg.total_updates
log_every = 10 if MODO == "humo" else 50

env_crudo = CarreraEnv(meta_dist=META_DIST, n_obstaculos=N_OBST, y_inicio=Y_INICIO,
                       meta_lat=META_LAT, ancho=ANCHO, max_pasos=MAX_PASOS)

#referencias sin aprendizaje
if AGENTE in ("aleatorio", "heuristico", "heuristico_directo"):
    env_eval = RegistroEpisodios(env_crudo, ruta_eval, NOMBRE, SEMILLA)
    if AGENTE == "aleatorio":
        ev = P.random_baseline(env_eval, n_episodes=N_EVAL, seed=1000 + SEMILLA)
    else:
        ev = P.heuristic_baseline(env_eval, n_episodes=N_EVAL, seed=1000 + SEMILLA,
                                  evadir=(AGENTE == "heuristico"))
    print(f"evaluacion {AGENTE}: retorno {ev[0]:.1f} +- {ev[1]:.1f} | llegada a la meta {ev[2]:.0%}")
    raise SystemExit

#solo evaluar un checkpoint ya entrenado
if MODO == "evaluar":
    model, disc, ck = P.load_checkpoint(base + ".pt")
    env_eval = RegistroEpisodios(env_crudo, ruta_eval, NOMBRE, SEMILLA)
    if AGENTE == "pva":
        ev = P.evaluate_pva(model, disc, env_eval, n_episodes=N_EVAL, seed=1000 + SEMILLA,
                            use_abort=FLAGS_PVA.get("use_abort", True), c=P.Cfg(**ck["cfg"]))
    else:
        ev = P.evaluate_policy(model, env_eval, n_episodes=N_EVAL, seed=1000 + SEMILLA)
    print(f"evaluacion {NOMBRE}: retorno {ev[0]:.1f} +- {ev[1]:.1f} | llegada a la meta {ev[2]:.0%}")
    raise SystemExit

#entrenamiento
if MODO == "entrenar" and os.path.exists(base + ".pt") and not SOBRESCRIBIR:
    print(f"ERROR: ya existe {base}.pt. Cambia NOMBRE/SEMILLA o pon SOBRESCRIBIR = True.")
    raise SystemExit
env = RegistroEpisodios(env_crudo, base + "_train.csv", NOMBRE, SEMILLA)
t0 = time.time()
if AGENTE == "pva":
    model, disc, hist = P.train_pva(env, cfg, n_updates=n_upd, log_every=log_every,
                                    lam_abs=LAM_ABS, **FLAGS_PVA)
else:
    model, hist = P.train_a2c(env, cfg, n_updates=n_upd, log_every=log_every)
    disc = None
dt = time.time() - t0
print(f"\n{dt / n_upd:.2f} s/update  ->  {cfg.total_updates} updates ~ "
      f"{dt / n_upd * cfg.total_updates / 60:.0f} min por corrida")

P.save_checkpoint(base + ".pt", model, disc, cfg)
with open(base + "_hist.json", "w") as f:
    json.dump(hist, f)

env_eval = RegistroEpisodios(env_crudo, ruta_eval, NOMBRE, SEMILLA)
if AGENTE == "pva":
    ev = P.evaluate_pva(model, disc, env_eval, n_episodes=N_EVAL, seed=1000 + SEMILLA,
                        use_abort=FLAGS_PVA.get("use_abort", True), c=cfg)
else:
    ev = P.evaluate_policy(model, env_eval, n_episodes=N_EVAL, seed=1000 + SEMILLA)
print(f"evaluacion: retorno {ev[0]:.1f} +- {ev[1]:.1f} | llegada a la meta {ev[2]:.0%}")