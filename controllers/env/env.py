"""Controlador Supervisor (controllers/env/env.py): entrena y evalua un agente.

Edita las constantes de abajo y dale Play en Webots (velocidad rapida).
Resultados en  <proyecto>/resultados/ :
    <NOMBRE>_s<SEMILLA>_train.csv   un episodio por fila (entrenamiento)
    <NOMBRE>_s<SEMILLA>_eval.csv    un episodio por fila (evaluacion)
    <NOMBRE>_s<SEMILLA>.pt          checkpoint
    <NOMBRE>_s<SEMILLA>_hist.json   metricas por bloque de updates
"""
import json
import os
import time
from dataclasses import replace

from carrera_env import CarreraEnv
from registro import RegistroEpisodios
import pva_ac as P

# ----------------------------------------------------------- configuracion
MODO = "entrenar"          # "humo" = prueba corta (30 updates) | "entrenar" = corrida completa
AGENTE = "pva"         # "pva" | "a2c"
SEMILLA = 0
NOMBRE = AGENTE        # nombre del experimento en los archivos (p. ej. "pva_sin_aborto")
CAMBIOS_CFG = {}       # p. ej. {"K": 1}, {"lam_s": 0.0}, {"H": 4}
FLAGS_PVA = {}         # p. ej. {"use_abort": False}, {"use_abs": False}
N_EVAL = 10            # episodios de evaluacion
# -------------------------------------------------------------------------

RAIZ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
RES = os.path.join(RAIZ, "resultados")
base = os.path.join(RES, f"{NOMBRE}_s{SEMILLA}")

cfg = replace(P.Cfg(seed=SEMILLA), **CAMBIOS_CFG)
n_upd = 30 if MODO == "humo" else cfg.total_updates
log_every = 10 if MODO == "humo" else 50

env_crudo = CarreraEnv()
env = RegistroEpisodios(env_crudo, base + "_train.csv", NOMBRE, SEMILLA)

t0 = time.time()
if AGENTE == "pva":
    model, disc, hist = P.train_pva(env, cfg, n_updates=n_upd, log_every=log_every, **FLAGS_PVA)
else:
    model, hist = P.train_a2c(env, cfg, n_updates=n_upd, log_every=log_every)
    disc = None
dt = time.time() - t0
print(f"\n{dt / n_upd:.2f} s/update  ->  {cfg.total_updates} updates ~ "
      f"{dt / n_upd * cfg.total_updates / 60:.0f} min por corrida")

P.save_checkpoint(base + ".pt", model, disc, cfg)
with open(base + "_hist.json", "w") as f:
    json.dump(hist, f)

env_eval = RegistroEpisodios(env_crudo, base + "_eval.csv", NOMBRE, SEMILLA)
if AGENTE == "pva":
    ev = P.evaluate_pva(model, disc, env_eval, n_episodes=N_EVAL, seed=1000 + SEMILLA,
                        use_abort=FLAGS_PVA.get("use_abort", True), c=cfg)
else:
    ev = P.evaluate_policy(model, env_eval, n_episodes=N_EVAL, seed=1000 + SEMILLA)
print(f"evaluacion: retorno {ev[0]:.1f} +- {ev[1]:.1f} | llegada a la meta {ev[2]:.0%}")
