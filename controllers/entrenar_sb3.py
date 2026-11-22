import json
import os
import platform
import sys
import time

import gymnasium as gym
import torch
import stable_baselines3 as sb3
from stable_baselines3 import DQN, PPO

RAIZ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(RAIZ, "controllers", "env"))      # carrera_env, registro

from carrera_env import CarreraEnv  # noqa: E402
from registro import RegistroEpisodios  # noqa: E402

# ----------------------------------------------------------- configuracion
MODO = "humo"                 # "humo" (prueba corta) | "entrenar" (entrena y evalua) |
                              # "evaluar" (carga los .zip y evalua) | "variantes" (4 variantes del entorno)
ALGORITMOS = ["ppo", "dqn"]   # puedes correr uno solo para repartir el trabajo en varias sesiones
SEMILLAS = [0, 1, 2]
SEMILLAS_VARIANTES = [0]      # el enunciado pide ppo_s0 y dqn_s0
TOTAL_TIMESTEPS = 40000       # presupuesto obligatorio: exactamente 40 000 decisiones
N_EVAL = 30
MAX_PASOS = 250
REW_SCALE = 0.1               # igual que A2C/PVA-AC: solo escala la recompensa de ENTRENAMIENTO
EVAL_DETERMINISTA = True      # accion = argmax (PPO) / greedy (DQN); es lo estandar en SB3
SOBRESCRIBIR = False          # True solo para repetir una corrida que ya tiene su .zip

# Hiperparametros.  PPO: n_steps * n_envs divide a 40 000, asi el entrenamiento se detiene en
# exactamente 40 000 decisiones (con n_steps=2048 SB3 haria 40 960).
PPO_HP = dict(learning_rate=3e-4, n_steps=250, batch_size=50, n_epochs=10, gamma=0.99,
              gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
              policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])))
DQN_HP = dict(learning_rate=1e-3, buffer_size=40000, learning_starts=1000, batch_size=64, tau=1.0,
              gamma=0.99, train_freq=4, gradient_steps=1, target_update_interval=500,
              exploration_initial_eps=1.0, exploration_final_eps=0.05, exploration_fraction=0.3,
              max_grad_norm=10, policy_kwargs=dict(net_arch=[128, 128]))
CLASES = {"ppo": (PPO, PPO_HP), "dqn": (DQN, DQN_HP)}

# variante -> parametros (los mismos que usaste con PVA-AC y A2C; ver nombres en resultados/)
VARIANTES = {
    "obs5":     dict(n_obst=5),
    "largo":    dict(meta_dist=4.5, max_pasos=350),
    "estrecha": dict(meta_lat=0.35, ancho=0.35),
    "lateral":  dict(y_inicio=0.4),
}
# -------------------------------------------------------------------------

RES = os.path.join(RAIZ, "resultados")
os.makedirs(RES, exist_ok=True)
RUTA_CFG = os.path.join(RES, "sb3_config.json")

if MODO == "humo":
    TOTAL_TIMESTEPS, N_EVAL = 500, 3
    DQN_HP = dict(DQN_HP, learning_starts=100, buffer_size=1000)


class EscalaRecompensa(gym.RewardWrapper):
    """Escala la recompensa que ve el algoritmo; el CSV (RegistroEpisodios) guarda la real."""

    def __init__(self, env, k):
        super().__init__(env)
        self.k = k

    def reward(self, r):
        return r * self.k


def configurar(env, n_obst=3, meta_dist=3.0, y_inicio=0.0, meta_lat=0.5, ancho=0.6,
               max_pasos=MAX_PASOS):
    """Cambia la variante del entorno sin crear otro Supervisor (solo puede haber uno)."""
    env.n_obstaculos, env.meta_dist, env.y_inicio = n_obst, meta_dist, y_inicio
    env.meta_lat, env.ancho, env.max_pasos = meta_lat, ancho, max_pasos
    return (n_obst, meta_dist, y_inicio, meta_lat, ancho, max_pasos)


def sufijo(p):
    n_obst, meta_dist, y_inicio, meta_lat, ancho, max_pasos = p
    if p == (3, 3.0, 0.0, 0.5, 0.6, 250):
        return "_eval.csv"
    return f"_eval_obs{n_obst}_d{meta_dist:g}_y{y_inicio:g}_l{meta_lat:g}_a{ancho:g}_t{max_pasos}.csv"


def nombre(algo):
    return algo + ("_humo" if MODO == "humo" else "")


def evaluar(env_crudo, model, algo, semilla, ruta_csv):
    """30 episodios con las mismas pistas que el resto: reset(seed=1000 + semilla + ep)."""
    env_ev = RegistroEpisodios(env_crudo, ruta_csv, algo, semilla)
    rets, ok = [], []
    for ep in range(N_EVAL):
        o, _ = env_ev.reset(seed=1000 + semilla + ep)
        total, done, info = 0.0, False, {}
        while not done:
            a, _ = model.predict(o, deterministic=EVAL_DETERMINISTA)
            o, r, term, trunc, info = env_ev.step(int(a))
            total += r
            done = term or trunc
        rets.append(total)
        ok.append(float(info.get("motivo") == "meta"))
    env_ev.close()
    media = sum(rets) / len(rets)
    desv = (sum((x - media) ** 2 for x in rets) / len(rets)) ** 0.5
    print(f"  evaluacion {algo} s{semilla}: retorno {media:.1f} +- {desv:.1f} | "
          f"llegada a la meta {sum(ok) / len(ok):.0%}  -> {os.path.basename(ruta_csv)}", flush=True)


def guardar_cfg(extra=None):
    cfg = {}
    if os.path.exists(RUTA_CFG):
        with open(RUTA_CFG) as f:
            cfg = json.load(f)
    cfg.update(
        versiones=dict(stable_baselines3=sb3.__version__, gymnasium=gym.__version__,
                       torch=torch.__version__, python=platform.python_version()),
        presupuesto_decisiones=TOTAL_TIMESTEPS if MODO != "humo" else cfg.get("presupuesto_decisiones", 40000),
        max_pasos=MAX_PASOS, rew_scale_entrenamiento=REW_SCALE, n_eval=N_EVAL,
        semillas_pista_eval="1000 + semilla + episodio", eval_determinista=EVAL_DETERMINISTA,
        hiperparametros=dict(ppo=json.loads(json.dumps(PPO_HP)), dqn=json.loads(json.dumps(DQN_HP))))
    if extra:
        cfg.setdefault("tiempos_min", {}).update(extra)
    with open(RUTA_CFG, "w") as f:
        json.dump(cfg, f, indent=2)


def entrenar(env_crudo, algo, semilla):
    Clase, hp = CLASES[algo]
    base = os.path.join(RES, f"{nombre(algo)}_s{semilla}")
    if os.path.exists(base + ".zip") and not SOBRESCRIBIR:
        print(f"[omitido] ya existe {base}.zip (SOBRESCRIBIR = True para repetirlo)")
        return Clase.load(base)
    configurar(env_crudo)
    registro = RegistroEpisodios(env_crudo, base + "_train.csv", algo, semilla)
    env_tr = EscalaRecompensa(registro, REW_SCALE)
    model = Clase("MlpPolicy", env_tr, seed=semilla, device="cpu", verbose=1, **hp)
    print(f"\n=== entrenando {algo.upper()} s{semilla}: {TOTAL_TIMESTEPS} decisiones ===", flush=True)
    t0 = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=10)
    minutos = (time.time() - t0) / 60
    print(f"  listo: {model.num_timesteps} timesteps SB3 | {registro.decisiones} decisiones "
          f"registradas | {registro.episodio} episodios | {minutos:.1f} min", flush=True)
    if MODO != "humo" and registro.decisiones != 40000:
        print("  AVISO: el presupuesto no es exactamente 40 000; revisa n_steps / train_freq.")
    model.save(base)
    registro.close()
    guardar_cfg({f"{algo}_s{semilla}": round(minutos, 1)})
    return model


def main():
    guardar_cfg()
    env_crudo = CarreraEnv(max_pasos=MAX_PASOS)
    if MODO in ("humo", "entrenar", "evaluar"):
        for algo in ALGORITMOS:
            for s in SEMILLAS:
                base = os.path.join(RES, f"{nombre(algo)}_s{s}")
                if MODO == "evaluar":
                    model = CLASES[algo][0].load(base)
                else:
                    model = entrenar(env_crudo, algo, s)
                p = configurar(env_crudo)
                evaluar(env_crudo, model, nombre(algo), s, base + sufijo(p))
    elif MODO == "variantes":
        for algo in ALGORITMOS:
            for s in SEMILLAS_VARIANTES:
                base = os.path.join(RES, f"{algo}_s{s}")
                model = CLASES[algo][0].load(base)
                for var, kw in VARIANTES.items():
                    p = configurar(env_crudo, **kw)
                    print(f"[{var}] {p}")
                    evaluar(env_crudo, model, algo, s, base + sufijo(p))
        configurar(env_crudo)
    else:
        raise SystemExit(f"MODO desconocido: {MODO}")
    print("\nTerminado.")


main()
