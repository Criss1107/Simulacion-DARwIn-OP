"""Registro de episodios en CSV, igual para todos los agentes (PVA-AC, PPO, DQN, A2C).

Uso:
    env = RegistroEpisodios(CarreraEnv(), "resultados/ppo_s0_train.csv", "ppo", 0)
    # ... entrenar o evaluar con `env` como cualquier entorno de Gymnasium ...
Cada episodio terminado agrega una fila al CSV.
"""
import csv
import os
import time

import gymnasium as gym

COLUMNAS = ["agente", "semilla", "episodio", "decisiones", "retorno",
            "pasos", "motivo", "dist_final", "segundos"]


class RegistroEpisodios(gym.Wrapper):
    def __init__(self, env, ruta_csv, agente, semilla):
        super().__init__(env)
        os.makedirs(os.path.dirname(os.path.abspath(ruta_csv)), exist_ok=True)
        self.agente, self.semilla = agente, semilla
        self.episodio = 0
        self.decisiones = 0          # pasos de RL acumulados
        self._ret, self._pasos, self._t0 = 0.0, 0, time.time()
        self._f = open(ruta_csv, "w", newline="")   # cada corrida sobrescribe
        self._w = csv.writer(self._f)
        self._w.writerow(COLUMNAS)
        self._f.flush()

    def reset(self, **kwargs):
        self._ret, self._pasos, self._t0 = 0.0, 0, time.time()
        return self.env.reset(**kwargs)

    def step(self, accion):
        obs, r, term, trunc, info = self.env.step(accion)
        self._ret += r
        self._pasos += 1
        self.decisiones += 1
        if term or trunc:
            self.episodio += 1
            self._w.writerow([self.agente, self.semilla, self.episodio, self.decisiones,
                              round(self._ret, 4), self._pasos, info.get("motivo", ""),
                              round(info.get("dist", float("nan")), 3),
                              round(time.time() - self._t0, 2)])
            self._f.flush()
        return obs, r, term, trunc, info

    def close(self):
        self._f.close()
        super().close()
