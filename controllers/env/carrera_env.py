"""Entorno Gymnasium para la carrera del robot en Webots 

Se usa desde un controlador Supervisor de Webots. La accion se envia al caminador en C++ mediante el campo customData del robot (DEF ROBOT).
"""
import math

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from controller import Supervisor

# accion -> (amplitud X de avance, amplitud A de giro) del GaitManager
ACCIONES = {
    0: (1.0, 0.0),    # recto
    1: (1.0, 0.5),    # recto + giro izquierda
    2: (1.0, -0.5),   # recto + giro derecha
    3: (0.0, 0.5),    # girar en sitio a la izquierda
    4: (0.0, -0.5),   # girar en sitio a la derecha
}
ANGULOS_RAYOS = np.radians([-60, -30, 0, 30, 60])
RANGO_MAX = 1.5  # metros


def _rayo_circulo(px, py, ang, cx, cy, r):
    """Distancia desde (px, py) en la direccion ang hasta un circulo, o None."""
    dx, dy = math.cos(ang), math.sin(ang)
    fx, fy = px - cx, py - cy
    b = fx * dx + fy * dy
    c = fx * fx + fy * fy - r * r
    disc = b * b - c
    if disc < 0:
        return None
    t = -b - math.sqrt(disc)
    return t if t >= 0 else None


class CarreraEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, meta_dist=3.0, n_obstaculos=3, repeticion=25,
                 max_pasos=150, z_caida=0.20, r_obstaculo=0.15,
                 r_robot=0.12, pasos_parada=100,
                 y_inicio=0.0, meta_lat=0.5, ancho=0.6, yaw_max=0.2):
        super().__init__()
        # variantes del entorno (los valores por defecto reproducen el entorno estandar
        self.y_inicio = y_inicio      #desplazamiento lateral de la posicion inicial (m)
        self.meta_lat = meta_lat      #la meta cae a +-meta_lat m del eje de salida
        self.ancho = ancho            #los obstaculos caen a +-ancho m del eje de salida
        self.yaw_max = yaw_max        #giro inicial aleatorio maximo (rad)
        self.pasos_parada = pasos_parada  #pasos para volver a postura inicial
        self.meta_dist = meta_dist
        self.n_obstaculos = n_obstaculos
        self.repeticion = repeticion      #pasos de simulacion por paso de RL
        self.max_pasos = max_pasos
        self.z_caida = z_caida            # CALIBRAR altura por debajo = caida
        self.r_obs = r_obstaculo
        self.r_robot = r_robot

        self.sup = Supervisor()
        self.dt = int(self.sup.getBasicTimeStep())
        self.robot = self.sup.getFromDef("ROBOT")
        self.f_custom = self.robot.getField("customData")
        self.f_trans = self.robot.getField("translation")
        self.f_rot = self.robot.getField("rotation")
        # esperar a que el robot se levante (parado, postura inicial) y medir su altura de pie: se usa para reiniciar sin enterrarle los pies
        self.f_custom.setSFString("0 0 0")
        for _ in range(2 * self.pasos_parada):
            self._sim_step()
        x0, y0, z0 = self.robot.getPosition()
        self.pos0 = [x0, y0, z0 + 0.005]
        print("Posicion de reinicio (de pie):", self.pos0)

        self.hijos = self.sup.getRoot().getField("children")
        self.nodos_obs = []
        self.meta = (self.pos0[0] + meta_dist, self.pos0[1])
        self.meta_nodo = self._crear_nodo(
            "Solid { translation 0 0 0.005 children [ Shape { appearance "
            "PBRAppearance { baseColor 0 0.8 0 roughness 1 metalness 0 } "
            "geometry Box { size 0.5 0.5 0.01 } } ] }")

        self.action_space = spaces.Discrete(len(ACCIONES))
        bajo = np.array([0, -1, -1, 0, 0, 0, 0, 0], dtype=np.float32)
        alto = np.array([2, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(bajo, alto, dtype=np.float32)

        self.pasos = 0
        self.dist_prev = meta_dist
        self.obstaculos = []

   
    def _sim_step(self):
        if self.sup.step(self.dt) == -1:
            raise SystemExit

    def _crear_nodo(self, texto):
        self.hijos.importMFNodeFromString(-1, texto)
        return self.hijos.getMFNode(self.hijos.getCount() - 1)

    def _estado(self):
        x, y, z = self.robot.getPosition()
        R = self.robot.getOrientation()
        yaw = math.atan2(R[3], R[0])   # VERIFICAR debe crecer al girar a la izq.
        return x, y, z, yaw

    #observacion
    def _obs(self):
        x, y, z, yaw = self._estado()
        dx, dy = self.meta[0] - x, self.meta[1] - y
        dist = math.hypot(dx, dy)
        ang = math.atan2(dy, dx) - yaw
        rayos = []
        for a in ANGULOS_RAYOS:
            m = RANGO_MAX
            for (cx, cy) in self.obstaculos:
                t = _rayo_circulo(x, y, yaw + a, cx, cy, self.r_obs)
                if t is not None and t < m:
                    m = t
            rayos.append(m / RANGO_MAX)
        obs = [min(dist / self.meta_dist, 2.0), math.sin(ang), math.cos(ang)]
        return np.array(obs + rayos, dtype=np.float32)

    # API de Gymnasium
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        x0, y0 = self.pos0[0], self.pos0[1] + self.y_inicio

        for n in self.nodos_obs:
            n.remove()
        self.nodos_obs = []
        self.obstaculos = []

        # parar el caminador y volver a la postura inicial antes de mover
        self.f_custom.setSFString("0 0 0")
        for _ in range(self.pasos_parada):
            self._sim_step()
        self.f_trans.setSFVec3f([x0, y0, self.pos0[2]])
        self.f_rot.setSFRotation([0, 0, 1, float(rng.uniform(-self.yaw_max, self.yaw_max))])
        self.robot.resetPhysics()

        self.meta = (x0 + self.meta_dist, y0 + float(rng.uniform(-self.meta_lat, self.meta_lat)))
        self.meta_nodo.getField("translation").setSFVec3f(
            [self.meta[0], self.meta[1], 0.005])

        intentos = 0
        while len(self.obstaculos) < self.n_obstaculos and intentos < 200:
            intentos += 1
            ox = float(rng.uniform(x0 + 0.8, self.meta[0] - 0.6))
            oy = float(rng.uniform(y0 - self.ancho, y0 + self.ancho))
            if all(math.hypot(ox - a, oy - b) > 0.5 for a, b in self.obstaculos):
                self.obstaculos.append((ox, oy))
                self.nodos_obs.append(self._crear_nodo(
                    f"Solid {{ translation {ox} {oy} 0.25 children [ Shape {{ "
                    "appearance PBRAppearance { baseColor 0.9 0.2 0.2 "
                    "roughness 1 metalness 0 } geometry Box { size 0.3 0.3 0.5 } "
                    "} ] boundingObject Box { size 0.3 0.3 0.5 } }"))

        for _ in range(5):
            self._sim_step()
        self.pasos = 0
        x, y, _, _ = self._estado()
        self.dist_prev = math.hypot(self.meta[0] - x, self.meta[1] - y)
        return self._obs(), {}

    def step(self, accion):
        x_a, a_a = ACCIONES[int(accion)]
        self.f_custom.setSFString(f"{x_a} {a_a} 1")
        for _ in range(self.repeticion):
            self._sim_step()

        x, y, z, yaw = self._estado()
        dist = math.hypot(self.meta[0] - x, self.meta[1] - y)
        recompensa = 5.0 * (self.dist_prev - dist) - 0.01
        self.dist_prev = dist
        self.pasos += 1

        terminado, motivo = False, ""
        if z < self.z_caida:
            recompensa -= 10.0
            terminado, motivo = True, "caida"
        elif any(math.hypot(x - a, y - b) < self.r_obs + self.r_robot
                 for a, b in self.obstaculos):
            recompensa -= 10.0
            terminado, motivo = True, "colision"
        elif dist < 0.3:
            recompensa += 20.0
            terminado, motivo = True, "meta"

        truncado = (not terminado) and self.pasos >= self.max_pasos
        if truncado:
            motivo = "tiempo"
        return self._obs(), recompensa, terminado, truncado, {"motivo": motivo, "dist": dist}
