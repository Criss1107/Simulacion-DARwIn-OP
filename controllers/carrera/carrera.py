"""Supervisor de la carrera simultanea de 4 robots (controllers/carrera/carrera.py).

Pista con curvas: cada robot sigue una lista de puntos intermedios (WAYPOPINTS, relativos a su salida)
el ultimo es la meta. Todos los carriles tienen la misma forma y los mismos obstaculos (carrera justa).
Los robots se ven entre si como obstaculos en los rayos. Un robot sale de la carrera si se cae,
choca con un obstaculo otoca a otro robot. Gana el primero en llegar a la meta.
Los agentes se entrenaron solo para ir a una meta, aqui la "meta" que ven es el siguiente punto.
"""
import csv
import math
import os
import random
import sys

import numpy as np

RAIZ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
RES = os.path.join(RAIZ, "resultados")
sys.path.insert(0, os.path.join(RAIZ, "controllers", "env"))              # pva_ac, carrera_env
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))            # agentes

from controller import Supervisor                                          # noqa: E402
from carrera_env import ACCIONES, ANGULOS_RAYOS, RANGO_MAX, _rayo_circulo  # noqa: E402
import agentes                                                             # noqa: E402

PARTICIPANTES = [
    ("PVA-AC", "pva", "pva_sin_abs_s2"),
    ("A2C", "a2c", "a2c_s2"),
    ("PPO", "ppo", "ppo_sX"),
    ("DQN", "dqn", "dqn_sX"),
]
#configuracion
SEMILLA = 0            # cambia el reparto de carriles y la posicion de los obstaculos
PARTICIPANTES = [
    ("PVA-AC", "pva", "pva_sin_abs_s2"),
    ("A2C", "a2c", "a2c_s2"),
    ("PPO", "ppo", "ppo_s0"),
    ("DQN", "dqn", "dqn_s0"),
]
WAYPOINTS = [(1.5, 0.6), (3.0, -0.6), (4.5, 0.0)]   # relativos a la salida, el ultimo = meta
N_OBST = 3
MAX_DECISIONES = 400
REPETICION = 25        #pasos de simulacion por decision (igual que en el entrenamiento)
R_OBS, R_ROBOT = 0.15, 0.12     #mismos radios que CarreraEnv
R_ENTRE_ROBOTS = 0.30           #ontacto entre robots
RADIO_WP, RADIO_META = 0.5, 0.3
Z_CAIDA = 0.20
META_DIST_NORM = 3.0            # normalizacion de la distancia como en el entrenamiento
ELIMINAR_POR_CHOQUE = True      # False: solo se cuentan los contactos
GRABAR_VIDEO = False            # API de Webots por verificar, si falla solo avisa


N = 4
COLORES = ["1 0.2 0.2", "0.2 0.4 1", "0.2 0.8 0.2", "1 0.8 0.1"]   # carril 1.4

sup = Supervisor()
dt = int(sup.getBasicTimeStep())
nodos = [sup.getFromDef(f"ROBOT{i + 1}") for i in range(N)]
if any(n is None for n in nodos):
    raise SystemExit("Faltan DEF ROBOT1..ROBOT4 en el mundo (usa generar_mundo_carrera.py).")
customs = [n.getField("customData") for n in nodos]
hijos = sup.getRoot().getField("children")


def avanzar(n):
    for _ in range(n):
        if sup.step(dt) == -1:
            raise SystemExit


def crear(texto):
    hijos.importMFNodeFromString(-1, texto)


def pose(j):
    x, y, z = nodos[j].getPosition()
    R = nodos[j].getOrientation()
    return x, y, z, math.atan2(R[3], R[0])


#reparto de carriles: participante p corre en el robot (carril) orden[p]
rng = random.Random(SEMILLA)
orden = list(range(N))
rng.shuffle(orden)
part_de_robot = {orden[p]: p for p in range(N)}
ags = []
for p, (nombre, tipo, archivo) in enumerate(PARTICIPANTES):
    ags.append(agentes.cargar(tipo, archivo, RES, SEMILLA + p))

# los robots se levantan y se leen las posiciones de salida
for c in customs:
    c.setSFString("0 0 0")
avanzar(200)
salida = [pose(j)[:2] for j in range(N)]
meta = [w[-1] for w in wps]

# obstaculos: mismo patron relativo en cada carril
ro = random.Random(1000 + SEMILLA)
ruta = [(0.0, 0.0)] + WAYPOINTS
patron, intentos = [], 0
while len(patron) < N_OBST and intentos < 200:
    intentos += 1
    s = ro.randrange(len(ruta) - 1)
    t = ro.uniform(0.25, 0.75)
    bx = ruta[s][0] + t * (ruta[s + 1][0] - ruta[s][0])
    by = ruta[s][1] + t * (ruta[s + 1][1] - ruta[s][1]) + ro.uniform(-0.15, 0.15)
    if all(math.hypot(bx - a, by - b) > 0.5 for a, b in patron):
        patron.append((bx, by))
obstaculos = [(salida[j][0] + bx, salida[j][1] + by) for j in range(N) for bx, by in patron]
for ox, oy in obstaculos:
    crear(f"Solid {{ translation {ox} {oy} 0.25 children [ Shape {{ appearance PBRAppearance "
          "{ baseColor 0.9 0.2 0.2 roughness 1 metalness 0 } geometry Box { size 0.3 0.3 0.5 } } ] "
          "boundingObject Box { size 0.3 0.3 0.5 } }")
#marcas visuales cmo puntos intermedios por carril y linea de meta
for j in range(N):
    for (wx, wy) in wps[j][:-1]:
        crear(f"Solid {{ translation {wx} {wy} 0.005 children [ Shape {{ appearance PBRAppearance "
              f"{{ baseColor {COLORES[j]} roughness 1 metalness 0 }} geometry Box {{ size 0.2 0.2 0.01 }} }} ] }}")
yc = sum(m[1] for m in meta) / N
crear(f"Solid {{ translation {meta[0][0]} {yc} 0.005 children [ Shape {{ appearance PBRAppearance "
      "{ baseColor 1 1 1 roughness 1 metalness 0 } geometry Box { size 0.1 3.6 0.01 } } ] }")

print("Carriles (1 = y menor):")
for j in sorted(range(N), key=lambda k: salida[k][1]):
    print(f"  carril ROBOT{j + 1} (y={salida[j][1]:+.2f}, marcas {COLORES[j]}): {PARTICIPANTES[part_de_robot[j]][0]}")

if GRABAR_VIDEO:
    try:
        sup.movieStartRecording(os.path.join(RES, f"carrera_s{SEMILLA}.mp4"), 1280, 720, 0, 90, 1, False)
    except Exception as e:                                                   # noqa: BLE001
        print("No se pudo iniciar la grabacion:", e)
        GRABAR_VIDEO = False

#estado de cada robot
est = []
for j in range(N):
    x, y, _, _ = pose(j)
    est.append(dict(activo=True, motivo="", wp=0, dec=0, t_fin=None, contactos=0, r_prev=0.0,
                    dist_prev=math.hypot(wps[j][0][0] - x, wps[j][0][1] - y)))
for a in ags:
    a.reiniciar()


def terminar(j, motivo):
    est[j]["activo"], est[j]["motivo"] = False, motivo
    customs[j].setSFString("0 0 0")
    print(f"  [{sup.getTime():6.1f} s] {PARTICIPANTES[part_de_robot[j]][0]}: {motivo}")


def observacion(j):
    x, y, _, yaw = pose(j)
    tx, ty = wps[j][est[j]["wp"]]
    dx, dy = tx - x, ty - y
    ang = math.atan2(dy, dx) - yaw
    circulos = [(ox, oy, R_OBS) for ox, oy in obstaculos]
    circulos += [(pose(k)[0], pose(k)[1], R_ROBOT) for k in range(N) if k != j]
    rayos = []
    for a in ANGULOS_RAYOS:
        m = RANGO_MAX
        for cx, cy, r in circulos:
            t = _rayo_circulo(x, y, yaw + a, cx, cy, r)
            if t is not None and t < m:
                m = t
        rayos.append(m / RANGO_MAX)
    return np.array([min(math.hypot(dx, dy) / META_DIST_NORM, 2.0), math.sin(ang), math.cos(ang)] + rayos,
                    dtype=np.float32)


#carrera
print("¡Salida!")
for d in range(MAX_DECISIONES):
    activos = [j for j in range(N) if est[j]["activo"]]
    if not activos:
        break
    for j in activos:
        a = ags[part_de_robot[j]].actuar(observacion(j), est[j]["r_prev"])
        xa, aa = ACCIONES[int(a)]
        customs[j].setSFString(f"{xa} {aa} 1")
        est[j]["dec"] += 1
    avanzar(REPETICION)

    poses = [pose(j) for j in range(N)]
    for j in activos:
        x, y, z, _ = poses[j]
        if z < Z_CAIDA:
            terminar(j, "caida")
            continue
        if any(math.hypot(x - ox, y - oy) < R_OBS + R_ROBOT for ox, oy in obstaculos):
            terminar(j, "colision")
            continue
        tx, ty = wps[j][est[j]["wp"]]
        dist = math.hypot(tx - x, ty - y)
        est[j]["r_prev"] = 5.0 * (est[j]["dist_prev"] - dist) - 0.01
        ultimo = est[j]["wp"] == len(WAYPOINTS) - 1
        if dist < (RADIO_META if ultimo else RADIO_WP):
            if ultimo:
                est[j]["t_fin"] = sup.getTime()
                terminar(j, "meta")
            else:
                est[j]["wp"] += 1
                nx, ny = wps[j][est[j]["wp"]]
                est[j]["dist_prev"] = math.hypot(nx - x, ny - y)
        else:
            est[j]["dist_prev"] = dist
    # contactos entre robots
    for j in range(N):
        for k in range(j + 1, N):
            if (j in activos or k in activos) and \
                    math.hypot(poses[j][0] - poses[k][0], poses[j][1] - poses[k][1]) < R_ENTRE_ROBOTS:
                for q in (j, k):
                    est[q]["contactos"] += 1
                    if ELIMINAR_POR_CHOQUE and est[q]["activo"]:
                        terminar(q, "choque_robot")

for j in range(N):
    if est[j]["activo"]:
        terminar(j, "tiempo")
if GRABAR_VIDEO:
    try:
        sup.movieStopRecording()
    except Exception as e:                                                   # noqa: BLE001
        print("No se pudo cerrar la grabacion:", e)

#resultados
filas = []
for j in range(N):
    x, y, _, _ = pose(j)
    filas.append(dict(nombre=PARTICIPANTES[part_de_robot[j]][0], carril=j + 1, motivo=est[j]["motivo"],
                      t=est[j]["t_fin"], dec=est[j]["dec"], wp=est[j]["wp"], contactos=est[j]["contactos"],
                      dist=math.hypot(meta[j][0] - x, meta[j][1] - y)))
filas.sort(key=lambda f: (f["t"] is None, f["t"] if f["t"] is not None else 0.0, f["dist"]))
os.makedirs(RES, exist_ok=True)
ruta_csv = os.path.join(RES, f"carrera_s{SEMILLA}.csv")
with open(ruta_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["puesto", "participante", "carril", "motivo", "tiempo_sim_s", "decisiones",
                "puntos_alcanzados", "contactos_robots", "dist_meta_m"])
    for i, r in enumerate(filas, 1):
        w.writerow([i, r["nombre"], r["carril"], r["motivo"],
                    "" if r["t"] is None else round(r["t"], 1), r["dec"], r["wp"],
                    r["contactos"], round(r["dist"], 2)])
print("\nPuesto | Participante | Resultado | Tiempo | Dist. a la meta")
for i, r in enumerate(filas, 1):
    tt = "-" if r["t"] is None else f"{r['t']:.1f} s"
    print(f"  {i}  | {r['nombre']:<8} | {r['motivo']:<12} | {tt:>8} | {r['dist']:.2f} m")
print("Guardado:", ruta_csv)
