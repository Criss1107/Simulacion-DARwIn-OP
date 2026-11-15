import math
from carrera_env import CarreraEnv

env = CarreraEnv(n_obstaculos=0)
for ep in range(5):
    obs, _ = env.reset(seed=ep)
    total = 0.0
    while True:
        ang = math.atan2(obs[1], obs[2])   # >0: la meta está a la izquierda
        if ang > 0.5:
            a = 3          # girar en sitio a la izquierda
        elif ang > 0.15:
            a = 1          # recto + izquierda
        elif ang < -0.5:
            a = 4
        elif ang < -0.15:
            a = 2
        else:
            a = 0
        obs, r, term, trunc, info = env.step(a)
        total += r
        if term or trunc:
            break
    print(f"ep {ep}: retorno={total:.2f} pasos={env.pasos} motivo={info['motivo']}")