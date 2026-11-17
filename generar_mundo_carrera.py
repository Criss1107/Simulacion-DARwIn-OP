"""Genera worlds/carrera_4robots.wbt a partir de worlds/carrera.wbt (tu mundo de un solo robot).

Uso (desde la carpeta del proyecto, p. ej. ~/Documents/carrera_ia):
    python3 generar_mundo_carrera.py

- Toma el bloque `DEF ROBOT RobotisOp2 { ... }` y lo copia 4 veces: DEF ROBOT1..ROBOT4 en la linea de
  salida (y = -0.9, -0.3, 0.3, 0.9), con nombres distintos y el mismo controlador "walk".
- El Robot supervisor pasa a usar el controlador "carrera".
- Revisa en Webots que el suelo (Floor) sea lo bastante grande: la pista usa x de 0 a 5 e y de -2 a 2.
"""
import os
import re
import sys

RAIZ = os.path.dirname(os.path.abspath(__file__))
ORIGEN = os.path.join(RAIZ, "worlds", "carrera.wbt")
DESTINO = os.path.join(RAIZ, "worlds", "carrera_4robots.wbt")
CARRILES_Y = [-0.9, -0.3, 0.3, 0.9]

txt = open(ORIGEN, encoding="utf-8").read()
m = re.search(r"DEF\s+ROBOT\s+RobotisOp2\s*\{", txt)
if not m:
    sys.exit("No encuentro 'DEF ROBOT RobotisOp2 {' en " + ORIGEN)

# bloque completo por balance de llaves
i, prof = m.end() - 1, 0
for k in range(i, len(txt)):
    if txt[k] == "{":
        prof += 1
    elif txt[k] == "}":
        prof -= 1
        if prof == 0:
            fin = k + 1
            break
else:
    sys.exit("Llaves desbalanceadas en el bloque del robot")
bloque = txt[m.start():fin]
cuerpo = bloque[bloque.index("{") + 1:-1]

mt = re.search(r"translation\s+(\S+)\s+(\S+)\s+(\S+)", cuerpo)
z = mt.group(3) if mt else "0.236"


def fijar(cuerpo, campo, valor_linea):
    patron = re.compile(r"^ {2}" + campo + r"\s+.*$", re.M)
    if patron.search(cuerpo):
        return patron.sub("  " + valor_linea, cuerpo, count=1)
    return "\n  " + valor_linea + cuerpo


copias = []
for n, y in enumerate(CARRILES_Y, 1):
    c = fijar(cuerpo, "translation", f"translation 0 {y} {z}")
    c = fijar(c, "name", f'name "robot{n}"')
    copias.append(f"DEF ROBOT{n} RobotisOp2 {{{c}}}")
nuevo = txt[:m.start()] + "\n".join(copias) + txt[fin:]

# supervisor -> controlador "carrera"
if re.search(r'controller\s+"env"', nuevo):
    nuevo = re.sub(r'controller\s+"env"', 'controller "carrera"', nuevo)
else:
    nuevo += '\nRobot {\n  name "supervisor"\n  controller "carrera"\n  supervisor TRUE\n}\n'

open(DESTINO, "w", encoding="utf-8").write(nuevo)
print("Escrito", DESTINO)
print("Siguiente: copia carrera.py y agentes.py a controllers/carrera/ y abre el mundo en Webots.")
