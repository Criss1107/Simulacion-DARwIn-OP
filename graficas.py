import argparse
import glob
import math
import os
import textwrap
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEMILLAS = (0, 1, 2)
MODELOS = [("PVA-AC (sin abs.)", "pva_sin_abs"), ("PVA-AC (completo)", "pva"),
           ("A2C", "a2c"), ("PPO", "ppo"), ("DQN", "dqn")]
REFS = [("Aleatoria", "aleatorio"), ("Heurística con evasión", "heuristico"),
        ("Heurística directa", "heuristico_directo")]
COLOR = {"pva_sin_abs": "#1b7837", "pva": "#7fbf7b", "a2c": "#2166ac", "ppo": "#d6604d",
         "dqn": "#e08214", "aleatorio": "#999999", "heuristico": "#777777",
         "heuristico_directo": "#bbbbbb"}
VARIANTES = [("Estándar", "_eval.csv"),
             ("5 obstáculos", "_eval_obs5_d3_y0_l0.5_a0.6_t250.csv"),
             ("Pista larga 4.5 m", "_eval_obs3_d4.5_y0_l0.5_a0.6_t350.csv"),
             ("Pista estrecha", "_eval_obs3_d3_y0_l0.35_a0.35_t250.csv"),
             ("Salida lateral", "_eval_obs3_d3_y0.4_l0.5_a0.6_t250.csv")]
MOTIVOS = [("meta", "Meta", "#2ca25f"), ("colision", "Colisión", "#de2d26"),
           ("caida", "Caída", "#fd8d3c"), ("tiempo", "Tiempo", "#969696")]
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 120})


# ------------------------------------------------------------------ utilidades
def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def leer(ruta):
    return pd.read_csv(ruta) if os.path.exists(ruta) else None


def cargar_eval(res, pref, sufijo="_eval.csv", semillas=SEMILLAS):
    dfs = {}
    for s in semillas:
        d = leer(os.path.join(res, f"{pref}_s{s}{sufijo}"))
        if d is not None and len(d):
            dfs[s] = d
    return dfs


def resumen(dfs, ddof=0):
    df = pd.concat(dfs.values())
    n = len(df)
    k = int((df.motivo == "meta").sum())
    lo, hi = wilson(k, n)
    ok = df[df.motivo == "meta"]
    return dict(n=n, k=k, exito=k / n, lo=lo, hi=hi,
                ret=df.retorno.mean(), sd=df.retorno.std(ddof=ddof),
                col=(df.motivo == "colision").mean(), caida=(df.motivo == "caida").mean(),
                tiempo=(df.motivo == "tiempo").mean(),
                pasos=ok.pasos.mean() if len(ok) else float("nan"),
                por_semilla={s: (d.motivo == "meta").mean() for s, d in dfs.items()},
                ret_semilla={s: d.retorno.mean() for s, d in dfs.items()})


def pct(x, d=0):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{d}f} %"


def faltan(msg):
    print(f"[aviso] {msg}")


# ------------------------------------------------------------------ datos
def reunir(res):
    modelos = {}
    for etiqueta, pref in MODELOS:
        dfs = cargar_eval(res, pref)
        if dfs:
            modelos[pref] = (etiqueta, dfs, resumen(dfs))
        else:
            faltan(f"no hay *_eval.csv de {etiqueta} ({pref}_s*_eval.csv)")
    refs = {}
    for etiqueta, pref in REFS:
        dfs = cargar_eval(res, pref, semillas=(0,))
        if dfs:
            # ddof=1 para igualar las cifras de las referencias que ya estan en tu informe
            refs[pref] = (etiqueta, dfs, resumen(dfs, ddof=1))
    return modelos, refs


# ------------------------------------------------------------------ figuras
def guardar(fig, out, nombre):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{nombre}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {nombre}")


def fig_curvas(res, out):
    rejilla = np.arange(0, 40001, 500)
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    hay = False
    for etiqueta, pref in MODELOS:
        curvas = []
        for s in SEMILLAS:
            d = leer(os.path.join(res, f"{pref}_s{s}_train.csv"))
            if d is None or len(d) < 5:
                continue
            m = d.retorno.rolling(20, min_periods=5).mean()
            x, y = d.decisiones.to_numpy(float), m.to_numpy(float)
            v = ~np.isnan(y)
            c = np.interp(rejilla, x[v], y[v])
            c[(rejilla < x[v].min()) | (rejilla > x[v].max())] = np.nan
            curvas.append(c)
        if not curvas:
            continue
        hay = True
        C = np.vstack(curvas)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mu = np.nanmean(C, 0)
        ax.plot(rejilla, mu, color=COLOR[pref], lw=1.4, label=f"{etiqueta} (n={len(curvas)})")
        if len(curvas) > 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                sd = np.nanstd(C, 0)
            ax.fill_between(rejilla, mu - sd, mu + sd, color=COLOR[pref], alpha=0.15, lw=0)
    if not hay:
        faltan("no hay *_train.csv; se omite curvas_aprendizaje")
        plt.close(fig)
        return
    ax.set_xlabel("Decisiones de entrenamiento")
    ax.set_ylabel("Retorno (media móvil de 20 episodios)")
    ax.legend(fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    guardar(fig, out, "curvas_aprendizaje")


def fig_exito(modelos, refs, out):
    filas = [(e, p, r) for p, (e, _, r) in list(refs.items()) + list(modelos.items())]
    if not filas:
        return
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    x = np.arange(len(filas))
    for i, (e, p, r) in enumerate(filas):
        ax.bar(i, 100 * r["exito"], color=COLOR[p], width=0.65,
               yerr=[[100 * (r["exito"] - r["lo"])], [100 * (r["hi"] - r["exito"])]],
               capsize=3, ecolor="black", error_kw=dict(lw=0.8))
        tope = max([r["hi"]] + list(r["por_semilla"].values()) if len(r["por_semilla"]) > 1 else [r["hi"]])
        ax.text(i, 100 * tope + 2.5, f"{100 * r['exito']:.0f} %", ha="center", fontsize=7)
        for j, v in r["por_semilla"].items() if len(r["por_semilla"]) > 1 else []:
            ax.plot(i + (j - 1) * 0.18, 100 * v, "o", ms=3, color="black", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([textwrap.fill(e, 12) + f"\n(n={r['n']})" for e, _, r in filas], fontsize=7)
    ax.set_ylabel("Tasa de éxito (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Barras: éxito agregado con IC95 % (Wilson); puntos: cada semilla", fontsize=7.5)
    guardar(fig, out, "barras_exito")


def fig_resultados(modelos, refs, out):
    filas = [(e, r) for _, (e, _, r) in list(refs.items()) + list(modelos.items())]
    if not filas:
        return
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    base = np.zeros(len(filas))
    for clave, etiqueta, color in MOTIVOS:
        vals = np.array([100 * r[{"meta": "exito", "colision": "col", "caida": "caida",
                                   "tiempo": "tiempo"}[clave]] for _, r in filas])
        ax.bar(range(len(filas)), vals, bottom=base, color=color, label=etiqueta, width=0.65)
        base += vals
    ax.set_xticks(range(len(filas)))
    ax.set_xticklabels([textwrap.fill(e, 12) for e, _ in filas], fontsize=7)
    ax.set_ylabel("Episodios (%)")
    ax.legend(ncol=4, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    guardar(fig, out, "barras_resultados")


def datos_variantes(res):
    modelos = [("PVA-AC (sin abs.)", "pva_sin_abs"), ("PVA-AC (completo)", "pva"),
               ("A2C", "a2c"), ("PPO", "ppo"), ("DQN", "dqn")]
    tabla = {}
    for etiqueta, pref in modelos:
        fila = {}
        for nom, suf in VARIANTES:
            d = leer(os.path.join(res, f"{pref}_s0{suf}"))
            if d is not None and len(d):
                fila[nom] = dict(exito=(d.motivo == "meta").mean(), ret=d.retorno.mean(), n=len(d))
        if len(fila) > 1:     # al menos una variante ademas de la estandar
            tabla[pref] = (etiqueta, fila)
    return tabla


def fig_variantes(tv, out):
    if not tv:
        faltan("no hay evaluaciones en variantes; se omite barras_variantes")
        return
    nombres = [n for n, _ in VARIANTES]
    fig, ax = plt.subplots(figsize=(7.0, 2.7))
    w = 0.8 / len(tv)
    for k, (pref, (e, fila)) in enumerate(tv.items()):
        vals = [100 * fila[n]["exito"] if n in fila else np.nan for n in nombres]
        ax.bar(np.arange(len(nombres)) + (k - (len(tv) - 1) / 2) * w, vals, width=w,
               color=COLOR[pref], label=e)
    ax.set_xticks(range(len(nombres)))
    ax.set_xticklabels(nombres, fontsize=7.5)
    ax.set_ylabel("Tasa de éxito (%) — semilla 0")
    ax.set_ylim(0, 105)
    ax.legend(ncol=len(tv), fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.17))
    guardar(fig, out, "barras_variantes")


# ------------------------------------------------------------------ tablas
def tabla_principal(modelos, refs):
    cab = ["Modelo", "Éxito s0", "Éxito s1", "Éxito s2", "Éxito agregado [IC95 %]",
           "Retorno (μ ± σ)", "Colisión", "Caída", "Tiempo", "Pasos a meta"]
    md = ["| " + " | ".join(cab) + " |", "|" + "---|" * len(cab)]
    tex = []
    for e, _, r in [v for v in refs.values()] + [v for v in modelos.values()]:
        ps = [pct(r["por_semilla"].get(s)) if len(r["por_semilla"]) > 1 else "—" for s in SEMILLAS]
        fila = [e] + ps + [f"{pct(r['exito'])} [{100 * r['lo']:.0f}–{100 * r['hi']:.0f}] (n={r['n']})",
                           f"{r['ret']:.1f} ± {r['sd']:.1f}", pct(r["col"]), pct(r["caida"]),
                           pct(r["tiempo"]), "—" if math.isnan(r["pasos"]) else f"{r['pasos']:.1f}"]
        md.append("| " + " | ".join(fila) + " |")
        tex.append(" & ".join(c.replace("%", r"\%").replace("±", r"$\pm$").replace("μ", r"$\mu$")
                              .replace("σ", r"$\sigma$").replace("–", "--") for c in fila) + r" \\")
    t = (r"\begin{table*}[t]" "\n" r"\caption{Comparación principal (30 episodios por semilla; las referencias "
         r"sin aprendizaje usan una sola corrida).}" "\n" r"\label{tab:principal}" "\n"
         r"\centering\footnotesize" "\n" r"\begin{tabular}{lccccccccc}" "\n" r"\hline" "\n"
         + " & ".join(c.replace("%", r"\%").replace("μ", r"$\mu$").replace("σ", r"$\sigma$")
                      for c in cab) + r" \\ \hline" "\n" + "\n".join(tex) + "\n" r"\hline" "\n"
         r"\end{tabular}" "\n" r"\end{table*}")
    return "\n".join(md), t


def tabla_variantes(tv):
    if not tv:
        return "", ""
    nombres = [n for n, _ in VARIANTES]
    cab = ["Variante"] + [e for e, _ in tv.values()]
    md = ["| " + " | ".join(cab) + " |", "|" + "---|" * len(cab)]
    tex = []
    for n in nombres:
        celdas = []
        for _, (_, fila) in tv.items():
            celdas.append(f"{pct(fila[n]['exito'])} ({fila[n]['ret']:.1f})" if n in fila else "—")
        md.append("| " + " | ".join([n] + celdas) + " |")
        tex.append(" & ".join([n] + [c.replace("%", r"\%") for c in celdas]) + r" \\")
    t = (r"\begin{table}[t]" "\n" r"\caption{Éxito (retorno medio) en variantes del entorno sin reentrenar, "
         r"semilla 0, 30 episodios.}" "\n" r"\label{tab:variantes}" "\n" r"\centering\footnotesize" "\n"
         r"\begin{tabular}{l" + "c" * (len(cab) - 1) + "}\n" r"\hline" "\n"
         + " & ".join(cab) + r" \\ \hline" "\n" + "\n".join(tex) + "\n" r"\hline" "\n"
         r"\end{tabular}" "\n" r"\end{table}")
    return "\n".join(md), t


def cargar_carreras(res):
    filas = []
    for ruta in sorted(glob.glob(os.path.join(res, "carrera_s*.csv"))):
        s = os.path.basename(ruta)[len("carrera_s"):-len(".csv")]
        d = pd.read_csv(ruta)
        d.insert(0, "carrera", f"s{s}")
        filas.append(d)
    return pd.concat(filas) if filas else None


def tabla_carreras(res):
    d = cargar_carreras(res)
    if d is None:
        faltan("no hay carrera_s*.csv todavia")
        return "", "", ""
    cab = ["Carrera", "Puesto", "Participante", "Resultado", "Tiempo (s)", "Dist. a meta (m)",
           "Contactos"]
    md = ["| " + " | ".join(cab) + " |", "|" + "---|" * len(cab)]
    tex = []
    for _, r in d.iterrows():
        t = "—" if pd.isna(r.tiempo_sim_s) else f"{r.tiempo_sim_s:.1f}"
        fila = [r.carrera, str(int(r.puesto)), r.participante, r.motivo.replace("_", " "), t,
                f"{r.dist_meta_m:.2f}", str(int(r.contactos_robots))]
        md.append("| " + " | ".join(fila) + " |")
        tex.append(" & ".join(c for c in fila) + r" \\")
    tt = (r"\begin{table}[t]" "\n" r"\caption{Resultados de la carrera simultánea de 4 robots.}" "\n"
          r"\label{tab:carrera}" "\n" r"\centering\footnotesize" "\n" r"\begin{tabular}{lccccc c}" "\n"
          r"\hline" "\n" + " & ".join(cab) + r" \\ \hline" "\n" + "\n".join(tex) + "\n" r"\hline" "\n"
          r"\end{tabular}" "\n" r"\end{table}")
    # resumen: ganadores y contactos
    ganadores = d[(d.puesto == 1) & (d.motivo == "meta")].participante.value_counts().to_dict()
    n_carr = d.carrera.nunique()
    suma_contactos = int(d.contactos_robots.sum())
    sin_ganador = n_carr - sum(ganadores.values())
    texto = [f"Se corrieron {n_carr} carreras ({', '.join(sorted(d.carrera.unique()))})."]
    if ganadores:
        texto.append("Victorias: " + ", ".join(f"{k}: {v}" for k, v in ganadores.items()) + ".")
    if sin_ganador:
        texto.append(f"{sin_ganador} carrera(s) sin ganador (ningún robot llegó a la meta).")
    llegadas = d[d.motivo == "meta"].groupby("participante").size().to_dict()
    if llegadas:
        texto.append("Llegadas a la meta: " + ", ".join(f"{k}: {v}" for k, v in llegadas.items()) + ".")
    texto.append(f"Contactos entre robots: la columna contactos_robots suma {suma_contactos}; "
                 f"como cada contacto se anota en los dos robots involucrados, equivale a "
                 f"{suma_contactos / 2:g} eventos de contacto en {n_carr} carreras.")
    return "\n".join(md), tt, " ".join(texto)


# ------------------------------------------------------------------ texto con cifras
def comparar(r, ref, nombre_ref):
    if r["lo"] > ref["hi"]:
        rel = "superior, con intervalos de confianza que no se solapan"
    elif r["hi"] < ref["lo"]:
        rel = "inferior, con intervalos de confianza que no se solapan"
    else:
        rel = "comparable (los IC95 % se solapan)"
    return f"{rel} a {nombre_ref} ({pct(ref['exito'])})"


def parrafos(modelos):
    out = []
    a2c, mejor = modelos.get("a2c"), modelos.get("pva_sin_abs")
    for pref, nom in (("ppo", "PPO"), ("dqn", "DQN")):
        if pref not in modelos:
            continue
        r = modelos[pref][2]
        por = ", ".join(f"s{s}: {pct(v)}" for s, v in r["por_semilla"].items())
        txt = (f"{nom} alcanzó un {pct(r['exito'])} de éxito agregado sobre {r['n']} episodios "
               f"(IC95 % {100 * r['lo']:.0f}–{100 * r['hi']:.0f}; por semilla {por}), con retorno "
               f"medio {r['ret']:.1f} ± {r['sd']:.1f}, {pct(r['col'])} de colisiones, "
               f"{pct(r['caida'])} de caídas y {pct(r['tiempo'])} de episodios por tiempo")
        if not math.isnan(r["pasos"]):
            txt += f"; llega a la meta en ~{r['pasos']:.1f} pasos"
        txt += "."
        comp = []
        if mejor:
            comp.append(comparar(r, mejor[2], "PVA-AC sin absorción"))
        if a2c:
            comp.append(comparar(r, a2c[2], "A2C"))
        if comp:
            txt += f" Su desempeño es {'; y '.join(comp)}."
        out.append(txt)
    if out:
        out.append("Nota: el IC95 % se calcula sobre los episodios agregados de las tres semillas; "
                   "las semillas varían entre sí, por lo que la comparación es indicativa.")
    return "\n\n".join(out)


def bloque_participantes(modelos):
    def mejor(pref):
        if pref not in modelos:
            return None
        r = modelos[pref][2]
        return max(r["por_semilla"], key=lambda s: (r["por_semilla"][s], r["ret_semilla"][s]))
    filas = [("PVA-AC", "pva", "pva_sin_abs", "pva_sin_abs"), ("A2C", "a2c", "a2c", "a2c"),
             ("PPO", "ppo", "ppo", "ppo"), ("DQN", "dqn", "dqn", "dqn")]
    lin = ["PARTICIPANTES = [      # mejor semilla de cada algoritmo por su tasa de éxito en evaluación"]
    for nombre, tipo, pref, arch in filas:
        s = mejor(pref)
        lin.append(f'    ("{nombre}", "{tipo}", "{arch}_s{s}"),' if s is not None
                   else f'    ("{nombre}", "{tipo}", "{arch}_s0"),   # (aún sin evaluar)')
    lin.append("]")
    return "\n".join(lin)


# ------------------------------------------------------------------ principal
def main():
    aqui = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=os.path.join(aqui, "resultados"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.res, "figuras")
    os.makedirs(out, exist_ok=True)

    modelos, refs = reunir(a.res)
    fig_curvas(a.res, out)
    fig_exito(modelos, refs, out)
    fig_resultados(modelos, refs, out)
    tv = datos_variantes(a.res)
    fig_variantes(tv, out)

    md1, tex1 = tabla_principal(modelos, refs)
    md2, tex2 = tabla_variantes(tv)
    md3, tex3, res3 = tabla_carreras(a.res)
    partes_md = ["# Tablas y textos del informe (generado por graficas.py)\n",
                 "## Tabla principal de comparación\n", md1, "\n",
                 "## Texto de análisis de PPO y DQN\n", parrafos(modelos) or "(aún no hay PPO/DQN)", "\n"]
    if md2:
        partes_md += ["## Variantes del entorno (éxito y, entre paréntesis, retorno medio)\n", md2, "\n"]
    if md3:
        partes_md += ["## Carrera simultánea\n", md3, "\n", res3, "\n"]
    partes_md += ["## Bloque para controllers/carrera/carrera.py\n", "```python",
                  bloque_participantes(modelos), "```\n"]
    with open(os.path.join(a.res, "tablas_informe.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(partes_md))
    with open(os.path.join(a.res, "tablas_informe.tex"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(t for t in (tex1, tex2, tex3) if t))
    print("[ok] tablas_informe.md / tablas_informe.tex")
    print("\n" + bloque_participantes(modelos))


if __name__ == "__main__":
    main()
