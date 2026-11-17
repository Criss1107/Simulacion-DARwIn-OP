"""Agentes para la carrera: cargan un modelo entrenado y devuelven una accion por observacion.

Interfaz comun:
    agente.reiniciar()
    agente.actuar(obs, r_prev) -> int   (accion 0..4)
`obs` es la observacion de 8 valores de CarreraEnv. `r_prev` es la recompensa del paso anterior
(5 * progreso - 0.01); solo la usa PVA-AC para decidir si aborta una habilidad.

Colocar en controllers/carrera/. Importa pva_ac desde controllers/env/ (carrera.py ajusta sys.path).
"""
import os


class AgenteAleatorio:
    def __init__(self, semilla=0):
        import random
        self.rng = random.Random(semilla)

    def reiniciar(self):
        pass

    def actuar(self, obs, r_prev=0.0):
        return self.rng.randrange(5)


class AgenteHeuristico:
    def __init__(self, evadir):
        from pva_ac import _politica_heuristica
        self._f = _politica_heuristica
        self.evadir = evadir

    def reiniciar(self):
        pass

    def actuar(self, obs, r_prev=0.0):
        return self._f(obs, self.evadir)


class AgenteA2C:
    def __init__(self, ruta):
        import torch
        import pva_ac as P
        self.torch, self.P = torch, P
        self.model, _, _ = P.load_checkpoint(ruta)

    def reiniciar(self):
        pass

    def actuar(self, obs, r_prev=0.0):
        torch = self.torch
        with torch.no_grad():
            logits, _, _ = self.model(torch.as_tensor(obs[None]))
            return torch.distributions.Categorical(logits=logits[:, :5]).sample().item()


class AgentePVA:
    """Replica un paso de P.evaluate_pva (politica + ask + verificacion + habilidad + aborto)."""

    def __init__(self, ruta, use_abort=True):
        import torch
        import pva_ac as P
        self.torch, self.P = torch, P
        self.model, self.disc, ck = P.load_checkpoint(ruta)
        self.cfg = P.Cfg(**ck["cfg"])
        self.use_abort = use_abort
        self.sk = None

    def reiniciar(self):
        self.sk = None

    def actuar(self, obs, r_prev=0.0):
        torch, P, c = self.torch, self.P, self.cfg
        A = 5
        with torch.no_grad():
            if self.sk is not None:      # la accion anterior formaba parte de una habilidad
                self.sk["partial"] += (c.gamma ** self.sk["j"]) * (r_prev * c.rew_scale)
                self.sk["j"] += 1
            logits, V, feat = self.model(torch.as_tensor(obs[None]))
            vbar = V.mean(-1)
            sk = self.sk
            if sk is not None and (not sk["queue"] or (self.use_abort and P.should_abort(
                    sk["partial"], sk["j"], vbar.item(), sk["mu0"], sk["sig0"], c))):
                sk = None
            if sk is not None:
                a = sk["queue"].pop(0)
            else:
                a = torch.distributions.Categorical(logits=logits).sample().item()
                if a == A:
                    z, acc, mu, sig = self.disc.verify(feat, vbar, c)
                    if acc[0].item():
                        q = self.disc.decode_actions(feat, z, greedy=True)[0, 0].tolist()
                        a = q[0]
                        sk = dict(queue=q[1:], partial=0.0, j=0, mu0=mu.item(), sig0=sig.item())
                    else:
                        a = torch.distributions.Categorical(logits=logits[:, :A]).sample().item()
            self.sk = sk
        return int(a)


class AgenteSB3:
    """PPO / DQN / A2C de Stable-Baselines3 (modelo guardado con model.save(ruta))."""

    def __init__(self, tipo, ruta, deterministico=True):
        from stable_baselines3 import PPO, DQN, A2C
        self.model = {"ppo": PPO, "dqn": DQN, "a2c": A2C}[tipo].load(ruta)
        self.det = deterministico

    def reiniciar(self):
        pass

    def actuar(self, obs, r_prev=0.0):
        a, _ = self.model.predict(obs, deterministic=self.det)
        return int(a)


def cargar(tipo, archivo, res_dir, semilla=0):
    """tipo: pva | pva_sin_aborto | a2c | ppo | dqn | sb3_a2c | heuristico | heuristico_directo | aleatorio.
    archivo: nombre dentro de resultados/ sin extension (.pt para pva/a2c, .zip para SB3)."""
    if tipo == "aleatorio":
        return AgenteAleatorio(semilla)
    if tipo in ("heuristico", "heuristico_directo"):
        return AgenteHeuristico(evadir=(tipo == "heuristico"))
    ext = ".pt" if tipo in ("pva", "pva_sin_aborto", "a2c") else ".zip"
    ruta = os.path.join(res_dir, archivo + ext)
    if not os.path.exists(ruta):
        raise SystemExit(f"No existe el modelo {ruta}")
    if tipo == "pva":
        return AgentePVA(ruta)
    if tipo == "pva_sin_aborto":
        return AgentePVA(ruta, use_abort=False)
    if tipo == "a2c":
        return AgenteA2C(ruta)
    if tipo == "sb3_a2c":
        return AgenteSB3("a2c", ruta)
    if tipo in ("ppo", "dqn"):
        return AgenteSB3(tipo, ruta)
    raise SystemExit(f"Tipo de agente desconocido: {tipo}")
