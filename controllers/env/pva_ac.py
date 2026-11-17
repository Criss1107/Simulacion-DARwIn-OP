"""PVA-AC y A2C adaptados al entorno de la carrera (Parte 1).

Basado en el notebook del Taller 2 (Asteroids). Cambios respecto al original:
  - Encoder MLP (observacion de 8 valores) en vez de CNN sobre imagenes.
  - Un solo entorno (un robot en Webots): n_envs = 1.
  - La recompensa NO se recorta al signo: se escala con cfg.rew_scale solo para
    entrenar (el retorno que se registra es el real).
  - Se completaron los fallos del notebook: lam_s, claves de cnt, metricas
    corr/div y el warm-up (existia en Cfg pero no se usaba).
"""
import math
import time
import random
import collections
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cpu")
N_ACTIONS = 5   # = len(ACCIONES) en carrera_env.py
OBS_DIM = 8     # = tamano de la observacion en carrera_env.py


@dataclass
class Cfg:
    n_steps: int = 20           # con 1 solo entorno se usa un lote mayor que 5
    gamma: float = 0.99
    lr: float = 7e-4
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    total_updates: int = 2000
    seed: int = 0
    rew_scale: float = 0.1      # escala de la recompensa (solo entrenamiento)
    # Discovery
    K: int = 5                  # miembros del ensamble del critic Q
    H: int = 8                  # largo de la habilidad
    zdim: int = 8
    N_cand: int = 8             # candidatos por consulta
    beta_o: float = 1.0         # optimismo del proponedor
    beta_p: float = 0.5         # pesimismo de la verificacion
    lam_s: float = 0.1          # peso de la sorpresa (VALOR ASUMIDO: el informe no lo da)
    warmup_updates: int = 300   # updates iniciales donde se aceptan habilidades al azar
    warmup_p: float = 0.5       # prob. de aceptar durante el warm-up
    lam_kl: float = 0.01
    vae_beta: float = 0.1
    eps: float = 0.0            # margen para aceptar una habilidad
    kappa: float = 2.0          # umbral de aborto
    eta: float = 0.1
    ask_cost: float = 0.01


# ---------------------------------------------------------------- redes
class Encoder(nn.Module):
    def __init__(self, feat_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(OBS_DIM, 128), nn.ReLU(),
                                 nn.Linear(128, feat_dim), nn.ReLU())

    def forward(self, x):
        return self.net(x.float())


class ActorCritic(nn.Module):
    def __init__(self, n_actions, use_ask=False, K=1, feat_dim=128):
        super().__init__()
        self.n_actions, self.use_ask = n_actions, use_ask
        self.enc = Encoder(feat_dim)
        self.pi = nn.Linear(feat_dim, n_actions + int(use_ask))   # ultima salida = a_ask
        self.v = nn.Linear(feat_dim, K)                           # K cabezas de V

    def forward(self, x):
        f = self.enc(x)
        return self.pi(f), self.v(f), f


def mlp(i, o, h=256):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))


class Discovery(nn.Module):
    def __init__(self, n_actions, feat_dim=128, H=8, zdim=8, K=5):
        super().__init__()
        self.A, self.H, self.zdim, self.K = n_actions, H, zdim, K
        self.prop = mlp(feat_dim, 2 * zdim)                    # G_psi(z|s)
        self.q = mlp(feat_dim + zdim, K)                       # Q_k(s,z)
        self.enc_q = mlp(feat_dim + H * n_actions, 2 * zdim)   # q_omega(z|s,a_1:H)
        self.dec = mlp(feat_dim + zdim, H * n_actions)         # d_omega(a_1:H|s,z)

    def propose(self, feat, N):
        mu, logstd = self.prop(feat).chunk(2, -1)
        logstd = logstd.clamp(-4, 1)
        eps = torch.randn(feat.size(0), N, self.zdim, device=feat.device)
        z = mu.unsqueeze(1) + logstd.exp().unsqueeze(1) * eps
        kl = 0.5 * (mu ** 2 + (2 * logstd).exp() - 1 - 2 * logstd).sum(-1).mean()
        return z, kl

    def q_stats(self, feat, z):
        f = feat.unsqueeze(1).expand(-1, z.size(1), -1)
        q = self.q(torch.cat([f, z], -1))                                   # (B,N,K)
        sig = q.std(-1) if q.size(-1) > 1 else torch.zeros_like(q[..., 0])  # K=1 -> sigma=0
        return q.mean(-1), sig, q

    def decode_logits(self, feat, z):
        if z.dim() == 2:
            z = z.unsqueeze(1)
        f = feat.unsqueeze(1).expand(-1, z.size(1), -1)
        return self.dec(torch.cat([f, z], -1)).view(feat.size(0), z.size(1), self.H, self.A)

    def decode_actions(self, feat, z, greedy=True):
        lg = self.decode_logits(feat, z)
        if greedy:
            return lg.argmax(-1)
        return torch.distributions.Categorical(logits=lg).sample()

    def vae_loss(self, feat, acts, beta):
        oh = F.one_hot(acts, self.A).float().flatten(1)
        mu, logstd = self.enc_q(torch.cat([feat, oh], -1)).chunk(2, -1)
        logstd = logstd.clamp(-4, 1)
        z = mu + logstd.exp() * torch.randn_like(mu)
        lg = self.decode_logits(feat, z).squeeze(1)
        rec = F.cross_entropy(lg.reshape(-1, self.A), acts.reshape(-1))
        kl = 0.5 * (mu ** 2 + (2 * logstd).exp() - 1 - 2 * logstd).sum(-1).mean()
        return rec * self.H + beta * kl

    def surprise(self, feat, z, pi_logits, clip=10.0):
        d = F.softmax(self.decode_logits(feat, z), -1)
        logp = F.log_softmax(pi_logits[:, :self.A], -1)
        s = -(d * logp[:, None, None, :]).sum(-1).mean(-1)
        return s.clamp(max=clip)

    def proposer_objective(self, feat, pi_logits, cfg):
        z, kl = self.propose(feat, cfg.N_cand)
        mu_q, sig_q, _ = self.q_stats(feat, z)
        S = self.surprise(feat, z, pi_logits)
        return (mu_q + cfg.beta_o * sig_q + cfg.lam_s * S).mean() - cfg.lam_kl * kl

    @torch.no_grad()
    def verify(self, feat, v_bar, cfg):
        z, _ = self.propose(feat, cfg.N_cand)
        mu_q, sig_q, _ = self.q_stats(feat, z)
        lcb = mu_q - cfg.beta_p * sig_q
        best = lcb.argmax(-1)
        idx = torch.arange(feat.size(0), device=feat.device)
        z_star, lcb_star = z[idx, best], lcb[idx, best]
        mu_star, sig_star = mu_q[idx, best], sig_q[idx, best]
        accept = (lcb_star - v_bar) > cfg.eps
        return z_star, accept, mu_star, sig_star


def should_abort(partial_return, j, v_next, q_mu0, q_sig0, cfg):
    delta = partial_return + (cfg.gamma ** j) * v_next - q_mu0
    return delta < -(cfg.kappa * q_sig0 + cfg.eta)


# ---------------------------------------------------------------- checkpoints
def save_checkpoint(path, model, disc=None, c=None):
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(),
                "disc": disc.state_dict() if disc is not None else None,
                "cfg": asdict(c) if c is not None else None,
                "use_ask": model.use_ask}, path)
    print("guardado:", path)


def load_checkpoint(path):
    ck = torch.load(path, map_location=device)
    c = ck["cfg"]
    model = ActorCritic(N_ACTIONS, use_ask=ck["use_ask"],
                        K=c["K"] if ck["use_ask"] else 1).to(device)
    model.load_state_dict(ck["model"])
    disc = None
    if ck["disc"] is not None:
        disc = Discovery(N_ACTIONS, H=c["H"], zdim=c["zdim"], K=c["K"]).to(device)
        disc.load_state_dict(ck["disc"])
    return model.eval(), (disc.eval() if disc is not None else None), ck


class SegBuf:
    """Ventanas de H acciones con su retorno. El VAE solo aprende de las COMPLETAS de retorno alto."""
    def __init__(self, capacity=20000):
        self.data = collections.deque(maxlen=capacity)

    def add(self, obs, acts, ret, aborted=False):
        self.data.append((obs, np.asarray(acts, dtype=np.int64), float(ret), bool(aborted)))

    def sample_top(self, batch, q=0.75):
        full = [d for d in self.data if not d[3]]
        if len(full) < 8:
            return None
        thr = np.quantile([d[2] for d in full], q)
        top = [d for d in full if d[2] >= thr]
        pick = random.sample(top, min(batch, len(top)))
        o = torch.as_tensor(np.stack([p[0] for p in pick]), device=device)
        a = torch.as_tensor(np.stack([p[1] for p in pick]), device=device)
        return o, a

    def __len__(self):
        return len(self.data)


# ---------------------------------------------------------------- A2C base
def train_a2c(env, cfg, n_updates=None, log_every=50, verbose=True):
    n_updates = n_updates or cfg.total_updates
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); random.seed(cfg.seed)
    obs = np.stack([env.reset(seed=cfg.seed)[0]])
    model = ActorCritic(N_ACTIONS).to(device)
    opt = torch.optim.RMSprop(model.parameters(), lr=cfg.lr, alpha=0.99, eps=1e-5)
    ep_ret = 0.0; done_returns = []; history = []
    t0 = time.time()
    for upd in range(1, n_updates + 1):
        buf = dict(obs=[], act=[], rew=[], done=[])
        for _ in range(cfg.n_steps):
            o = torch.as_tensor(obs, device=device)
            with torch.no_grad():
                logits, _, _ = model(o)
                a = torch.distributions.Categorical(logits=logits).sample()
            ob, r, term, trunc, _ = env.step(int(a.item()))
            ep_ret += r
            d = term or trunc
            if d:
                done_returns.append(ep_ret); ep_ret = 0.0
                ob, _ = env.reset()
            buf["obs"].append(o); buf["act"].append(a)
            buf["rew"].append(torch.tensor([r * cfg.rew_scale], device=device, dtype=torch.float32))
            buf["done"].append(torch.tensor([float(d)], device=device))
            obs = np.stack([ob])
        with torch.no_grad():
            _, v_last, _ = model(torch.as_tensor(obs, device=device))
            R = v_last.mean(-1)
        rets = []
        for t in reversed(range(cfg.n_steps)):
            R = buf["rew"][t] + cfg.gamma * R * (1 - buf["done"][t])
            rets.append(R)
        rets = torch.stack(rets[::-1])
        o = torch.cat(buf["obs"]); a = torch.cat(buf["act"]); rets = rets.flatten()
        logits, v, _ = model(o)
        v = v.mean(-1)
        dist = torch.distributions.Categorical(logits=logits)
        adv = (rets - v).detach()
        loss = (-(dist.log_prob(a) * adv).mean()
                + cfg.vf_coef * F.mse_loss(v, rets)
                - cfg.ent_coef * dist.entropy().mean())
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        opt.step()
        if upd % log_every == 0:
            m = float(np.mean(done_returns[-20:])) if done_returns else float("nan")
            history.append(dict(decisiones=upd * cfg.n_steps, ret=m))
            if verbose:
                print(f"upd {upd} | decisiones {upd * cfg.n_steps} | ret(ult20) {m:.1f} | "
                      f"{time.time() - t0:.0f}s", flush=True)
    return model, history


# ---------------------------------------------------------------- PVA-AC
def train_pva(env, cfg, n_updates=None, log_every=50, use_abort=True, use_abs=True,
              lam_abs=0.5, use_warmup=True, verbose=True):
    n_updates = n_updates or cfg.total_updates
    A, H, K = N_ACTIONS, cfg.H, cfg.K
    E = 1
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); random.seed(cfg.seed)

    envs = [env]
    obs = np.stack([env.reset(seed=cfg.seed)[0]])
    model = ActorCritic(A, use_ask=True, K=K).to(device)
    disc = Discovery(A, H=H, zdim=cfg.zdim, K=K).to(device)

    opt = torch.optim.RMSprop(model.parameters(), lr=cfg.lr, alpha=0.99, eps=1e-5)
    opt_q = torch.optim.Adam(disc.q.parameters(), lr=1e-3)
    opt_v = torch.optim.Adam(list(disc.enc_q.parameters()) + list(disc.dec.parameters()), lr=1e-3)
    opt_p = torch.optim.Adam(disc.prop.parameters(), lr=1e-4)

    segs = SegBuf(); qbuf = collections.deque(maxlen=5000); absbuf = collections.deque(maxlen=5000)
    st = [dict(active=False) for _ in range(E)]
    cnt = dict(dec=0, ask=0, acc=0, skills=0, aborts=0, errs=[], seqs=[])
    ep_ret = np.zeros(E); done_returns = []; history = []; t0 = time.time()

    def finalize(i, v_next, aborted, done):
        s = st[i]
        y = s["partial"] + (cfg.gamma ** s["j"]) * v_next * (1.0 - float(done))
        acts = s["acts"] + [0] * (H - len(s["acts"]))
        segs.add(s["obs0"], acts, y, aborted or len(s["acts"]) < H)
        qbuf.append((s["obs0"], s["z"], y))
        if use_abs and (y - s["v0"]) > 0:                    # absorcion: ventaja positiva
            for o_, a_ in zip(s["traj"], s["acts"]):
                absbuf.append((o_, a_))
        cnt["skills"] += 1; cnt["aborts"] += int(aborted)
        cnt["errs"].append((abs(y - s["mu0"]), s["sig0"]))   # error real vs sigma_Q
        cnt["seqs"].append(tuple(s["acts"]))                 # para la diversidad
        st[i] = dict(active=False)

    for upd in range(1, n_updates + 1):
        buf = dict(obs=[], act=[], rew=[], done=[], mask=[])
        for _ in range(cfg.n_steps):
            o = torch.as_tensor(obs, device=device)
            with torch.no_grad():
                logits, V, feat = model(o)
                vbar = V.mean(-1)
            vbar_np = vbar.cpu().numpy()
            # terminar o abortar habilidades activas
            for i in range(E):
                s = st[i]
                if s["active"]:
                    if len(s["queue"]) == 0:
                        finalize(i, float(vbar_np[i]), False, False)
                    elif use_abort and should_abort(s["partial"], s["j"], float(vbar_np[i]),
                                                    s["mu0"], s["sig0"], cfg):
                        finalize(i, float(vbar_np[i]), True, False)
            # actor + verificacion pesimista
            with torch.no_grad():
                a_act = torch.distributions.Categorical(logits=logits).sample()
                z_star, accept, mu_s, sig_s = disc.verify(feat, vbar, cfg)
                dec_acts = disc.decode_actions(feat, z_star, greedy=False).squeeze(1)
            a_np = a_act.cpu().numpy(); acc_np = accept.cpu().numpy()
            mu_np = mu_s.cpu().numpy(); sig_np = sig_s.cpu().numpy()
            z_np = z_star.cpu().numpy().astype(np.float32); dec_np = dec_acts.cpu().numpy()

            nxt, rews, dones, acts_st, masks = [], [], [], [], []
            for i, e in enumerate(envs):
                s = st[i]; pen = 0.0; mask = 1.0
                if s["active"]:
                    prim = s["queue"].pop(0); mask = 0.0; a_store = prim
                else:
                    a = int(a_np[i]); a_store = a
                    cnt["dec"] += 1
                    if a == A:                                       # el actor pregunta
                        cnt["ask"] += 1
                        acepta = bool(acc_np[i])
                        if use_warmup and upd <= cfg.warmup_updates and random.random() < cfg.warmup_p:
                            acepta = True                            # warm-up: acepta al azar
                        if acepta:
                            cnt["acc"] += 1
                            q = dec_np[i].tolist()
                            st[i] = dict(active=True, queue=q[1:], acts=[], traj=[], partial=0.0, j=0,
                                         mu0=float(mu_np[i]), sig0=float(sig_np[i]), v0=float(vbar_np[i]),
                                         z=z_np[i].copy(), obs0=obs[i].copy())
                            s = st[i]; prim = q[0]
                        else:                                        # rechazada: paga y usa primitiva
                            pen = cfg.ask_cost
                            prim = torch.distributions.Categorical(logits=logits[i, :A]).sample().item()
                    else:
                        prim = a
                ob, r, term, trunc, _ = e.step(int(prim))
                ep_ret[i] += r; d = term or trunc; rc = float(r) * cfg.rew_scale
                if s["active"]:
                    s["partial"] += (cfg.gamma ** s["j"]) * rc; s["j"] += 1
                    s["acts"].append(int(prim)); s["traj"].append(obs[i].copy())
                if d:
                    if st[i]["active"]:
                        finalize(i, 0.0, True, True)
                    done_returns.append(ep_ret[i]); ep_ret[i] = 0
                    ob, _ = e.reset()
                nxt.append(ob); rews.append(rc - pen); dones.append(float(d))
                acts_st.append(int(a_store)); masks.append(mask)
            buf["obs"].append(o)
            buf["act"].append(torch.tensor(acts_st, device=device))
            buf["rew"].append(torch.tensor(rews, device=device, dtype=torch.float32))
            buf["done"].append(torch.tensor(dones, device=device))
            buf["mask"].append(torch.tensor(masks, device=device, dtype=torch.float32))
            obs = np.stack(nxt)

        # ---- retornos n-step
        with torch.no_grad():
            _, v_last, _ = model(torch.as_tensor(obs, device=device))
            R = v_last.mean(-1)
        rets = []
        for t in reversed(range(cfg.n_steps)):
            R = buf["rew"][t] + cfg.gamma * R * (1 - buf["done"][t]); rets.append(R)
        rets = torch.stack(rets[::-1]).flatten()
        o = torch.cat(buf["obs"]); a = torch.cat(buf["act"]); m = torch.cat(buf["mask"])
        # ---- actor-critic (el actor solo aprende de decisiones que tomo el mismo)
        logits, v, feat = model(o)
        vm = v.mean(-1); adv = (rets - vm).detach()
        dist = torch.distributions.Categorical(logits=logits)
        msum = m.sum().clamp(min=1)
        pg = -(dist.log_prob(a) * adv * m).sum() / msum
        ent = (dist.entropy() * m).sum() / msum
        vloss = ((v - rets[:, None]) ** 2).mean()
        loss = pg + cfg.vf_coef * vloss - cfg.ent_coef * ent
        if use_abs and len(absbuf) >= 64:                           # absorcion
            b = random.sample(absbuf, 64)
            ob = torch.as_tensor(np.stack([x[0] for x in b]), device=device)
            ab = torch.as_tensor([x[1] for x in b], device=device)
            lg, _, _ = model(ob)
            loss = loss + lam_abs * F.cross_entropy(lg[:, :A], ab)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm); opt.step()

        # ---- ensamble Q con mascaras Bernoulli
        if len(qbuf) >= 64:
            b = random.sample(qbuf, 64)
            ob = torch.as_tensor(np.stack([x[0] for x in b]), device=device)
            zb = torch.as_tensor(np.stack([x[1] for x in b]), device=device)
            yb = torch.as_tensor([x[2] for x in b], device=device, dtype=torch.float32)
            with torch.no_grad():
                fb = model.enc(ob)
            mk = (torch.rand(64, K, device=device) < 0.8).float()
            qv = disc.q(torch.cat([fb, zb], -1))
            lq = (mk * (qv - yb[:, None]) ** 2).sum() / mk.sum().clamp(min=1)
            opt_q.zero_grad(); lq.backward(); opt_q.step()
        # ---- VAE de habilidades (solo retorno alto)
        sb = segs.sample_top(64)
        if sb is not None:
            ob, ab = sb
            with torch.no_grad():
                fb = model.enc(ob)
            lv = disc.vae_loss(fb, ab, cfg.vae_beta)
            opt_v.zero_grad(); lv.backward(); opt_v.step()
        # ---- proponedor optimista (solo actualiza disc.prop)
        if len(qbuf) >= 64:
            J = disc.proposer_objective(feat.detach(), logits.detach(), cfg)
            opt_p.zero_grad(); (-J).backward(); opt_p.step()

        if upd % log_every == 0:
            mret = float(np.mean(done_returns[-20:])) if done_returns else float("nan")
            decisiones = upd * cfg.n_steps
            ask_r = cnt["ask"] / max(cnt["dec"], 1); acc_r = cnt["acc"] / max(cnt["ask"], 1)
            abo_r = cnt["aborts"] / max(cnt["skills"], 1)
            # correlacion entre sigma_Q y el error real del critic, y diversidad de secuencias
            corr, div = float("nan"), float("nan")
            if len(cnt["errs"]) > 5:
                er = np.array(cnt["errs"], dtype=float)
                if er[:, 0].std() > 0 and er[:, 1].std() > 0:
                    corr = float(np.corrcoef(er[:, 1], er[:, 0])[0, 1])
            if len(cnt["seqs"]) > 0:
                div = len(set(cnt["seqs"])) / len(cnt["seqs"])   # fraccion de secuencias distintas
            history.append(dict(decisiones=decisiones, ret=mret, ask=ask_r, accept=acc_r,
                                abort=abo_r, corr=corr, div=div))
            if verbose:
                print(f"upd {upd} | decisiones {decisiones} | ret {mret:.1f} | ask {ask_r:.2f} | "
                      f"accept {acc_r:.2f} | abort {abo_r:.2f} | {time.time() - t0:.0f}s", flush=True)
            cnt = dict(dec=0, ask=0, acc=0, skills=0, aborts=0, errs=[], seqs=[])
    return model, disc, history


# ---------------------------------------------------------------- evaluacion
def _exito(info):
    return float(info.get("motivo") == "meta")


@torch.no_grad()
def evaluate_policy(model, env, n_episodes=10, seed=1000, max_steps=1000):
    """A2C: politica estocastica del actor. Devuelve (media, desv, tasa de meta)."""
    rets, ok = [], []
    for ep in range(n_episodes):
        o, _ = env.reset(seed=seed + ep); total, done, steps, info = 0.0, False, 0, {}
        while not done and steps < max_steps:
            logits, _, _ = model(torch.as_tensor(o[None], device=device))
            a = torch.distributions.Categorical(logits=logits[:, :N_ACTIONS]).sample().item()
            o, r, term, trunc, info = env.step(a); total += r; done = term or trunc; steps += 1
        rets.append(total); ok.append(_exito(info))
    return float(np.mean(rets)), float(np.std(rets)), float(np.mean(ok))


@torch.no_grad()
def evaluate_pva(model, disc, env, n_episodes=10, seed=1000, max_steps=1000,
                 use_abort=True, c=None):
    """PVA-AC: politica estocastica + ask/verificacion/habilidades/aborto (sin entrenar).
    Devuelve (media, desv, tasa de meta)."""
    c = c or Cfg()
    A = N_ACTIONS; rets, ok = [], []
    for ep in range(n_episodes):
        o, _ = env.reset(seed=seed + ep)
        total, done, steps, sk, info = 0.0, False, 0, None, {}
        while not done and steps < max_steps:
            logits, V, feat = model(torch.as_tensor(o[None], device=device)); vbar = V.mean(-1)
            if sk is not None and (not sk["queue"] or (use_abort and should_abort(
                    sk["partial"], sk["j"], vbar.item(), sk["mu0"], sk["sig0"], c))):
                sk = None
            if sk is not None:
                a = sk["queue"].pop(0)
            else:
                a = torch.distributions.Categorical(logits=logits).sample().item()
                if a == A:
                    z, acc, mu, sig = disc.verify(feat, vbar, c)
                    if acc[0].item():
                        q = disc.decode_actions(feat, z, greedy=True)[0, 0].tolist()
                        a = q[0]
                        sk = dict(queue=q[1:], partial=0.0, j=0, mu0=mu.item(), sig0=sig.item())
                    else:
                        a = torch.distributions.Categorical(logits=logits[:, :A]).sample().item()
            o, r, term, trunc, info = env.step(int(a)); total += r; done = term or trunc; steps += 1
            if sk is not None:
                sk["partial"] += (c.gamma ** sk["j"]) * (r * c.rew_scale); sk["j"] += 1
        rets.append(total); ok.append(_exito(info))
    return float(np.mean(rets)), float(np.std(rets)), float(np.mean(ok))


def random_baseline(env, n_episodes=10, seed=1000, max_steps=1000):
    """Politica aleatoria. Devuelve (media, desv, tasa de meta)."""
    rets, ok = [], []
    for ep in range(n_episodes):
        env.reset(seed=seed + ep); total, done, steps, info = 0.0, False, 0, {}
        while not done and steps < max_steps:
            _, r, term, trunc, info = env.step(env.action_space.sample())
            total += r; done = term or trunc; steps += 1
        rets.append(total); ok.append(_exito(info))
    return float(np.mean(rets)), float(np.std(rets)), float(np.mean(ok))


def _politica_heuristica(obs, evadir=True):
    """Referencia sin aprendizaje: va hacia la meta y esquiva obstaculos cercanos.
    obs[1], obs[2] = sen/cos del angulo a la meta (>0: meta a la izquierda);
    obs[3:8] = rayos a -60, -30, 0, +30, +60 grados (izquierda = angulo positivo)."""
    r_der, r_izq = min(obs[3], obs[4]), min(obs[6], obs[7])
    if evadir and min(obs[4], obs[5], obs[6]) < 0.35:   # obstaculo a ~0.5 m
        return 3 if r_izq >= r_der else 4            # girar hacia el lado mas libre
    ang = math.atan2(obs[1], obs[2])
    if ang > 0.5:
        return 3
    if ang > 0.15:
        return 1
    if ang < -0.5:
        return 4
    if ang < -0.15:
        return 2
    return 0


def heuristic_baseline(env, n_episodes=10, seed=1000, max_steps=1000, evadir=True):
    """Politica heuristica (evadir=False: va directo a la meta ignorando obstaculos).
    Devuelve (media, desv, tasa de meta)."""
    rets, ok = [], []
    for ep in range(n_episodes):
        o, _ = env.reset(seed=seed + ep); total, done, steps, info = 0.0, False, 0, {}
        while not done and steps < max_steps:
            o, r, term, trunc, info = env.step(_politica_heuristica(o, evadir))
            total += r; done = term or trunc; steps += 1
        rets.append(total); ok.append(_exito(info))
    return float(np.mean(rets)), float(np.std(rets)), float(np.mean(ok))
