# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELDA 1 · MOTOR MATEMÁTICO (modelo MILP del artículo, generalizado) ║
# ╚══════════════════════════════════════════════════════════════════════╝
import time
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import milp, linprog, LinearConstraint, Bounds
from scipy.sparse import coo_matrix


@dataclass
class Datos:
    productos: list      # nombres de productos
    J: list              # J[i-1] = nº de fuentes de la etapa i   (M = len(J))
    T: int               # nº de períodos
    meses: float         # meses que dura cada período (para pérdidas por mantenimiento)
    efic: float          # factor de eficiencia (0.8 en el artículo)
    dias: list           # días hábiles por período (len T)
    fuente: dict         # (i,j) -> dict(N, perd, turnos, horas)
    tasa: dict           # (p,i,j) -> unidades/hora   (0 = esa fuente no procesa ese producto)
    setup: dict          # (p,i,j) -> costo de preparación
    costo: dict          # (p,i,j,t) -> costo unitario de producción
    mantener: dict       # (p,i) -> costo de mantener inventario por unidad y período
    coef: dict           # (p,i) -> unidades de la etapa i necesarias por unidad de la etapa i+1
    demanda: dict        # (p,t) -> demanda
    inv_ini: list        # inventario inicial de producto terminado, por producto
    inv_fin: list        # inventario final de producto terminado, por producto
    c_falt: list         # costo de escasez (unidad pendiente) por producto
    faltantes: bool = True    # ¿se permite demanda pendiente (escasez)?
    enteras: bool = True      # ¿X enteras?
    t_limite: float = 60.0    # segundos máx. del solver
    gap: float = 0.0          # gap relativo tolerado (%)


def horas(d, i, j, t):
    """Horas-máquina disponibles de la fuente (i,j) en el período t.
       Cd = N · tasa · efic · ( días·turnos·h_turno − pérdida_mant·meses )   (Tabla III del artículo)"""
    f = d.fuente[(i, j)]
    h = f['N'] * d.efic * (d.dias[t - 1] * f['turnos'] * f['horas'] - f['perd'] * d.meses)
    return max(h, 0.0)


class _Modelo:
    def __init__(s):
        s.nom, s.lb, s.ub, s.ent, s.c = [], [], [], [], []
        s.r, s.k, s.v, s.lo, s.hi = [], [], [], [], []

    def var(s, nom, lb, ub, ent, c):
        s.nom.append(nom); s.lb.append(lb); s.ub.append(ub)
        s.ent.append(1 if ent else 0); s.c.append(c)
        return len(s.nom) - 1

    def con(s, terms, lo, hi):
        r = len(s.lo)
        for k, v in terms:
            s.r.append(r); s.k.append(k); s.v.append(v)
        s.lo.append(lo); s.hi.append(hi)

    def A(s):
        return coo_matrix((s.v, (s.r, s.k)), shape=(len(s.lo), len(s.nom))).tocsr()


ESTADOS = {0: "Óptima", 1: "Límite de tiempo (mejor solución encontrada)",
           2: "INFACTIBLE", 3: "No acotado", 4: "Error numérico"}


def resolver(d: Datos):
    P, M, T = len(d.productos), len(d.J), d.T
    sep = "" if max(M, max(d.J), T) < 10 else "_"
    m = _Modelo()
    X, Y, I, Ip, Im, CAP = {}, {}, {}, {}, {}, {}

    def nm(letra, p, *ix):
        s = letra + sep.join(str(a) for a in ix)
        return s + (f" [{d.productos[p]}]" if P > 1 else "")

    # ---------------- variables (en el mismo orden que la tabla solución del artículo)
    for p in range(P):
        for i in range(1, M + 1):
            for t in range(1, T + 1):
                for j in range(1, d.J[i - 1] + 1):
                    cap = horas(d, i, j, t) * d.tasa[(p, i, j)]
                    if d.enteras: cap = float(np.floor(cap + 1e-9))   # X entera -> capacidad entera (evita problemas numéricos)
                    CAP[p, i, j, t] = cap
                    X[p, i, j, t] = m.var(nm("X", p, i, j, t), 0, cap, d.enteras, d.costo[(p, i, j, t)])
    for p in range(P):
        for i in range(1, M):
            for t in range(1, T + 1):
                I[p, i, t] = m.var(nm("I", p, i, t), 0, np.inf if t < T else 0, False, d.mantener[(p, i)])
        for t in range(1, T + 1):
            if t < T: Ip[p, t] = m.var(nm("I+", p, M, t), 0, np.inf, False, d.mantener[(p, M)])
            else:     Ip[p, t] = m.var(nm("I+", p, M, t), d.inv_fin[p], d.inv_fin[p], False, d.mantener[(p, M)])
        for t in range(1, T + 1):
            ub = np.inf if (d.faltantes and t < T) else 0
            Im[p, t] = m.var(nm("I-", p, M, t), 0, ub, False, d.c_falt[p])
    for p in range(P):
        for i in range(1, M + 1):
            for t in range(1, T + 1):
                for j in range(1, d.J[i - 1] + 1):
                    Y[p, i, j, t] = m.var(nm("Y", p, i, j, t), 0, 1 if CAP[p, i, j, t] > 0 else 0,
                                          True, d.setup[(p, i, j)])

    # ---------------- restricciones
    for p in range(P):
        for t in range(1, T + 1):
            for i in range(1, M):                      # balance de las etapas 1..M-1
                terms = [(X[p, i, j, t], 1) for j in range(1, d.J[i - 1] + 1)]
                if t > 1: terms.append((I[p, i, t - 1], 1))
                terms.append((I[p, i, t], -1))
                terms += [(X[p, i + 1, j, t], -d.coef[(p, i)]) for j in range(1, d.J[i] + 1)]
                m.con(terms, 0, 0)
            terms = [(X[p, M, j, t], 1) for j in range(1, d.J[M - 1] + 1)]   # balance etapa final
            rhs = d.demanda[(p, t)]
            if t > 1: terms += [(Ip[p, t - 1], 1), (Im[p, t - 1], -1)]
            else:     rhs -= d.inv_ini[p]
            terms += [(Ip[p, t], -1), (Im[p, t], 1)]
            m.con(terms, rhs, rhs)
    for (p, i, j, t), cap in CAP.items():              # X ≤ Cd · Y
        if cap > 0:
            m.con([(X[p, i, j, t], 1), (Y[p, i, j, t], -cap)], -np.inf, 0)
    if P > 1:                                           # horas-máquina compartidas entre productos
        for i in range(1, M + 1):
            for j in range(1, d.J[i - 1] + 1):
                for t in range(1, T + 1):
                    terms = [(X[p, i, j, t], 1.0 / d.tasa[(p, i, j)]) for p in range(P) if d.tasa[(p, i, j)] > 0]
                    if terms: m.con(terms, -np.inf, horas(d, i, j, t))

    # ---------------- resolver MILP
    c, A = np.array(m.c, float), m.A()
    lo, hi = np.array(m.lo, float), np.array(m.hi, float)
    lb, ub = np.array(m.lb, float), np.array(m.ub, float)
    t0 = time.time()
    res = milp(c, constraints=LinearConstraint(A, lo, hi), integrality=np.array(m.ent),
               bounds=Bounds(lb, ub),
               options={"time_limit": float(d.t_limite), "mip_rel_gap": d.gap / 100.0, "disp": False})
    seg = time.time() - t0
    if res.x is None:
        msg = ESTADOS.get(res.status, res.message)
        if res.status == 2:
            msg += (" → no hay forma de cumplir la demanda con las capacidades dadas. "
                    "Subir máquinas, turnos o días, bajar la demanda, o activar 'permitir escasez'.")
        return {"ok": False, "estado": msg, "seg": seg}

    x = res.x.copy()
    for k in range(len(x)):
        if m.ent[k]: x[k] = round(x[k])

    # ---------------- costos reducidos: LP con los Y fijos en su valor óptimo
    rc, xlp = np.full(len(x), np.nan), x
    try:
        lb2, ub2 = lb.copy(), ub.copy()
        for k in Y.values(): lb2[k] = ub2[k] = x[k]
        eq = lo == hi
        r2 = linprog(c, A_ub=A[~eq] if (~eq).any() else None, b_ub=hi[~eq] if (~eq).any() else None,
                     A_eq=A[eq] if eq.any() else None, b_eq=lo[eq] if eq.any() else None,
                     bounds=np.column_stack([lb2, ub2]), method="highs")
        if r2.status == 0:
            rc = r2.lower.marginals + r2.upper.marginals
            xlp = r2.x
    except Exception:
        pass

    # ---------------- tabla solución (igual a la del artículo)
    yset, filas = set(Y.values()), []
    for k in range(len(x)):
        if k in yset:
            est, rcv = "Fija (binaria)", np.nan
        else:
            tol = 1e-6 * (1 + abs(xlp[k]))
            enc = abs(xlp[k] - lb[k]) < tol or abs(xlp[k] - ub[k]) < tol
            est, rcv = ("En cota" if enc else "Básica"), rc[k]
        filas.append([m.nom[k], x[k], m.c[k], x[k] * m.c[k], rcv, est])
    tabla = pd.DataFrame(filas, columns=["Variable", "Valor solución", "Costo unitario",
                                         "Contribución total", "Costo reducido", "Estado base"])

    # ---------------- resúmenes
    cost = dict(
        prod=sum(m.c[k] * x[k] for k in X.values()),
        setup=sum(m.c[k] * x[k] for k in Y.values()),
        mant=sum(m.c[k] * x[k] for k in list(I.values()) + list(Ip.values())),
        falt=sum(m.c[k] * x[k] for k in Im.values()))
    cols = [f"T{t}" for t in range(1, T + 1)]
    plan, inv = {}, {}
    for p, nombre in enumerate(d.productos):
        idx, vals = [], []
        for i in range(1, M + 1):
            for j in range(1, d.J[i - 1] + 1):
                idx.append(f"Etapa {i} · Fuente {j}")
                vals.append([x[X[p, i, j, t]] for t in range(1, T + 1)])
        plan[nombre] = pd.DataFrame(vals, index=idx, columns=cols)
        idx, vals = [], []
        idx.append("Demanda"); vals.append([d.demanda[(p, t)] for t in range(1, T + 1)])
        idx.append(f"Producción etapa {M}")
        vals.append([sum(x[X[p, M, j, t]] for j in range(1, d.J[M - 1] + 1)) for t in range(1, T + 1)])
        for i in range(1, M):
            idx.append(f"Inventario etapa {i}"); vals.append([x[I[p, i, t]] for t in range(1, T + 1)])
        idx.append(f"Inv. prod. terminado (I+)"); vals.append([x[Ip[p, t]] for t in range(1, T + 1)])
        idx.append(f"Demanda pendiente (I-)");    vals.append([x[Im[p, t]] for t in range(1, T + 1)])
        inv[nombre] = pd.DataFrame(vals, index=idx, columns=cols)
    util = {}
    for i in range(1, M + 1):
        for j in range(1, d.J[i - 1] + 1):
            fila = []
            for t in range(1, T + 1):
                H = horas(d, i, j, t)
                usado = sum(x[X[p, i, j, t]] / d.tasa[(p, i, j)] for p in range(P) if d.tasa[(p, i, j)] > 0)
                fila.append(usado / H if H > 0 else 0.0)
            util[f"E{i}-F{j}"] = fila
    util = pd.DataFrame(util, index=cols).T

    return {"ok": True, "estado": ESTADOS.get(res.status, str(res.status)), "obj": float(res.fun),
            "seg": seg, "tabla": tabla, "costos": cost, "plan": plan, "inv": inv, "util": util,
            "gap": getattr(res, "mip_gap", None)}


# ───────────── caso del artículo (Tablas I–III) ─────────────
def caso_articulo():
    J = [3, 2, 3, 4]
    fte = {  # (i,j): (N máquinas, pérdida h/mes/máq)
        (1, 1): (5, 12.2), (1, 2): (5, 10.3), (1, 3): (10, 11.2),
        (2, 1): (12, 14.2), (2, 2): (11, 16.1),
        (3, 1): (4, 12.2), (3, 2): (5, 12.5), (3, 3): (11, 13.5),
        (4, 1): (3, 12.5), (4, 2): (4, 11.2), (4, 3): (5, 10.3), (4, 4): (6, 11.4)}
    rate = {(1, 1): 20, (1, 2): 25, (1, 3): 28, (2, 1): 35, (2, 2): 40, (3, 1): 25, (3, 2): 28, (3, 3): 30,
            (4, 1): 24, (4, 2): 25, (4, 3): 28, (4, 4): 30}
    setup = {(1, 1): 25000, (1, 2): 24000, (1, 3): 20000, (2, 1): 24000, (2, 2): 22000,
             (3, 1): 12000, (3, 2): 10000, (3, 3): 12000,
             (4, 1): 23000, (4, 2): 25000, (4, 3): 23000, (4, 4): 22000}
    cu = {(1, 1): [25, 28, 30, 28], (1, 2): [28, 30, 30, 32], (1, 3): [30, 32, 34, 32],
          (2, 1): [45, 46, 44, 48], (2, 2): [48, 50, 49, 50],
          (3, 1): [12, 15, 18, 19], (3, 2): [14, 16, 18, 20], (3, 3): [13, 16, 20, 22],
          (4, 1): [24, 29, 24, 28], (4, 2): [24, 29, 24, 27], (4, 3): [26, 30, 27, 25], (4, 4): [28, 33, 27, 25]}
    return dict(J=J, T=4, meses=3, efic=0.8, dias=[67, 68, 76, 67], fte=fte, rate=rate, setup=setup, cu=cu,
                mantener=[2, 3, 4, 5], demanda=[50500, 65500, 82000, 92000],
                inv_ini=1000, inv_fin=1000, c_falt=4)


def datos_articulo():
    a = caso_articulo(); T = a['T']
    return Datos(
        productos=["Producto"], J=a['J'], T=T, meses=a['meses'], efic=a['efic'], dias=a['dias'],
        fuente={k: dict(N=v[0], perd=v[1], turnos=1, horas=8) for k, v in a['fte'].items()},
        tasa={(0,) + k: v for k, v in a['rate'].items()},
        setup={(0,) + k: v for k, v in a['setup'].items()},
        costo={(0,) + k + (t + 1,): v[t] for k, v in a['cu'].items() for t in range(T)},
        mantener={(0, i + 1): a['mantener'][i] for i in range(4)},
        coef={(0, i): 1.0 for i in range(1, 5)},
        demanda={(0, t + 1): a['demanda'][t] for t in range(T)},
        inv_ini=[a['inv_ini']], inv_fin=[a['inv_fin']], c_falt=[a['c_falt']])

# %%
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELDA 2 · INTERFAZ GRÁFICA (ipywidgets)                             ║
# ╚══════════════════════════════════════════════════════════════════════╝
import ipywidgets as W
from dataclasses import replace
import matplotlib.pyplot as plt
from IPython.display import display, HTML

S = {"est": None, "g": {}, "R": None, "d": None}


def ES(x, dec=2):
    if x is None or (isinstance(x, float) and np.isnan(x)): return "—"
    return f"{x:,.{dec}f}".replace(",", "§").replace(".", ",").replace("§", ".")


PAL = dict(total="#27ae60", prod="#2980b9", setup="#e67e22", mant="#8e44ad", falt="#e74c3c",
           info="#16a085", gris="#7f8c8d")
TIPO_COLOR = {"X": PAL["prod"], "I": PAL["mant"], "Y": PAL["setup"]}


def rgba(hexc, a):
    h = hexc.lstrip("#")
    return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})"


def seccion(titulo, color, cuerpo, icono=""):
    """Tarjeta con encabezado de color (funciona en modo claro y oscuro)."""
    return (f"<div style='border:1px solid {rgba(color,.55)};border-radius:12px;margin:12px 0;overflow:hidden'>"
            f"<div style='background:{rgba(color,.9)};color:white;padding:8px 14px;font-weight:700;font-size:15px'>"
            f"{(icono + " " + titulo).strip()}</div><div style='padding:10px 14px'>{cuerpo}</div></div>")


CSS_TABLA = ("<style>.pa{border-collapse:collapse;width:100%}.pa td,.pa th{padding:4px 12px;"
             "border-bottom:1px solid #8883;text-align:right;white-space:nowrap}"
             ".pa th{position:sticky;top:0;color:white;z-index:1}"
             ".pa td:first-child,.pa th:first-child{text-align:left}"
             ".pa tr:hover td{background:#8882}</style>")


def tabla_html(df, alto=450, dec=2, color=None, calor=False, fila_color=None):
    """Tabla HTML con encabezado de color. calor=True sombrea las celdas numéricas según su magnitud.
       fila_color: función(fila)->color hex para pintar una barra a la izquierda de cada fila."""
    color = color or PAL["prod"]
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    vmax = max([float(np.nanmax(np.abs(df[num].values))) if len(num) else 1.0, 1e-9])
    h = "<table class='pa'><thead><tr>" + "".join(
        f"<th style='background:{color}'>{c}</th>" for c in df.columns) + "</tr></thead><tbody>"
    for _, fila in df.iterrows():
        borde = f"border-left:5px solid {fila_color(fila)};" if fila_color else ""
        h += "<tr>"
        for n, c in enumerate(df.columns):
            v = fila[c]; est = borde if n == 0 else ""
            if c in num:
                txt = ES(v, dec)
                if calor and v == v and abs(v) > 1e-9: est += f"background:{rgba(color, .12 + .55*min(abs(v)/vmax,1))};"
                elif abs(v) < 1e-9 if v == v else False: est += "opacity:.4;"
            else: txt = str(v)
            h += f"<td style='{est}'>{txt}</td>"
        h += "</tr>"
    return f"{CSS_TABLA}<div style='max-height:{alto}px;overflow:auto'>{h}</tbody></table></div>"


# ───────── colores y estilos de las tablas de entrada
COL_T = {"dem": "#2980b9", "cap": "#16a085", "tasa": "#e67e22", "costo": "#8e44ad", "etp": "#d4ac0d"}
CSS_UI = "<style>" + "".join(
    f".pa-{k} input{{border:2px solid {c}!important;background:{rgba(c, .22)}!important;"
    f"font-weight:700!important;text-align:right!important}}" for k, c in COL_T.items()) + (
    ".pa-config input,.pa-config select{border:2px solid #7f8c8d!important;font-weight:600!important}"
    "</style>")


def ayuda(titulo, color, que_es, campos=(), consejo=""):
    """Caja explicativa: qué es la tabla, qué significa cada columna y un consejo."""
    li = "".join(f"<li style='margin:3px 0'><b>{n}:</b> {d}</li>" for n, d in campos)
    lista = f"<ul style='margin:6px 0 6px 18px;padding:0'>{li}</ul>" if li else ""
    tip = (f"<div style='margin-top:6px'><i>{consejo}</i></div>" if consejo else "")
    return W.HTML(f"<div style='border-left:6px solid {color};background:{rgba(color, .13)};padding:10px 16px;"
                  f"border-radius:8px;margin:6px 0 12px 0;max-width:1000px'>"
                  f"<div style='font-weight:800;font-size:15px;color:{color}'>{titulo}</div>"
                  f"<div style='margin-top:4px'>{que_es}</div>{lista}{tip}</div>")


class Grid:
    def __init__(self, filas, cols, defaults=None, clave="dem", unidades=None, ancho=100, ancho_et=290):
        color = COL_T[clave]
        defaults = defaults if defaults is not None else [0.0] * len(cols)
        unidades = unidades or [""] * len(cols)
        items = [W.HTML("")]
        for c, u in zip(cols, unidades):
            u_html = f"<div style='font-weight:400;font-size:11px'>{u}</div>" if u else ""
            items.append(W.HTML(f"<div style='background:{color};color:white;padding:5px 6px;border-radius:8px;"
                                f"text-align:center;font-weight:700;font-size:12px;line-height:1.25'>{c}{u_html}</div>"))
        self.celdas = []
        for r, f in enumerate(filas):
            items.append(W.HTML(f"<div style='border-left:6px solid {color};background:{rgba(color, .10 if r % 2 == 0 else .20)};"
                                f"padding:7px 10px;border-radius:6px;font-weight:600'>{f}</div>"))
            fila = []
            for c in range(len(cols)):
                w = W.FloatText(value=float(defaults[c]), layout=W.Layout(width=f"{ancho}px"))
                w.add_class(f"pa-{clave}"); fila.append(w)
            items += fila; self.celdas.append(fila)
        self.box = W.GridBox(items, layout=W.Layout(
            grid_template_columns=f"{ancho_et}px repeat({len(cols)}, {ancho + 12}px)", grid_gap="6px 8px"))

    def get(self):
        return np.array([[c.value for c in fila] for fila in self.celdas], float)

    def set(self, M):
        for r, fila in enumerate(M):
            for c, v in enumerate(fila): self.celdas[r][c].value = float(v)


# ───────── widgets de estructura (cada uno con su explicación al lado)
def _ancho(w, n=270):
    w.layout = W.Layout(width=f"{n}px"); w.add_class("pa-config"); return w


w_prod = _ancho(W.Text(value="Producto A, Producto B"))
w_J = _ancho(W.Text(value="3,2,3,4"))
w_T = _ancho(W.IntText(value=4))
w_meses = _ancho(W.FloatText(value=3))
w_efic = _ancho(W.FloatText(value=0.8))
w_falt = W.Checkbox(value=True, description="Sí, permitir escasez", indent=False)
w_modo = _ancho(W.Dropdown(options=["Ambas (comparar)", "Entera (con aproximación)", "Continua (sin aproximación)"],
                           value="Ambas (comparar)"))
w_tl = _ancho(W.IntText(value=60))
w_gap = _ancho(W.FloatText(value=0.0))
w_ver = W.Dropdown(options=[("Con aproximación (X entera)", "ent"), ("Sin aproximación (X continua)", "cont")],
                   value="ent", description="Ver solución:", style={"description_width": "110px"},
                   layout=W.Layout(width="420px"))
b_gen = W.Button(description="Paso 2 · Generar tablas", button_style="info", icon="table",
                 tooltip="Crear las tablas de datos según la estructura de arriba", layout=W.Layout(width="230px", height="40px"))
b_ej = W.Button(description="Cargar caso del artículo", icon="book", button_style="warning",
                tooltip="Rellenar todo con el ejemplo del paper (4 etapas, 12 fuentes, 4 trimestres)",
                layout=W.Layout(width="230px", height="40px"))
b_res = W.Button(description="Paso 4 · RESOLVER", button_style="success", icon="play",
                 tooltip="Calcular el plan óptimo", layout=W.Layout(width="230px", height="44px"))
b_xls = W.Button(description="Descargar Excel", icon="download", layout=W.Layout(width="190px", height="44px"))
msg = W.HTML("")
w_vista = W.HTML("")
caja_tablas = W.VBox([])
o_res, o_tab, o_plan, o_grafo, o_graf, o_util = (W.Output() for _ in range(6))
w_filtro = W.Dropdown(options=["Todas", "Solo valor ≠ 0", "Producción (X)", "Inventarios (I)", "Preparaciones (Y)"],
                      value="Solo valor ≠ 0", description="Mostrar:")
res_tabs = W.Tab(children=[o_res, W.VBox([w_filtro, o_tab]), o_plan, o_grafo, o_graf, o_util])
for k, t in enumerate(["Resumen", "Tabla solución", "Plan por etapa", "Grafo", "Gráficas", "Utilización"]):
    res_tabs.set_title(k, t)


def aviso(txt, color="#c0392b"):
    msg.value = (f"<div style='border-left:6px solid {color};background:{rgba(color, .13)};padding:8px 14px;"
                 f"border-radius:8px;margin:6px 0;max-width:1000px'><b style='color:{color}'>{txt}</b></div>")


def _etq(t, c):
    return (f"<span style='background:{rgba(c, .25)};border:1px solid {c};border-radius:14px;padding:2px 10px;"
            f"margin:2px;display:inline-block;font-weight:700'>{t}</span>")


# ───────── estructura
def leer_estructura():
    prods = [s.strip() for s in w_prod.value.split(",") if s.strip()]
    try: J = [int(s) for s in w_J.value.replace(";", ",").split(",") if s.strip()]
    except ValueError: raise ValueError("'Fuentes por etapa' debe tener solo números separados por coma. Ejemplo: 3,2,3,4")
    T = int(w_T.value)
    if not prods: raise ValueError("Debe haber al menos un producto.")
    if not J or min(J) < 1: raise ValueError("Cada etapa necesita al menos 1 fuente. Ejemplo: 3,2,3,4")
    if T < 1: raise ValueError("Debe haber al menos 1 período.")
    return prods, J, T


def actualizar_vista(_=None):
    try: prods, J, T = leer_estructura()
    except Exception as e:
        w_vista.value = f"<div style='color:#c0392b'><b>{e}</b></div>"; return
    M = len(J)
    flujo = " <b>→</b> ".join(_etq(f"Etapa {i} · {n} fuente{'s' if n > 1 else ''}", COL_T["cap"]) for i, n in enumerate(J, 1))
    w_vista.value = (
        f"<div style='border:2px dashed #8886;border-radius:10px;padding:10px 14px;max-width:1000px'>"
        f"<b>Estructura resultante:</b><div style='margin-top:6px'>Productos: "
        + "".join(_etq(p, COL_T["dem"]) for p in prods) + "</div>"
        f"<div style='margin-top:6px'>Proceso: {flujo} <b>→</b> {_etq('Producto terminado', '#27ae60')}</div>"
        f"<div style='margin-top:6px'>Períodos: " + "".join(_etq(f"T{t}", COL_T["etp"]) for t in range(1, T + 1)) + "</div>"
        f"<div style='margin-top:6px;opacity:.85'>= {len(prods)} producto(s) · {M} etapa(s) · {sum(J)} fuente(s) en total · {T} período(s)</div></div>")


for _w in (w_prod, w_J, w_T): _w.observe(actualizar_vista, names="value")


def generar_tablas(_=None):
    try: prods, J, T = leer_estructura()
    except Exception as e: aviso(str(e)); return
    M = len(J); P = len(prods)
    src = [(i, j) for i in range(1, M + 1) for j in range(1, J[i - 1] + 1)]
    cT = [f"T{t}" for t in range(1, T + 1)]
    rows_pf = [f"{p} · Etapa {i} · Fuente {j}" for p in prods for (i, j) in src]
    g = {}
    g["dem"] = Grid(prods, ["Inv. inicial", "Inv. final", "Costo escasez"] + [f"Demanda T{t}" for t in range(1, T + 1)],
                    [0, 0, 10] + [10000] * T, "dem", ["unidades", "unidades", "$ por unidad"] + ["unidades"] * T)
    g["dias"] = Grid(["Días hábiles del período"], cT, [65] * T, "cap", ["días"] * T)
    g["fte"] = Grid([f"Etapa {i} · Fuente {j}" for i, j in src],
                    ["Nº máquinas", "Pérdida mantenim.", "Turnos", "Horas por turno"], [5, 12, 1, 8], "cap",
                    ["máquinas", "horas/mes por máquina", "turnos al día", "horas"], ancho=130)
    g["tasa"] = Grid(rows_pf, ["Tasa de producción", "Costo de preparación"], [25, 20000], "tasa",
                     ["unidades/hora por máquina", "$ por vez"], ancho=150)
    g["costo"] = Grid(rows_pf, cT, [20] * T, "costo", ["$ por unidad"] * T)
    g["etp"] = Grid([f"{p} · Etapa {i}" for p in prods for i in range(1, M + 1)],
                    ["Costo de mantener", "Consumo por unidad"], [1, 1], "etp",
                    ["$ por unidad y período", "unid. de esta etapa por 1 de la siguiente"], ancho=170)
    S["g"], S["est"] = g, (prods, J, T)
    H = lambda *a, **k: ayuda(*a, **k)
    tabs = W.Tab(children=[
        W.VBox([H("Tabla 1 · Demanda e inventarios", COL_T["dem"],
                  "Datos de lo que hay que <b>entregar</b> de cada producto y del estado de la bodega al empezar y al terminar el plan. "
                  "Una fila por producto.",
                  [("Inv. inicial", "unidades de producto terminado que ya hay en bodega al empezar el período 1."),
                   ("Inv. final", "unidades que deben quedar en bodega al terminar el último período."),
                   ("Costo escasez", "lo que cuesta cada unidad que se queda sin entregar a tiempo (demanda pendiente). "
                                     "Si es muy bajo, al modelo le puede salir más barato dejar de entregar que producir."),
                   ("Demanda Tn", "unidades que piden los clientes en el período n.")],
                  "Si se desmarca 'permitir escasez' en la estructura, el modelo está obligado a cumplir toda la demanda."), g["dem"].box]),
        W.VBox([H("Tabla 2 · Capacidad de las máquinas", COL_T["cap"],
                  "Tiempo que <b>puede trabajar</b> cada fuente. Una <b>etapa</b> es un paso del proceso (la 1 es la "
                  "primera, la última entrega el producto terminado). Una <b>fuente</b> es una máquina o grupo de máquinas que puede "
                  "hacer ese paso: el modelo decide cuál usar.",
                  [("Días hábiles", "días que se trabaja en cada período (arriba, una sola fila para todo el sistema)."),
                   ("Nº máquinas", "máquinas iguales que tiene esa fuente."),
                   ("Pérdida mantenim.", "horas por mes que cada máquina está parada por mantenimiento."),
                   ("Turnos / Horas por turno", "turnos que se trabajan cada día y cuántas horas dura cada uno.")],
                  "Horas disponibles = máquinas × eficiencia × (días × turnos × horas por turno − pérdida × meses del período). "
                  "Si hay varios productos, comparten estas horas."),
            g["dias"].box, W.HTML("<div style='height:14px'></div>"), g["fte"].box]),
        W.VBox([H("Tabla 3 · Velocidad y preparación", COL_T["tasa"],
                  "Una fila por <b>producto y fuente</b>: qué tan rápido produce y cuánto cuesta arrancar.",
                  [("Tasa de producción", "unidades por hora que saca UNA máquina de esa fuente. Poner <b>0</b> si esa fuente no puede "
                                          "hacer ese producto."),
                   ("Costo de preparación (setup)", "costo fijo de poner a trabajar esa fuente en un período. Se paga una vez si "
                                                    "produce algo, sin importar cuánto.")],
                  "Por el costo de preparación, al modelo le conviene concentrar la producción en pocas fuentes y períodos."),
            g["tasa"].box]),
        W.VBox([H("Tabla 4 · Costo de producir", COL_T["costo"],
                  "Cuánto cuesta producir <b>una unidad</b> en cada fuente, período por período (materiales, mano de obra, energía...).",
                  [("T1, T2, ...", "costo unitario en ese período. Puede cambiar entre períodos.")],
                  "Una fuente más barata no siempre gana: depende también de su capacidad y de su costo de preparación."),
            g["costo"].box]),
        W.VBox([H("Tabla 5 · Inventarios entre etapas", COL_T["etp"],
                  "Qué cuesta guardar piezas en la bodega de cada etapa y cuánto material de una etapa se gasta en la siguiente.",
                  [("Costo de mantener", "costo de guardar 1 unidad de esa etapa durante un período."),
                   ("Consumo por unidad", "unidades de esta etapa que se necesitan para hacer 1 unidad de la etapa siguiente "
                                          "(normalmente 1; en la última etapa no se usa).")],
                  "Mantener inventario cuesta más cuanto más avanzada está la etapa (el producto ya tiene más valor)."),
            g["etp"].box])])
    for k, t in enumerate(["1 · Demanda", "2 · Capacidad", "3 · Velocidad y setup", "4 · Costos", "5 · Inventarios"]):
        tabs.set_title(k, t)
    caja_tablas.children = [W.HTML(
        "<div style='font-size:18px;font-weight:800;margin-top:10px'>Paso 3 · Editar los datos</div>"
        "<div style='opacity:.85;max-width:1000px'>Recorrer las 5 pestañas de izquierda a derecha. Los números que aparecen son "
        "<b>valores de ejemplo</b> y hay que reemplazarlos por los del problema. Cada pestaña explica qué es cada dato.</div>"), tabs]
    aviso(f"Tablas generadas: {P} producto(s), {M} etapas, {sum(J)} fuentes, {T} períodos. "
          f"Ahora se pueden editar los datos y pulsar RESOLVER.", "#27ae60")


def cargar_ejemplo(_=None):
    a = caso_articulo()
    w_prod.value, w_J.value, w_T.value = "Producto", ",".join(map(str, a["J"])), a["T"]
    w_meses.value, w_efic.value, w_falt.value = a["meses"], a["efic"], True
    generar_tablas()
    g = S["g"]; src = sorted(a["fte"])
    g["dem"].set([[a["inv_ini"], a["inv_fin"], a["c_falt"]] + a["demanda"]])
    g["dias"].set([a["dias"]])
    g["fte"].set([[a["fte"][k][0], a["fte"][k][1], 1, 8] for k in src])
    g["tasa"].set([[a["rate"][k], a["setup"][k]] for k in src])
    g["costo"].set([a["cu"][k] for k in src])
    g["etp"].set([[h, 1] for h in a["mantener"]])
    aviso("Caso del artículo cargado (4 etapas, 12 fuentes, 4 trimestres). Se puede pulsar RESOLVER: el costo esperado es ≈ 34.208.760.",
          "#27ae60")


def leer_datos():
    if S["est"] is None: raise ValueError("Primero hay que pulsar 'Generar tablas'.")
    prods, J, T = S["est"]; g = S["g"]; M = len(J); P = len(prods)
    src = [(i, j) for i in range(1, M + 1) for j in range(1, J[i - 1] + 1)]
    dem, dias, fte = g["dem"].get(), g["dias"].get()[0], g["fte"].get()
    tasa, costo, etp = g["tasa"].get(), g["costo"].get(), g["etp"].get()
    if w_efic.value <= 0 or w_efic.value > 1.5: raise ValueError("El factor de eficiencia debe estar entre 0 y 1 (ej. 0,8).")
    if (dias <= 0).any(): raise ValueError("Pestaña 2 · Capacidad: los días hábiles deben ser mayores que 0.")
    if (dem[:, 3:] < 0).any(): raise ValueError("Pestaña 1 · Demanda: la demanda no puede ser negativa.")
    fuente = {k: dict(N=fte[n][0], perd=fte[n][1], turnos=fte[n][2], horas=fte[n][3]) for n, k in enumerate(src)}
    d = Datos(productos=prods, J=J, T=T, meses=w_meses.value, efic=w_efic.value, dias=list(dias), fuente=fuente,
              tasa={}, setup={}, costo={}, mantener={}, coef={},
              demanda={(p, t + 1): dem[p][3 + t] for p in range(P) for t in range(T)},
              inv_ini=list(dem[:, 0]), inv_fin=list(dem[:, 1]), c_falt=list(dem[:, 2]),
              faltantes=w_falt.value, enteras=True, t_limite=w_tl.value, gap=w_gap.value)
    for p in range(P):
        for n, (i, j) in enumerate(src):
            r = p * len(src) + n
            d.tasa[(p, i, j)], d.setup[(p, i, j)] = tasa[r][0], tasa[r][1]
            for t in range(T): d.costo[(p, i, j, t + 1)] = costo[r][t]
        for i in range(1, M + 1):
            d.mantener[(p, i)], d.coef[(p, i)] = etp[p * M + i - 1][0], etp[p * M + i - 1][1]
        for i in range(1, M + 1):
            if all(d.tasa[(p, i, j)] <= 0 or fuente[(i, j)]["N"] <= 0 for j in range(1, J[i - 1] + 1)):
                raise ValueError(f"Pestaña 3 · Velocidad: '{prods[p]}' no tiene ninguna fuente con tasa > 0 "
                                 f"(y máquinas > 0) en la Etapa {i}, así que no se puede fabricar.")
    return d



# ───────── resultados
def pintar_tabla(_=None):
    R = S["R"]
    if not R: return
    t = R["tabla"]; f = w_filtro.value
    if f == "Solo valor ≠ 0": t = t[t["Valor solución"].abs() > 1e-9]
    elif f == "Producción (X)": t = t[t["Variable"].str.startswith("X")]
    elif f == "Inventarios (I)": t = t[t["Variable"].str.startswith("I")]
    elif f == "Preparaciones (Y)": t = t[t["Variable"].str.startswith("Y")]
    o_tab.clear_output()
    leyenda = ("<div style='margin-bottom:8px'>"
               + "".join(f"<span style='border-left:5px solid {c};padding:2px 10px;margin-right:14px'>{n}</span>"
                         for n, c in [("X · producción", PAL["prod"]), ("I · inventarios", PAL["mant"]),
                                      ("Y · preparación", PAL["setup"])]) + "</div>")
    with o_tab:
        M_ = len(S["d"].J) if S.get("d") else 1
        guia = caja("Cómo leer esta tabla", PAL["prod"],
                    "<b>Nombres:</b> <b>X123</b> = unidades producidas en la <b>etapa 1</b>, <b>fuente 2</b>, <b>período 3</b> · "
                    "<b>Y123</b> = 1 si esa fuente se prepara (se usa) en ese período, 0 si no · "
                    "<b>I12</b> = inventario de la etapa 1 al final del período 2 · "
                    f"<b>I+{M_}3</b> = producto terminado en bodega al final del período 3 · <b>I-{M_}3</b> = demanda pendiente.<br>"
                    "<b>Columnas:</b> <i>Valor solución</i> = lo que decide el modelo · <i>Costo unitario</i> = costo de cada unidad · "
                    "<i>Contribución total</i> = valor × costo (la suma de esta columna es el costo total) · "
                    "<i>Costo reducido</i> = cuánto subiría el costo total (aprox.) si se forzara 1 unidad de una variable que vale 0 · "
                    "<i>Estado base</i>: 'Básica' = variable activa del plan; 'En cota' = en su límite (0 o capacidad máxima).")
        display(HTML(seccion(f"Tabla solución · {len(t)} variables", PAL["prod"], guia + leyenda +
                  tabla_html(t, 480, color=PAL["gris"],
                             fila_color=lambda r: TIPO_COLOR.get(r["Variable"][0], PAL["gris"])), "")))


w_filtro.observe(pintar_tabla, names="value")


def caja(titulo, color, texto):
    return (f"<div style='border-left:6px solid {color};background:{rgba(color, .13)};padding:9px 16px;border-radius:8px;"
            f"margin:8px 0;max-width:1000px'><b style='color:{color}'>{titulo}</b><div style='margin-top:3px'>{texto}</div></div>")


def kpi(titulo, valor, color, sub=""):
    sub = f"<div style='opacity:.75;font-size:12px'>{sub}</div>" if sub else ""
    return (f"<div style='flex:1;min-width:170px;border-radius:12px;padding:12px 16px;"
            f"background:{rgba(color,.16)};border-top:5px solid {color}'>"
            f"<div style='opacity:.75;font-size:12px;text-transform:uppercase;letter-spacing:.5px'>{titulo}</div>"
            f"<div style='font-size:21px;font-weight:800;color:{color}'>{valor}</div>{sub}</div>")


def mostrar(R, d, etiqueta="", Rs=None):
    for o in (o_res, o_plan, o_grafo, o_graf, o_util): o.clear_output()
    c, tot = R["costos"], max(R["obj"], 1e-9)
    pct = lambda v: f"{100*v/tot:.1f}% del total"
    ok = R["estado"].startswith("Óptima")
    with o_res:
        display(HTML(
            "<div style='display:flex;gap:12px;flex-wrap:wrap;margin:6px 0'>"
            + kpi("Costo total (F.O.)", "$ " + ES(R["obj"], 0), PAL["total"], "valor de la función objetivo")
            + kpi("Estado", R["estado"], PAL["total"] if ok else PAL["setup"],
                  f"{etiqueta} · {R['seg']:.1f} s") + "</div>"
            "<div style='display:flex;gap:12px;flex-wrap:wrap;margin:6px 0'>"
            + kpi("Producción", "$ " + ES(c["prod"], 0), PAL["prod"], pct(c["prod"]))
            + kpi("Preparación (setup)", "$ " + ES(c["setup"], 0), PAL["setup"], pct(c["setup"]))
            + kpi("Mantener inventario", "$ " + ES(c["mant"], 0), PAL["mant"], pct(c["mant"]))
            + kpi("Escasez", "$ " + ES(c["falt"], 0), PAL["falt"], pct(c["falt"])) + "</div>"))
    with o_res:
        display(HTML(caja("Cómo leer el resumen", PAL["total"],
                          "El <b>costo total</b> es lo mínimo que cuesta cumplir la demanda con las máquinas disponibles. Se reparte en: "
                          "<b>producción</b> (costo por unidad fabricada), <b>preparación</b> (costo fijo de arrancar cada fuente), "
                          "<b>mantener inventario</b> (guardar unidades) y <b>escasez</b> (demanda que no se entrega a tiempo). "
                          "Si 'Estado' dice <b>INFACTIBLE</b>, la capacidad no alcanza: subir máquinas, turnos o días, o permitir escasez.")))
    if Rs and len(Rs) == 2:
        A_, B_ = Rs["ent"], Rs["cont"]
        filas = [("Costo total (F.O.)", A_["obj"], B_["obj"]), ("Producción", A_["costos"]["prod"], B_["costos"]["prod"]),
                 ("Preparación (setup)", A_["costos"]["setup"], B_["costos"]["setup"]),
                 ("Mantener inventario", A_["costos"]["mant"], B_["costos"]["mant"]),
                 ("Escasez", A_["costos"]["falt"], B_["costos"]["falt"])]
        dfc = pd.DataFrame([[n, a_, b_, a_ - b_] for n, a_, b_ in filas],
                           columns=["Concepto", "Con aproximación (X entera)", "Sin aproximación (X continua)", "Diferencia"])
        nota = ("<p style='opacity:.8;margin:8px 0 0'>Sin aproximación: X continua y capacidades exactas (cota inferior). "
                "Con aproximación: X entera y capacidad redondeada hacia abajo, que es lo que se puede producir en la práctica. "
                f"La diferencia en el costo total es de $ {ES(A_['obj'] - B_['obj'], 2)} "
                f"({ES(100 * (A_['obj'] - B_['obj']) / max(B_['obj'], 1e-9), 6)} %).</p>")
        with o_res:
            display(HTML(seccion("Comparación: con vs sin aproximaciones", PAL["info"],
                                 tabla_html(dfc, 300, 2, PAL["info"]) + nota, "")))
    pintar_tabla()
    T = d.T
    with o_plan:
        display(HTML(caja("Cómo leer estas tablas", PAL["prod"],
                          "Cada fila es una fuente (máquina) y cada columna un período. El número es cuántas unidades produce; "
                          "<b>cuanto más oscuro, más produce</b> y una celda atenuada con 0 significa que esa fuente no trabaja ese período. "
                          "En la tabla de inventarios: <b>Demanda</b> es lo que piden los clientes, <b>Inventario etapa i</b> son piezas guardadas "
                          "entre etapas, <b>I+</b> es producto terminado en bodega e <b>I−</b> es demanda que quedó sin entregar.")))
        for nombre in d.productos:
            df = R["plan"][nombre].reset_index().rename(columns={"index": "Fuente"})
            display(HTML(seccion(f"{nombre} · Producción por fuente (unidades)", PAL["prod"],
                                 tabla_html(df, 380, 0, PAL["prod"], calor=True), "")))
            df = R["inv"][nombre].reset_index().rename(columns={"index": ""})
            display(HTML(seccion(f"{nombre} · Inventarios y cumplimiento de demanda", PAL["mant"],
                                 tabla_html(df, 380, 0, PAL["mant"], calor=True), "")))
    with o_grafo:
        display(HTML(caja("Cómo leer el grafo", PAL["mant"], TEXTO_GRAFO)))
        for nombre in d.productos:
            fig = figura_grafo(R, d, nombre, etiqueta); plt.show()
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": .25, "axes.axisbelow": True})
    ciclo = ["#2980b9", "#e67e22", "#8e44ad", "#16a085", "#e74c3c", "#f1c40f", "#34495e"]
    with o_graf:
        cols = list(R["plan"][d.productos[0]].columns); x = np.arange(T); M = len(d.J)
        for nombre in d.productos:
            nc = min(M, 4); nr = int(np.ceil(M / nc))
            fig, axs = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.2 * nr), squeeze=False)
            for i in range(1, M + 1):
                ax = axs[(i - 1) // nc][(i - 1) % nc]; base = np.zeros(T)
                for j in range(1, d.J[i - 1] + 1):
                    v = R["plan"][nombre].loc[f"Etapa {i} · Fuente {j}"].values.astype(float)
                    ax.bar(x, v, bottom=base, label=f"F{j}", color=ciclo[(j - 1) % len(ciclo)]); base += v
                ax.set_title(f"Etapa {i}", fontweight="bold"); ax.set_xticks(x); ax.set_xticklabels(cols)
                ax.legend(fontsize=7)
            for k in range(M, nr * nc): axs[k // nc][k % nc].axis("off")
            fig.suptitle(f"{nombre}: producción por etapa y fuente", fontweight="bold"); fig.tight_layout(); plt.show()
            inv = R["inv"][nombre]
            fig, ax = plt.subplots(figsize=(7.5, 3.6))
            ax.bar(x - 0.2, inv.loc["Demanda"], 0.4, label="Demanda", color=PAL["gris"])
            ax.bar(x + 0.2, inv.loc[f"Producción etapa {M}"], 0.4, label="Producción final", color=PAL["prod"])
            ax.plot(x, inv.loc["Inv. prod. terminado (I+)"], "o-", label="Inventario I+", color=PAL["mant"], lw=2)
            ax.plot(x, inv.loc["Demanda pendiente (I-)"], "s--", label="Pendiente I-", color=PAL["falt"], lw=2)
            ax.set_xticks(x); ax.set_xticklabels(cols); ax.legend(fontsize=8)
            ax.set_title(f"{nombre}: demanda vs producción", fontweight="bold"); fig.tight_layout(); plt.show()
        fig, ax = plt.subplots(figsize=(6, 3.4))
        nombres = ["Producción", "Setup", "Mantener", "Escasez"]; vals = [c["prod"], c["setup"], c["mant"], c["falt"]]
        bars = ax.bar(nombres, vals, color=[PAL["prod"], PAL["setup"], PAL["mant"], PAL["falt"]])
        for bb, v in zip(bars, vals): ax.text(bb.get_x() + bb.get_width() / 2, v, ES(v, 0), ha="center", va="bottom", fontsize=8)
        ax.set_title("Desglose del costo total", fontweight="bold"); fig.tight_layout(); plt.show()
    with o_util:
        display(HTML(caja("Cómo leer este mapa", PAL["setup"],
                          "Muestra qué porcentaje de las horas disponibles de cada fuente usa el plan en cada período. "
                          "<b>Verde</b> = sobra capacidad. <b>Rojo (cerca de 100 %)</b> = cuello de botella: esa fuente no puede producir más, "
                          "y si la demanda crece habrá que agregar máquinas, turnos o días.")))
        U = R["util"]
        fig, ax = plt.subplots(figsize=(1.2 * T + 3, 0.45 * len(U) + 1.5))
        im = ax.imshow(U.values, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
        ax.grid(False)
        ax.set_xticks(range(T)); ax.set_xticklabels(U.columns); ax.set_yticks(range(len(U))); ax.set_yticklabels(U.index)
        for a_ in range(U.shape[0]):
            for b_ in range(U.shape[1]): ax.text(b_, a_, f"{U.values[a_][b_]*100:.0f}%", ha="center", va="center", fontsize=8)
        fig.colorbar(im); ax.set_title("Utilización de la capacidad (horas usadas / disponibles)", fontweight="bold")
        fig.tight_layout(); plt.show()


ETIQ = {"ent": "Con aproximación (X entera)", "cont": "Sin aproximación (X continua)"}


def _pintar_version(_=None):
    if S.get("mute") or not S.get("Rs"): return
    k = w_ver.value
    if k not in S["Rs"]: return
    S["R"] = S["Rs"][k]
    mostrar(S["R"], S["d"], ETIQ[k], S["Rs"])


w_ver.observe(_pintar_version, names="value")


def resolver_click(_=None):
    try: d = leer_datos()
    except Exception as e: aviso(str(e)); return
    modo = w_modo.value
    corridas = (["ent"] if modo[0] in "AE" else []) + (["cont"] if modo[0] in "AC" else [])
    aviso("Resolviendo… (puede tardar hasta el tiempo máximo configurado por corrida)", "#2980b9")
    b_res.disabled = True
    Rs = {}
    try:
        for k in corridas:
            Rs[k] = resolver(replace(d, enteras=(k == "ent")))
    except Exception as e:
        aviso(f"Error al resolver: {e}"); b_res.disabled = False; return
    b_res.disabled = False
    for k, R in Rs.items():
        if not R["ok"]:
            aviso(f"{ETIQ[k]}: {R['estado']}"); return
    S["Rs"], S["d"] = Rs, d
    k = "ent" if "ent" in Rs else "cont"
    S["mute"] = True; w_ver.value = k; S["mute"] = False
    w_ver.disabled = len(Rs) < 2
    S["R"] = Rs[k]
    txt = " · ".join(f"{ETIQ[q]}: $ {ES(Rs[q]['obj'], 2)}" for q in Rs)
    aviso(f"{Rs[k]['estado']} · {txt}", "#27ae60")
    mostrar(Rs[k], d, ETIQ[k], Rs)


# ───────── grafo de la solución (red por etapas y períodos, como en el Excel del profesor)
GR_COLORES = ["#2980b9", "#e67e22", "#16a085", "#8e44ad", "#c0392b", "#d4ac0d", "#34495e", "#27ae60"]


def figura_grafo(R, d, nombre, etiqueta=""):
    from matplotlib.patches import Circle, FancyArrowPatch
    from matplotlib.lines import Line2D
    M, T = len(d.J), d.T
    p = d.productos.index(nombre)
    plan, inv = R["plan"][nombre], R["inv"][nombre]
    dx, dy = 5.2, 3.6
    ancho_u, alto_u = T * dx + 8.0, M * dy + 6.5
    fig, ax = plt.subplots(figsize=(max(9, 0.5 * ancho_u), 0.5 * alto_u + 1.0))
    ax.set_xlim(-4.2, T * dx + 3.8); ax.set_ylim(-M * dy - 3.6, 1.6)
    ax.set_aspect("equal"); ax.axis("off")
    pos = lambda i, t: (t * dx, -i * dy)
    vmax = max(float(plan.values.max()), 1.0)
    nodos = {}
    for i in range(0, M + 1):
        for t in range(1, T + 1):
            x, y = pos(i, t)
            c = Circle((x, y), 0.62, fc="#ecf0f1" if i == 0 else "white", ec="#34495e", lw=1.6, zorder=3)
            ax.add_patch(c); nodos[i, t] = c
            ax.text(x, y, "Ent." if i == 0 else f"T{t}", ha="center", va="center", fontsize=8, fontweight="bold",
                    zorder=4, color="#2c3e50")
    for i in range(1, M + 1):
        ax.text(-4.0, -i * dy, f"Etapa {i}\nJ = {', '.join(str(j) for j in range(1, d.J[i-1] + 1))} máquinas",
                ha="left", va="center", fontsize=9, fontweight="bold")
    ax.text(-4.0, 0, "Entrada", ha="left", va="center", fontsize=9, fontweight="bold")

    def flecha(a, b, rad, color, lw, estilo="-", ps=None, pb=None, z=2):
        f = FancyArrowPatch(a, b, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>", mutation_scale=11,
                            color=color, lw=lw, linestyle=estilo, patchA=ps, patchB=pb, shrinkA=0, shrinkB=0, zorder=z)
        ax.add_patch(f)

    def etiqueta_(x, y, txt, color, blanco=True, fs=7.5):
        ax.text(x, y, txt, ha="center", va="center", fontsize=fs, zorder=6, color="white" if blanco else "#2c3e50",
                bbox=dict(boxstyle="round,pad=0.22", fc=color, ec="none"))

    for i in range(1, M + 1):                                   # producción: flechas que bajan
        J = d.J[i - 1]
        rads = np.linspace(-0.85, 0.85, J) if J > 1 else [0.0]
        for t in range(1, T + 1):
            a, b = pos(i - 1, t), pos(i, t)
            for j in range(1, J + 1):
                v = float(plan.loc[f"Etapa {i} · Fuente {j}", f"T{t}"]); col = GR_COLORES[(j - 1) % len(GR_COLORES)]
                if v > 1e-9:
                    flecha(a, b, rads[j - 1], col, 1.2 + 2.6 * v / vmax, ps=nodos[i - 1, t], pb=nodos[i, t], z=2)
                    etiqueta_((a[0] + b[0]) / 2 + 0.5 * rads[j - 1] * (b[1] - a[1]),
                              (a[1] + b[1]) / 2 - 0.5 * rads[j - 1] * (b[0] - a[0]), ES(v, 0), col)
                else:
                    flecha(a, b, rads[j - 1], "#95a5a6", 0.6, ":", ps=nodos[i - 1, t], pb=nodos[i, t], z=1)
    filas_inv = [f"Inventario etapa {i}" for i in range(1, M)] + ["Inv. prod. terminado (I+)"]
    for i in range(1, M + 1):                                   # inventarios: flechas horizontales
        for t in range(1, T):
            v = float(inv.loc[filas_inv[i - 1], f"T{t}"]); a, b = pos(i, t), pos(i, t + 1)
            col = "#27ae60" if i == M else "#7f8c8d"
            if v > 1e-9:
                flecha(a, b, 0.0, col, 2.0, ps=nodos[i, t], pb=nodos[i, t + 1], z=2)
                etiqueta_((a[0] + b[0]) / 2, a[1] + 0.55, ES(v, 0), col)
            else:
                flecha(a, b, 0.0, "#bdc3c7", 0.7, "--", ps=nodos[i, t], pb=nodos[i, t + 1], z=1)
    for t in range(1, T):                                       # demanda pendiente (rojo)
        v = float(inv.loc["Demanda pendiente (I-)", f"T{t}"])
        if v > 1e-9:
            a, b = pos(M, t), pos(M, t + 1)
            flecha(a, b, -0.55, "#e74c3c", 1.8, "--", ps=nodos[M, t], pb=nodos[M, t + 1], z=2)
            etiqueta_((a[0] + b[0]) / 2, a[1] - 1.05, ES(v, 0), "#e74c3c")
    for i in range(1, M + 1):                                   # inventario al final del horizonte
        a = pos(i, T); b = (a[0] + 2.6, a[1])
        v = float(inv.loc[filas_inv[i - 1], f"T{T}"])
        flecha(a, b, 0.0, "#27ae60" if i == M else "#7f8c8d", 1.4, ps=nodos[i, T])
        etiqueta_((a[0] + b[0]) / 2 + 0.2, a[1] + 0.5, ES(v, 0), "#27ae60" if i == M else "#7f8c8d")
    a = pos(M, 1); b = (a[0] - 2.6, a[1])                       # inventario inicial
    flecha(b, a, 0.0, "#27ae60", 1.4, pb=nodos[M, 1])
    etiqueta_((a[0] + b[0]) / 2 - 0.2, a[1] + 0.5, ES(d.inv_ini[p], 0), "#27ae60")
    for t in range(1, T + 1):                                   # demanda
        a = pos(M, t); b = (a[0], a[1] - 2.3)
        flecha(a, b, 0.0, "#2c3e50", 1.6, ps=nodos[M, t])
        ax.text(b[0], b[1] - 0.15, f"Demanda T{t}\n{ES(d.demanda[(p, t)], 0)}", ha="center", va="top", fontsize=8,
                fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="#ecf0f1", ec="#2c3e50"), color="#2c3e50")
    h = [Line2D([0], [0], color=GR_COLORES[(j - 1) % len(GR_COLORES)], lw=3, label=f"Fuente {j}")
         for j in range(1, max(d.J) + 1)]
    h += [Line2D([0], [0], color="#7f8c8d", lw=2, label="Inventario entre períodos"),
          Line2D([0], [0], color="#27ae60", lw=2, label="Producto terminado en bodega"),
          Line2D([0], [0], color="#e74c3c", lw=2, ls="--", label="Demanda pendiente"),
          Line2D([0], [0], color="#95a5a6", lw=1, ls=":", label="Sin flujo (0)")]
    fig.legend(handles=h, loc="lower center", ncol=min(len(h), 6), fontsize=8, frameon=False)
    fig.suptitle(f"Grafo de la solución · {nombre}" + (f" · {etiqueta}" if etiqueta else ""), fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    return fig


TEXTO_GRAFO = ("Cada fila de círculos es una etapa y cada círculo un período (T1, T2, ...). Las flechas que bajan son producción: "
               "cada color es una fuente (máquina) y el número indica las unidades que fabrica en ese período; cuanto más gruesa, "
               "más produce. Las flechas horizontales son inventario que pasa al período siguiente (verde en la última etapa = producto "
               "terminado en bodega). Las flechas a la derecha son el inventario al final del plan, y debajo de cada período aparece la demanda.")


# ───────── exportar a Excel
def construir_excel(Rs, d, ruta):
    import io
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    from openpyxl.formatting.rule import ColorScaleRule
    from openpyxl.drawing.image import Image as XLImage

    P, M, T = len(d.productos), len(d.J), d.T
    src = [(i, j) for i in range(1, M + 1) for j in range(1, d.J[i - 1] + 1)]
    cT = [f"T{t}" for t in range(1, T + 1)]
    borde = Side(style="thin", color="BDC3C7"); BORDE = Border(left=borde, right=borde, top=borde, bottom=borde)
    fill = lambda c: PatternFill("solid", fgColor=c.lstrip("#"))
    blanco = Font(bold=True, color="FFFFFF")
    nombres = {"ent": "entera", "cont": "continua"}
    wb = Workbook(); wb.remove(wb.active)

    def titulo(ws, txt, color, ancho=8):
        ws["A1"] = txt; ws["A1"].font = Font(size=16, bold=True, color="FFFFFF"); ws["A1"].fill = fill(color)
        ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=max(ancho, 2))
        ws.row_dimensions[1].height = 28; ws["A1"].alignment = Alignment(vertical="center")

    def nota(ws, fila, txt, ancho=8, alto=62):
        ws.cell(fila, 1, txt).alignment = Alignment(wrap_text=True, vertical="top")
        ws.merge_cells(start_row=fila, start_column=1, end_row=fila, end_column=max(ancho, 2))
        ws.row_dimensions[fila].height = alto; ws.cell(fila, 1).fill = fill("#F4F6F7")

    def fmt(v):
        return "#,##0" if abs(float(v) - round(float(v))) < 1e-6 else "#,##0.00"

    def volcar(ws, fila, txt, df, color, col0=1):
        ws.cell(fila, col0, txt).font = Font(bold=True, size=12, color=color.lstrip("#"))
        fila += 1
        for c, h in enumerate(df.columns):
            x = ws.cell(fila, col0 + c, str(h)); x.font = blanco; x.fill = fill(color); x.border = BORDE
            x.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        for r, row in enumerate(df.itertuples(index=False), 1):
            for c, v in enumerate(row):
                if isinstance(v, (float, np.floating)) and abs(v - round(v)) < 1e-6: v = int(round(v))
                x = ws.cell(fila + r, col0 + c, v); x.border = BORDE
                if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
                    x.number_format = fmt(v); x.alignment = Alignment(horizontal="right")
                if r % 2 == 0: x.fill = fill("#F8F9F9")
        return fila + len(df) + 2

    def anchos(ws, base=14, primera=30):
        ws.column_dimensions["A"].width = primera
        for c in range(2, 40): ws.column_dimensions[get_column_letter(c)].width = base

    # ---------- Resumen
    ws = wb.create_sheet("Resumen"); ws.sheet_properties.tabColor = "27AE60"
    titulo(ws, "Planeación agregada · Resumen de la solución", "#27ae60", 5)
    ks = list(Rs.keys()); cab = ["Concepto"] + [f"Versión {nombres[k]}" for k in ks] + (["Diferencia (entera − continua)"] if len(ks) == 2 else [])
    filas = [("Estado", [Rs[k]["estado"] for k in ks], False), ("Tiempo de cálculo (s)", [round(Rs[k]["seg"], 2) for k in ks], False),
             ("Costo total (función objetivo)", [Rs[k]["obj"] for k in ks], True),
             ("Costo de producción", [Rs[k]["costos"]["prod"] for k in ks], True),
             ("Costo de preparación (setup)", [Rs[k]["costos"]["setup"] for k in ks], True),
             ("Costo de mantener inventario", [Rs[k]["costos"]["mant"] for k in ks], True),
             ("Costo de escasez", [Rs[k]["costos"]["falt"] for k in ks], True)]
    for c, h in enumerate(cab, 1):
        x = ws.cell(3, c, h); x.font = blanco; x.fill = fill("#27ae60"); x.border = BORDE
        x.alignment = Alignment(horizontal="center", wrap_text=True)
    for r, (n, vals, num) in enumerate(filas, 4):
        ws.cell(r, 1, n).font = Font(bold=True); ws.cell(r, 1).border = BORDE
        for c, v in enumerate(vals, 2):
            x = ws.cell(r, c, v); x.border = BORDE
            if num: x.number_format = "#,##0.00"
        if len(ks) == 2:
            x = ws.cell(r, 4, f"=B{r}-C{r}" if num else ""); x.border = BORDE; x.number_format = "#,##0.00"
        if r == 6:
            for c in range(1, len(cab) + 1): ws.cell(r, c).font = Font(bold=True, color="27AE60", size=12)
    nota(ws, 12, "La versión continua permite decimales y usa capacidades exactas (límite teórico). La versión entera produce unidades "
                 "enteras y redondea la capacidad hacia abajo (lo que se puede hacer en la práctica); nunca cuesta menos que la continua.",
         5, 50)
    anchos(ws, 24, 34)

    # ---------- Parámetros
    ws = wb.create_sheet("Parámetros"); ws.sheet_properties.tabColor = "2980B9"
    titulo(ws, "Datos de entrada del problema", "#2980b9", 9)
    f = 3
    f = volcar(ws, f, "Datos generales", pd.DataFrame({"Parámetro": ["Productos", "Etapas", "Fuentes por etapa", "Períodos",
                "Meses por período", "Factor de eficiencia", "Escasez permitida"],
                "Valor": [", ".join(d.productos), M, ", ".join(map(str, d.J)), T, d.meses, d.efic, "Sí" if d.faltantes else "No"]}), "#2980b9")
    f = volcar(ws, f, "Demanda e inventarios", pd.DataFrame(
        [[n, d.inv_ini[p], d.inv_fin[p], d.c_falt[p]] + [d.demanda[(p, t)] for t in range(1, T + 1)] for p, n in enumerate(d.productos)],
        columns=["Producto", "Inv. inicial", "Inv. final", "Costo escasez/u"] + [f"Demanda {c}" for c in cT]), "#2980b9")
    f = volcar(ws, f, "Días hábiles por período", pd.DataFrame([d.dias], columns=cT), "#16a085")
    f = volcar(ws, f, "Fuentes y horas disponibles (N · eficiencia · (días · turnos · h/turno − pérdida · meses))", pd.DataFrame(
        [[f"Etapa {i} · Fuente {j}", d.fuente[(i, j)]["N"], d.fuente[(i, j)]["perd"], d.fuente[(i, j)]["turnos"], d.fuente[(i, j)]["horas"]]
         + [round(horas(d, i, j, t), 2) for t in range(1, T + 1)] for i, j in src],
        columns=["Fuente", "Nº máquinas", "Pérdida mant. (h/mes/máq)", "Turnos", "Horas/turno"] + [f"Horas {c}" for c in cT]), "#16a085")
    f = volcar(ws, f, "Tasas, preparación y capacidad en unidades (horas × tasa)", pd.DataFrame(
        [[n, f"Etapa {i} · Fuente {j}", d.tasa[(p, i, j)], d.setup[(p, i, j)]]
         + [round(horas(d, i, j, t) * d.tasa[(p, i, j)], 1) for t in range(1, T + 1)]
         for p, n in enumerate(d.productos) for i, j in src],
        columns=["Producto", "Fuente", "Tasa (u/h)", "Costo preparación"] + [f"Capacidad {c}" for c in cT]), "#e67e22")
    f = volcar(ws, f, "Costo unitario de producción", pd.DataFrame(
        [[n, f"Etapa {i} · Fuente {j}"] + [d.costo[(p, i, j, t)] for t in range(1, T + 1)]
         for p, n in enumerate(d.productos) for i, j in src], columns=["Producto", "Fuente"] + cT), "#8e44ad")
    f = volcar(ws, f, "Inventarios entre etapas", pd.DataFrame(
        [[n, f"Etapa {i}", d.mantener[(p, i)], d.coef[(p, i)]] for p, n in enumerate(d.productos) for i in range(1, M + 1)],
        columns=["Producto", "Etapa", "Costo de mantener", "Consumo por unidad"]), "#d4ac0d")
    anchos(ws, 16, 34)

    # ---------- por versión
    for k, R in Rs.items():
        nm = nombres[k]; col = "#2980b9" if k == "ent" else "#16a085"
        # Solución
        ws = wb.create_sheet(f"Solución {nm}"); ws.sheet_properties.tabColor = col.lstrip("#")
        enc = ["Variable", "Valor solución", "Costo unitario", "Contribución total", "Costo reducido", "Estado base"]
        for c, h in enumerate(enc, 1):
            x = ws.cell(1, c, h); x.font = blanco; x.fill = fill(col); x.border = BORDE
            x.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        tint = {"X": "#D6EAF8", "I": "#E8DAEF", "Y": "#FDEBD0"}
        for r, row in enumerate(R["tabla"].itertuples(index=False), 2):
            ws.cell(r, 1, row[0]).fill = fill(tint.get(row[0][0], "#FFFFFF"))
            ws.cell(r, 2, float(row[1])); ws.cell(r, 3, float(row[2])); ws.cell(r, 4, f"=B{r}*C{r}")
            if row[4] == row[4]: ws.cell(r, 5, float(row[4]))
            ws.cell(r, 6, row[5])
            for c in range(1, 7): ws.cell(r, c).border = BORDE
            for c in (2, 3, 4, 5): ws.cell(r, c).number_format = "#,##0.00"
            if abs(row[1]) < 1e-9:
                for c in range(1, 7): ws.cell(r, c).font = Font(color="95A5A6")
        n = len(R["tabla"]) + 1
        ws.cell(n + 1, 1, "FUNCIÓN OBJETIVO (suma de contribuciones)").font = Font(bold=True)
        ws.cell(n + 1, 4, f"=SUM(D2:D{n})").number_format = "#,##0.00"; ws.cell(n + 1, 4).font = Font(bold=True, color="27AE60", size=12)
        ws.freeze_panes = "B2"; ws.auto_filter.ref = f"A1:F{n}"
        for c, w in zip("ABCDEF", (18, 16, 16, 20, 16, 16)): ws.column_dimensions[c].width = w
        ws.column_dimensions["H"].width = 70
        ws["H1"] = "Cómo leer"; ws["H1"].font = Font(bold=True)
        ws["H2"] = ("X123 = producción en etapa 1, fuente 2, período 3 · Y123 = 1 si esa fuente se prepara · I12 = inventario de la "
                    "etapa 1 al final del período 2 · I+ = producto terminado en bodega · I- = demanda pendiente. "
                    "Colores: azul = X, morado = I, naranja = Y.")
        ws["H2"].alignment = Alignment(wrap_text=True, vertical="top"); ws.row_dimensions[2].height = 90

        # Plan
        ws = wb.create_sheet(f"Plan {nm}"); ws.sheet_properties.tabColor = col.lstrip("#")
        titulo(ws, f"Plan de producción e inventarios · versión {nm}", col, T + 2)
        f = 3
        for nombre in d.productos:
            dfp = R["plan"][nombre].reset_index().rename(columns={"index": "Fuente"})
            f0 = f; f = volcar(ws, f, f"{nombre} · Producción por fuente (unidades)", dfp, "#2980b9")
            ws.conditional_formatting.add(f"B{f0+2}:{get_column_letter(T+1)}{f0+1+len(dfp)}",
                                          ColorScaleRule(start_type="min", start_color="FFFFFF", end_type="max", end_color="5DADE2"))
            dfi = R["inv"][nombre].reset_index().rename(columns={"index": ""})
            f0 = f; f = volcar(ws, f, f"{nombre} · Inventarios y cumplimiento de demanda", dfi, "#8e44ad")
            ws.conditional_formatting.add(f"B{f0+2}:{get_column_letter(T+1)}{f0+1+len(dfi)}",
                                          ColorScaleRule(start_type="min", start_color="FFFFFF", end_type="max", end_color="BB8FCE"))
        anchos(ws, 15, 34)

        # Utilización
        ws = wb.create_sheet(f"Utilización {nm}"); ws.sheet_properties.tabColor = "E67E22"
        titulo(ws, f"Utilización de la capacidad · versión {nm}", "#e67e22", T + 1)
        U = R["util"].reset_index().rename(columns={"index": "Fuente"})
        volcar(ws, 3, "Horas usadas / horas disponibles", U, "#e67e22")
        for r in range(5, 5 + len(U)):
            for c in range(2, T + 2): ws.cell(r, c).number_format = "0%"
        ws.conditional_formatting.add(f"B5:{get_column_letter(T+1)}{4+len(U)}",
                                      ColorScaleRule(start_type="num", start_value=0, start_color="63BE7B", mid_type="num", mid_value=0.5,
                                                     mid_color="FFEB84", end_type="num", end_value=1, end_color="F8696B"))
        anchos(ws, 12, 18)

        # Grafo
        ws = wb.create_sheet(f"Grafo {nm}"); ws.sheet_properties.tabColor = "8E44AD"
        titulo(ws, f"Grafo de la solución · versión {nm}", "#8e44ad", 20)
        nota(ws, 2, TEXTO_GRAFO, 20, 62)
        fila = 4
        for nombre in d.productos:
            fig = figura_grafo(R, d, nombre, f"versión {nm}")
            buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110); plt.close(fig); buf.seek(0)
            img = XLImage(buf); esc = min(1.0, 1500 / img.width); img.width, img.height = img.width * esc, img.height * esc
            ws.add_image(img, f"A{fila}"); fila += int(img.height / 20) + 3
    wb.save(ruta)


def exportar(_=None):
    Rs = S.get("Rs")
    if not Rs: aviso("Primero hay que resolver el modelo."); return
    ruta = "plan_agregado.xlsx"
    try:
        construir_excel(Rs, S["d"], ruta)
    except Exception as e:
        aviso(f"No se pudo crear el Excel: {e}"); return
    try:
        from google.colab import files; files.download(ruta)
    except Exception:
        aviso(f"Archivo guardado como {ruta}", "#27ae60")



b_gen.on_click(generar_tablas); b_ej.on_click(cargar_ejemplo)
b_res.on_click(resolver_click); b_xls.on_click(exportar)


def _form(filas):
    items = []
    for t, w, h in filas:
        items += [W.HTML(f"<b>{t}</b>"), w, W.HTML(f"<span style='opacity:.85'>{h}</span>")]
    return W.GridBox(items, layout=W.Layout(grid_template_columns="170px 290px 1fr", grid_gap="8px 14px", align_items="center"))


def _paso(n, titulo, texto, color):
    return (f"<div style='flex:1;min-width:210px;border-radius:12px;padding:10px 14px;background:{rgba(color, .15)};"
            f"border-top:5px solid {color}'><div style='font-size:22px;font-weight:800;color:{color}'>{n}</div>"
            f"<div style='font-weight:700'>{titulo}</div><div style='opacity:.85;font-size:13px'>{texto}</div></div>")


PASOS = ("<div style='display:flex;gap:12px;flex-wrap:wrap;max-width:1000px;margin:8px 0'>"
         + _paso("1", "Definir la estructura", "Productos, etapas, fuentes y períodos (abajo).", COL_T["dem"])
         + _paso("2", "Generar las tablas", "Pulsar el botón azul. Para probar, cargar el caso del artículo.", COL_T["cap"])
         + _paso("3", "Editar los datos", "5 pestañas con colores. Cada una explica sus columnas.", COL_T["tasa"])
         + _paso("4", "Resolver", "Pulsar RESOLVER y revisar las pestañas de resultados.", "#27ae60") + "</div>")

GLOSARIO = """
<dl style='max-width:1000px;line-height:1.5'>
<dt><b>Etapa</b></dt><dd>Un paso del proceso productivo. La etapa 1 es la primera; la última entrega el producto terminado.</dd>
<dt><b>Fuente</b></dt><dd>Una máquina (o grupo de máquinas iguales) que puede realizar una etapa. Si una etapa tiene varias fuentes, el modelo elige cuánto produce cada una.</dd>
<dt><b>Período</b></dt><dd>Cada tramo de tiempo del plan (por ejemplo un trimestre). T1, T2, ...</dd>
<dt><b>Tasa de producción</b></dt><dd>Unidades por hora que produce una máquina.</dd>
<dt><b>Capacidad</b></dt><dd>Máximo de unidades que una fuente puede producir en un período, según máquinas, días, turnos y mantenimiento.</dd>
<dt><b>Costo de preparación (setup)</b></dt><dd>Costo fijo por poner a trabajar una fuente en un período, produzca poco o mucho.</dd>
<dt><b>Inventario</b></dt><dd>Unidades guardadas entre una etapa y la siguiente (o producto terminado en bodega).</dd>
<dt><b>Escasez / demanda pendiente</b></dt><dd>Unidades que no se entregan a tiempo y quedan debiéndose al cliente. Tienen un costo por unidad.</dd>
<dt><b>Variables del modelo</b></dt><dd><b>X</b> = unidades producidas · <b>Y</b> = 1 si se prepara una fuente (0 si no) · <b>I</b> = inventario entre etapas · <b>I+</b> = producto terminado en bodega · <b>I−</b> = demanda pendiente.</dd>
<dt><b>Con aproximación / sin aproximación</b></dt><dd><i>Con</i>: la producción es un número entero y la capacidad se redondea hacia abajo (lo que se puede hacer en la práctica). <i>Sin</i>: se permiten decimales (da un costo ligeramente menor, es un límite teórico).</dd>
</dl>"""

acc = W.Accordion(children=[W.HTML(GLOSARIO)])
acc.set_title(0, "Glosario: ¿qué significa cada palabra?")
try: acc.selected_index = None
except Exception: pass

ui = W.VBox([
    W.HTML(CSS_UI + "<h2>Planeación agregada · multietapa · multifuente · multiproducto</h2>"
           "<p style='opacity:.85;max-width:1000px'>Esta herramienta calcula <b>cuánto producir, dónde y cuándo</b> para cumplir la "
           "demanda al <b>menor costo posible</b>, respetando la capacidad de las máquinas (modelo MILP de Kalenatic, López y González)."
           "</p>"),
    W.HTML("<div style='font-size:18px;font-weight:800'>¿Cómo se usa?</div>"), W.HTML(PASOS), acc,
    W.HTML("<div style='font-size:18px;font-weight:800;margin-top:14px'>Paso 1 · Definir la estructura</div>"
           "<div style='opacity:.85'>Datos generales del problema. A la derecha de cada campo aparece qué escribir.</div>"),
    _form([
        ("Productos", w_prod, "Nombres separados por coma. Ej: <i>Camisa, Pantalón</i> = 2 productos."),
        ("Fuentes por etapa", w_J, "Un número por etapa, separado por comas. <i>3,2,3,4</i> = 4 etapas: la 1 tiene 3 fuentes, la 2 tiene 2, la 3 tiene 3 y la 4 tiene 4."),
        ("Nº de períodos", w_T, "Cuántos períodos tiene el plan. Ej: 4 trimestres."),
        ("Meses por período", w_meses, "Duración de cada período en meses (trimestre = 3). Se usa para descontar el mantenimiento."),
        ("Factor de eficiencia", w_efic, "Fracción del tiempo que la máquina realmente produce. 0,8 = 80 % (es el valor del artículo)."),
        ("Escasez", w_falt, "Al desmarcar, el modelo debe cumplir toda la demanda. Si la capacidad no alcanza, el resultado es INFACTIBLE."),
        ("Resolver", w_modo, "<b>Ambas</b>: calcular las dos versiones y compararlas. Entera = práctica; Continua = límite teórico."),
        ("Tiempo máximo (s)", w_tl, "Segundos máximos de cálculo por versión. Subir si el problema es grande."),
        ("Gap tolerado (%)", w_gap, "0 = exigir el óptimo exacto. 1 = aceptar un plan a menos de 1 % del mejor posible (más rápido)."),
    ]),
    W.HTML("<div style='height:8px'></div>"), w_vista,
    W.HTML("<div style='font-size:18px;font-weight:800;margin-top:14px'>Paso 2 · Generar las tablas</div>"
           "<div style='opacity:.85;max-width:1000px'>Al cambiar productos, etapas, fuentes o períodos hay que volver a pulsar "
           "<b>Generar tablas</b> (los datos escritos se reemplazan por valores de ejemplo).</div>"),
    W.HBox([b_gen, b_ej], layout=W.Layout(gap="12px", margin="6px 0")), msg, caja_tablas,
    W.HTML("<div style='font-size:18px;font-weight:800;margin-top:16px'>Paso 4 · Resolver y revisar los resultados</div>"),
    W.HBox([b_res, b_xls], layout=W.Layout(gap="12px", margin="6px 0")), w_ver, res_tabs])
display(ui)
actualizar_vista()
generar_tablas()
