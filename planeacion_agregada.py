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
                    "Sube máquinas/turnos/días, baja la demanda, o activa 'permitir escasez'.")
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
            f"{icono} {titulo}</div><div style='padding:10px 14px'>{cuerpo}</div></div>")


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


class Grid:
    def __init__(self, filas, cols, defaults=None, ancho=95, ancho_et=250):
        defaults = defaults if defaults is not None else [0.0] * len(cols)
        self.celdas = [[W.FloatText(value=float(defaults[c]), layout=W.Layout(width=f"{ancho}px"))
                        for c in range(len(cols))] for _ in filas]
        items = [W.HTML("")] + [W.HTML(f"<b>{c}</b>") for c in cols]
        for r, f in enumerate(filas):
            items.append(W.HTML(f))
            items += self.celdas[r]
        self.box = W.GridBox(items, layout=W.Layout(
            grid_template_columns=f"{ancho_et}px repeat({len(cols)}, {ancho + 8}px)", grid_gap="4px 8px"))

    def get(self):
        return np.array([[c.value for c in fila] for fila in self.celdas], float)

    def set(self, M):
        for r, fila in enumerate(M):
            for c, v in enumerate(fila): self.celdas[r][c].value = float(v)


# ───────── widgets de estructura
est = dict(style={"description_width": "150px"}, layout=W.Layout(width="430px"))
w_prod = W.Text(value="Producto A, Producto B", description="Productos (coma)", **est)
w_J = W.Text(value="3,2,3,4", description="Fuentes por etapa", **est)
w_T = W.IntText(value=4, description="Nº de períodos", **est)
w_meses = W.FloatText(value=3, description="Meses por período", **est)
w_efic = W.FloatText(value=0.8, description="Factor eficiencia", **est)
w_falt = W.Checkbox(value=True, description="Permitir escasez (demanda pendiente)", indent=False)
w_modo = W.Dropdown(options=["Ambas (comparar)", "Entera (con aproximación)", "Continua (sin aproximación)"],
                    value="Ambas (comparar)", description="Resolver", **est)
w_ver = W.Dropdown(options=[("Con aproximación (X entera)", "ent"), ("Sin aproximación (X continua)", "cont")],
                   value="ent", description="Ver solución:", style={"description_width": "100px"},
                   layout=W.Layout(width="400px"))
w_tl = W.IntText(value=60, description="Tiempo máx. (s)", **est)
w_gap = W.FloatText(value=0.0, description="Gap tolerado (%)", **est)
b_gen = W.Button(description="1 · Generar tablas", button_style="info", icon="table", layout=W.Layout(width="190px"))
b_ej = W.Button(description="Cargar caso del artículo", icon="book", layout=W.Layout(width="210px"))
b_res = W.Button(description="2 · RESOLVER", button_style="success", icon="play", layout=W.Layout(width="190px"))
b_xls = W.Button(description="Descargar Excel", icon="download", layout=W.Layout(width="170px"))
msg = W.HTML("")
caja_tablas = W.VBox([])
o_res, o_tab, o_plan, o_graf, o_util = (W.Output() for _ in range(5))
w_filtro = W.Dropdown(options=["Todas", "Solo valor ≠ 0", "Producción (X)", "Inventarios (I)", "Preparaciones (Y)"],
                      value="Solo valor ≠ 0", description="Mostrar:")
res_tabs = W.Tab(children=[o_res, W.VBox([w_filtro, o_tab]), o_plan, o_graf, o_util])
for k, t in enumerate(["Resumen", "Tabla solución", "Plan por etapa", "Gráficas", "Utilización"]): res_tabs.set_title(k, t)


def aviso(txt, color="#c0392b"):
    msg.value = f"<div style='color:{color};padding:4px 0'><b>{txt}</b></div>"


# ───────── generar tablas
def leer_estructura():
    prods = [s.strip() for s in w_prod.value.split(",") if s.strip()]
    J = [int(s) for s in w_J.value.replace(";", ",").split(",") if s.strip()]
    T = int(w_T.value)
    if not prods or not J or T < 1 or min(J) < 1: raise ValueError("Revisa productos, fuentes por etapa y períodos.")
    return prods, J, T


def generar_tablas(_=None):
    try: prods, J, T = leer_estructura()
    except Exception as e: aviso(str(e)); return
    M = len(J); P = len(prods)
    src = [(i, j) for i in range(1, M + 1) for j in range(1, J[i - 1] + 1)]
    cT = [f"T{t}" for t in range(1, T + 1)]
    rows_pf = [f"{p} · E{i}-F{j}" for p in prods for (i, j) in src]
    g = {}
    g["dem"] = Grid(prods, ["Inv. inicial", "Inv. final", "Costo escasez/u"] + [f"Demanda T{t}" for t in range(1, T + 1)],
                    [0, 0, 10] + [10000] * T)
    g["dias"] = Grid(["Días hábiles"], cT, [65] * T)
    g["fte"] = Grid([f"Etapa {i} · Fuente {j}" for i, j in src],
                    ["Nº máquinas", "Pérdida mant. (h/mes/máq)", "Turnos", "Horas/turno"], [5, 12, 1, 8], ancho=120)
    g["tasa"] = Grid(rows_pf, ["Tasa (u/hora)", "Costo preparación"], [25, 20000])
    g["costo"] = Grid(rows_pf, cT, [20] * T)
    g["etp"] = Grid([f"{p} · Etapa {i}" for p in prods for i in range(1, M + 1)],
                    ["Costo mantener/u·período", "u etapa i por u etapa i+1"], [1, 1], ancho=150)
    S["g"], S["est"] = g, (prods, J, T)
    H = lambda s: W.HTML(f"<p style='opacity:.8'>{s}</p>")
    tabs = W.Tab(children=[
        W.VBox([H("Una fila por producto. Demanda por período, inventarios y costo de escasez."), g["dem"].box]),
        W.VBox([H("Días hábiles de cada período."), g["dias"].box,
                H("Máquinas por fuente. Capacidad (h) = N·efic·(días·turnos·h/turno − pérdida·meses). "
                  "Las máquinas se comparten entre productos."), g["fte"].box]),
        W.VBox([H("Tasa de producción de cada producto en cada fuente (0 = esa fuente no lo procesa) y costo de preparación."),
                g["tasa"].box]),
        W.VBox([H("Costo unitario de producción por período."), g["costo"].box]),
        W.VBox([H("Costo de mantener inventario por etapa; 'u etapa i por u etapa i+1' normalmente 1."), g["etp"].box])])
    for k, t in enumerate(["Demanda", "Capacidad", "Tasas y setup", "Costos unitarios", "Etapas"]): tabs.set_title(k, t)
    caja_tablas.children = [W.HTML("<h3>Datos del problema</h3>"), tabs]
    aviso(f"Tablas generadas: {P} producto(s), {M} etapas, {sum(J)} fuentes, {T} períodos. Edita los valores y pulsa RESOLVER.",
          "#27ae60")


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
    aviso("Caso del artículo cargado (4 etapas, 12 fuentes, 4 trimestres). Costo esperado ≈ 34.208.760.", "#27ae60")


def leer_datos():
    if S["est"] is None: raise ValueError("Primero pulsa 'Generar tablas'.")
    prods, J, T = S["est"]; g = S["g"]; M = len(J); P = len(prods)
    src = [(i, j) for i in range(1, M + 1) for j in range(1, J[i - 1] + 1)]
    dem, dias, fte = g["dem"].get(), g["dias"].get()[0], g["fte"].get()
    tasa, costo, etp = g["tasa"].get(), g["costo"].get(), g["etp"].get()
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
        display(HTML(seccion(f"Tabla solución · {len(t)} variables", PAL["prod"], leyenda +
                  tabla_html(t, 480, color=PAL["gris"],
                             fila_color=lambda r: TIPO_COLOR.get(r["Variable"][0], PAL["gris"])), "📋")))


w_filtro.observe(pintar_tabla, names="value")


def kpi(titulo, valor, color, sub=""):
    sub = f"<div style='opacity:.75;font-size:12px'>{sub}</div>" if sub else ""
    return (f"<div style='flex:1;min-width:170px;border-radius:12px;padding:12px 16px;"
            f"background:{rgba(color,.16)};border-top:5px solid {color}'>"
            f"<div style='opacity:.75;font-size:12px;text-transform:uppercase;letter-spacing:.5px'>{titulo}</div>"
            f"<div style='font-size:21px;font-weight:800;color:{color}'>{valor}</div>{sub}</div>")


def mostrar(R, d, etiqueta="", Rs=None):
    for o in (o_res, o_plan, o_graf, o_util): o.clear_output()
    c, tot = R["costos"], max(R["obj"], 1e-9)
    pct = lambda v: f"{100*v/tot:.1f}% del total"
    ok = R["estado"].startswith("Óptima")
    with o_res:
        display(HTML(
            "<div style='display:flex;gap:12px;flex-wrap:wrap;margin:6px 0'>"
            + kpi("Costo total (F.O.)", "$ " + ES(R["obj"], 0), PAL["total"], "valor de la función objetivo")
            + kpi("Estado", "✔ " + R["estado"] if ok else "⚠ " + R["estado"], PAL["total"] if ok else PAL["setup"],
                  f"{etiqueta} · {R['seg']:.1f} s") + "</div>"
            "<div style='display:flex;gap:12px;flex-wrap:wrap;margin:6px 0'>"
            + kpi("Producción", "$ " + ES(c["prod"], 0), PAL["prod"], pct(c["prod"]))
            + kpi("Preparación (setup)", "$ " + ES(c["setup"], 0), PAL["setup"], pct(c["setup"]))
            + kpi("Mantener inventario", "$ " + ES(c["mant"], 0), PAL["mant"], pct(c["mant"]))
            + kpi("Escasez", "$ " + ES(c["falt"], 0), PAL["falt"], pct(c["falt"])) + "</div>"))
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
                                 tabla_html(dfc, 300, 2, PAL["info"]) + nota, "⚖️")))
    pintar_tabla()
    T = d.T
    with o_plan:
        for nombre in d.productos:
            df = R["plan"][nombre].reset_index().rename(columns={"index": "Fuente"})
            display(HTML(seccion(f"{nombre} · Producción por fuente (unidades)", PAL["prod"],
                                 tabla_html(df, 380, 0, PAL["prod"], calor=True), "🏭")))
            df = R["inv"][nombre].reset_index().rename(columns={"index": ""})
            display(HTML(seccion(f"{nombre} · Inventarios y cumplimiento de demanda", PAL["mant"],
                                 tabla_html(df, 380, 0, PAL["mant"], calor=True), "📦")))
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


def exportar(_=None):
    R = S["R"]
    if not R: aviso("Primero resuelve el modelo."); return
    ruta = "plan_agregado.xlsx"
    with pd.ExcelWriter(ruta) as xw:
        pd.DataFrame({"Concepto": ["Estado", "Costo total", *R["costos"].keys()],
                      "Valor": [R["estado"], R["obj"], *R["costos"].values()]}).to_excel(xw, sheet_name="Resumen", index=False)
        R["tabla"].to_excel(xw, sheet_name="Solucion", index=False)
        for k, (n, df) in enumerate(R["plan"].items()):
            df.to_excel(xw, sheet_name=f"Plan {k+1}"); R["inv"][n].to_excel(xw, sheet_name=f"Inventarios {k+1}")
        R["util"].to_excel(xw, sheet_name="Utilizacion")
    try:
        from google.colab import files; files.download(ruta)
    except Exception:
        aviso(f"Archivo guardado como {ruta}", "#27ae60")


b_gen.on_click(generar_tablas); b_ej.on_click(cargar_ejemplo)
b_res.on_click(resolver_click); b_xls.on_click(exportar)

ui = W.VBox([
    W.HTML("<h2>🏭 Planeación agregada · multietapa · multifuente · multiproducto</h2>"
           "<p style='opacity:.8'>Modelo MILP (Kalenatic, López, González) generalizado. Cambia la estructura y pulsa "
           "<b>Generar tablas</b>.</p>"),
    W.HBox([W.VBox([w_prod, w_J, w_T, w_meses]), W.VBox([w_efic, w_tl, w_gap, w_modo, w_falt])]),
    W.HBox([b_gen, b_ej]), msg, caja_tablas, W.HBox([b_res, b_xls]), w_ver, res_tabs])
display(ui)
generar_tablas()
