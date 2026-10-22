#!/usr/bin/env python3
"""Simulateur de caméra pour la base déportée — Coupe de France de robotique 2027 (Camelot).

On décrit une ou plusieurs caméras et leur position dans un fichier de configuration (camera.toml,
à côté de ce script) ; le programme calcule ce que chaque caméra voit de la table et surtout la
RÉSOLUTION AU SOL (combien de mm de table couvre un pixel), point par point.

Lancer :
  python3 simulateur_camera.py                 # lit camera.toml à côté du script (créé s'il n'existe pas)
  python3 simulateur_camera.py autre.toml      # lit un autre fichier de configuration

Sorties :
  - pour chaque caméra active, une figure à 4 panneaux : résolution au sol, taille d'un marqueur ArUco
    dans l'image, vue depuis la caméra (cadrage), résumé chiffré ;
  - des statistiques dans le terminal : champ, pixels par degré, couverture, pire marqueur,
    détail aux carrières et aux sites de construction, marqueurs fixes vus, netteté, flou de bougé ;
  - avec plusieurs caméras actives, un tableau comparatif.

Repère : celui de la stratégie (origine dans un coin de la table, x le long des 3 m vers le côté bleu,
y le long des 2 m, z vers le haut). La base déportée jaune est en (1,275 ; 2,12).
Visée : CAP (0° = vers x croissant, -90° = droit vers la table depuis la base) et INCLINAISON
sous l'horizontale (90° = verticale).

Limites du modèle : caméra parfaite (sténopé), sans distorsion d'objectif ni occultation par les robots
ou les piles. La distorsion réelle réduit un peu la résolution sur les bords de l'image.
Prérequis : Python 3.11 ou plus (tomllib), numpy, matplotlib.
"""

import math
import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:          # Python < 3.11 : pip install tomli
    import tomli as tomllib

import numpy as np

# ═══════════════════════════════════════════════════════════════════ données

# Caméras connues : résolution max (px), champ horizontal × vertical (°), taille de pixel (µm),
# ouverture (f/…). Valeurs constructeur, à vérifier sur la fiche du modèle exact.
PRESETS = {
    'imx519': dict(nom='Arducam IMX519 16 Mpx (B0386)', res=(4656, 3496), champ=(66.0, 49.5), pixel=1.22),
    'imx219': dict(nom='IMX219 8 Mpx (Pi Camera v2, objectif standard)', res=(3280, 2464), champ=(62.2, 48.8), pixel=1.12, ouverture=2.0),
    'cm3': dict(nom='Raspberry Pi Camera Module 3 (IMX708)', res=(4608, 2592), champ=(66.0, 41.0), pixel=1.4, ouverture=1.8),
    'cm3_wide': dict(nom='Raspberry Pi Camera Module 3 Wide (IMX708)', res=(4608, 2592), champ=(102.0, 67.0), pixel=1.4, ouverture=2.2),
    'ov5647_21': dict(nom='OV5647 5 Mpx + objectif M12 2,1 mm (champ estimé)', res=(2592, 1944), champ=(82.0, 66.0), pixel=1.4, ouverture=2.0),
    'webcam_1080p': dict(nom='Webcam 1080p générique ~78° diagonal', res=(1920, 1080), champ=(70.0, 43.0), pixel=3.0, ouverture=2.0),
}

TABLE_X, TABLE_Y = 3.0, 2.0
BASE_JAUNE = (1.275, 2.12)            # milieu de notre base déportée (plan général du règlement)
MARQUEUR_PIERRE = 0.070               # m : ArUco sur les grandes faces des pierres
MARQUEUR_FIXE = 0.100                 # m : ArUco fixes de la table

# Zones à surveiller (x_min, x_max, y_min, y_max)
ZONES = {
    'jaune': (0.0, 1.70, 0.0, 2.0),   # notre moitié + carrières du centre (x = 1.5 ± 0.16)
    'table': (0.0, 3.0, 0.0, 2.0),
}
HAUTEURS = (0.0, 0.11, 0.33)          # sol, dessus d'une pierre, dessus d'une pile de 3

# Pierres en carrière (x, y, angle de la grande longueur) et sites de construction de notre côté
CARRIERES = {
    'centre_sud': (1.500, 0.505, 90), 'centre_nord': (1.500, 1.495, 90),
    'bord_sud_1': (0.750, 0.057, 0), 'bord_sud_2': (1.250, 0.057, 0),
    'bord_sud_3': (1.750, 0.057, 0), 'bord_sud_4': (2.250, 0.057, 0),
    'bord_nord_1': (0.750, 1.943, 0), 'bord_nord_2': (1.250, 1.943, 0),
    'bord_nord_3': (1.750, 1.943, 0), 'bord_nord_4': (2.250, 1.943, 0),
}
SITES = {
    'mur_nord': (0.250, 1.577), 'mur_sud': (0.250, 0.423),
    'tour_nord_ext': (0.496, 1.590), 'tour_nord_int': (0.843, 1.248),
    'tour_sud_int': (0.843, 0.752), 'tour_sud_ext': (0.496, 0.410),
}
MARQUEURS_FIXES = {20: (0.6, 1.4), 21: (2.4, 1.4), 22: (0.6, 0.6), 23: (2.4, 0.6)}


# ═══════════════════════════════════════════════════════════════════ modèle de caméra

class Camera:
    """Caméra sténopé : position (m), cap / inclinaison / roulis (°), résolution (px), champ (°)."""

    def __init__(self, position, cap, inclinaison, roulis, res, champ):
        self.C = np.asarray(position, dtype=float)
        self.W, self.H = res
        self.hf, self.vf = champ
        self.fx = (self.W / 2) / math.tan(math.radians(self.hf / 2))
        self.fy = (self.H / 2) / math.tan(math.radians(self.vf / 2))
        self.orienter(cap, inclinaison, roulis)

    def orienter(self, cap, inclinaison, roulis):
        self.cap, self.inclinaison, self.roulis = cap, inclinaison, roulis
        y, p, r = math.radians(cap), -math.radians(inclinaison), math.radians(roulis)
        axe = np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)])
        dx = np.cross(axe, [0.0, 0.0, 1.0])
        dx /= np.linalg.norm(dx)
        dy = np.cross(dx, axe)
        c, s = math.cos(r), math.sin(r)
        self.axe, self.ex, self.ey = axe, c * dx + s * dy, -s * dx + c * dy

    def projeter(self, P):
        """Points (N,3) -> pixels (N,2) et profondeur (N,). Profondeur <= 0 : derrière la caméra."""
        v = P - self.C
        z = v @ self.axe
        zs = np.where(z > 1e-6, z, np.nan)
        u = self.W / 2 + self.fx * (v @ self.ex) / zs
        w = self.H / 2 - self.fy * (v @ self.ey) / zs
        return np.column_stack([u, w]), z

    def dans_image(self, uv, z, marge=0.0):
        m = marge * min(self.W, self.H)
        return ((z > 0) & (uv[:, 0] >= m) & (uv[:, 0] <= self.W - m)
                & (uv[:, 1] >= m) & (uv[:, 1] <= self.H - m))

    def resolution(self, P):
        """Pour des points d'un plan horizontal : (px par m dans la meilleure direction,
        px par m dans la pire direction, dans l'image ?). Jacobienne de la projection plan -> image."""
        e = 1e-4
        uv, z = self.projeter(P)
        ux, _ = self.projeter(P + [e, 0, 0])
        uy, _ = self.projeter(P + [0, e, 0])
        a, c = (ux - uv).T / e
        b, d = (uy - uv).T / e
        t = a * a + b * b + c * c + d * d
        det = a * d - b * c
        r = np.sqrt(np.maximum(t * t - 4 * det * det, 0))
        s_max = np.sqrt((t + r) / 2)
        s_min = np.sqrt(np.maximum((t - r) / 2, 0))
        return s_max, s_min, self.dans_image(uv, z)


# ═══════════════════════════════════════════════════════════════════ calculs

def grille_zone(zone, n=(36, 24), hauteurs=HAUTEURS):
    x0, x1, y0, y1 = zone
    return np.array([[x, y, h] for x in np.linspace(x0 + 0.01, x1 - 0.01, n[0])
                     for y in np.linspace(y0 + 0.01, y1 - 0.01, n[1]) for h in hauteurs])


def couverture(cam, pts, seuil, taille=MARQUEUR_PIERRE):
    """(part des points dans l'image, part des points où un marqueur est lisible, plus petit marqueur vu en px)."""
    s_max, s_min, dedans = cam.resolution(pts)
    dedans &= cam.dans_image(*cam.projeter(pts), marge=0.01)
    px_min = s_min * taille
    lisible = dedans & (px_min >= seuil)
    pire = np.nanmin(px_min[dedans]) if dedans.any() else 0.0
    return dedans.mean(), lisible.mean(), pire


def meilleure_visee(cam, zone, seuil, roulis_possibles=(0, 90)):
    """Cherche cap / inclinaison / roulis qui rendent lisible la plus grande part de la zone,
    puis qui maximisent le plus petit marqueur vu. Recherche grossière puis fine."""
    pts = grille_zone(zone, n=(26, 18))
    cx = (zone[0] + zone[1]) / 2 - cam.C[0]
    cy = (zone[2] + zone[3]) / 2 - cam.C[1]
    cap0 = math.degrees(math.atan2(cy, cx))
    meilleur = None

    def essai(cap, inc, rou):
        nonlocal meilleur
        cam.orienter(cap, inc, rou)
        _, lis, pire = couverture(cam, pts, seuil)
        score = (round(lis, 3), pire)
        if meilleur is None or score > meilleur[0]:
            meilleur = (score, cap, inc, rou)

    for rou in roulis_possibles:
        for cap in np.arange(cap0 - 60, cap0 + 61, 4):
            for inc in np.arange(10, 90, 3):
                essai(cap, inc, rou)
    _, cap_b, inc_b, rou_b = meilleur
    for cap in np.arange(cap_b - 4, cap_b + 4.1, 1):
        for inc in np.arange(max(5, inc_b - 3), min(90, inc_b + 3.1), 0.5):
            essai(cap, inc, rou_b)
    _, cap, inc, rou = meilleur
    cam.orienter(cap, inc, rou)
    return cap, inc, rou


def champ_depuis_optique(res, focale_mm, pixel_um):
    lx = res[0] * pixel_um / 1000
    ly = res[1] * pixel_um / 1000
    return (2 * math.degrees(math.atan(lx / (2 * focale_mm))), 2 * math.degrees(math.atan(ly / (2 * focale_mm))))


def focale_depuis_champ(res, champ, pixel_um):
    return res[0] * pixel_um / 1000 / (2 * math.tan(math.radians(champ[0] / 2)))


def point_info(cam, x, y, z, taille):
    P = np.array([[x, y, z]])
    s_max, s_min, dedans = cam.resolution(P)
    v = P[0] - cam.C
    D = float(np.linalg.norm(v))
    angle = math.degrees(math.asin(-v[2] / D)) if D > 0 else 90.0
    if not dedans[0]:
        return dict(vu=False, D=D, angle=angle)
    return dict(vu=True, D=D, angle=angle, mm_bon=1000 / s_max[0], mm_mauvais=1000 / s_min[0],
                px=(taille * s_max[0], taille * s_min[0]))


def analyser(cam, zone, seuil, ouverture, pixel_um, vitesse, pose_ms):
    """Toutes les statistiques chiffrées, dans un dictionnaire."""
    st = {}
    st['px_par_degre'] = cam.W / cam.hf
    st['f_px'] = cam.fx
    # couverture de la zone, au sol / dessus d'une pierre / dessus d'une pile
    for h in HAUTEURS:
        st[f'couverture_{h}'] = couverture(cam, grille_zone(zone, hauteurs=(h,)), seuil)
    tous = grille_zone(zone)
    st['couverture'] = couverture(cam, tous, seuil)
    # résolution au sol dans la zone (sol)
    sol = grille_zone(zone, n=(90, 60), hauteurs=(0.0,))
    s_max, s_min, dedans = cam.resolution(sol)
    mm = 1000 / s_min[dedans]
    st['mm_px'] = (mm.min(), np.median(mm), np.percentile(mm, 95), mm.max()) if dedans.any() else None
    D = np.linalg.norm(sol[dedans] - cam.C, axis=1)
    st['distances'] = (D.min(), D.max()) if dedans.any() else None
    # marqueurs fixes
    st['fixes'] = {i: point_info(cam, x, y, 0.0, MARQUEUR_FIXE) for i, (x, y) in MARQUEURS_FIXES.items()}
    # lieux de jeu (dessus des piles et des constructions)
    st['lieux'] = {}
    for n, (x, y, _) in CARRIERES.items():
        st['lieux']['carrière ' + n] = point_info(cam, x, y, 0.33, MARQUEUR_PIERRE)
    for n, (x, y) in SITES.items():
        st['lieux'][n] = point_info(cam, x, y, 0.33 if n.startswith('mur') else 0.32, MARQUEUR_PIERRE)
    # netteté (hyperfocale) et flou de bougé
    if pixel_um:
        f_mm = focale_depuis_champ((cam.W, cam.H), (cam.hf, cam.vf), pixel_um)
        c_mm = 2 * pixel_um / 1000                     # flou acceptable : 2 pixels
        hyper = f_mm * f_mm / (ouverture * c_mm) + f_mm
        st['optique'] = dict(focale=f_mm, hyper=hyper / 1000, ouverture=ouverture,
                             capteur=(cam.W * pixel_um / 1000, cam.H * pixel_um / 1000))
    if st['distances']:
        d_med = float(np.median(D))
        st['flou'] = vitesse * pose_ms / 1000 * cam.fx / d_med
    return st


# ═══════════════════════════════════════════════════════════════════ affichage

def fmt_pct(v):
    return f'{100 * v:.0f} %'


def afficher(nom, cam, st, zone_nom, seuil, vitesse, pose_ms):
    print(f'\n══ {nom}')
    print(f'  Image {cam.W} × {cam.H} px ({cam.W * cam.H / 1e6:.1f} Mpx), champ {cam.hf:.1f}° × {cam.vf:.1f}°, '
          f'{st["px_par_degre"]:.0f} px/°, focale {st["f_px"]:.0f} px')
    if 'optique' in st:
        o = st['optique']
        print(f'  Capteur {o["capteur"][0]:.2f} × {o["capteur"][1]:.2f} mm, focale ≈ {o["focale"]:.2f} mm, '
              f'hyperfocale ≈ {o["hyper"]:.1f} m à f/{o["ouverture"]:g} (net de {o["hyper"] / 2:.1f} m à l\'infini '
              f'si mise au point sur l\'hyperfocale)')
    print(f'  Caméra en ({cam.C[0]:.3f} ; {cam.C[1]:.3f}), {cam.C[2]:.2f} m au-dessus de la table ; '
          f'visée cap {cam.cap:.1f}°, inclinaison {cam.inclinaison:.1f}° sous l\'horizontale, '
          f'{"portrait" if abs(cam.roulis) == 90 else "paysage" if cam.roulis == 0 else f"roulis {cam.roulis:g}°"}')
    print(f'  Zone « {zone_nom} » — dans le champ / marqueur de 70 mm lisible (≥ {seuil:g} px) :')
    for h in HAUTEURS:
        dedans, lis, pire = st[f'couverture_{h}']
        lib = {0.0: 'au sol', 0.11: 'dessus d\'une pierre', 0.33: 'dessus d\'une pile de 3'}[h]
        print(f'    {lib:24s} {fmt_pct(dedans):>5s} / {fmt_pct(lis):>5s}   (plus petit marqueur vu : {pire:.0f} px)')
    if st['mm_px']:
        a, med, p95, b = st['mm_px']
        print(f'  Résolution au sol (direction la moins bonne) : de {a:.2f} à {b:.2f} mm/px, médiane {med:.2f}, '
              f'95 % de la zone sous {p95:.2f} mm/px')
        print(f'  Distances caméra-table : {st["distances"][0]:.2f} à {st["distances"][1]:.2f} m')
    if 'flou' in st:
        print(f'  Flou de bougé d\'un robot à {vitesse:g} m/s, pose de {pose_ms:g} ms : ≈ {st["flou"]:.0f} px')
    vus = [i for i, f in st['fixes'].items() if f['vu']]
    print(f'  Marqueurs fixes de la table vus : {vus or "aucun"}'
          + ('  → se repère seule' if len(vus) >= 2 else '  → ⚠ moins de 2 : ne peut pas se repérer seule'))
    print('  Détail (dessus des piles / constructions) :')
    print(f'    {"lieu":28s} {"distance":>8s} {"angle":>6s} {"mm/px":>13s} {"marqueur 70 mm":>16s}')
    for n, p in st['lieux'].items():
        if p['vu']:
            print(f'    {n:28s} {p["D"]:7.2f} m {p["angle"]:5.0f}° {p["mm_bon"]:5.2f} × {p["mm_mauvais"]:4.2f}'
                  f' {p["px"][0]:7.0f} × {p["px"][1]:3.0f} px' + ('  ⚠' if p['px'][1] < seuil else ''))
        else:
            print(f'    {n:28s} {p["D"]:7.2f} m {p["angle"]:5.0f}°   hors champ')


def figure(nom, cam, st, zone, zone_nom, seuil, fichier, tapis=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle

    xs = np.linspace(0, TABLE_X, 151)
    ys = np.linspace(0, TABLE_Y, 101)
    X, Y = np.meshgrid(xs, ys)

    def carte(h, taille=None):
        P = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, h)])
        s_max, s_min, dedans = cam.resolution(P)
        val = 1000 / s_min if taille is None else taille * s_min
        return np.where(dedans, val, np.nan).reshape(X.shape), dedans.reshape(X.shape)

    def decor(ax, titre):
        if tapis:
            try:
                ax.imshow(plt.imread(tapis), extent=(0, TABLE_X, 0, TABLE_Y), alpha=0.35, zorder=0)
            except OSError:
                pass
        ax.add_patch(Rectangle((0, 0), TABLE_X, TABLE_Y, fill=False, lw=2, zorder=4))
        ax.add_patch(Rectangle((zone[0], zone[2]), zone[1] - zone[0], zone[3] - zone[2], fill=False,
                               ls='--', lw=1.2, ec='#dc2626', zorder=4))
        for (x, y, a) in CARRIERES.values():
            c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
            pts = [(x + c * dx - s * dy, y + s * dx + c * dy)
                   for dx, dy in ((-.16, -.055), (.16, -.055), (.16, .055), (-.16, .055))]
            ax.add_patch(Polygon(pts, fill=False, ec='k', lw=0.8, zorder=4))
        for (x, y) in SITES.values():
            ax.plot(x, y, 'o', ms=4, mfc='white', mec='k', zorder=5)
        for i, (x, y) in MARQUEURS_FIXES.items():
            ax.add_patch(Rectangle((x - .05, y - .05), .1, .1, fc='k', zorder=5))
            ax.text(x, y + .07, str(i), ha='center', fontsize=7, zorder=5)
        ax.plot(cam.C[0], cam.C[1], 'r^', ms=11, clip_on=False, zorder=6)
        ax.set_xlim(-0.05, TABLE_X + 0.05)
        ax.set_ylim(-0.05, max(TABLE_Y, cam.C[1]) + 0.12)
        ax.set_aspect('equal')
        ax.set_title(titre, fontsize=10)

    fig, axs = plt.subplots(2, 2, figsize=(16, 11))
    # 1. résolution au sol
    ax = axs[0, 0]
    res, dedans = carte(0.0)
    zx = (X >= zone[0]) & (X <= zone[1]) & (Y >= zone[2]) & (Y <= zone[3]) & dedans
    vals = res[zx] if zx.any() else res[dedans]
    bas, haut = np.nanmin(vals), np.nanpercentile(vals, 99)
    pas = [p for p in (0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5) if (haut - bas) / p <= 12][0]
    niveaux = list(np.arange(math.floor(bas / pas) * pas, math.ceil(haut / pas) * pas + pas / 2, pas))
    cs = ax.contourf(X, Y, res, levels=niveaux, cmap='viridis_r', alpha=0.8, zorder=1, extend='max')
    traits = [n for n in niveaux[1:-1:2] if np.nanmin(res) < n < np.nanmax(res)]
    if traits:                         # étiquettes seulement s'il y a des lignes (vieux matplotlib)
        cl = ax.contour(X, Y, res, levels=traits, colors='k', linewidths=0.7, zorder=2)
        ax.clabel(cl, fmt={n: f'{n:.2g} mm' for n in traits}, fontsize=7)
    ax.contourf(X, Y, np.where(dedans, np.nan, 1), levels=[0.5, 1.5], colors='none', hatches=['//'], zorder=2)
    fig.colorbar(cs, ax=ax, shrink=0.85).set_label('mm de table par pixel (direction la moins bonne)')
    decor(ax, 'Résolution au sol (au niveau du tapis) — hachuré : hors champ')
    # 2. taille d'un marqueur de pierre
    ax = axs[0, 1]
    px, dedans = carte(0.11, MARQUEUR_PIERRE)
    from matplotlib.colors import BoundaryNorm, ListedColormap
    bornes = [k * seuil for k in (0, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12)]
    couleurs = ['#991b1b', '#dc2626', '#f97316'] + [plt.cm.Greens(t) for t in np.linspace(0.25, 0.95, len(bornes) - 3)]
    palette = ListedColormap(couleurs[:-1])
    palette.set_over(couleurs[-1])
    cs = ax.contourf(X, Y, px, levels=bornes, cmap=palette, norm=BoundaryNorm(bornes, len(bornes) - 1),
                     alpha=0.8, zorder=1, extend='max')
    if np.nanmin(px) < seuil < np.nanmax(px):   # la ligne du seuil n'existe que si on passe dessous
        cl = ax.contour(X, Y, px, levels=[seuil], colors='#b91c1c', linewidths=2, zorder=3)
        ax.clabel(cl, fmt={seuil: f'seuil {seuil:g} px'}, fontsize=8)
    ax.contourf(X, Y, np.where(dedans, np.nan, 1), levels=[0.5, 1.5], colors='none', hatches=['//'], zorder=2)
    fig.colorbar(cs, ax=ax, shrink=0.85).set_label(f'côté le plus court d\'un ArUco de 70 mm (px) — rouge : sous le seuil de {seuil:g} px')
    decor(ax, 'Taille d\'un marqueur de pierre (dessus d\'une pierre couchée)')
    # 3. vue caméra
    ax = axs[1, 0]
    ax.add_patch(Rectangle((0, 0), cam.W, cam.H, fc='#f3f4f6', ec='k', lw=1.5))

    def trace(points, **kw):
        uv, z = cam.projeter(np.asarray(points, dtype=float))
        ok = (z > 0)
        if ok.sum() >= 2:
            ax.plot(uv[ok, 0], uv[ok, 1], **kw)

    for gx in np.arange(0, TABLE_X + 1e-9, 0.5):
        trace([[gx, y, 0] for y in np.linspace(0, TABLE_Y, 60)], color='#cbd5e1', lw=0.7)
    for gy in np.arange(0, TABLE_Y + 1e-9, 0.5):
        trace([[x, gy, 0] for x in np.linspace(0, TABLE_X, 90)], color='#cbd5e1', lw=0.7)
    bord = [[0, 0, 0], [TABLE_X, 0, 0], [TABLE_X, TABLE_Y, 0], [0, TABLE_Y, 0], [0, 0, 0]]
    trace([p for a, b in zip(bord, bord[1:]) for p in np.linspace(a, b, 60)], color='k', lw=1.5)
    zb = [[zone[0], zone[2], 0], [zone[1], zone[2], 0], [zone[1], zone[3], 0], [zone[0], zone[3], 0], [zone[0], zone[2], 0]]
    trace([p for a, b in zip(zb, zb[1:]) for p in np.linspace(a, b, 40)], color='#dc2626', ls='--', lw=1.2)
    for (x, y, a) in CARRIERES.values():
        c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
        for h, col in ((0.0, '#9ca3af'), (0.33, '#b45309')):
            pts = [[x + c * dx - s * dy, y + s * dx + c * dy, h]
                   for dx, dy in ((-.16, -.055), (.16, -.055), (.16, .055), (-.16, .055), (-.16, -.055))]
            trace(pts, color=col, lw=1)
    for i, (x, y) in MARQUEURS_FIXES.items():
        uv, z = cam.projeter(np.array([[x, y, 0.0]]))
        if z[0] > 0:
            ax.plot(*uv[0], 's', color='k', ms=5)
            ax.text(uv[0, 0], uv[0, 1] - 40, str(i), ha='center', fontsize=8)
    ax.set_xlim(-0.15 * cam.W, 1.15 * cam.W)
    ax.set_ylim(1.15 * cam.H, -0.15 * cam.H)
    ax.set_aspect('equal')
    ax.set_title('Vue depuis la caméra (gris : table, grille 50 cm ; orange : dessus des piles)', fontsize=10)
    # 4. résumé
    ax = axs[1, 1]
    ax.axis('off')
    lignes = [nom, '',
              f'Image : {cam.W} × {cam.H} px ({cam.W * cam.H / 1e6:.1f} Mpx), champ {cam.hf:.1f}° × {cam.vf:.1f}°',
              f'Pixels par degré : {st["px_par_degre"]:.0f}     focale : {st["f_px"]:.0f} px',
              f'Position : ({cam.C[0]:.2f} ; {cam.C[1]:.2f}), {cam.C[2]:.2f} m au-dessus de la table',
              f'Visée : cap {cam.cap:.0f}°, inclinaison {cam.inclinaison:.1f}°, '
              + ('portrait' if abs(cam.roulis) == 90 else 'paysage'), '',
              f'Zone « {zone_nom} » (dans le champ / lisible ≥ {seuil:g} px) :']
    for h in HAUTEURS:
        d, l, p = st[f'couverture_{h}']
        lib = {0.0: 'au sol', 0.11: 'dessus 1 pierre', 0.33: 'dessus pile de 3'}[h]
        lignes.append(f'   {lib:18s} {fmt_pct(d):>5s} / {fmt_pct(l):>5s}   (min {p:.0f} px)')
    if st['mm_px']:
        a, med, p95, b = st['mm_px']
        lignes += ['', f'Résolution au sol : {a:.2f} à {b:.2f} mm/px (médiane {med:.2f})',
                   f'Distances : {st["distances"][0]:.2f} à {st["distances"][1]:.2f} m']
    if 'optique' in st:
        o = st['optique']
        lignes.append(f'Focale ≈ {o["focale"]:.2f} mm, hyperfocale ≈ {o["hyper"]:.1f} m (f/{o["ouverture"]:g})')
    if 'flou' in st:
        lignes.append(f'Flou d\'un robot en mouvement : ≈ {st["flou"]:.0f} px')
    vus = [i for i, f in st['fixes'].items() if f['vu']]
    lignes.append(f'Marqueurs fixes vus : {vus or "aucun"}' + ('' if len(vus) >= 2 else '  ⚠'))
    lignes += ['', 'Pire lieu de jeu :']
    vus_l = {n: p for n, p in st['lieux'].items() if p['vu']}
    hors = [n for n, p in st['lieux'].items() if not p['vu']]
    if vus_l:
        n, p = min(vus_l.items(), key=lambda t: t[1]['px'][1])
        lignes.append(f'   {n} : {p["mm_mauvais"]:.2f} mm/px, marqueur {p["px"][0]:.0f} × {p["px"][1]:.0f} px')
    if hors:
        import textwrap
        lignes += textwrap.wrap('Hors champ : ' + ', '.join(n.replace('carrière ', '') for n in hors), 62)
    ax.text(0, 1, '\n'.join(lignes), va='top', family='monospace', fontsize=9.5)
    fig.suptitle('Simulation caméra — base déportée Camelot 2027', fontsize=13)
    fig.tight_layout()
    fig.savefig(fichier, dpi=110)
    print(f'\n  Figure : {fichier}')




# ═══════════════════════════════════════════════════════════════════ configuration

FICHIER_CONFIG = 'camera.toml'

CONFIG_DEFAUT = '''\
# ═══ Simulateur caméra — base déportée Camelot 2027 ═══
# Lancer : python3 simulateur_camera.py             (lit ce fichier, posé à côté du script)
#          python3 simulateur_camera.py autre.toml
# Repère de la table : origine dans un coin, x le long des 3 m (vers le côté bleu), y le long des 2 m.

[position]                 # où est la caméra (m)
x = 1.275                  # milieu de notre base déportée (jaune)
y = 2.12
hauteur = 2.0              # au-dessus du tapis

[visee]
auto = true                # cherche la meilleure orientation pour la zone
portrait = true            # en auto : peut aussi essayer la caméra tournée de 90°
# Utilisés seulement si auto = false :
cap = -100.0               # 0° = vers x croissant, -90° = droit vers la table depuis la base
inclinaison = 61.0         # sous l'horizontale (90° = verticale)
roulis = 0.0               # 0 = paysage, 90 = portrait

[analyse]
zone = "jaune"             # "jaune" (notre moitié + carrières du centre), "table",
                           # ou une zone [x_min, x_max, y_min, y_max] en m
seuil_px = 20              # taille mini d'un marqueur ArUco lisible (px) ; 30 = confortable
vitesse_robot = 1.0        # m/s, pour estimer le flou de bougé
pose_ms = 10               # temps de pose (ms)

[sortie]
dossier = "."              # où écrire les figures (relatif à ce fichier)
tapis = ""                 # image du tapis recadrée sur la table, en fond des cartes ("" = sans)

# ─── Caméras : une section [[camera]] par caméra ───────────────────────────────────────────────────
# Pour chaque caméra, au choix :
#   - modele = "..." : imx519, imx219, cm3, cm3_wide, ov5647_21, webcam_1080p
#     (valeurs constructeur, à vérifier ; toute valeur écrite en dessous remplace celle du modèle) ;
#   - resolution + champ ;
#   - resolution + focale_mm + pixel_um (le champ est alors calculé).
# Optionnel : nom, actif (false pour l'ignorer), pixel_um et ouverture (pour la netteté),
#             fichier (nom de la figure), visee = { auto = false, cap = ..., inclinaison = ..., roulis = ... }
# Avec plusieurs caméras actives, un tableau comparatif est affiché en plus.

[[camera]]
nom = "Arducam IMX519 16 Mpx (B0386)"
resolution = [4656, 3496]  # px
champ = [66.0, 49.5]       # ° horizontal, vertical
pixel_um = 1.22            # taille d'un pixel (µm)
# ouverture = 2.0          # f/N : non indiquée sur la fiche, f/2 supposée

[[camera]]
nom = "Exemple : 12 Mpx, objectif 3,2 mm"
actif = false
resolution = [4056, 3040]
focale_mm = 3.2
pixel_um = 1.55

[[camera]]
modele = "cm3_wide"
actif = false
'''

CLES = {
    'position': {'x', 'y', 'hauteur'},
    'visee': {'auto', 'portrait', 'cap', 'inclinaison', 'roulis'},
    'analyse': {'zone', 'seuil_px', 'vitesse_robot', 'pose_ms'},
    'sortie': {'dossier', 'tapis'},
    'camera': {'nom', 'modele', 'actif', 'resolution', 'champ', 'focale_mm', 'pixel_um', 'ouverture', 'fichier', 'visee'},
}


class ErreurConfig(Exception):
    pass


def verifier_cles(section, valeurs, autorisees):
    inconnues = set(valeurs) - autorisees
    if inconnues:
        raise ErreurConfig(f'[{section}] : clé(s) inconnue(s) {", ".join(sorted(inconnues))} '
                           f'(possibles : {", ".join(sorted(autorisees))})')


def nombres(valeur, n, quoi):
    if not (isinstance(valeur, list) and len(valeur) == n and all(isinstance(v, (int, float)) for v in valeur)):
        raise ErreurConfig(f'{quoi} doit être une liste de {n} nombres, par exemple {[1] * n}')
    return [float(v) for v in valeur]


def lire_config(chemin):
    try:
        with open(chemin, 'rb') as f:
            cfg = tomllib.load(f)
    except tomllib.TOMLDecodeError as err:
        raise ErreurConfig(f'{chemin} : erreur de syntaxe TOML — {err}') from err
    for section in cfg:
        if section not in CLES:
            raise ErreurConfig(f'section [{section}] inconnue (possibles : {", ".join(CLES)})')
    for section in ('position', 'visee', 'analyse', 'sortie'):
        verifier_cles(section, cfg.get(section, {}), CLES[section])
    cameras = cfg.get('camera', [])
    if isinstance(cameras, dict):
        cameras = [cameras]
    if not cameras:
        raise ErreurConfig('aucune caméra : ajoute au moins une section [[camera]]')
    for i, c in enumerate(cameras, 1):
        verifier_cles(f'camera n°{i}', c, CLES['camera'])
    return cfg, cameras


def decrire_camera(c, i):
    """Caractéristiques complètes d'une caméra de la config (modèle éventuel + valeurs écrites)."""
    base = {}
    if 'modele' in c:
        if c['modele'] not in PRESETS:
            raise ErreurConfig(f'camera n°{i} : modèle « {c["modele"]} » inconnu (connus : {", ".join(PRESETS)})')
        base = PRESETS[c['modele']]
    res = nombres(c['resolution'], 2, f'camera n°{i} : resolution') if 'resolution' in c else base.get('res')
    pixel = c.get('pixel_um', base.get('pixel'))
    if 'pixel_um' not in c and base.get('pixel') and 'resolution' in c:
        pixel = base['pixel'] * base['res'][0] / res[0]   # mode réduit (binning) : même capteur, pixels regroupés
    if 'champ' in c:
        champ = nombres(c['champ'], 2, f'camera n°{i} : champ')
    elif 'focale_mm' in c:
        if not (res and pixel):
            raise ErreurConfig(f'camera n°{i} : focale_mm demande aussi resolution et pixel_um')
        champ = champ_depuis_optique(res, float(c['focale_mm']), float(pixel))
    else:
        champ = base.get('champ')
    if not res or not champ:
        raise ErreurConfig(f'camera n°{i} : il faut modele, ou resolution + champ, ou resolution + focale_mm + pixel_um')
    if not (0 < champ[0] < 179 and 0 < champ[1] < 179):
        raise ErreurConfig(f'camera n°{i} : champ {champ} hors limites (entre 0 et 179°)')
    nom = c.get('nom') or base.get('nom') or f'Caméra {int(res[0])}×{int(res[1])}, {champ[0]:.0f}°×{champ[1]:.0f}°'
    ouverture = c.get('ouverture', base.get('ouverture'))
    return dict(nom=nom, res=(int(res[0]), int(res[1])), champ=tuple(champ), pixel=pixel,
                ouverture=ouverture, fichier=c.get('fichier'), visee=c.get('visee', {}))


def zone_depuis(valeur):
    if isinstance(valeur, str):
        if valeur not in ZONES:
            raise ErreurConfig(f'[analyse] zone « {valeur} » inconnue (« jaune », « table » ou [x_min, x_max, y_min, y_max])')
        return valeur, ZONES[valeur]
    z = nombres(valeur, 4, '[analyse] zone')
    if not (z[0] < z[1] and z[2] < z[3]):
        raise ErreurConfig('[analyse] zone : il faut x_min < x_max et y_min < y_max')
    return f'x {z[0]:g}–{z[1]:g}, y {z[2]:g}–{z[3]:g}', tuple(z)


def nom_fichier(nom):
    import unicodedata
    simple = unicodedata.normalize('NFKD', nom).encode('ascii', 'ignore').decode().lower()
    return 'simulation_' + (re.sub(r'[^a-z0-9]+', '_', simple).strip('_')[:40] or 'camera') + '.png'


# ═══════════════════════════════════════════════════════════════════ lancement

def main():
    chemin = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().with_name(FICHIER_CONFIG)
    if len(sys.argv) > 1 and sys.argv[1] in ('-h', '--help'):
        print(__doc__)
        return
    if not chemin.exists():
        if len(sys.argv) > 1:
            sys.exit(f'Fichier de configuration introuvable : {chemin}')
        chemin.write_text(CONFIG_DEFAUT, encoding='utf-8')
        print(f'{chemin.name} n\'existait pas : un exemple a été créé ({chemin}).\n'
              'Modifie-le si besoin ; simulation avec ces valeurs :')
    try:
        cfg, cameras = lire_config(chemin)
        pos = cfg.get('position', {})
        position = (float(pos.get('x', BASE_JAUNE[0])), float(pos.get('y', BASE_JAUNE[1])), float(pos.get('hauteur', 2.0)))
        visee_g = cfg.get('visee', {})
        an = cfg.get('analyse', {})
        zone_nom, zone = zone_depuis(an.get('zone', 'jaune'))
        seuil = float(an.get('seuil_px', 20))
        vitesse = float(an.get('vitesse_robot', 1.0))
        pose_ms = float(an.get('pose_ms', 10))
        so = cfg.get('sortie', {})
        dossier = (chemin.parent / so.get('dossier', '.')).resolve()
        tapis = so.get('tapis') or None
        if tapis:
            tapis = (chemin.parent / tapis).resolve()
            if not tapis.exists():
                print(f'  ⚠ image du tapis introuvable ({tapis}) : cartes sans fond')
                tapis = None
        actives = [(i, c) for i, c in enumerate(cameras, 1) if c.get('actif', True)]
        if not actives:
            raise ErreurConfig('aucune caméra active (actif = true)')
        descriptions = [(i, decrire_camera(c, i)) for i, c in actives]
    except ErreurConfig as err:
        sys.exit(f'Erreur dans {chemin.name} : {err}')

    dossier.mkdir(parents=True, exist_ok=True)
    print(f'Configuration : {chemin}')
    resultats = []
    for i, d in descriptions:
        visee = {**visee_g, **d['visee']}
        if not d['ouverture']:
            print(f'  ({d["nom"]} : ouverture non indiquée, f/2 supposée pour la netteté)')
        cam = Camera(position, 0, 45, 0, d['res'], d['champ'])
        if visee.get('auto', True):
            meilleure_visee(cam, zone, seuil, (0, 90) if visee.get('portrait', True) else (0,))
        else:
            cam.orienter(float(visee.get('cap', -90)), float(visee.get('inclinaison', 60)), float(visee.get('roulis', 0)))
        st = analyser(cam, zone, seuil, d['ouverture'] or 2.0, d['pixel'], vitesse, pose_ms)
        afficher(d['nom'], cam, st, zone_nom, seuil, vitesse, pose_ms)
        fichier = dossier / (d['fichier'] or nom_fichier(d['nom']))
        figure(d['nom'], cam, st, zone, zone_nom, seuil, str(fichier), str(tapis) if tapis else None)
        resultats.append((d['nom'], cam, st))

    if len(resultats) > 1:
        lignes = []
        for nom, cam, st in resultats:
            _, lis, pire = st['couverture']
            vu = st['couverture'][0]
            mm = st['mm_px']
            lignes.append((nom[:34], f'{cam.W * cam.H / 1e6:.1f}', f'{cam.hf:.0f}×{cam.vf:.0f}', f'{st["px_par_degre"]:.0f}',
                           f'{100 * vu:.0f} %', f'{100 * lis:.0f} %', f'{pire:.0f}',
                           f'{mm[1]:.2f}' if mm else '-', f'{mm[3]:.2f}' if mm else '-',
                           str(sum(f['vu'] for f in st['fixes'].values())),
                           f'{cam.cap:.0f}/{cam.inclinaison:.0f}' + ('P' if abs(cam.roulis) == 90 else '')))
        tete = ('caméra', 'Mpx', 'champ °', 'px/°', 'vu', 'lisible', 'pire px', 'mm/px méd', 'mm/px max', 'fixes', 'visée')
        larg = [max(len(t), *(len(l[k]) for l in lignes)) for k, t in enumerate(tete)]
        print(f'\n══ Comparaison — zone « {zone_nom} », caméra en ({position[0]:.2f} ; {position[1]:.2f}) à '
              f'{position[2]:.2f} m, seuil {seuil:g} px')
        print('  ' + '  '.join(t.ljust(w) for t, w in zip(tete, larg)))
        for l in lignes:
            print('  ' + '  '.join(v.ljust(w) for v, w in zip(l, larg)))
        print('  (vu / lisible : sol + dessus de pierre + dessus de pile ; visée : cap/inclinaison, P = portrait)')


if __name__ == '__main__':
    main()