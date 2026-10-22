#!/usr/bin/env python3
"""Pupitre du robot_omni : ventouses, ascenseur, poignet et base mobile.

Services (nœud ventouse_service) :
  /robot_omni/ventouse/set       les 3 étages ensemble
  /robot_omni/ventouse_<n>/set   un étage (1 = carton du bas quand le poignet est à 0°)
  /robot_omni/poignet/set        true = 90° (cartons debout), false = 0°
Actions (nœud robot_control) :
  /go_to_pose       aller en (x, y, theta)  — feedback : elapsed, message
  /rock_centering   centrage sur le marqueur — feedback : marker_visible, err_fwd, err_lat, err_theta, elapsed
Affiche aussi la position de l'ascenseur et du poignet lue sur /joint_states.

Lancer :  python3 interface_ventouses.py    (après avoir sourcé install/setup.bash)
Prérequis : sudo apt install python3-tk
"""

import math
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk

import rclpy
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from rclpy.action.graph import get_action_names_and_types
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rosidl_runtime_py.utilities import get_action
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool

SERVICES = {
    'tous': '/robot_omni/ventouse/set',
    'etage_1': '/robot_omni/ventouse_1/set',
    'etage_2': '/robot_omni/ventouse_2/set',
    'etage_3': '/robot_omni/ventouse_3/set',
    'poignet': '/robot_omni/poignet/set',
}

# 'type' = None : détecté automatiquement sur le graphe ROS (comme `ros2 action info -t`).
# On peut le forcer, par exemple 'mon_paquet/action/GoToPose'.
ACTIONS = {
    'go_to_pose': {'nom': '/go_to_pose', 'type': None},
    'rock_centering': {'nom': '/rock_centering', 'type': None},
}

ECHELLE_CIBLE = 0.10   # m : erreur affichée au bord de la cible de centrage

C = {
    'fond': '#f4f5f7', 'carte': '#ffffff', 'texte': '#1f2328', 'doux': '#6b7280',
    'accent': '#2563eb', 'accent_actif': '#1d4ed8', 'accent_off': '#a9bff0',
    'danger': '#dc2626', 'danger_actif': '#b91c1c', 'danger_off': '#efb4b4',
    'ok': '#16a34a', 'erreur': '#b3261e', 'bordure': '#d0d4da', 'grille': '#e5e7eb',
}


class Pont(Node):
    """Côté ROS : clients de service, clients d'action et abonnement à /joint_states.

    Tourne dans un thread à part ; tout ce qui doit toucher l'interface passe par la file `sortie`.
    """

    def __init__(self, sortie):
        super().__init__('interface_ventouses')
        self.sortie = sortie
        self.clients_srv = {k: self.create_client(SetBool, v) for k, v in SERVICES.items()}
        self.create_subscription(JointState, '/joint_states', self.sur_joints, 10)

        self.actions = {}            # cle -> ActionClient
        self.types_action = {}       # cle -> classe d'action
        self.goals = {}              # cle -> goal handle en cours
        self.annulation_demandee = set()
        self.types_signales = set()
        self.create_timer(1.0, self.chercher_actions)

    # --- services ---
    def appeler(self, cle, valeur, libelle):
        client = self.clients_srv[cle]
        if not client.service_is_ready():
            self.sortie.put(('fin', cle, False, f'{libelle} : service {SERVICES[cle]} indisponible'))
            return
        requete = SetBool.Request()
        requete.data = valeur
        debut = time.monotonic()
        futur = client.call_async(requete)

        def termine(f):
            duree = time.monotonic() - debut
            try:
                rep = f.result()
                self.sortie.put(('fin', cle, rep.success, f'{libelle} : {rep.message} ({duree:.1f} s)'))
            except Exception as err:  # noqa: BLE001 - toute erreur est affichée telle quelle
                self.sortie.put(('fin', cle, False, f'{libelle} : erreur {err}'))

        futur.add_done_callback(termine)

    # --- actions ---
    def chercher_actions(self):
        """Crée les clients d'action dès que leur type est connu (appelé par un timer)."""
        manquantes = [k for k in ACTIONS if k not in self.actions]
        if not manquantes:
            return
        connues = dict(get_action_names_and_types(self))
        for cle in manquantes:
            conf = ACTIONS[cle]
            type_str = conf['type'] or (connues.get(conf['nom']) or [None])[0]
            if not type_str:
                continue
            try:
                classe = get_action(type_str)
            except Exception as err:  # noqa: BLE001
                if type_str not in self.types_signales:
                    self.types_signales.add(type_str)
                    self.sortie.put(('info', f'Type {type_str} non importable ({err}). '
                                             'As-tu sourcé le workspace qui le définit ?'))
                continue
            self.types_action[cle] = classe
            self.actions[cle] = ActionClient(self, classe, conf['nom'])
            self.sortie.put(('info', f'Action {conf["nom"]} : type {type_str}'))

    def action_prete(self, cle):
        client = self.actions.get(cle)
        return client is not None and client.server_is_ready()

    def envoyer_action(self, cle, champs, libelle):
        nom = ACTIONS[cle]['nom']
        if not self.action_prete(cle):
            self.sortie.put(('action_fin', cle, False, f'{libelle} : serveur {nom} indisponible'))
            return
        but = self.types_action[cle].Goal()
        for champ, valeur in champs.items():
            setattr(but, champ, valeur)
        self.annulation_demandee.discard(cle)
        debut = time.monotonic()
        futur = self.actions[cle].send_goal_async(
            but, feedback_callback=lambda fb: self.sortie.put(('fb', cle, fb.feedback)))

        def accepte(f):
            try:
                gh = f.result()
            except Exception as err:  # noqa: BLE001
                self.sortie.put(('action_fin', cle, False, f'{libelle} : erreur {err}'))
                return
            if not gh.accepted:
                self.sortie.put(('action_fin', cle, False, f'{libelle} : refusé par le serveur'))
                return
            self.goals[cle] = gh
            self.sortie.put(('info', f'{libelle} : accepté'))
            if cle in self.annulation_demandee:
                gh.cancel_goal_async()
            gh.get_result_async().add_done_callback(lambda r: self.resultat(cle, libelle, debut, r))

        futur.add_done_callback(accepte)

    def resultat(self, cle, libelle, debut, futur):
        self.goals.pop(cle, None)
        duree = time.monotonic() - debut
        try:
            res = futur.result()
        except Exception as err:  # noqa: BLE001
            self.sortie.put(('action_fin', cle, False, f'{libelle} : erreur {err}'))
            return
        rep = res.result
        if res.status == GoalStatus.STATUS_SUCCEEDED:
            ok = rep.success
            texte = rep.message or ('réussi' if ok else 'échec')
        elif res.status == GoalStatus.STATUS_CANCELED:
            ok, texte = False, 'annulé' + (f' — {rep.message}' if rep.message else '')
        else:
            ok, texte = False, 'abandonné' + (f' — {rep.message}' if rep.message else '')
        self.sortie.put(('action_fin', cle, ok, f'{libelle} : {texte} ({duree:.1f} s)'))

    def annuler(self, cle):
        self.annulation_demandee.add(cle)
        gh = self.goals.get(cle)
        if gh is not None:
            gh.cancel_goal_async()

    def annuler_tout(self):
        en_cours = list(self.goals)
        for cle in en_cours:
            self.annuler(cle)
        return bool(en_cours)

    # --- état ---
    def disponibilites(self):
        dispo = {SERVICES[k]: c.service_is_ready() for k, c in self.clients_srv.items()}
        dispo.update({ACTIONS[k]['nom']: self.action_prete(k) for k in ACTIONS})
        return dispo

    def sur_joints(self, msg):
        pos = dict(zip(msg.name, msg.position))
        self.sortie.put(('joints', pos.get('ascenseur_joint'), pos.get('poignet_joint')))


def appliquer_style(racine):
    s = ttk.Style(racine)
    if 'clam' in s.theme_names():
        s.theme_use('clam')
    racine.configure(background=C['fond'])
    s.configure('.', background=C['fond'], foreground=C['texte'])
    s.configure('TLabelframe', background=C['fond'], bordercolor=C['bordure'])
    s.configure('TLabelframe.Label', background=C['fond'], foreground=C['texte'],
                font=('TkDefaultFont', 10, 'bold'))
    s.configure('Titre.TLabel', font=('TkDefaultFont', 14, 'bold'))
    s.configure('Valeur.TLabel', font=('TkFixedFont', 12))
    s.configure('Doux.TLabel', foreground=C['doux'])
    s.configure('TButton', padding=(10, 5))
    s.configure('Accent.TButton', background=C['accent'], foreground='white', bordercolor=C['accent'])
    s.map('Accent.TButton',
          background=[('disabled', C['accent_off']), ('active', C['accent_actif'])],
          foreground=[('disabled', '#f1f5fd')])
    s.configure('Danger.TButton', background=C['danger'], foreground='white', bordercolor=C['danger'])
    s.map('Danger.TButton',
          background=[('disabled', C['danger_off']), ('active', C['danger_actif'])],
          foreground=[('disabled', '#fdf3f3')])
    s.configure('Horizontal.TProgressbar', background=C['accent'], troughcolor=C['grille'],
                bordercolor=C['bordure'])


class Interface:
    def __init__(self, racine, pont, file_ros):
        self.racine = racine
        self.pont = pont
        self.file = file_ros
        self.en_cours = None          # une seule commande de service à la fois, comme le nœud
        self.boutons = []
        self.action_en_cours = None   # une seule action de base à la fois (les deux bougent la base)

        racine.title('Pupitre robot_omni')
        racine.minsize(880, 0)
        appliquer_style(racine)

        cadre = ttk.Frame(racine, padding=14)
        cadre.grid(sticky='nsew')
        racine.columnconfigure(0, weight=1)
        racine.rowconfigure(0, weight=1)
        cadre.columnconfigure((0, 1), weight=1, uniform='col')

        ttk.Label(cadre, text='Pupitre robot_omni', style='Titre.TLabel').grid(
            row=0, column=0, columnspan=2, sticky='w', pady=(0, 10))

        gauche = ttk.Frame(cadre)
        gauche.grid(row=1, column=0, sticky='nsew', padx=(0, 8))
        droite = ttk.Frame(cadre)
        droite.grid(row=1, column=1, sticky='nsew', padx=(8, 0))
        self.construire_prehension(gauche)
        self.construire_base(droite)

        # --- état des connexions et journal ---
        self.lbl_srv = ttk.Label(cadre, text='Recherche des services et actions…',
                                 style='Doux.TLabel', wraplength=850)
        self.lbl_srv.grid(row=2, column=0, columnspan=2, sticky='w', pady=(12, 0))
        bloc = ttk.Frame(cadre)
        bloc.grid(row=3, column=0, columnspan=2, sticky='nsew', pady=(4, 0))
        cadre.rowconfigure(3, weight=1)
        bloc.columnconfigure(0, weight=1)
        bloc.rowconfigure(0, weight=1)
        self.journal = tk.Text(bloc, height=9, state='disabled', wrap='word', relief='flat',
                               background=C['carte'], foreground=C['texte'], padx=8, pady=6,
                               highlightthickness=1, highlightbackground=C['bordure'],
                               font=('TkFixedFont', 10))
        self.journal.grid(row=0, column=0, sticky='nsew')
        defil = ttk.Scrollbar(bloc, orient='vertical', command=self.journal.yview)
        defil.grid(row=0, column=1, sticky='ns')
        self.journal.configure(yscrollcommand=defil.set)
        self.journal.tag_configure('ok', foreground='#1d7a3e')
        self.journal.tag_configure('erreur', foreground=C['erreur'])
        self.journal.tag_configure('info', foreground=C['doux'])

        racine.protocol('WM_DELETE_WINDOW', self.fermer)
        self.racine.after(50, self.lire_file)
        self.racine.after(500, self.verifier_services)

    # ------------------------------------------------------------------ construction
    def construire_prehension(self, parent):
        parent.columnconfigure(0, weight=1)
        ligne = 0

        etat = ttk.LabelFrame(parent, text='État (joint_states)', padding=10)
        etat.grid(row=ligne, column=0, sticky='ew'); ligne += 1
        ttk.Label(etat, text='Ascenseur').grid(row=0, column=0, sticky='w')
        self.lbl_asc = ttk.Label(etat, text='—', style='Valeur.TLabel', width=10, anchor='e')
        self.lbl_asc.grid(row=0, column=1, sticky='e')
        ttk.Label(etat, text='Poignet').grid(row=1, column=0, sticky='w')
        self.lbl_poi = ttk.Label(etat, text='—', style='Valeur.TLabel', width=10, anchor='e')
        self.lbl_poi.grid(row=1, column=1, sticky='e')
        etat.columnconfigure(0, weight=1)

        tous = ttk.LabelFrame(parent, text='Les 3 étages ensemble', padding=10)
        tous.grid(row=ligne, column=0, sticky='ew', pady=(10, 0)); ligne += 1
        self.bouton(tous, 'Prendre la pile', 'tous', True, 'Prendre les 3', 'Accent.TButton').grid(
            row=0, column=0, sticky='ew', padx=(0, 4))
        self.bouton(tous, 'Tout lâcher', 'tous', False, 'Lâcher les 3').grid(
            row=0, column=1, sticky='ew', padx=(4, 0))
        tous.columnconfigure((0, 1), weight=1)

        # étages affichés du haut vers le bas, comme sur le robot à 0°
        etages = ttk.LabelFrame(parent, text='Étage par étage', padding=10)
        etages.grid(row=ligne, column=0, sticky='ew', pady=(10, 0)); ligne += 1
        for i, n in enumerate((3, 2, 1)):
            nom = {3: 'Étage 3 (haut)', 2: 'Étage 2 (milieu)', 1: 'Étage 1 (bas)'}[n]
            ttk.Label(etages, text=nom).grid(row=i, column=0, sticky='w', pady=2)
            cle = f'etage_{n}'
            self.bouton(etages, 'Prendre', cle, True, f'Prendre étage {n}').grid(row=i, column=1, padx=4, pady=2)
            self.bouton(etages, 'Poser', cle, False, f'Poser étage {n}').grid(row=i, column=2, pady=2)
        etages.columnconfigure(0, weight=1)
        ttk.Label(etages, text='Poser : l\'ascenseur descend, la ventouse lâche, puis il remonte\n'
                               'si d\'autres étages tiennent encore un carton.',
                  style='Doux.TLabel').grid(row=3, column=0, columnspan=3, sticky='w', pady=(6, 0))

        poignet = ttk.LabelFrame(parent, text='Poignet', padding=10)
        poignet.grid(row=ligne, column=0, sticky='ew', pady=(10, 0)); ligne += 1
        self.bouton(poignet, 'Couché (0°)', 'poignet', False, 'Poignet à 0°').grid(
            row=0, column=0, sticky='ew', padx=(0, 4))
        self.bouton(poignet, 'Debout (90°)', 'poignet', True, 'Poignet à 90°').grid(
            row=0, column=1, sticky='ew', padx=(4, 0))
        poignet.columnconfigure((0, 1), weight=1)

    def construire_base(self, parent):
        parent.columnconfigure(0, weight=1)
        self.annulers = {}
        self.progres = {}

        # --- aller à une pose ---
        nav = ttk.LabelFrame(parent, text='Aller à une pose   /go_to_pose', padding=10)
        nav.grid(row=0, column=0, sticky='ew')
        self.var_x = tk.StringVar(value='0.00')
        self.var_y = tk.StringVar(value='0.00')
        self.var_t = tk.StringVar(value='0')
        champs = (('x (m)', self.var_x, 0.1, '%.2f'),
                  ('y (m)', self.var_y, 0.1, '%.2f'),
                  ('θ (°)', self.var_t, 15, '%.0f'))
        for i, (texte, var, pas, fmt) in enumerate(champs):
            ttk.Label(nav, text=texte).grid(row=0, column=2 * i, sticky='w', padx=(0 if i == 0 else 12, 4))
            sp = ttk.Spinbox(nav, textvariable=var, from_=-1000, to=1000, increment=pas,
                             format=fmt, width=7, justify='right')
            sp.grid(row=0, column=2 * i + 1, sticky='ew')
            sp.bind('<Return>', lambda _e: self.aller())
            nav.columnconfigure(2 * i + 1, weight=1)

        boutons = ttk.Frame(nav)
        boutons.grid(row=1, column=0, columnspan=6, sticky='ew', pady=(10, 0))
        boutons.columnconfigure((0, 1), weight=1)
        self.btn_aller = ttk.Button(boutons, text='Aller', style='Accent.TButton', command=self.aller)
        self.btn_aller.grid(row=0, column=0, sticky='ew', padx=(0, 4))
        self.annulers['go_to_pose'] = ttk.Button(boutons, text='Annuler', style='Danger.TButton',
                                                 command=lambda: self.annuler('go_to_pose'))
        self.annulers['go_to_pose'].grid(row=0, column=1, sticky='ew', padx=(4, 0))

        self.progres['go_to_pose'] = ttk.Progressbar(nav, mode='indeterminate')
        self.progres['go_to_pose'].grid(row=2, column=0, columnspan=6, sticky='ew', pady=(10, 4))
        info = ttk.Frame(nav)
        info.grid(row=3, column=0, columnspan=6, sticky='ew')
        info.columnconfigure(0, weight=1)
        self.lbl_nav_msg = ttk.Label(info, text='En attente', style='Doux.TLabel', wraplength=300)
        self.lbl_nav_msg.grid(row=0, column=0, sticky='w')
        self.lbl_nav_t = ttk.Label(info, text='', style='Valeur.TLabel')
        self.lbl_nav_t.grid(row=0, column=1, sticky='e')

        # --- centrage sur le rocher ---
        cen = ttk.LabelFrame(parent, text='Centrage sur le rocher   /rock_centering', padding=10)
        cen.grid(row=1, column=0, sticky='ew', pady=(10, 0))
        cen.columnconfigure(1, weight=1)
        self.cible = tk.Canvas(cen, width=170, height=170, background=C['carte'],
                               highlightthickness=1, highlightbackground=C['bordure'])
        self.cible.grid(row=0, column=0, rowspan=6, padx=(0, 12), sticky='n')

        self.lbl_marqueur = ttk.Label(cen, text='●  Marqueur : —', foreground=C['doux'],
                                      font=('TkDefaultFont', 10, 'bold'))
        self.lbl_marqueur.grid(row=0, column=1, columnspan=2, sticky='w', pady=(0, 6))
        self.lbl_cen = {}
        mesures = (('Erreur avant', 'err_fwd'), ('Erreur latérale', 'err_lat'),
                   ('Erreur angle', 'err_theta'), ('Durée', 'elapsed'))
        for i, (texte, champ) in enumerate(mesures):
            ttk.Label(cen, text=texte).grid(row=1 + i, column=1, sticky='w')
            lbl = ttk.Label(cen, text='—', style='Valeur.TLabel', width=10, anchor='e')
            lbl.grid(row=1 + i, column=2, sticky='e')
            self.lbl_cen[champ] = lbl

        bcen = ttk.Frame(cen)
        bcen.grid(row=5, column=1, columnspan=2, sticky='sew', pady=(8, 0))
        bcen.columnconfigure((0, 1), weight=1)
        self.btn_centrer = ttk.Button(bcen, text='Centrer', style='Accent.TButton',
                                      command=lambda: self.lancer_action('rock_centering', {}, 'Centrage rocher'))
        self.btn_centrer.grid(row=0, column=0, sticky='ew', padx=(0, 4))
        self.annulers['rock_centering'] = ttk.Button(bcen, text='Annuler', style='Danger.TButton',
                                                     command=lambda: self.annuler('rock_centering'))
        self.annulers['rock_centering'].grid(row=0, column=1, sticky='ew', padx=(4, 0))

        self.progres['rock_centering'] = ttk.Progressbar(cen, mode='indeterminate')
        self.progres['rock_centering'].grid(row=6, column=0, columnspan=3, sticky='ew', pady=(10, 0))

        for b in self.annulers.values():
            b.state(['disabled'])
        self.dessiner_cible()

    def bouton(self, parent, texte, cle, valeur, libelle, style='TButton'):
        b = ttk.Button(parent, text=texte, style=style, command=lambda: self.envoyer(cle, valeur, libelle))
        self.boutons.append(b)
        return b

    # ------------------------------------------------------------------ cible de centrage
    def dessiner_cible(self, fwd=None, lat=None, theta=None, visible=False):
        c = self.cible
        c.delete('all')
        cx = cy = 85
        r = 70
        for rr in (r, r / 2):
            c.create_oval(cx - rr, cy - rr, cx + rr, cy + rr, outline=C['grille'])
        c.create_line(cx - r, cy, cx + r, cy, fill=C['grille'])
        c.create_line(cx, cy - r, cx, cy + r, fill=C['grille'])
        c.create_text(cx, 7, text='avant', fill=C['doux'], font=('TkDefaultFont', 8))
        c.create_text(14, cy - 8, text='gauche', fill=C['doux'], font=('TkDefaultFont', 8), anchor='w')
        c.create_text(166, 164, text=f'±{ECHELLE_CIBLE * 100:.0f} cm', fill=C['doux'],
                      font=('TkDefaultFont', 8), anchor='se')
        if fwd is None:
            return
        k = r / ECHELLE_CIBLE
        dx, dy = -lat * k, -fwd * k          # avant vers le haut, gauche vers la gauche
        d = math.hypot(dx, dy)
        hors_champ = d > r
        if hors_champ:
            dx, dy = dx * r / d, dy * r / d
        x, y = cx + dx, cy + dy
        couleur = C['ok'] if visible else C['doux']
        c.create_line(cx, cy, x, y, fill=couleur, dash=(2, 3))
        c.create_line(x, y, x - math.sin(theta) * 22, y - math.cos(theta) * 22,
                      fill=couleur, width=3, arrow='last')
        c.create_oval(x - 6, y - 6, x + 6, y + 6, fill=couleur if not hors_champ else C['carte'],
                      outline=couleur, width=2)

    # ------------------------------------------------------------------ commandes
    def envoyer(self, cle, valeur, libelle):
        if self.en_cours:
            return
        self.en_cours = libelle
        for b in self.boutons:
            b.state(['disabled'])
        self.ecrire(f'{libelle}…', 'info')
        self.pont.appeler(cle, valeur, libelle)

    def aller(self):
        if self.action_en_cours:
            return
        try:
            x = float(self.var_x.get().replace(',', '.'))
            y = float(self.var_y.get().replace(',', '.'))
            t_deg = float(self.var_t.get().replace(',', '.'))
        except ValueError:
            self.ecrire('Pose invalide : x, y et θ doivent être des nombres', 'erreur')
            return
        self.lancer_action('go_to_pose', {'x': x, 'y': y, 'theta': math.radians(t_deg)},
                           f'Aller à ({x:.2f} m, {y:.2f} m, {t_deg:.0f}°)')

    def lancer_action(self, cle, champs, libelle):
        if self.action_en_cours:
            return
        self.action_en_cours = cle
        self.btn_aller.state(['disabled'])
        self.btn_centrer.state(['disabled'])
        self.annulers[cle].state(['!disabled'])
        self.progres[cle].start(12)
        if cle == 'go_to_pose':
            self.lbl_nav_msg.configure(text='Envoi…')
            self.lbl_nav_t.configure(text='')
        else:
            self.lbl_marqueur.configure(text='●  Marqueur : —', foreground=C['doux'])
            for lbl in self.lbl_cen.values():
                lbl.configure(text='—')
            self.dessiner_cible()
        self.ecrire(f'{libelle}…', 'info')
        self.pont.envoyer_action(cle, champs, libelle)

    def annuler(self, cle):
        if self.action_en_cours != cle:
            return
        self.annulers[cle].state(['disabled'])
        self.ecrire('Annulation demandée…', 'info')
        self.pont.annuler(cle)

    def fin_action(self, cle, ok, texte):
        self.ecrire(texte, 'ok' if ok else 'erreur')
        self.action_en_cours = None
        self.btn_aller.state(['!disabled'])
        self.btn_centrer.state(['!disabled'])
        for b in self.annulers.values():
            b.state(['disabled'])
        for p in self.progres.values():
            p.stop()
        if cle == 'go_to_pose':
            self.lbl_nav_msg.configure(text=texte, foreground=C['ok'] if ok else C['erreur'])

    def sur_feedback(self, cle, fb):
        if cle == 'go_to_pose':
            self.lbl_nav_msg.configure(text=fb.message or 'En route…', foreground=C['doux'])
            self.lbl_nav_t.configure(text=f'{fb.elapsed:5.1f} s')
            return
        vis = fb.marker_visible
        self.lbl_marqueur.configure(text='●  Marqueur visible' if vis else '●  Marqueur perdu',
                                    foreground=C['ok'] if vis else C['erreur'])
        self.lbl_cen['err_fwd'].configure(text=f'{fb.err_fwd * 1000:7.1f} mm')
        self.lbl_cen['err_lat'].configure(text=f'{fb.err_lat * 1000:7.1f} mm')
        self.lbl_cen['err_theta'].configure(text=f'{math.degrees(fb.err_theta):7.1f} °')
        self.lbl_cen['elapsed'].configure(text=f'{fb.elapsed:7.1f} s')
        self.dessiner_cible(fb.err_fwd, fb.err_lat, fb.err_theta, vis)

    # ------------------------------------------------------------------ boucle
    def ecrire(self, texte, tag):
        self.journal.configure(state='normal')
        self.journal.insert('end', time.strftime('%H:%M:%S ') + texte + '\n', tag)
        self.journal.see('end')
        self.journal.configure(state='disabled')

    def lire_file(self):
        try:
            while True:
                evt = self.file.get_nowait()
                if evt[0] == 'fin':
                    _, _, ok, texte = evt
                    self.ecrire(texte, 'ok' if ok else 'erreur')
                    self.en_cours = None
                    for b in self.boutons:
                        b.state(['!disabled'])
                elif evt[0] == 'joints':
                    _, asc, poi = evt
                    if asc is not None:
                        self.lbl_asc.configure(text=f'{asc * 1000:6.1f} mm')
                    if poi is not None:
                        self.lbl_poi.configure(text=f'{math.degrees(poi):6.1f} °')
                elif evt[0] == 'fb':
                    self.sur_feedback(evt[1], evt[2])
                elif evt[0] == 'action_fin':
                    _, cle, ok, texte = evt
                    self.fin_action(cle, ok, texte)
                elif evt[0] == 'info':
                    self.ecrire(evt[1], 'info')
        except queue.Empty:
            pass
        self.racine.after(50, self.lire_file)

    def verifier_services(self):
        dispo = self.pont.disponibilites()
        manquants = [nom for nom, ok in dispo.items() if not ok]
        if manquants:
            self.lbl_srv.configure(text='Absents : ' + ', '.join(manquants), foreground=C['erreur'])
        else:
            self.lbl_srv.configure(text='Tous les services et actions sont disponibles', foreground=C['ok'])
        self.racine.after(1000, self.verifier_services)

    def fermer(self):
        """Annule les actions en cours avant de fermer, pour ne pas laisser la base partir seule."""
        if self.pont.annuler_tout():
            self.ecrire('Fermeture : annulation des actions en cours…', 'info')
            self.racine.after(400, self.racine.destroy)
        else:
            self.racine.destroy()


def main():
    rclpy.init()
    file_ros = queue.Queue()
    pont = Pont(file_ros)
    executeur = SingleThreadedExecutor()
    executeur.add_node(pont)
    fil = threading.Thread(target=executeur.spin, daemon=True)
    fil.start()

    racine = tk.Tk()
    Interface(racine, pont, file_ros)
    try:
        racine.mainloop()
    finally:
        executeur.shutdown()
        pont.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()