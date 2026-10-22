#!/usr/bin/env python3
"""Petite interface pour piloter les ventouses, l'ascenseur et le poignet du robot_omni.

Appelle les services du nœud ventouse_service :
  /robot_omni/ventouse/set       les 3 étages ensemble
  /robot_omni/ventouse_<n>/set   un étage (1 = carton du bas quand le poignet est à 0°)
  /robot_omni/poignet/set        true = 90° (cartons debout), false = 0°
et affiche la position de l'ascenseur et du poignet lue sur /joint_states.

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
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool

SERVICES = {
    'tous': '/robot_omni/ventouse/set',
    'etage_1': '/robot_omni/ventouse_1/set',
    'etage_2': '/robot_omni/ventouse_2/set',
    'etage_3': '/robot_omni/ventouse_3/set',
    'poignet': '/robot_omni/poignet/set',
}


class Pont(Node):
    """Côté ROS : clients de service et abonnement à /joint_states.

    Tourne dans un thread à part ; tout ce qui doit toucher l'interface passe par la file `sortie`.
    """

    def __init__(self, sortie):
        super().__init__('interface_ventouses')
        self.sortie = sortie
        self.clients_srv = {k: self.create_client(SetBool, v) for k, v in SERVICES.items()}
        self.create_subscription(JointState, '/joint_states', self.sur_joints, 10)

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

    def disponibilites(self):
        return {k: c.service_is_ready() for k, c in self.clients_srv.items()}

    def sur_joints(self, msg):
        pos = dict(zip(msg.name, msg.position))
        self.sortie.put(('joints', pos.get('ascenseur_joint'), pos.get('poignet_joint')))


class Interface:
    def __init__(self, racine, pont, file_ros):
        self.racine = racine
        self.pont = pont
        self.file = file_ros
        self.en_cours = None          # une seule commande à la fois, comme le nœud
        self.boutons = []

        racine.title('Ventouses robot_omni')
        racine.minsize(420, 0)
        style = ttk.Style()
        style.configure('Titre.TLabel', font=('TkDefaultFont', 11, 'bold'))
        style.configure('Valeur.TLabel', font=('TkFixedFont', 12))

        cadre = ttk.Frame(racine, padding=12)
        cadre.grid(sticky='nsew')
        racine.columnconfigure(0, weight=1)
        cadre.columnconfigure(0, weight=1)
        ligne = 0

        # --- état mesuré ---
        etat = ttk.LabelFrame(cadre, text='État (joint_states)', padding=8)
        etat.grid(row=ligne, column=0, sticky='ew'); ligne += 1
        ttk.Label(etat, text='Ascenseur').grid(row=0, column=0, sticky='w')
        self.lbl_asc = ttk.Label(etat, text='—', style='Valeur.TLabel', width=10, anchor='e')
        self.lbl_asc.grid(row=0, column=1, sticky='e')
        ttk.Label(etat, text='Poignet').grid(row=1, column=0, sticky='w')
        self.lbl_poi = ttk.Label(etat, text='—', style='Valeur.TLabel', width=10, anchor='e')
        self.lbl_poi.grid(row=1, column=1, sticky='e')
        etat.columnconfigure(0, weight=1)

        # --- toutes les ventouses ---
        tous = ttk.LabelFrame(cadre, text='Les 3 étages ensemble', padding=8)
        tous.grid(row=ligne, column=0, sticky='ew', pady=(10, 0)); ligne += 1
        self.bouton(tous, 'Prendre la pile', 'tous', True, 'Prendre les 3').grid(row=0, column=0, sticky='ew', padx=(0, 4))
        self.bouton(tous, 'Tout lâcher', 'tous', False, 'Lâcher les 3').grid(row=0, column=1, sticky='ew', padx=(4, 0))
        tous.columnconfigure((0, 1), weight=1)

        # --- étage par étage (affichés du haut vers le bas, comme sur le robot à 0°) ---
        etages = ttk.LabelFrame(cadre, text='Étage par étage', padding=8)
        etages.grid(row=ligne, column=0, sticky='ew', pady=(10, 0)); ligne += 1
        self.voyants = {}
        for i, n in enumerate((3, 2, 1)):
            nom = {3: 'Étage 3 (haut)', 2: 'Étage 2 (milieu)', 1: 'Étage 1 (bas)'}[n]
            ttk.Label(etages, text=nom).grid(row=i, column=0, sticky='w', pady=2)
            cle = f'etage_{n}'
            self.bouton(etages, 'Prendre', cle, True, f'Prendre étage {n}').grid(row=i, column=1, padx=4, pady=2)
            self.bouton(etages, 'Poser', cle, False, f'Poser étage {n}').grid(row=i, column=2, pady=2)
        etages.columnconfigure(0, weight=1)
        ttk.Label(etages, text='Poser : l\'ascenseur descend, la ventouse lâche, puis il remonte\n'
                               'si d\'autres étages tiennent encore un carton.',
                  foreground='gray').grid(row=3, column=0, columnspan=3, sticky='w', pady=(6, 0))

        # --- poignet ---
        poignet = ttk.LabelFrame(cadre, text='Poignet', padding=8)
        poignet.grid(row=ligne, column=0, sticky='ew', pady=(10, 0)); ligne += 1
        self.bouton(poignet, 'Couché (0°)', 'poignet', False, 'Poignet à 0°').grid(row=0, column=0, sticky='ew', padx=(0, 4))
        self.bouton(poignet, 'Debout (90°)', 'poignet', True, 'Poignet à 90°').grid(row=0, column=1, sticky='ew', padx=(4, 0))
        poignet.columnconfigure((0, 1), weight=1)

        # --- état des services et journal ---
        self.lbl_srv = ttk.Label(cadre, text='Recherche des services…', foreground='gray')
        self.lbl_srv.grid(row=ligne, column=0, sticky='w', pady=(10, 0)); ligne += 1
        self.journal = tk.Text(cadre, height=8, width=56, state='disabled', wrap='word')
        self.journal.grid(row=ligne, column=0, sticky='nsew', pady=(4, 0))
        cadre.rowconfigure(ligne, weight=1)
        self.journal.tag_configure('ok', foreground='#1d7a3e')
        self.journal.tag_configure('erreur', foreground='#b3261e')
        self.journal.tag_configure('info', foreground='gray')

        self.racine.after(50, self.lire_file)
        self.racine.after(500, self.verifier_services)

    def bouton(self, parent, texte, cle, valeur, libelle):
        b = ttk.Button(parent, text=texte, command=lambda: self.envoyer(cle, valeur, libelle))
        self.boutons.append(b)
        return b

    def envoyer(self, cle, valeur, libelle):
        if self.en_cours:
            return
        self.en_cours = libelle
        for b in self.boutons:
            b.state(['disabled'])
        self.ecrire(f'{libelle}…', 'info')
        self.pont.appeler(cle, valeur, libelle)

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
        except queue.Empty:
            pass
        self.racine.after(50, self.lire_file)

    def verifier_services(self):
        dispo = self.pont.disponibilites()
        manquants = [SERVICES[k] for k, ok in dispo.items() if not ok]
        if manquants:
            self.lbl_srv.configure(text='Services absents : ' + ', '.join(manquants), foreground='#b3261e')
        else:
            self.lbl_srv.configure(text='Tous les services sont disponibles', foreground='#1d7a3e')
        self.racine.after(1000, self.verifier_services)


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