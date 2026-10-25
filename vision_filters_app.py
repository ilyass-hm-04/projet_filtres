"""
=============================================================
  Vision Filters App -- TP 7 : Les Filtres a convolution
  Interactive real-time filtering, camera & image file support
=============================================================
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time


# LIGHT MODE COLOR PALETTE
COLORS = {
    "bg_dark":      "#f8f9fa",
    "bg_mid":       "#ffffff",
    "bg_card":      "#f1f3f5",
    "bg_hover":     "#e9ecef",
    "accent":       "#2b8a3e",     # Forest green
    "accent_hover": "#2f9e44",
    "accent2":      "#1971c2",     # Royal blue
    "accent3":      "#d9480f",     # Orange/Red
    "text_bright":  "#212529",     # Very dark charcoal
    "text_dim":     "#495057",     # Dark grey
    "border":       "#dee2e6",
    "red":          "#e03131",
    "purple":       "#7048e8",
    "cyan":         "#0c8599",
    "slider_bg":    "#e9ecef",
    "slider_trough":"#ced4da",
}

FONT_TITLE = ("Segoe UI", 16, "bold")
FONT_H2    = ("Segoe UI", 12, "bold")
FONT_LABEL = ("Segoe UI", 10)
FONT_BOLD  = ("Segoe UI", 10, "bold")
FONT_SMALL = ("Segoe UI", 9)
FONT_MONO  = ("Consolas", 9)


def calculate_mse(img1, img2):
    if img1 is None or img2 is None:
        return 0.0
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    if (h1, w1) != (h2, w2):
        img2 = cv2.resize(img2, (w1, h1))
    if len(img1.shape) != len(img2.shape):
        if len(img1.shape) == 2:
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        if len(img2.shape) == 2:
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    diff = img1.astype(np.float32) - img2.astype(np.float32)
    err = np.sum(diff ** 2) / float(img1.shape[0] * img1.shape[1] * (img1.shape[2] if len(img1.shape) == 3 else 1))
    return float(err)


def generate_histogram_image(frame, width=280, height=70):
    if frame is None or frame.size == 0:
        return np.zeros((height, width, 3), dtype=np.uint8) + 245
    hist_img = np.zeros((height, width, 3), dtype=np.uint8) + 245
    for y in range(0, height, 20):
        cv2.line(hist_img, (0, y), (width, y), (230, 230, 230), 1)
    
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        colors = [(220, 60, 60), (40, 160, 40), (50, 50, 220)] # BGR
        for i, col in enumerate(colors):
            hist = cv2.calcHist([frame], [i], None, [256], [0, 256])
            cv2.normalize(hist, hist, 0, height - 8, cv2.NORM_MINMAX)
            pts = []
            for x in range(256):
                px = int(x * (width / 256.0))
                py = int(height - hist[x][0])
                pts.append((px, py))
            for k in range(len(pts) - 1):
                cv2.line(hist_img, pts[k], pts[k+1], col, 1, cv2.LINE_AA)
    else:
        gray = frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist, 0, height - 8, cv2.NORM_MINMAX)
        pts = [(int(x * (width / 256.0)), int(height - hist[x][0])) for x in range(256)]
        for k in range(len(pts) - 1):
            cv2.line(hist_img, pts[k], pts[k+1], (80, 80, 80), 1, cv2.LINE_AA)
    return hist_img


class FilterEngine:
    @staticmethod
    def filtre_moyenneur(frame, ksize=5, **_):
        ksize = max(1, int(ksize) | 1)
        return cv2.blur(frame, (ksize, ksize))

    @staticmethod
    def filtre_gaussien(frame, ksize=5, sigma=0, **_):
        ksize = max(1, int(ksize) | 1)
        return cv2.GaussianBlur(frame, (ksize, ksize), float(sigma))

    @staticmethod
    def filtre_median(frame, ksize=5, **_):
        ksize = max(1, int(ksize) | 1)
        return cv2.medianBlur(frame, ksize)

    @staticmethod
    def filtre_bilateral(frame, d=9, sigma_color=75, sigma_space=75, **_):
        d = max(1, int(d) | 1)
        return cv2.bilateralFilter(frame, d, float(sigma_color), float(sigma_space))

    @staticmethod
    def bruit_gaussien(frame, mean=0, std=25, **_):
        bruit = np.random.normal(float(mean), max(0.1, float(std)), frame.shape).astype(np.float32)
        img   = frame.astype(np.float32) + bruit
        return np.clip(img, 0, 255).astype(np.uint8)

    @staticmethod
    def bruit_sel_poivre(frame, density=5, **_):
        img   = frame.copy()
        dens  = float(density) / 100.0
        total = img.shape[0] * img.shape[1]
        n_sp  = int(total * dens)
        coords = [np.random.randint(0, d, n_sp) for d in img.shape[:2]]
        img[coords[0], coords[1]] = 255
        coords = [np.random.randint(0, d, n_sp) for d in img.shape[:2]]
        img[coords[0], coords[1]] = 0
        return img

    @staticmethod
    def bruit_uniforme(frame, minimum=-30, maximum=30, **_):
        mn  = float(minimum)
        mx  = float(maximum)
        if mn >= mx:
            mx = mn + 1
        bruit = np.random.uniform(mn, mx, frame.shape).astype(np.float32)
        img   = frame.astype(np.float32) + bruit
        return np.clip(img, 0, 255).astype(np.uint8)

    @staticmethod
    def filtre_sobel(frame, ksize=3, direction="magnitude", **_):
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        ksize   = max(1, int(ksize) | 1)
        if ksize > 7:
            ksize = 7
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        if direction == "x":
            result = cv2.convertScaleAbs(sobel_x)
        elif direction == "y":
            result = cv2.convertScaleAbs(sobel_y)
        else:
            mag    = cv2.magnitude(sobel_x.astype(np.float32), sobel_y.astype(np.float32))
            result = cv2.convertScaleAbs(mag)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def filtre_laplacien(frame, ksize=3, sharpen=False, **_):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        if sharpen:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            result = cv2.filter2D(gray, -1, kernel)
        else:
            ksize  = max(1, int(ksize) | 1)
            if ksize > 31:
                ksize = 31
            lap    = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
            result = cv2.convertScaleAbs(lap)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def filtre_canny(frame, threshold1=50, threshold2=150, blur_ksize=5, **_):
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        bk    = max(1, int(blur_ksize) | 1)
        blur  = cv2.GaussianBlur(gray, (bk, bk), 0)
        edges = cv2.Canny(blur, float(threshold1), float(threshold2))
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def pipeline_gaussien_canny(frame, ksize=5, t1=50, t2=150, **_):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        bk   = max(1, int(ksize) | 1)
        blur = cv2.GaussianBlur(gray, (bk, bk), 0)
        out  = cv2.Canny(blur, float(t1), float(t2))
        return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def pipeline_median_canny(frame, ksize=5, t1=50, t2=150, **_):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        bk   = max(1, int(ksize) | 1)
        blur = cv2.medianBlur(gray, bk)
        out  = cv2.Canny(blur, float(t1), float(t2))
        return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def pipeline_bilateral_canny(frame, d=9, sigma=75, t1=50, t2=150, **_):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        dd   = max(1, int(d) | 1)
        blur = cv2.bilateralFilter(gray, dd, float(sigma), float(sigma))
        out  = cv2.Canny(blur, float(t1), float(t2))
        return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def original(frame, **_):
        return frame


FILTERS = {
    "Original": {
        "fn": FilterEngine.original,
        "category": "Affichage",
        "color": COLORS["text_dim"],
        "params": [],
    },
    "Moyenneur (Box)": {
        "fn": FilterEngine.filtre_moyenneur,
        "category": "Lissage",
        "color": COLORS["accent2"],
        "params": [("Taille noyau", "ksize", 1, 31, 5, 2)],
    },
    "Gaussien": {
        "fn": FilterEngine.filtre_gaussien,
        "category": "Lissage",
        "color": COLORS["accent2"],
        "params": [
            ("Taille noyau", "ksize", 1, 31, 5, 2),
            ("Sigma",        "sigma", 0, 50, 0, 1),
        ],
    },
    "Median": {
        "fn": FilterEngine.filtre_median,
        "category": "Lissage",
        "color": COLORS["accent2"],
        "params": [("Taille noyau", "ksize", 1, 31, 5, 2)],
    },
    "Bilateral": {
        "fn": FilterEngine.filtre_bilateral,
        "category": "Lissage",
        "color": COLORS["accent2"],
        "params": [
            ("Diametre (d)", "d",           1, 25,  9, 2),
            ("Sigma Color",  "sigma_color",  1, 200, 75, 1),
            ("Sigma Space",  "sigma_space",  1, 200, 75, 1),
        ],
    },
    "Bruit Gaussien": {
        "fn": FilterEngine.bruit_gaussien,
        "category": "Bruit",
        "color": COLORS["accent3"],
        "params": [
            ("Moyenne",    "mean",  -50, 50,  0, 1),
            ("Ecart-type", "std",     1, 100, 25, 1),
        ],
    },
    "Bruit Sel et Poivre": {
        "fn": FilterEngine.bruit_sel_poivre,
        "category": "Bruit",
        "color": COLORS["accent3"],
        "params": [("Densite (%)", "density", 1, 50, 5, 1)],
    },
    "Bruit Uniforme": {
        "fn": FilterEngine.bruit_uniforme,
        "category": "Bruit",
        "color": COLORS["accent3"],
        "params": [
            ("Min", "minimum", -100, 0,   -30, 1),
            ("Max", "maximum",    0, 100,  30, 1),
        ],
    },
    "Sobel": {
        "fn": FilterEngine.filtre_sobel,
        "category": "Contours",
        "color": COLORS["purple"],
        "params": [("Taille noyau", "ksize", 1, 7, 3, 2)],
        "extra": "direction",
    },
    "Laplacien": {
        "fn": FilterEngine.filtre_laplacien,
        "category": "Contours",
        "color": COLORS["purple"],
        "params": [("Taille noyau", "ksize", 1, 31, 3, 2)],
        "extra": "sharpen",
    },
    "Canny": {
        "fn": FilterEngine.filtre_canny,
        "category": "Contours",
        "color": COLORS["purple"],
        "params": [
            ("Seuil bas",  "threshold1",  0, 300,  50, 1),
            ("Seuil haut", "threshold2",  0, 300, 150, 1),
            ("Blur ksize", "blur_ksize",  1, 21,   5, 2),
        ],
    },
    "Gaussien + Canny": {
        "fn": FilterEngine.pipeline_gaussien_canny,
        "category": "Pipeline",
        "color": COLORS["cyan"],
        "params": [
            ("Taille blur", "ksize", 1, 21,  5, 2),
            ("Seuil bas",   "t1",    0, 300, 50, 1),
            ("Seuil haut",  "t2",    0, 300,150, 1),
        ],
    },
    "Median + Canny": {
        "fn": FilterEngine.pipeline_median_canny,
        "category": "Pipeline",
        "color": COLORS["cyan"],
        "params": [
            ("Taille blur", "ksize", 1, 21,  5, 2),
            ("Seuil bas",   "t1",    0, 300, 50, 1),
            ("Seuil haut",  "t2",    0, 300,150, 1),
        ],
    },
    "Bilateral + Canny": {
        "fn": FilterEngine.pipeline_bilateral_canny,
        "category": "Pipeline",
        "color": COLORS["cyan"],
        "params": [
            ("Diametre",   "d",     1, 25,   9, 2),
            ("Sigma",      "sigma", 1, 200, 75, 1),
            ("Seuil bas",  "t1",    0, 300, 50, 1),
            ("Seuil haut", "t2",    0, 300,150, 1),
        ],
    },
}

CATEGORY_ORDER = ["Affichage", "Lissage", "Bruit", "Contours", "Pipeline"]


class VisionFiltersApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Vision Filters -- TP 7 : Les Filtres a convolution")
        self.configure(bg=COLORS["bg_dark"])
        self.resizable(True, True)
        self.geometry("1280x760")
        self.minsize(1050, 650)

        self.cap              = None
        self.running          = False
        self.current_filter   = "Original"
        self.param_vars       = {}
        self.extra_vars       = {}
        self.input_mode       = tk.StringVar(value="camera") # "camera" or "image"
        self.uploaded_image   = None
        self.camera_index     = tk.IntVar(value=0)
        self.fps_label        = None
        self.mse_label        = None
        self._latest_orig     = None
        self._latest_filt     = None
        self._last_frame_time = time.time()
        self._fps             = 0.0

        self._build_ui()
        self._apply_styles()
        self.after(200, self._start_camera)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        # 1. TOP HEADER BAR
        top = tk.Frame(self, bg=COLORS["bg_mid"], height=62, bd=0, highlightthickness=1, highlightbackground=COLORS["border"])
        top.pack(fill="x", side="top")
        top.pack_propagate(False)

        # Title section
        title_frame = tk.Frame(top, bg=COLORS["bg_mid"])
        title_frame.pack(side="left", padx=15, pady=8)
        tk.Label(title_frame, text="Vision Filters", font=FONT_TITLE,
                 bg=COLORS["bg_mid"], fg=COLORS["accent"]).pack(anchor="w")
        tk.Label(title_frame, text="TP 7 - Filtres a convolution & Traitement d'image", font=FONT_SMALL,
                 bg=COLORS["bg_mid"], fg=COLORS["text_dim"]).pack(anchor="w")

        # Mode Selection Radio Frame
        mode_frame = tk.Frame(top, bg=COLORS["bg_card"], padx=10, pady=4, bd=1, relief="solid")
        mode_frame.pack(side="left", padx=20)
        
        tk.Label(mode_frame, text="Source : ", font=FONT_BOLD,
                 bg=COLORS["bg_card"], fg=COLORS["text_bright"]).pack(side="left")
        
        rb_cam = tk.Radiobutton(mode_frame, text="🎥 Camera Live", variable=self.input_mode,
                                value="camera", font=FONT_LABEL, bg=COLORS["bg_card"],
                                fg=COLORS["accent2"], activebackground=COLORS["bg_card"],
                                command=self._on_input_mode_change)
        rb_cam.pack(side="left", padx=5)

        rb_img = tk.Radiobutton(mode_frame, text="📁 Importer Image", variable=self.input_mode,
                                value="image", font=FONT_LABEL, bg=COLORS["bg_card"],
                                fg=COLORS["purple"], activebackground=COLORS["bg_card"],
                                command=self._on_input_mode_change)
        rb_img.pack(side="left", padx=5)

        # Action bar right (Upload & Download & Cam Controls)
        self.top_actions = tk.Frame(top, bg=COLORS["bg_mid"])
        self.top_actions.pack(side="right", padx=15)

        self.fps_label = tk.Label(self.top_actions, text="FPS: --", font=FONT_MONO,
                                  bg=COLORS["bg_card"], fg=COLORS["cyan"], padx=8, pady=4)
        self.fps_label.pack(side="right", padx=6)

        self.download_btn = tk.Button(self.top_actions, text="💾 Telecharger Resultat", font=FONT_BOLD,
                                      bg=COLORS["accent"], fg="#ffffff", activebackground=COLORS["accent_hover"],
                                      activeforeground="#ffffff", relief="flat", padx=12, pady=6, cursor="hand2",
                                      command=self._download_filtered_image)
        self.download_btn.pack(side="right", padx=6)

        self.upload_btn = tk.Button(self.top_actions, text="📂 Choisir une Image", font=FONT_BOLD,
                                    bg=COLORS["purple"], fg="#ffffff", activebackground="#5f3dc4",
                                    activeforeground="#ffffff", relief="flat", padx=12, pady=6, cursor="hand2",
                                    command=self._upload_image_file)
        
        self.cam_controls = tk.Frame(self.top_actions, bg=COLORS["bg_mid"])
        self.cam_controls.pack(side="right", padx=6)
        tk.Label(self.cam_controls, text="Cam ID:", font=FONT_SMALL,
                 bg=COLORS["bg_mid"], fg=COLORS["text_dim"]).pack(side="left")
        tk.Spinbox(self.cam_controls, from_=0, to=5, width=2,
                   textvariable=self.camera_index, font=FONT_SMALL,
                   bg=COLORS["bg_card"], fg=COLORS["text_bright"], relief="flat").pack(side="left", padx=4)
        tk.Button(self.cam_controls, text="Reconnecter", font=FONT_SMALL,
                  bg=COLORS["bg_card"], fg=COLORS["accent2"], relief="flat", padx=6, pady=2,
                  command=self._restart_camera).pack(side="left", padx=2)

        # 2. MAIN BODY
        body = tk.Frame(self, bg=COLORS["bg_dark"])
        body.pack(fill="both", expand=True)

        # Left filter list
        left = tk.Frame(body, bg=COLORS["bg_mid"], width=230, bd=0, highlightthickness=1, highlightbackground=COLORS["border"])
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        self._build_filter_list(left)

        # Center area (Images + Metrics)
        self.center_frame = tk.Frame(body, bg=COLORS["bg_dark"])
        self.center_frame.pack(side="left", fill="both", expand=True)
        self._build_video_area(self.center_frame)

        # Right params panel
        right = tk.Frame(body, bg=COLORS["bg_mid"], width=280, bd=0, highlightthickness=1, highlightbackground=COLORS["border"])
        right.pack(side="right", fill="y")
        right.pack_propagate(False)
        self.right_panel = right
        self._build_param_panel(right)

    def _build_filter_list(self, parent):
        header = tk.Frame(parent, bg=COLORS["bg_card"], height=40)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(header, text="SELECTION DU FILTRE", font=FONT_BOLD,
                 bg=COLORS["bg_card"], fg=COLORS["text_bright"]).pack(side="left", padx=12, pady=10)

        canvas    = tk.Canvas(parent, bg=COLORS["bg_mid"], bd=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        self.filter_list_frame = tk.Frame(canvas, bg=COLORS["bg_mid"])

        self.filter_list_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.filter_list_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        self.filter_buttons = {}
        for cat in CATEGORY_ORDER:
            members = [n for n, v in FILTERS.items() if v["category"] == cat]
            if not members:
                continue
            cat_lbl = tk.Label(self.filter_list_frame, text=f"  {cat.upper()}",
                               font=("Segoe UI", 8, "bold"), bg=COLORS["bg_card"],
                               fg=COLORS["text_dim"], anchor="w", pady=4)
            cat_lbl.pack(fill="x", padx=6, pady=(10, 4))
            
            for name in members:
                color = FILTERS[name]["color"]
                btn   = tk.Button(
                    self.filter_list_frame, text=f" {name}",
                    font=FONT_LABEL, bg=COLORS["bg_mid"], fg=color,
                    activebackground=COLORS["bg_hover"], activeforeground=color,
                    relief="flat", anchor="w", pady=6, padx=10, cursor="hand2",
                    bd=0, command=lambda n=name: self._select_filter(n))
                btn.pack(fill="x", padx=8, pady=2)
                self.filter_buttons[name] = btn

        self._highlight_button("Original")

    def _build_video_area(self, parent):
        # Info bar
        info_bar = tk.Frame(parent, bg=COLORS["bg_dark"])
        info_bar.pack(fill="x", padx=12, pady=(10, 4))

        self.filter_name_label = tk.Label(info_bar, text="Filtre Actif: Original",
                                          font=("Segoe UI", 11, "bold"),
                                          bg=COLORS["bg_dark"], fg=COLORS["accent2"])
        self.filter_name_label.pack(side="left")

        self.mse_label = tk.Label(info_bar, text="MSE: 0.00", font=FONT_MONO,
                                  bg=COLORS["bg_card"], fg=COLORS["text_bright"],
                                  padx=10, pady=3, bd=1, relief="solid")
        self.mse_label.pack(side="right")

        # Visual display frames
        video_frame = tk.Frame(parent, bg=COLORS["bg_dark"])
        video_frame.pack(fill="both", expand=True, padx=8, pady=4)

        # Original view
        orig_wrapper = tk.Frame(video_frame, bg=COLORS["bg_mid"], bd=1, relief="solid")
        orig_wrapper.pack(side="left", fill="both", expand=True, padx=(0, 4))
        
        orig_hdr = tk.Frame(orig_wrapper, bg=COLORS["bg_card"], height=30)
        orig_hdr.pack(fill="x")
        orig_hdr.pack_propagate(False)
        self.orig_title_lbl = tk.Label(orig_hdr, text="📷 Image Source (Originale)", font=FONT_BOLD,
                                       bg=COLORS["bg_card"], fg=COLORS["text_bright"])
        self.orig_title_lbl.pack(side="left", padx=10)

        self.canvas_orig = tk.Canvas(orig_wrapper, bg="#ffffff", bd=0, highlightthickness=0)
        self.canvas_orig.pack(fill="both", expand=True, padx=4, pady=4)

        # Filtered view
        filt_wrapper = tk.Frame(video_frame, bg=COLORS["bg_mid"], bd=1, relief="solid")
        filt_wrapper.pack(side="left", fill="both", expand=True, padx=(4, 0))
        
        filt_hdr = tk.Frame(filt_wrapper, bg=COLORS["bg_card"], height=30)
        filt_hdr.pack(fill="x")
        filt_hdr.pack_propagate(False)
        tk.Label(filt_hdr, text="✨ Resultat (Filtre Applique)", font=FONT_BOLD,
                 bg=COLORS["bg_card"], fg=COLORS["accent"]).pack(side="left", padx=10)

        self.canvas_filt = tk.Canvas(filt_wrapper, bg="#ffffff", bd=0, highlightthickness=0)
        self.canvas_filt.pack(fill="both", expand=True, padx=4, pady=4)

        # Bottom Histogram view bar
        hist_frame = tk.Frame(parent, bg=COLORS["bg_mid"], height=90, bd=1, relief="solid")
        hist_frame.pack(fill="x", padx=8, pady=(4, 10))
        
        tk.Label(hist_frame, text="Histogrammes d'Intensite (Original vs Filtre)", font=FONT_SMALL,
                 bg=COLORS["bg_mid"], fg=COLORS["text_dim"]).pack(anchor="w", padx=8, pady=(2, 0))
        
        hist_container = tk.Frame(hist_frame, bg=COLORS["bg_mid"])
        hist_container.pack(fill="both", expand=True, padx=4, pady=2)

        self.canvas_hist_orig = tk.Canvas(hist_container, bg="#ffffff", height=60, bd=0, highlightthickness=0)
        self.canvas_hist_orig.pack(side="left", fill="both", expand=True, padx=4)

        self.canvas_hist_filt = tk.Canvas(hist_container, bg="#ffffff", height=60, bd=0, highlightthickness=0)
        self.canvas_hist_filt.pack(side="right", fill="both", expand=True, padx=4)

    def _build_param_panel(self, parent):
        for w in parent.winfo_children():
            w.destroy()

        self.param_vars  = {}
        self.extra_vars  = {}
        name   = self.current_filter
        fdata  = FILTERS[name]
        params = fdata["params"]
        color  = fdata["color"]

        header = tk.Frame(parent, bg=COLORS["bg_card"], height=60)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(header, text=name, font=FONT_H2, bg=COLORS["bg_card"],
                 fg=color, wraplength=170, justify="left"
                 ).pack(side="left", padx=12, pady=10)
        tk.Label(header, text=fdata["category"], font=FONT_SMALL,
                 bg=color, fg="#ffffff", padx=8, pady=3
                 ).pack(side="right", padx=10, pady=14)

        tk.Frame(parent, bg=COLORS["border"], height=1).pack(fill="x")

        if not params and "extra" not in fdata:
            tk.Label(parent, text="Aucun parametre ajustable pour ce filtre.",
                     font=FONT_LABEL, bg=COLORS["bg_mid"], fg=COLORS["text_dim"],
                     justify="center").pack(expand=True)
        else:
            sc  = tk.Canvas(parent, bg=COLORS["bg_mid"], bd=0, highlightthickness=0)
            sc.pack(fill="both", expand=True)
            inner = tk.Frame(sc, bg=COLORS["bg_mid"])
            sc.create_window((0, 0), window=inner, anchor="nw")
            inner.bind("<Configure>",
                       lambda e: sc.configure(scrollregion=sc.bbox("all")))

            for label, key, lo, hi, default, res in params:
                self._add_slider(inner, label, key, lo, hi, default, res, color)

            extra = fdata.get("extra")
            if extra == "direction":
                self._add_direction_control(inner, color)
            elif extra == "sharpen":
                self._add_sharpen_control(inner, color)

        tk.Frame(parent, bg=COLORS["border"], height=1).pack(fill="x", side="bottom")
        tk.Button(parent, text="🔄 Reinitialiser Parametres",
                  font=FONT_BOLD, bg=COLORS["bg_card"], fg=COLORS["text_bright"],
                  activebackground=COLORS["bg_hover"], activeforeground=COLORS["text_bright"],
                  relief="flat", pady=8, cursor="hand2",
                  command=self._reset_params).pack(fill="x", side="bottom", padx=10, pady=10)

    def _add_slider(self, parent, label, key, lo, hi, default, res, color):
        frame = tk.Frame(parent, bg=COLORS["bg_mid"])
        frame.pack(fill="x", padx=14, pady=(12, 0))

        top_row = tk.Frame(frame, bg=COLORS["bg_mid"])
        top_row.pack(fill="x")
        tk.Label(top_row, text=label, font=FONT_LABEL,
                 bg=COLORS["bg_mid"], fg=COLORS["text_bright"], anchor="w").pack(side="left")

        var                  = tk.DoubleVar(value=default)
        self.param_vars[key] = var

        val_label = tk.Label(top_row, text=str(int(default)), font=FONT_MONO,
                             bg=COLORS["bg_mid"], fg=color, width=5, anchor="e")
        val_label.pack(side="right")

        def on_change(v, vl=val_label):
            vl.config(text=f"{float(v):.0f}")
            if self.input_mode.get() == "image":
                self._update_static_image_view()

        scale = tk.Scale(frame, variable=var, from_=lo, to=hi, resolution=res,
                         orient="horizontal", showvalue=False,
                         bg=COLORS["bg_mid"], fg=COLORS["text_bright"],
                         troughcolor=COLORS["slider_trough"], activebackground=color,
                         highlightthickness=0, bd=0, command=on_change,
                         sliderlength=18, length=220)
        scale.pack(fill="x", pady=(4, 0))

        range_row = tk.Frame(frame, bg=COLORS["bg_mid"])
        range_row.pack(fill="x")
        tk.Label(range_row, text=str(lo), font=FONT_SMALL,
                 bg=COLORS["bg_mid"], fg=COLORS["text_dim"]).pack(side="left")
        tk.Label(range_row, text=str(hi), font=FONT_SMALL,
                 bg=COLORS["bg_mid"], fg=COLORS["text_dim"]).pack(side="right")

    def _add_direction_control(self, parent, color):
        frame = tk.Frame(parent, bg=COLORS["bg_mid"])
        frame.pack(fill="x", padx=14, pady=(16, 0))
        tk.Label(frame, text="Direction du gradient", font=FONT_BOLD,
                 bg=COLORS["bg_mid"], fg=COLORS["text_bright"]).pack(anchor="w")
        var = tk.StringVar(value="magnitude")
        self.extra_vars["direction"] = var
        
        def on_rad_change():
            if self.input_mode.get() == "image":
                self._update_static_image_view()

        for val, lbl in [("magnitude", "Magnitude (X+Y)"), ("x", "Horizontal (Sobel X)"), ("y", "Vertical (Sobel Y)")]:
            tk.Radiobutton(frame, text=lbl, variable=var, value=val,
                           font=FONT_SMALL, bg=COLORS["bg_mid"], fg=color,
                           activebackground=COLORS["bg_mid"], selectcolor=COLORS["bg_card"],
                           command=on_rad_change).pack(anchor="w", padx=6, pady=2)

    def _add_sharpen_control(self, parent, color):
        frame = tk.Frame(parent, bg=COLORS["bg_mid"])
        frame.pack(fill="x", padx=14, pady=(16, 0))
        var = tk.BooleanVar(value=False)
        self.extra_vars["sharpen"] = var
        
        def on_chk_change():
            if self.input_mode.get() == "image":
                self._update_static_image_view()

        tk.Checkbutton(frame, text="Mode Sharpening (Rehaussement)", variable=var,
                       font=FONT_LABEL, bg=COLORS["bg_mid"], fg=color,
                       activebackground=COLORS["bg_mid"], selectcolor=COLORS["bg_card"],
                       command=on_chk_change).pack(anchor="w")

    def _apply_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Vertical.TScrollbar",
                        background=COLORS["bg_card"], troughcolor=COLORS["bg_mid"],
                        bordercolor=COLORS["border"], arrowcolor=COLORS["text_dim"])

    def _select_filter(self, name):
        self.current_filter = name
        self._highlight_button(name)
        self.filter_name_label.config(text=f"Filtre Actif: {name}")
        self._build_param_panel(self.right_panel)
        if self.input_mode.get() == "image":
            self._update_static_image_view()

    def _highlight_button(self, name):
        for n, btn in self.filter_buttons.items():
            if n == name:
                btn.config(bg=COLORS["accent2"], fg="#ffffff", font=FONT_BOLD)
            else:
                btn.config(bg=COLORS["bg_mid"], fg=FILTERS[n]["color"], font=FONT_LABEL)

    def _reset_params(self):
        fdata = FILTERS[self.current_filter]
        for label, key, lo, hi, default, res in fdata["params"]:
            if key in self.param_vars:
                self.param_vars[key].set(default)
        if "direction" in self.extra_vars:
            self.extra_vars["direction"].set("magnitude")
        if "sharpen" in self.extra_vars:
            self.extra_vars["sharpen"].set(False)
        if self.input_mode.get() == "image":
            self._update_static_image_view()

    def _on_input_mode_change(self):
        mode = self.input_mode.get()
        if mode == "camera":
            self.upload_btn.pack_forget()
            self.cam_controls.pack(side="right", padx=6)
            self.orig_title_lbl.config(text="📷 Camera Live (Originale)")
            self._start_camera()
        else:
            self.cam_controls.pack_forget()
            self.upload_btn.pack(side="right", padx=6)
            self.orig_title_lbl.config(text="📁 Image Importee (Originale)")
            self.running = False
            if self.cap:
                self.cap.release()
            self.fps_label.config(text="Source: Image")
            if self.uploaded_image is None:
                self._upload_image_file()
            else:
                self._update_static_image_view()

    def _upload_image_file(self):
        file_path = filedialog.askopenfilename(
            title="Selectionner une Image",
            filetypes=[("Fichiers Image", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff"), ("Tous les fichiers", "*.*")]
        )
        if not file_path:
            return
        img = cv2.imread(file_path)
        if img is None:
            messagebox.showerror("Erreur", "Impossible de charger l'image selectionnee.")
            return
        self.uploaded_image = img
        self._update_static_image_view()

    def _download_filtered_image(self):
        if self._latest_filt is None:
            messagebox.showinfo("Information", "Aucune image filtree a enregistrer.")
            return
        file_path = filedialog.asksaveasfilename(
            title="Enregistrer l'Image Filtree",
            defaultextension=".png",
            filetypes=[("Image PNG", "*.png"), ("Image JPEG", "*.jpg"), ("Image BMP", "*.bmp")]
        )
        if file_path:
            cv2.imwrite(file_path, self._latest_filt)
            messagebox.showinfo("Succes", f"Image filtree enregistree avec succes sous:\n{file_path}")

    def _update_static_image_view(self):
        if self.uploaded_image is None:
            return
        orig = self.uploaded_image.copy()
        try:
            kwargs   = self._get_params()
            fn       = FILTERS[self.current_filter]["fn"]
            filtered = fn(orig, **kwargs)
        except Exception as e:
            filtered = orig.copy()
            cv2.putText(filtered, f"Err: {str(e)[:50]}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        self._latest_orig = orig.copy()
        self._latest_filt = filtered.copy()

        mse_val = calculate_mse(orig, filtered)
        self.mse_label.config(text=f"MSE: {mse_val:.2f}")

        w_orig = self.canvas_orig.winfo_width()
        h_orig = self.canvas_orig.winfo_height()
        w_filt = self.canvas_filt.winfo_width()
        h_filt = self.canvas_filt.winfo_height()

        if w_orig > 10 and h_orig > 10:
            self._show_frame(self.canvas_orig, orig, w_orig, h_orig)
        if w_filt > 10 and h_filt > 10:
            self._show_frame(self.canvas_filt, filtered, w_filt, h_filt)

        self._update_histograms(orig, filtered)

    def _start_camera(self):
        idx = self.camera_index.get()
        self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(idx)
        if not self.cap.isOpened():
            messagebox.showwarning("Camera non detectee",
                "Impossible d'acceder a la camera ID " + str(idx) + ".\n"
                "Vous pouvez basculer en mode 'Importer Image' pour tester les filtres.")
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.running = True
        self._thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._thread.start()

    def _restart_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
        time.sleep(0.3)
        if self.input_mode.get() == "camera":
            self._start_camera()

    def _on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.destroy()

    def _camera_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            try:
                kwargs   = self._get_params()
                fn       = FILTERS[self.current_filter]["fn"]
                filtered = fn(frame, **kwargs)
            except Exception as e:
                filtered = frame.copy()
                cv2.putText(filtered, f"Err: {str(e)[:55]}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            self._latest_orig = frame.copy()
            self._latest_filt = filtered.copy()

            self.after(0, self._update_camera_display, frame.copy(), filtered)

            elapsed = time.time() - self._last_frame_time
            time.sleep(max(0, 1/30 - elapsed))
            now = time.time()
            self._fps = 1.0 / max(0.001, now - self._last_frame_time)
            self._last_frame_time = now

    def _get_params(self):
        kw    = {}
        fdata = FILTERS[self.current_filter]
        for label, key, lo, hi, default, res in fdata["params"]:
            if key in self.param_vars:
                val = self.param_vars[key].get()
                if res >= 1:
                    val = int(round(val))
                kw[key] = val
        if "direction" in self.extra_vars:
            kw["direction"] = self.extra_vars["direction"].get()
        if "sharpen" in self.extra_vars:
            kw["sharpen"] = self.extra_vars["sharpen"].get()
        return kw

    def _update_camera_display(self, orig, filtered):
        if self.input_mode.get() != "camera":
            return
        self.fps_label.config(text=f"FPS: {self._fps:.1f}")
        
        mse_val = calculate_mse(orig, filtered)
        self.mse_label.config(text=f"MSE: {mse_val:.2f}")

        w_orig = self.canvas_orig.winfo_width()
        h_orig = self.canvas_orig.winfo_height()
        w_filt = self.canvas_filt.winfo_width()
        h_filt = self.canvas_filt.winfo_height()

        if w_orig > 10 and h_orig > 10:
            self._show_frame(self.canvas_orig, orig, w_orig, h_orig)
        if w_filt > 10 and h_filt > 10:
            self._show_frame(self.canvas_filt, filtered, w_filt, h_filt)

        self._update_histograms(orig, filtered)

    def _update_histograms(self, orig, filtered):
        w_ho = self.canvas_hist_orig.winfo_width()
        h_ho = self.canvas_hist_orig.winfo_height()
        w_hf = self.canvas_hist_filt.winfo_width()
        h_hf = self.canvas_hist_filt.winfo_height()

        if w_ho > 10 and h_ho > 10:
            h_img_o = generate_histogram_image(orig, w_ho, h_ho)
            self._show_frame(self.canvas_hist_orig, h_img_o, w_ho, h_ho)
        if w_hf > 10 and h_hf > 10:
            h_img_f = generate_histogram_image(filtered, w_hf, h_hf)
            self._show_frame(self.canvas_hist_filt, h_img_f, w_hf, h_hf)

    def _show_frame(self, canvas, frame, cw, ch):
        h, w = frame.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) if len(resized.shape) == 3 else cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        img     = Image.fromarray(rgb)
        photo   = ImageTk.PhotoImage(image=img)
        x       = (cw - nw) // 2
        y       = (ch - nh) // 2
        canvas.delete("all")
        canvas.create_image(x, y, image=photo, anchor="nw")
        canvas._photo = photo


if __name__ == "__main__":
    app = VisionFiltersApp()
    app.mainloop()
