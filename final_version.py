
import threading
import time

import cv2
import customtkinter as ctk
import torch
from PIL import Image
from torchvision import models

# AI MODEL
weights = models.MobileNet_V3_Small_Weights.DEFAULT
model = models.mobilenet_v3_small(weights=weights).eval()
preprocess = weights.transforms()

# CAMERA
cap = cv2.VideoCapture(0)
frame = None
lock = threading.Lock()
busy = False
running = True
camera_available = cap.isOpened()


def set_camera_available(available):
    global camera_available
    camera_available = available
    if not busy:
        status_label.configure(text="Ready" if available else "Camera unavailable")
        analyze_button.configure(state="normal" if available else "disabled")
    if not available:
        video.configure(image=None, text="Camera unavailable\nCheck that a camera is connected.")
        video.image = None


def camera():
    global frame
    was_available = cap.isOpened()
    if not was_available:
        return

    while running:
        ok, img = cap.read()
        if ok:
            with lock:
                frame = img
            if not was_available:
                was_available = True
                if running:
                    root.after(0, lambda: set_camera_available(True))
        else:
            with lock:
                frame = None
            if was_available:
                was_available = False
                if running:
                    root.after(0, lambda: set_camera_available(False))
            time.sleep(0.1)


def finish_analysis(text=None, error=None):
    global busy
    busy = False
    analyze_button.configure(state="normal" if camera_available else "disabled")
    if error:
        status_label.configure(text="Analysis failed")
        result_labels[0].configure(text=error)
        for label in result_labels[1:]:
            label.configure(text="")
        for bar in result_bars:
            bar.set(0)
    else:
        status_label.configure(text="Analysis completed")
        for index, line in enumerate(text.splitlines()):
            result_labels[index].configure(text=line)
        for index in range(len(text.splitlines()), len(result_labels)):
            result_labels[index].configure(text="")
        for bar, value in zip(result_bars, confidences):
            bar.set(value / 100)


def analyze():
    global confidences
    try:
        with lock:
            img = None if frame is None else frame.copy()

        if img is None:
            if running:
                root.after(0, lambda: finish_analysis(error="No camera image available."))
            return

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = preprocess(Image.fromarray(img)).unsqueeze(0)

        with torch.no_grad():
            scores = model(x).softmax(1)[0]
            values, indices = scores.topk(3)

        confidences = [value.item() * 100 for value in values]
        text = "\n".join(
            f"{weights.meta['categories'][index]}  —  {value.item():.1%}"
            for value, index in zip(values, indices)
        )
        if running:
            root.after(0, lambda: finish_analysis(text=text))
    except Exception as exc:
        message = f"Could not complete analysis: {exc}"
        if running:
            root.after(0, lambda message=message: finish_analysis(error=message))


def start_analysis():
    global busy
    if busy:
        return
    if not camera_available:
        status_label.configure(text="Camera unavailable")
        return

    busy = True
    analyze_button.configure(state="disabled")
    status_label.configure(text="Analyzing...")
    for label in result_labels:
        label.configure(text="")
    for bar in result_bars:
        bar.set(0)
    try:
        threading.Thread(target=analyze, daemon=True).start()
    except RuntimeError as exc:
        finish_analysis(error=f"Could not start analysis: {exc}")


def update():
    with lock:
        img = None if frame is None else frame.copy()

    if img is not None:
        width = max(video.winfo_width() - 24, 1)
        height = max(video.winfo_height() - 24, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        display_image = Image.fromarray(img)
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        display_image.thumbnail((width, height), resampling)
        photo = ctk.CTkImage(
            light_image=display_image,
            dark_image=display_image,
            size=display_image.size,
        )
        video.configure(image=photo, text="")
        video.image = photo

    if running:
        root.after(30, update)


def close():
    global running
    running = False
    cap.release()
    root.destroy()


# GUI
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
root = ctk.CTk()
root.title("AI Vision – Object Recognition")
root.geometry("900x760")
root.minsize(640, 620)
root.configure(fg_color="#101820")
root.protocol("WM_DELETE_WINDOW", close)

root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=4)
root.grid_rowconfigure(2, weight=1)

header = ctk.CTkFrame(root, fg_color="transparent", corner_radius=0)
header.grid(row=0, column=0, sticky="ew")
header.grid_columnconfigure(0, weight=1)
ctk.CTkLabel(
    header,
    text="AI Vision – Object Recognition",
    font=("Segoe UI", 22, "bold"),
    text_color="#f2f6fa",
).grid(row=0, column=0, sticky="w", padx=24, pady=(20, 0))
ctk.CTkLabel(
    header,
    text="Live camera preview and on-device object classification",
    font=("Segoe UI", 10),
    text_color="#a9bac9",
).grid(row=1, column=0, sticky="w", padx=24, pady=(4, 12))

preview_card = ctk.CTkFrame(root, fg_color="#1b2733", corner_radius=14)
preview_card.grid(row=1, column=0, sticky="nsew", padx=24, pady=(0, 12))
preview_card.grid_rowconfigure(0, weight=1)
preview_card.grid_columnconfigure(0, weight=1)
video = ctk.CTkLabel(
    preview_card,
    text="Starting camera...",
    fg_color="#0b1118",
    text_color="#a9bac9",
    font=("Segoe UI", 12),
    corner_radius=10,
)
video.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)

results_card = ctk.CTkFrame(root, fg_color="#1b2733", corner_radius=14)
results_card.grid(row=2, column=0, sticky="nsew", padx=24, pady=(0, 12))
results_card.grid_columnconfigure(0, weight=1)
ctk.CTkLabel(
    results_card,
    text="Recognition results",
    font=("Segoe UI", 12, "bold"),
    text_color="#f2f6fa",
).grid(
    row=0, column=0, columnspan=2, sticky="w", padx=18, pady=(14, 8)
)

result_labels = []
result_bars = []
confidences = []
for row in range(3):
    result_label = ctk.CTkLabel(
        results_card,
        text="Run an analysis to see the top predictions." if row == 0 else "",
        font=("Segoe UI", 11),
        text_color="#f2f6fa",
        anchor="w",
    )
    result_labels.append(result_label)
    result_label.grid(row=1 + row, column=0, sticky="ew", padx=(18, 12), pady=(2, 4))
    bar = ctk.CTkProgressBar(
        results_card,
        height=10,
        corner_radius=5,
        progress_color="#3b8ed0",
        fg_color="#0b1118",
    )
    bar.set(0)
    bar.grid(row=1 + row, column=1, sticky="ew", padx=(0, 18), pady=(2, 4))
    result_bars.append(bar)
results_card.grid_columnconfigure(1, weight=1)

footer = ctk.CTkFrame(root, fg_color="transparent", corner_radius=0)
footer.grid(row=3, column=0, sticky="ew")
footer.grid_columnconfigure(0, weight=1)
status_label = ctk.CTkLabel(
    footer,
    text="Ready" if camera_available else "Camera unavailable",
    font=("Segoe UI", 10),
    text_color="#a9bac9",
)
status_label.grid(row=0, column=0, sticky="w", padx=24, pady=(0, 20))
actions = ctk.CTkFrame(footer, fg_color="transparent", corner_radius=0)
actions.grid(row=0, column=1, sticky="e", padx=24, pady=(0, 20))
analyze_button = ctk.CTkButton(
    actions,
    text="Start Analysis",
    command=start_analysis,
    font=("Segoe UI", 11, "bold"),
    height=40,
    corner_radius=9,
)
analyze_button.pack(side="left", padx=(0, 10))
ctk.CTkButton(
    actions,
    text="Exit",
    command=close,
    font=("Segoe UI", 10),
    height=40,
    corner_radius=9,
    fg_color="#364553",
    hover_color="#465969",
).pack(side="left")

if not camera_available:
    set_camera_available(False)
else:
    threading.Thread(target=camera, daemon=True).start()
root.after(30, update)
root.mainloop()
 