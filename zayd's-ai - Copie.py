import logging
import queue
import threading
import time
import tkinter as tk

import cv2
import torch
from PIL import Image, ImageDraw, ImageTk
from torchvision import models


PREVIEW_SIZE = (480, 360)
CLICK_CROP_RATIO = 0.25
PREVIEW_HIGHLIGHT_COLOR = "#00a000"
MARKER_COLOR = "#ff8c00"


class VisionApp:
    def __init__(self):
        self.weights = models.MobileNet_V3_Small_Weights.DEFAULT
        self.model = models.mobilenet_v3_small(weights=self.weights).eval()
        self.preprocess = self.weights.transforms()
        self.inference_lock = threading.Lock()

        self.capture = cv2.VideoCapture(0)
        self.frame = None
        self.frame_lock = threading.Lock()
        self.running = True
        self.analysis_busy = False
        self.click_busy = False
        self.selected_point = None
        self.messages = queue.Queue()
        self.camera_thread = None

        self.root = tk.Tk()
        self.root.title("AI Vision - PyTorch")

        self.preview_window = tk.Toplevel(self.root)
        self.preview_window.title("AI Vision - Live Preview")
        self.preview_window.protocol("WM_DELETE_WINDOW", self.close)

        self.result = tk.Label(
            self.root,
            text="Ready to recognize objects!",
            font=("Arial", 13),
            wraplength=420,
        )
        self.result.pack(padx=10, pady=10)

        tk.Button(self.root, text="Analyze!", command=self.start_analysis).pack(
            pady=5
        )
        tk.Button(self.root, text="Exit", command=self.close).pack(pady=5)

        self.preview_status = tk.Label(
            self.preview_window, text="Starting camera..."
        )
        self.preview_status.pack(pady=(8, 0))

        self.selected_result = tk.Label(
            self.preview_window,
            text="Left-click an object to identify it",
            fg=PREVIEW_HIGHLIGHT_COLOR,
            font=("Arial", 12, "bold"),
        )
        self.selected_result.pack(pady=8)

        self.video = tk.Label(self.preview_window)
        self.video.pack()
        self.video.bind("<Button-1>", self.identify_object_at_click)

        self.camera_ready = self.capture.isOpened()
        if self.camera_ready:
            self.camera_thread = threading.Thread(
                target=self.camera_worker, daemon=True
            )
            self.camera_thread.start()
            self.preview_status.config(text="Camera ready")
        else:
            self.preview_status.config(text="Camera unavailable")

        self.root.after(30, self.update_ui)

    def camera_worker(self):
        reported_failure = False
        while self.running:
            ok, image = self.capture.read()
            if not ok:
                if not reported_failure:
                    self.messages.put(("camera_error", "Camera read failed; retrying"))
                    reported_failure = True
                time.sleep(0.05)
                continue

            if reported_failure:
                self.messages.put(("camera_status", "Camera ready"))
                reported_failure = False

            with self.frame_lock:
                self.frame = image

    def get_current_frame(self):
        with self.frame_lock:
            return None if self.frame is None else self.frame.copy()

    def classify_image(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.preprocess(Image.fromarray(rgb_image)).unsqueeze(0)

        with self.inference_lock, torch.no_grad():
            scores = self.model(tensor).softmax(1)[0]
            values, indices = scores.topk(3)

        predictions = [
            (
                self.weights.meta["categories"][index.item()],
                value.item(),
            )
            for value, index in zip(values, indices)
        ]
        return predictions

    def start_analysis(self):
        if self.analysis_busy:
            return

        image = self.get_current_frame()
        if image is None:
            self.result.config(text="No camera image")
            return

        self.analysis_busy = True
        self.result.config(text="Analyzing...")
        threading.Thread(
            target=self.analyze_frame, args=(image,), daemon=True
        ).start()

    def analyze_frame(self, image):
        try:
            predictions = self.classify_image(image)
            text = "\n".join(
                f"{label}: {score:.1%}" for label, score in predictions
            )
            self.messages.put(("analysis_result", text))
        except Exception as error:
            logging.exception("Full-frame image recognition failed")
            self.messages.put(("analysis_error", str(error)))

    def identify_object_at_click(self, event):
        if self.click_busy:
            return

        image = self.get_current_frame()
        if image is None:
            self.selected_result.config(
                text="No camera image", fg=PREVIEW_HIGHLIGHT_COLOR
            )
            return

        widget_width = self.video.winfo_width()
        widget_height = self.video.winfo_height()
        if widget_width <= 0 or widget_height <= 0:
            return

        height, width = image.shape[:2]
        x = min(width - 1, max(0, int(event.x * width / widget_width)))
        y = min(height - 1, max(0, int(event.y * height / widget_height)))
        crop_width = max(1, int(width * CLICK_CROP_RATIO))
        crop_height = max(1, int(height * CLICK_CROP_RATIO))
        left = min(max(0, x - crop_width // 2), width - crop_width)
        top = min(max(0, y - crop_height // 2), height - crop_height)
        crop = image[top : top + crop_height, left : left + crop_width].copy()

        self.click_busy = True
        self.selected_point = (x, y)
        self.selected_result.config(
            text="Identifying clicked area...", fg=PREVIEW_HIGHLIGHT_COLOR
        )
        threading.Thread(
            target=self.analyze_clicked_area,
            args=(crop, (x, y)),
            daemon=True,
        ).start()

    def analyze_clicked_area(self, crop, point):
        try:
            label, score = self.classify_image(crop)[0]
            text = f"Selected: {label} ({score:.1%})"
            self.messages.put(("click_result", text, point))
        except Exception as error:
            logging.exception("Clicked-area image recognition failed")
            self.messages.put(("click_error", str(error)))

    def process_messages(self):
        while True:
            try:
                message = self.messages.get_nowait()
            except queue.Empty:
                break

            kind = message[0]
            if kind == "analysis_result":
                self.analysis_busy = False
                self.result.config(text=message[1])
            elif kind == "analysis_error":
                self.analysis_busy = False
                self.result.config(text=f"Recognition failed: {message[1]}")
            elif kind == "click_result":
                self.click_busy = False
                self.selected_result.config(
                    text=message[1], fg=PREVIEW_HIGHLIGHT_COLOR
                )
                self.selected_point = message[2]
            elif kind == "click_error":
                self.click_busy = False
                self.selected_result.config(
                    text=f"Recognition failed: {message[1]}",
                    fg=PREVIEW_HIGHLIGHT_COLOR,
                )
            elif kind == "camera_error":
                self.preview_status.config(text=message[1])
            elif kind == "camera_status":
                self.preview_status.config(text=message[1])

    def update_ui(self):
        self.process_messages()
        image = self.get_current_frame()

        if image is not None:
            height, width = image.shape[:2]
            preview = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            preview_image = Image.fromarray(preview).resize(PREVIEW_SIZE)
            if self.selected_point is not None:
                x, y = self.selected_point
                preview_x = int(x * PREVIEW_SIZE[0] / width)
                preview_y = int(y * PREVIEW_SIZE[1] / height)
                draw = ImageDraw.Draw(preview_image)
                draw.ellipse(
                    (preview_x - 8, preview_y - 8, preview_x + 8, preview_y + 8),
                    outline=MARKER_COLOR,
                    width=3,
                )

            photo = ImageTk.PhotoImage(preview_image)
            self.video.config(image=photo)
            self.video.image = photo

        if self.running:
            self.root.after(30, self.update_ui)

    def close(self):
        if not self.running:
            return

        self.running = False
        self.capture.release()
        if self.camera_thread is not None:
            self.camera_thread.join(timeout=1.0)
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    VisionApp().run()


if __name__ == "__main__":
    main()
