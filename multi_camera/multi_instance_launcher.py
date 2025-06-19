import subprocess
from multiprocessing import Process

# ---------------- Global Configuration ----------------
CAMERA_LIST = ["camera1", "camera2","camera3","camera4","camera5","camera6","camera7"]
LIVE_MODE = True  

# ------------------------------------------------------
def run_camera(camera_name, live_mode):
    args = ["python", "main.py", "--camera", camera_name]
    if live_mode:
        args.append("--live")
    subprocess.run(args)

if __name__ == "__main__":
    processes = []
    for cam in CAMERA_LIST:
        p = Process(target=run_camera, args=(cam, LIVE_MODE))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("✅ All camera detection processes completed.")
