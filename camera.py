import argparse
import time
from io import BytesIO

from picamera import PiCamera, PiCameraCircularIO
from threading import Thread


class ThreadedCamera(Thread):
    def __init__(self, fps, r_x, r_y):
        super(ThreadedCamera, self).__init__()
        self.fps = fps
        self.period = self.get_period(fps)
        self.r_x = r_x
        self.r_y = r_y
        self.buffer = BytesIO()

    def get_period(self, fps):
        return 1.0/fps

    def run(self):
        with PiCamera() as camera:
            camera.resolution = (self.r_x, self.r_y) # set resolution
            camera.start_recording(self.buffer, format='h264')
            camera.wait_recording()


class Consumer(Thread):
    def __init__(self, camera):
        super(Consumer, self).__init__()
        self.camera = camera
        # self.data = np.empty((self.camera.r_y * self.camera.r_x * 3,), dtype=np.uint8)
        self.data = self.camera.buffer

    def run(self):
        correction = 0 # This value is to make sure that the frame time is always as accurate as it can be
        next_time = int(time.time() + 2) # give some time to warm up
        while True:
                while (time.time() <= next_time):
                    pass  # Wait for the next time
                # Perform the calculation right away to avoid excessive waiting
                if correction < self.camera.fps:
                    next_time = next_time + self.camera.period

                next_time = next_time + self.camera.period if correction < self.camera.fps else int(time.time()) + self.camera.period
                self.data = self.camera.buffer.getvalue()
                print("Consume: ", time.time(), end="\r")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fps', type=int, default=10, help='Camera framerate')
    parser.add_argument('-x', type=int, default=1280, help='X resolution')
    parser.add_argument('-y', type=int, default=720, help='Y resolution')
    opt = parser.parse_args()

    picam = ThreadedCamera(opt.fps, opt.x, opt.y)
    consumer = Consumer(picam)
    picam.start()
    consumer.start()
    while True:
        pass
