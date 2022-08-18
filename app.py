from functools import partial
from typing import List
import blobconverter
from pathlib import Path

import depthai as dai
import numpy as np

from robothub_sdk import (
    App,
    IS_INTERACTIVE,
    CameraResolution,
    InputStream,
    StreamType,
    Config,
)

if IS_INTERACTIVE:
    import cv2

PREVIEW_SIZE = (640, 640)

class AgeGender(App):
    next_window_position = (0, 0)
    next_detection = 0
    camera_controls: List[InputStream] = []

    def on_initialize(self, unused_devices: List[dai.DeviceInfo]):
        self.config.add_defaults(
            send_still_picture=False,
            detect_threshold=0.5
        )
        self.fps = 20
        self.res = CameraResolution.THE_1080_P
        self.obj_detections = []

    def on_configuration(self, old_configuration: Config):
        pass

    def on_detection(self,
            device_id: str,
            obj_data: dai.NNData,
            obj_frame: dai.ImgFrame,
            frame: dai.ImgFrame):

        self.obj_detections = []
        cv_frame = obj_frame.getCvFrame()
        out = np.array(obj_data.getLayerFp16('detection_out'))
        out = out.reshape(len(out) // 7, 7)
        out = out[out[..., 2] > 0.5]
        for detection in out:
            xmin, ymin, xmax, ymax = (detection[3:7] * PREVIEW_SIZE[0]).astype('int')
            # print(f'[{xmin}, {ymin}], [{xmax}, {ymax}]')
            bbox = (xmin, ymin, xmax, ymax)
            self.obj_detections.append((detection, bbox))

    def on_setup(self, device):
        camera = device.configure_camera(
            dai.CameraBoardSocket.RGB,
            res=self.res,
            fps=self.fps,
            preview_size=PREVIEW_SIZE
        )
        camera.initialControl.setSceneMode(
                dai.CameraControl.SceneMode.FACE_PRIORITY)
        face_blob_path = Path(blobconverter.from_zoo(
            name="face-detection-retail-0004",
            shaves=6
        ))
        age_gender_blob_path = Path(blobconverter.from_zoo(
            name="age-gender-recognition-retail-0013",
            shaves=6
        ))

        _, nn_face_det_out, nn_face_det_passthrough = device.create_nn(
            device.streams.color_preview,
            face_blob_path,
            input_size=(300, 300)
        )
        _, nn_ag_det_out, nn_ag_det_passthrough = device.create_nn(
            nn_face_det_passthrough,
            age_gender_blob_path,
            input_size=(62, 62)
        )
        if IS_INTERACTIVE:
            device.streams.color_video.consume()
            device.streams.color_video.description = (
                f"{device.name} {device.streams.color_video.description}"
            )
            device.streams.synchronize(
                (nn_face_det_out, nn_face_det_passthrough, device.streams.color_video),
                partial(self.on_detection, device.id)
            )
        self.camera_controls.append(device.streams.color_control)

        if IS_INTERACTIVE:
            device.streams.color_preview.consume()
            stream_id = device.streams.color_preview.description = f"{device.name} {device.streams.color_preview.description}"

    def on_update(self):
        if IS_INTERACTIVE:
            for device in self.devices:
                for camera in device.cameras:
                    if camera != dai.CameraBoardSocket.RGB:
                        continue
                    if device.streams.color_preview.last_value:
                        frame = (device
                            .streams
                            .color_preview
                            .last_value
                            .getCvFrame())
                    else:
                        frame = np.empty([1, 1])
                    for detection, bbox in self.obj_detections:
                        # print(bbox)
                        cv2.rectangle(
                            frame,
                            (bbox[0], bbox[1]),
                            (bbox[2], bbox[3]),
                            (0, 255, 0),
                            2
                    )
                    cv2.imshow(
                        device.streams.color_preview.description,
                        frame
                    )

        if IS_INTERACTIVE and cv2.waitKey(1) == ord("q"):
            self.stop()


AgeGender().run()
