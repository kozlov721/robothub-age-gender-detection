from functools import partial
from typing import List, cast
import blobconverter
from pathlib import Path

import depthai as dai
import numpy as np

from robothub_sdk import (
    App,
    IS_INTERACTIVE,
    CameraResolution,
    Config,
)

from robothub_sdk.device import Device

if IS_INTERACTIVE:
    import cv2


class AgeGender(App):
    next_window_position = (0, 0)
    next_detection = 0

    def on_initialize(self, unused_devices: List[dai.DeviceInfo]):
        self.config.add_defaults(
            send_still_picture=False,
            detect_threshold=0.5
        )
        self.fps = 20
        self.res = CameraResolution.THE_1080_P
        self.preview_size = (640, 640)
        self.obj_detections = []

    def on_configuration(self, old_configuration: Config):
        pass

    def frame_norm(self, bbox):
        return (np.array(bbox) * self.preview_size[0]).astype('int')

    def on_detection(self,
            device_id: str,
            face_detection: dai.ImgDetections,
            # recognition: dai.NNData,
            frame: dai.ImgFrame):

        self.obj_detections = []
        # print(face_detection.detections)
        # ages = np.array(recognition.getLayerFp16('age_conv3'))
        # ages = (ages * 100).astype('int')
        # genders = np.array(recognition.getLayerFp16('prob'))
        # print(genders.shape)
        for i, det in enumerate(face_detection.detections):
            bbox = self.frame_norm(
                [det.xmin, det.ymin, det.xmax, det.ymax]
            )
            print(bbox)
            # self.obj_detections.append((bbox, ages[i]))
            self.obj_detections.append((bbox, 20))


    def on_setup(self, device: Device):
        camera = device.configure_camera(
            dai.CameraBoardSocket.RGB,
            res=self.res,
            fps=self.fps,
            preview_size=self.preview_size,
        )
        camera.initialControl.setSceneMode(
                dai.CameraControl.SceneMode.FACE_PRIORITY)

        copy_manip, copy_manip_stream = device.create_image_manipulator()
        copy_manip.setNumFramesPool(15)
        copy_manip.setMaxOutputFrameSize(
                self.preview_size[0] * self.preview_size[1] * 3)

        device.streams.color_preview.output_node.link(copy_manip.inputImage)
        _, face_det_nn, face_det_nn_passthrough = device.create_nn(
            copy_manip_stream,
            Path(blobconverter.from_zoo(
                name="face-detection-retail-0004",
                shaves=6)),
            nn_family='mobilenet',
            input_size=(300, 300),
            confidence=0.5,
        )

        rec_manip, rec_manip_stream = device.create_image_manipulator()
        rec_manip.setMaxOutputFrameSize(62 * 62 * 3)
        rec_manip.initialConfig.setResize((62, 62))
        rec_manip.inputConfig.setWaitForMessage(True)

        script = device.create_script(
            script_path=Path("./script.py"),
            inputs={
                'face_det_in': face_det_nn,
                'passthrough': face_det_nn_passthrough,
                'preview': copy_manip_stream,
            },
            outputs={
                'manip_cfg': rec_manip.inputConfig,
                'manip_img': rec_manip.inputImage,
            },
        )

        _, recognition_nn, _ = device.create_nn(
            rec_manip_stream,
            Path(blobconverter.from_zoo(
                name="age-gender-recognition-retail-0013",
                shaves=6)),
            input_size=(62, 62),
        )

        if IS_INTERACTIVE:
            device.streams.color_video.consume()
            device.streams.color_video.description = (
                f"{device.name} {device.streams.color_video.description}"
            )
            device.streams.synchronize(
                # (face_det_nn, recognition_nn, device.streams.color_video),
                (face_det_nn, device.streams.color_video),
                partial(self.on_detection, device.id) # type: ignore
            )

        if IS_INTERACTIVE:
            device.streams.color_preview.consume()
            device.streams.color_preview.description = f"{device.name} {device.streams.color_preview.description}"

    def on_update(self):
        if IS_INTERACTIVE:
            for device in self.devices:
                for camera in device.cameras:
                    if camera != dai.CameraBoardSocket.RGB:
                        continue
                    # device.getOutputQueue('recognition')
                    if device.streams.color_preview.last_value:
                        last_val = cast(dai.ImgFrame, device.streams.color_preview.last_value)
                        frame = cast(cv2.Mat, last_val.getCvFrame())
                        for bbox, age in self.obj_detections:
                            # print(bbox)
                            cv2.rectangle(
                                frame,
                                (bbox[0], bbox[1]),
                                (bbox[2], bbox[3]),
                                (0, 255, 0),
                                2
                            )
                            # print(age)
                        cv2.imshow(
                            device.streams.color_preview.description,
                            frame
                        )

        if IS_INTERACTIVE and cv2.waitKey(1) == ord("q"):
            self.stop()


AgeGender().run()
