from functools import partial
from typing import List, cast, Literal
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
        self.msgs = {}

    def on_configuration(self, old_configuration: Config):
        pass

    def make_bbox(self, det: dai.ImgDetection):
        if det.xmin < 0: det.xmin = 0.001
        if det.ymin < 0: det.ymin = 0.001
        if det.xmax > 1: det.xmax = 0.999
        if det.ymax > 1: det.ymax = 0.999
        return (np.array([det.xmin, det.ymin, det.xmax, det.ymax]) * self.preview_size[0]).astype('int')

    def add_msg(self, msg: dai.ImgFrame | dai.ImgDetections | dai.NNData):
        seq = str(msg.getSequenceNum())
        if seq not in self.msgs:
            self.msgs[seq] = {}
        if "recognition" not in self.msgs[seq]:
            self.msgs[seq]["recognition"] = []

        if isinstance(msg, dai.NNData): # name == "recognition":
            self.msgs[seq]["recognition"].append(msg)

        elif isinstance(msg, dai.ImgDetections): # name == "detection":
            self.msgs[seq]["detection"] = msg
            self.msgs[seq]["len"] = len(msg.detections)

        elif isinstance(msg, dai.ImgFrame): # name == "color": # color
            self.msgs[seq]["color"] = msg

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

        device.create_script(
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
            face_det_nn.consume(self.add_msg) # type: ignore
            recognition_nn.consume(self.add_msg) # type: ignore
            device.streams.color_preview.consume(self.add_msg) # type: ignore

        if IS_INTERACTIVE:
            device.streams.color_preview.consume()
            device.streams.color_preview.description = f"{device.name} {device.streams.color_preview.description}"

    def get_msgs(self):
        seq_remove = []

        for seq, msgs in self.msgs.items():
            seq_remove.append(seq)
            if "color" in msgs and "len" in msgs:
                if msgs["len"] == len(msgs["recognition"]):
                    for rm in seq_remove:
                        del self.msgs[rm]
                    return msgs
        return None

    def on_update(self):
        if IS_INTERACTIVE:
            for device in self.devices:
                for camera in device.cameras:
                    if camera != dai.CameraBoardSocket.RGB:
                        continue
                    msgs = self.get_msgs()
                    if not msgs:
                        continue
                    frame = cast(cv2.Mat, msgs['color'].getCvFrame())
                    detections = msgs['detection'].detections
                    recognitions = msgs['recognition']
                    for i, det in enumerate(detections):
                        bbox = self.make_bbox(det)
                        rec = recognitions[i]
                        age = int(rec.getLayerFp16('age_conv3')[0] * 100)
                        gender = rec.getLayerFp16('prob')
                        gender_str = "female" if gender[0] > gender[1] else "male"

                        cv2.rectangle(
                            frame,
                            (bbox[0], bbox[1]),
                            (bbox[2], bbox[3]),
                            (10, 245, 10),
                            2
                        )
                        y = (bbox[1] + bbox[3]) // 2
                        cv2.putText(
                            frame,
                            str(age),
                            (bbox[0], y),
                            cv2.FONT_HERSHEY_TRIPLEX,
                            1.5,
                            (0, 0, 0),
                            8
                        )
                        cv2.putText(
                            frame,
                            str(age),
                            (bbox[0], y),
                            cv2.FONT_HERSHEY_TRIPLEX,
                            1.5,
                            (255, 255, 255),
                            2
                        )
                        cv2.putText(
                            frame,
                            gender_str,
                            (bbox[0], y + 30),
                            cv2.FONT_HERSHEY_TRIPLEX,
                            1.5,
                            (0, 0, 0),
                            8
                        )
                        cv2.putText(
                            frame,
                            gender_str,
                            (bbox[0], y + 30),
                            cv2.FONT_HERSHEY_TRIPLEX,
                            1.5,
                            (255, 255, 255),
                            2
                        )
                    cv2.imshow(
                        device.streams.color_preview.description,
                        frame
                    )

        if IS_INTERACTIVE and cv2.waitKey(1) == ord("q"):
            self.stop()


AgeGender().run()
