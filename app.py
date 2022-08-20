from typing import cast
from functools import partial
from pathlib import Path

from robothub_sdk import App, IS_INTERACTIVE, CameraResolution, StreamType
from robothub_sdk.device import Device

import depthai as dai
import numpy as np
import time

if IS_INTERACTIVE:
    import cv2


class AgeGender(App):
    def on_initialize(self, _):
        self.config.add_defaults(
            detection_threshold=0.7,
            time_delta=10,
        )
        self.fps = 20
        self.res = CameraResolution.THE_1080_P
        self.preview_size = (1080, 1080)
        self.msgs = {}
        self.last_detection = int(time.monotonic())

    def make_bbox(self, det: dai.ImgDetection):
        bbox = np.array([det.xmin, det.ymin, det.xmax, det.ymax])
        bbox = np.clip(bbox, 0.0, 1.0)
        return (bbox * self.preview_size[0]).astype("int")

    def add_msg(self, msg: dai.ImgFrame | dai.ImgDetections | dai.NNData):
        seq = str(msg.getSequenceNum())
        if seq not in self.msgs:
            self.msgs[seq] = {}
        if "recognition" not in self.msgs[seq]:
            self.msgs[seq]["recognition"] = []

        if isinstance(msg, dai.NNData):
            self.msgs[seq]["recognition"].append(msg)

        elif isinstance(msg, dai.ImgDetections):
            self.msgs[seq]["detection"] = msg
            self.msgs[seq]["len"] = len(msg.detections)

        elif isinstance(msg, dai.ImgFrame):
            self.msgs[seq]["color"] = msg

        if len(self.msgs) > 15:
            self.msgs.popitem()

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

    def on_setup(self, device: Device):
        camera = device.configure_camera(
            dai.CameraBoardSocket.RGB,
            res=self.res,
            fps=self.fps,
            preview_size=self.preview_size,
        )
        camera.initialControl.setSceneMode(
                dai.CameraControl.SceneMode.FACE_PRIORITY)

        self.scale_x = camera.getVideoWidth() // self.preview_size[0]
        self.scale_y = camera.getVideoHeight() // self.preview_size[1]

        _, face_det_nn, face_det_nn_passthrough = device.create_nn(
            device.streams.color_preview,
            Path("./face-detection.blob"),
            nn_family="mobilenet",
            input_size=(300, 300),
            confidence=cast(float, self.config.detection_threshold),
        )

        rec_manip, rec_manip_stream = device.create_image_manipulator()
        rec_manip.inputConfig.setWaitForMessage(True)

        device.create_script(
            script_path=Path("./script.py"),
            inputs={
                "face_det_in": face_det_nn,
                "passthrough": face_det_nn_passthrough,
                "preview": device.streams.color_preview,
            },
            outputs={
                "manip_cfg": rec_manip.inputConfig,
                "manip_img": rec_manip.inputImage,
            },
        )

        _, rec_nn, _ = device.create_nn(
            rec_manip_stream,
            Path("./age-gender-recognition.blob"),
            input_size=(62, 62),
        )

        face_det_nn.consume(self.add_msg)  # type: ignore
        rec_nn.consume(self.add_msg)  # type: ignore
        device.streams.color_preview.consume(self.add_msg)  # type: ignore

        if IS_INTERACTIVE:
            device.streams.color_preview.description = (
                f"{device.name} {device.streams.color_preview.description}"
            )
        else:
            res_manip, res_manip_stream = device.create_image_manipulator()
            res_manip.initialConfig.setResize(1056, self.preview_size[1])
            res_manip.setMaxOutputFrameSize(1056 * self.preview_size[1] * 3)
            device.streams.color_video.output_node.link(res_manip.inputImage)
            encoder = device.create_encoder(
                res_manip_stream.output_node,
                fps=self.fps,
                profile=dai.VideoEncoderProperties.Profile.MJPEG,
                quality=80,
            )
            encoder_stream = device.streams.create(
                encoder,
                encoder.bitstream,
                stream_type=StreamType.FRAME,
                rate=self.fps,
            )
            encoder_stream.consume(
                    partial(self.on_recognition, device.id))  # type: ignore
            device.streams.color_video.publish()

    def _nndata_to_age_gender(self, rec: dai.NNData):
        age = int(rec.getLayerFp16("age_conv3")[0] * 100)
        gen = rec.getLayerFp16("prob")
        gender = "female" if gen[0] > gen[1] else "male"
        return age, gender

    def on_recognition(self, device_id: str, frame: dai.ImgFrame):
        msgs = self.get_msgs()
        if not msgs:
            return
        t = int(time.monotonic())
        if t - self.last_detection < cast(int, self.config.time_delta):
            return
        self.last_detection = t
        detections = msgs["detection"].detections
        recognitions = msgs["recognition"]
        for det, rec in zip(detections, recognitions):
            age, gender = self._nndata_to_age_gender(rec)
            data = {
                "confidence": det.confidence,
                "xmin": det.xmin,
                "ymin": det.ymin,
                "xmax": det.xmax,
                "ymax": det.ymax,
                "age": age,
                "gender": gender,
            }

            self.send_detection(
                f"Detection from device {device_id}",
                frames=[(frame, "jpeg")],
                data=data,
                tags=["detection"]
            )

    def on_update(self):
        if IS_INTERACTIVE:
            for device in self.devices:
                for camera in device.cameras:
                    if camera != dai.CameraBoardSocket.RGB:
                        continue
                    msgs = self.get_msgs()
                    if not msgs:
                        continue
                    frame = cast(cv2.Mat, msgs["color"].getCvFrame())
                    detections = msgs["detection"].detections
                    recognitions = msgs["recognition"]

                    for det, rec in zip(detections, recognitions):
                        bbox = self.make_bbox(det)
                        age, gender = self._nndata_to_age_gender(rec)
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
                            8,
                        )
                        cv2.putText(
                            frame,
                            str(age),
                            (bbox[0], y),
                            cv2.FONT_HERSHEY_TRIPLEX,
                            1.5,
                            (255, 255, 255),
                            2,
                        )
                        cv2.putText(
                            frame,
                            gender,
                            (bbox[0], y + 30),
                            cv2.FONT_HERSHEY_TRIPLEX,
                            1.5,
                            (0, 0, 0),
                            8,
                        )
                        cv2.putText(
                            frame,
                            gender,
                            (bbox[0], y + 30),
                            cv2.FONT_HERSHEY_TRIPLEX,
                            1.5,
                            (255, 255, 255),
                            2,
                        )
                    cv2.imshow(
                        device.streams.color_preview.description,
                        frame,
                    )

        if IS_INTERACTIVE and cv2.waitKey(1) == ord("q"):
            self.stop()


AgeGender().run()
