"""Draw the right hand as two stick-figure skeletons: current vs commanded joint state."""

import cv2
import numpy as np
import torch

from common import ensure_fk_ready

BACKGROUND_COLOR = (238, 239, 241)
CURRENT_COLOR = (120, 200, 120)   # green: where the hand is
DESIRED_COLOR = (70, 110, 220)    # blue: where the policy wants it
CYLINDER_COLOR = (180, 120, 110)
JOINT_RADIUS = 4
LINK_THICKNESS = 3
IMAGE_MARGIN = 12
# Fraction of the half-view the cylinder radius should take up -- smaller zooms in.
CYLINDER_VIEW_FRACTION = 0.32
# Half-view in metres when there is no object to frame against.
HAND_HALF_VIEW = 0.16

# Look at the palm from its -x (palmar) side, thumb-to-little-finger axis up.
CAMERA_POSITION = np.array([0.1, -1.2, 0.5])
VIEW_UP = np.array([0.0, 0.0, 1.0])


def _hand_link_chain(entity, side="right"):
    """``[(link, parent_link_or_None), ...]`` for the palm and every finger segment."""
    base = entity.links[0].idx
    by_local = {link.idx_local: link for link in entity.links}
    palm = next(link for link in entity.links if link.name.endswith(f"{side}_wuji_palm"))
    chain = [(palm, None)]
    for link in entity.links:
        if link.name.startswith(f"{side}_finger"):
            chain.append((link, by_local[link.parent_idx - base]))
    return chain


def _hand_qs_indexes(entity, side="right"):
    """Generalized-coordinate indexes of the hand joints, in finger/segment order."""
    indexes = []
    for joint in entity.joints:
        if joint.name.startswith(f"{side}_finger"):
            indexes.extend(int(i) for i in joint.qs_idx_local)
    return indexes


class JointView:
    """One image per frame: the right-hand URDF skeleton at the current and desired qpos."""

    def __init__(self, hand_controller, task_node, *, robot_name="tianji", side="right", width=420):
        self.entity = hand_controller.entity
        self.robot_name = robot_name
        self.side = side
        self.width = int(width) + int(width) % 2
        self.task_node = task_node
        self.chain = _hand_link_chain(self.entity, side)
        self.link_locals = [link.idx_local for link, _ in self.chain]
        self.parent_row = [
            None if parent is None else self.link_locals.index(parent.idx_local)
            for _, parent in self.chain
        ]
        self.palm_row = 0
        self.hand_qs = _hand_qs_indexes(self.entity, side)
        ensure_fk_ready(self.entity)
        cylinder = getattr(task_node, "cylinder", None)
        if hasattr(task_node, "drill"):
            drill = task_node.drill
            bounds = drill.get_local_bounding_box()
            extent = (bounds.high - bounds.low).detach().cpu().numpy()
            self.object_position_key = "drill_position"
            self.object_radius = float(max(extent) / 2)
            self.object_base_offset = 0.0
        elif cylinder is not None:
            self.object_position_key = "cylinder_position"
            self.object_radius = float(cylinder.radius)
            self.object_base_offset = float(cylinder.height) / 2
        elif getattr(task_node, "target_object", None) is not None:
            # tianji_sim_real tabletop tasks (drill board, hammer): no object marker,
            # frame the hand alone at a fixed scale.
            self.object_position_key = None
            self.object_radius = HAND_HALF_VIEW * CYLINDER_VIEW_FRACTION
            self.object_base_offset = 0.0
        else:
            raise TypeError("JointView task must expose a cylinder, drill or target_object")
        self._frames = []

    # -- geometry --------------------------------------------------------

    def _skeleton_in_palm(self, hand_qpos):
        """Joint origins of every chain link in world coordinates."""
        qpos = self.entity.get_qpos().detach().clone()
        if qpos.ndim > 1:
            qpos = qpos[0]
        qpos[self.hand_qs] = torch.as_tensor(hand_qpos).detach().to(qpos.dtype).to(qpos.device)
        positions, quaternions = self.entity.forward_kinematics(
            qpos, links_idx_local=self.link_locals
        )
        if positions.ndim == 3:
            positions, quaternions = positions[0], quaternions[0]
        positions = positions.detach().cpu().numpy()
        palm_pos = positions[self.palm_row]
        palm_quat = quaternions[self.palm_row].detach().cpu().numpy()
        return positions, palm_pos, palm_quat

    def _object_in_world(self, obs):
        task = obs[self.task_node.name] if self.task_node.name in obs else obs
        centre = np.asarray(
            task[self.object_position_key].detach().cpu(), float
        ).reshape(-1, 3)[0]
        return centre

    def _project(self, points, forward):
        """Orthographic projection using the scene camera direction."""
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, VIEW_UP)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        return np.stack([points @ right, points @ up], axis=-1)

    def _to_pixels(self, plane_points, centre, scale):
        x = IMAGE_MARGIN + (plane_points[:, 0] - centre[0]) * scale + (self.width - 2 * IMAGE_MARGIN) / 2
        y = IMAGE_MARGIN + (centre[1] - plane_points[:, 1]) * scale + (self.height - 2 * IMAGE_MARGIN) / 2
        return np.stack([x, y], axis=-1).round().astype(np.int32)

    # -- frame ----------------------------------------------------------

    def update(self, obs, action):
        current = action[self.robot_name] if self.robot_name in action else action
        desired_qpos = current[f"{self.side}_hand"] if f"{self.side}_hand" in current else current["right_hand"]
        robot_obs = obs[self.robot_name] if self.robot_name in obs else obs
        current_qpos = robot_obs[f"{self.side}_hand"]["joint_position"]

        skel_current, palm_pos, palm_quat = self._skeleton_in_palm(current_qpos)
        skel_desired, _, _ = self._skeleton_in_palm(desired_qpos)
        cylinder = self._object_in_world(obs) if self.object_position_key else palm_pos
        # The scene camera looks at the tabletop point beneath the cylinder, not
        # the cylinder centre. Match that target exactly.
        forward_target = cylinder.copy()
        forward_target[2] -= self.object_base_offset
        forward = forward_target - CAMERA_POSITION

        plane_current = self._project(skel_current, forward)
        plane_desired = self._project(skel_desired, forward)
        plane_cylinder = self._project(cylinder[None, :], forward)[0]

        self.height = self.width  # square panel
        # Zoom so the cylinder fills CYLINDER_VIEW_FRACTION of the half-view; frame it
        # on the midpoint between the cylinder and the hand so both stay in view.
        half = self.object_radius / CYLINDER_VIEW_FRACTION
        scale = (min(self.width, self.height) - 2 * IMAGE_MARGIN) / (2 * half)
        centre = 0.5 * (plane_cylinder + plane_current.mean(axis=0))
        if self.object_position_key is None:
            centre = plane_current.mean(axis=0)

        image = np.full((self.height, self.width, 3), BACKGROUND_COLOR, np.uint8)
        if self.object_position_key is not None:
            radius_px = int(round(self.object_radius * scale))
            cv2.circle(image, tuple(self._to_pixels(plane_cylinder[None, :], centre, scale)[0]),
                       radius_px, CYLINDER_COLOR, 2, cv2.LINE_AA)
        for plane, colour in ((plane_desired, DESIRED_COLOR), (plane_current, CURRENT_COLOR)):
            pixels = self._to_pixels(plane, centre, scale)
            for row, parent in enumerate(self.parent_row):
                if parent is not None:
                    cv2.line(image, tuple(pixels[parent]), tuple(pixels[row]), colour,
                             LINK_THICKNESS, cv2.LINE_AA)
            for point in pixels:
                cv2.circle(image, tuple(point), JOINT_RADIUS, colour, -1, cv2.LINE_AA)
        return image

    def record(self, obs, action):
        image = self.update(obs, action)
        self._frames.append(image)
        return image

    def frames(self):
        return self._frames


def _to_frame(points, origin, quat):
    """Express world ``points`` in the frame at ``origin`` with quaternion ``quat`` (wxyz)."""
    w, x, y, z = quat
    rotation = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return (np.asarray(points, float) - np.asarray(origin, float)) @ rotation
