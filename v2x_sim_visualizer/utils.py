import matplotlib.pyplot as plt
import numpy as np
from nuscenes.utils.geometry_utils import view_points, BoxVisibility, transform_matrix
from typing import Tuple, List
import os
import os.path as osp
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion
from matplotlib.axes import Axes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
import math
from PIL import Image
from nuscenes.lidarseg.lidarseg_utils import colormap_to_colors,  \
    get_labels_in_coloring, create_lidarseg_legend, paint_points_label

def render_box_translate(box,
               axis: Axes,
               view: np.ndarray = np.eye(3),
               normalize: bool = False,
               colors: Tuple = ('b', 'r', 'k'),
               linewidth: float = 2) -> None:

        """
        Renders the box in the provided Matplotlib axis.
        :param axis: Axis onto which the box should be drawn.
        :param view: <np.array: 3, 3>. Define a projection in needed (e.g. for drawing projection in an image).
        :param normalize: Whether to normalize the remaining coordinate.
        :param colors: (<Matplotlib.colors>: 3). Valid Matplotlib colors (<str> or normalized RGB tuple) for front,
            back and sides.
        :param linewidth: Width in pixel of the box sides.
        """
        temp = box.corners()

        corners = view_points(temp, view, normalize=normalize)[:2, :]

        def draw_rect(selected_corners, color):
            prev = selected_corners[-1]
            for corner in selected_corners:
                axis.plot([prev[0], corner[0]], [prev[1], corner[1]], color=color, linewidth=linewidth)
                prev = corner
        
        # Draw the sides
        for i in range(4):
            axis.plot([corners.T[i][0], corners.T[i + 4][0]],
                      [corners.T[i][1], corners.T[i + 4][1]],
                      color=colors[2], linewidth=linewidth)

        # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
        draw_rect(corners.T[:4], colors[0])
        draw_rect(corners.T[4:], colors[1])

        # Draw line indicating the front
        center_bottom_forward = np.mean(corners.T[2:4], axis=0)
        center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
        axis.plot([center_bottom[0], center_bottom_forward[0]],
               [center_bottom[1], center_bottom_forward[1]],
               color=colors[0], linewidth=linewidth)

def map_pointcloud_to_global(v2x_sim,
                            pointsensor_token: str) -> Tuple:
    """
    Given a point sensor (lidar/radar) token and camera sample_data token, load point-cloud and map it to the image
    plane.
    :param pointsensor_token: Lidar/radar sample_data token.
    :param render_intensity: Whether to render lidar intensity instead of point depth.
    :return (pointcloud <np.float: 2, n)>, coloring <np.float: n>, image <Image>).
    """
    pointsensor = v2x_sim.get('sample_data', pointsensor_token)
    pcl_path = osp.join(v2x_sim.dataroot, pointsensor['filename'])
    if pointsensor['sensor_modality'] == 'lidar':
        pc = LidarPointCloud.from_file(pcl_path)

    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = v2x_sim.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))

    # Second step: transform to the global frame.
    poserecord = v2x_sim.get('ego_pose', pointsensor['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))

    x_ego = poserecord['translation'][0]
    y_ego = poserecord['translation'][1]

    return pc, x_ego, y_ego, pointsensor


def render_scene_lidar(v2x_sim,
                           scene_token: str,
                           box_visualization: bool = True,
                           with_anns: bool = True,
                           axes_limit: float = 100,
                           out_path: str = None,
                           single_frame_idx = None,
                           ax = None) -> None:
        """
        Render lidar data of all sensors in a scene (multi-view).
        :param with_anns: Whether to draw annotations.
        :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
        :param axes_limit: Axes limit for lidar and radar (measured in meters).
        :param out_path: Optional path to save the rendered figure to disk.
        :param single_frame_idx: If given, will only render one single frame at the given index number.
        :param ax: Axes onto which to render.
        """
        # Get records from DB
        scene_record = v2x_sim.get('scene', scene_token)
        sample_record = v2x_sim.get('sample', scene_record['first_sample_token'])
        
        num_sensor = 6
        
        sd_rec = [[] for _ in range(num_sensor)]
        pointsensor_token = [[] for _ in range(num_sensor)]
        pc = [[] for _ in range(num_sensor)]
        x_ego = [[] for _ in range(num_sensor)]
        y_ego = [[] for _ in range(num_sensor)]

        viewpoint = np.eye(4)
        point_scale = 0.2
        point_cloud_color = ['lightslategrey', 'r', 'g', 'b', 'y', 'c', 'm']
        
        # Init axes.
        if ax is None:
            fig = plt.figure(figsize=(8,8))
            ax = fig.add_subplot(111)
        
        for k in range(num_sensor):
            pointsensor_token[k] = sample_record['data']['LIDAR_TOP_id_' + str(k)]
            pc[k], x_ego[k], y_ego[k], sd_rec[k] = map_pointcloud_to_global(v2x_sim, pointsensor_token[k])

        x_center = sum(x_ego) / num_sensor
        y_center = sum(y_ego) / num_sensor
        
        has_more_frames = True
        num_frame = 0

        def _go_to_next_frame():
            for k in range(num_sensor):
                if not sd_rec[k]['next'] == "":
                    sd_rec[k] = v2x_sim.get('sample_data', sd_rec[k]['next'])
                    pointsensor_token[k] = sd_rec[k]['token']
                else:
                    return False
            
            return True

        while has_more_frames:

            if single_frame_idx is not None and num_frame < single_frame_idx:
                has_more_frames = _go_to_next_frame()
                num_frame += 1
                continue

            for k in range(num_sensor):   
                pc[k], x_ego[k], y_ego[k], sd_rec[k] = map_pointcloud_to_global(v2x_sim, pointsensor_token[k])

                # Show point cloud.
                points = view_points(pc[k].points[:3, :], viewpoint, normalize=False)

                # k == 0 means RSU
                alpha = 0.1 if k == 0 else 1
                ax.scatter(points[0, :], points[1, :], c=point_cloud_color[k], alpha=alpha, s=point_scale)
                ax.plot(x_ego[k], y_ego[k], 'x', color=point_cloud_color[k]) 
            
                if box_visualization == True:
                    # Get boxes in lidar frame.
                    boxes = v2x_sim.get_boxes(pointsensor_token[k])
                    monitor_R = 32            
                # Show boxes.
                if with_anns:
                    for box in boxes:
                        c = np.array(v2x_sim.colormap[box.name]) / 255.0
                        if abs(box.center[0] - x_ego[k]) < monitor_R and abs(box.center[1] - y_ego[k]) < monitor_R:
                            render_box_translate(box, axis=ax, view=np.eye(4), colors=(c, c, c))

            has_more_frames = _go_to_next_frame()

            # Limit visible range.
            ax.set_xlim(x_center - axes_limit, x_center + axes_limit)
            ax.set_ylim(y_center - axes_limit, y_center + axes_limit)

            ax.axis('on')
            ax.set_aspect('equal')

            if not out_path == None:
                if not os.path.exists(out_path):
                    os.mkdir(out_path)
                plt.savefig(os.path.join(out_path, f"{num_frame}.jpg"))
                plt.cla()

            num_frame = num_frame + 1
            
            if single_frame_idx is not None and num_frame > single_frame_idx:
                break

def render_ego_centric_map(v2x_sim,
                               sample_data_token: str,
                               axes_limit: float = 40,
                               ax: Axes = None) -> None:
        """
        Render map centered around the associated ego pose.
        :param sample_data_token: Sample_data token.
        :param axes_limit: Axes limit measured in meters.
        :param ax: Axes onto which to render.
        """

        def crop_image(image: np.array,
                       x_px: int,
                       y_px: int,
                       axes_limit_px: int) -> np.array:
            x_min = int(x_px - axes_limit_px)
            x_max = int(x_px + axes_limit_px)
            y_min = int(y_px - axes_limit_px)
            y_max = int(y_px + axes_limit_px)

            cropped_image = image[y_min:y_max, x_min:x_max]

            return cropped_image

        # Get data.
        sd_record = v2x_sim.get('sample_data', sample_data_token)
        sample = v2x_sim.get('sample', sd_record['sample_token'])
        scene = v2x_sim.get('scene', sample['scene_token'])
        log = v2x_sim.get('log', scene['log_token'])
        map_ = v2x_sim.get('map', log['map_token'])
        map_mask = map_['mask']
        pose = v2x_sim.get('ego_pose', sd_record['ego_pose_token'])

        # Retrieve and crop mask.
        pixel_coords = map_mask.to_pixel_coords(pose['translation'][0], pose['translation'][1])
        scaled_limit_px = int(axes_limit * (1.0 / map_mask.resolution))
        mask_raster = map_mask.mask()
        cropped = crop_image(mask_raster, pixel_coords[0], pixel_coords[1], int(scaled_limit_px * math.sqrt(2)))

        # Rotate image.
        ypr_rad = Quaternion(pose['rotation']).yaw_pitch_roll
        yaw_deg = -math.degrees(ypr_rad[0])
        rotated_cropped = np.array(Image.fromarray(cropped).rotate(yaw_deg))

        # Crop image.
        ego_centric_map = crop_image(rotated_cropped, rotated_cropped.shape[1] / 2,
                                     rotated_cropped.shape[0] / 2,
                                     scaled_limit_px)

        # Init axes and show image.
        # Set background to white and foreground (semantic prior) to gray.
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(9, 9))
        ego_centric_map[ego_centric_map == map_mask.foreground] = 125
        ego_centric_map[ego_centric_map == map_mask.background] = 255
        ax.imshow(ego_centric_map, extent=[-axes_limit, axes_limit, -axes_limit, axes_limit],
                  cmap='gray', vmin=0, vmax=255)

def render_sample_data(v2x_sim,
                        sample_data_token: str,
                        with_anns: bool = True,
                        box_vis_level: BoxVisibility = BoxVisibility.ANY,
                        axes_limit: float = 40,
                        ax: Axes = None,
                        nsweeps: int = 1,
                        out_path: str = None,
                        underlay_map: bool = True,
                        use_flat_vehicle_coordinates: bool = True,
                        show_lidarseg: bool = False,
                        show_lidarseg_legend: bool = False,
                        filter_lidarseg_labels: List = None,
                        lidarseg_preds_bin_path: str = None,
                        verbose: bool = True,
                        pointsensor_channel: str = "LIDAR_TOP_id_0") -> None:
    """
    Render sample data onto axis.
    Can be used to render lidar data of a single sensor in a single sample.
    :param sample_data_token: Sample_data token.
    :param with_anns: Whether to draw box annotations.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param axes_limit: Axes limit for lidar and radar (measured in meters).
    :param ax: Axes onto which to render.
    :param nsweeps: Number of sweeps for lidar and radar.
    :param out_path: Optional path to save the rendered figure to disk.
    :param underlay_map: When set to true, lidar data is plotted onto the map. This can be slow.
    :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
        aligned to z-plane in the world. Note: Previously this method did not use flat vehicle coordinates, which
        can lead to small errors when the vertical axis of the global frame and lidar are not aligned. The new
        setting is more correct and rotates the plot by ~90 degrees.
    :param show_lidarseg: When set to True, the lidar data is colored with the segmentation labels. When set
        to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
    :param show_lidarseg_legend: Whether to display the legend for the lidarseg labels in the frame.
    :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes. If None
        or the list is empty, all classes will be displayed.
    :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                    predictions for the sample.
    :param verbose: Whether to display the image after it is rendered.
    """
    # Get sensor modality.
    sd_record = v2x_sim.get('sample_data', sample_data_token)
    sensor_modality = sd_record['sensor_modality']

    if sensor_modality in ['lidar', 'radar']:
        sample_rec = v2x_sim.get('sample', sd_record['sample_token'])
        chan = sd_record['channel']
        ref_chan = pointsensor_channel
        ref_sd_token = sample_rec['data'][ref_chan]
        ref_sd_record = v2x_sim.get('sample_data', ref_sd_token)

        if sensor_modality == 'lidar':
            if show_lidarseg:
                assert hasattr(v2x_sim, 'lidarseg'), 'Error: nuScenes-lidarseg not installed!'

                # Ensure that lidar pointcloud is from a keyframe.
                assert sd_record['is_key_frame'], \
                    'Error: Only pointclouds which are keyframes have lidar segmentation labels. Rendering aborted.'

                assert nsweeps == 1, \
                    'Error: Only pointclouds which are keyframes have lidar segmentation labels; nsweeps should ' \
                    'be set to 1.'

                # Load a single lidar point cloud.
                pcl_path = osp.join(v2x_sim.dataroot, ref_sd_record['filename'])
                pc = LidarPointCloud.from_file(pcl_path)
            else:
                # Get aggregated lidar point cloud in lidar frame.
                pc, times = LidarPointCloud.from_file_multisweep(v2x_sim, sample_rec, chan, ref_chan,
                                                                    nsweeps=nsweeps)
            velocities = None
        else:
            # Get aggregated radar point cloud in reference frame.
            # The point cloud is transformed to the reference frame for visualization purposes.
            pc, times = RadarPointCloud.from_file_multisweep(v2x_sim, sample_rec, chan, ref_chan, nsweeps=nsweeps)

            # Transform radar velocities (x is front, y is left), as these are not transformed when loading the
            # point cloud.
            radar_cs_record = v2x_sim.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
            ref_cs_record = v2x_sim.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
            velocities = pc.points[8:10, :]  # Compensated velocity
            velocities = np.vstack((velocities, np.zeros(pc.points.shape[1])))
            velocities = np.dot(Quaternion(radar_cs_record['rotation']).rotation_matrix, velocities)
            velocities = np.dot(Quaternion(ref_cs_record['rotation']).rotation_matrix.T, velocities)
            velocities[2, :] = np.zeros(pc.points.shape[1])

        # By default we render the sample_data top down in the sensor frame.
        # This is slightly inaccurate when rendering the map as the sensor frame may not be perfectly upright.
        # Using use_flat_vehicle_coordinates we can render the map in the ego frame instead.
        if use_flat_vehicle_coordinates:
            # Retrieve transformation matrices for reference point cloud.
            cs_record = v2x_sim.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
            pose_record = v2x_sim.get('ego_pose', ref_sd_record['ego_pose_token'])    # ego pose
            ref_to_ego = transform_matrix(translation=cs_record['translation'],
                                            rotation=Quaternion(cs_record["rotation"]))

            # Compute rotation between 3D vehicle pose and "flat" vehicle pose (parallel to global z plane).
            ego_yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
            rotation_vehicle_flat_from_vehicle = np.dot(
                Quaternion(scalar=np.cos(ego_yaw / 2), vector=[0, 0, np.sin(ego_yaw / 2)]).rotation_matrix,
                Quaternion(pose_record['rotation']).inverse.rotation_matrix)
            vehicle_flat_from_vehicle = np.eye(4)
            vehicle_flat_from_vehicle[:3, :3] = rotation_vehicle_flat_from_vehicle
            viewpoint = np.dot(vehicle_flat_from_vehicle, ref_to_ego)
        else:
            viewpoint = np.eye(4)

        # Init axes.
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(9, 9))

        # Render map if requested.
        if underlay_map:
            assert use_flat_vehicle_coordinates, 'Error: underlay_map requires use_flat_vehicle_coordinates, as ' \
                                                    'otherwise the location does not correspond to the map!'
            render_ego_centric_map(v2x_sim, sample_data_token=sample_data_token, axes_limit=axes_limit, ax=ax)

        # Show point cloud.
        points = view_points(pc.points[:3, :], viewpoint, normalize=False)
        dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
        if sensor_modality == 'lidar' and show_lidarseg:
            # Load labels for pointcloud.
            if lidarseg_preds_bin_path:
                sample_token = v2x_sim.get('sample_data', sample_data_token)['sample_token']
                lidarseg_labels_filename = lidarseg_preds_bin_path
                assert os.path.exists(lidarseg_labels_filename), \
                    'Error: Unable to find {} to load the predictions for sample token {} (lidar ' \
                    'sample data token {}) from.'.format(lidarseg_labels_filename, sample_token, sample_data_token)
            else:
                if len(v2x_sim.lidarseg) > 0:  # Ensure lidarseg.json is not empty (e.g. in case of v1.0-test).
                    lidarseg_labels_filename = osp.join(v2x_sim.dataroot,
                                                        v2x_sim.get('lidarseg', sample_data_token)['filename'])
                else:
                    lidarseg_labels_filename = None

            if lidarseg_labels_filename:
                # Paint each label in the pointcloud with a RGBA value.
                colors = paint_points_label(lidarseg_labels_filename, filter_lidarseg_labels,
                                            v2x_sim.lidarseg_name2idx_mapping, v2x_sim.colormap)

                if show_lidarseg_legend:
                    # Since the labels are stored as class indices, we get the RGB colors from the colormap
                    # in an array where the position of the RGB color corresponds to the index of the class
                    # it represents.
                    color_legend = colormap_to_colors(v2x_sim.colormap, v2x_sim.lidarseg_name2idx_mapping)

                    # If user does not specify a filter, then set the filter to contain the classes present in
                    # the pointcloud after it has been projected onto the image; this will allow displaying the
                    # legend only for classes which are present in the image (instead of all the classes).
                    if filter_lidarseg_labels is None:
                        filter_lidarseg_labels = get_labels_in_coloring(color_legend, colors)

                    create_lidarseg_legend(filter_lidarseg_labels,
                                            v2x_sim.lidarseg_idx2name_mapping, v2x_sim.colormap,
                                            loc='upper left', ncol=1, bbox_to_anchor=(1.05, 1.0))
            else:
                colors = np.minimum(1, dists / axes_limit / np.sqrt(2))
                print('Warning: There are no lidarseg labels in {}. Points will be colored according to distance '
                        'from the ego vehicle instead.'.format(v2x_sim.version))
        else:
            colors = np.minimum(1, dists / axes_limit / np.sqrt(2))
        point_scale = 0.2 if sensor_modality == 'lidar' else 3.0

        #print(points.shape)

        scatter = ax.scatter(points[0, :], points[1, :], c=colors, s=point_scale)

        # Show velocities.
        if sensor_modality == 'radar':
            points_vel = view_points(pc.points[:3, :] + velocities, viewpoint, normalize=False)
            deltas_vel = points_vel - points
            deltas_vel = 6 * deltas_vel  # Arbitrary scaling
            max_delta = 20
            deltas_vel = np.clip(deltas_vel, -max_delta, max_delta)  # Arbitrary clipping
            colors_rgba = scatter.to_rgba(colors)
            for i in range(points.shape[1]):
                ax.arrow(points[0, i], points[1, i], deltas_vel[0, i], deltas_vel[1, i], color=colors_rgba[i])

        # Show ego vehicle.
        ax.plot(0, 0, 'x', color='red')

        # Get boxes in lidar frame.
        _, boxes, _ = v2x_sim.get_sample_data(ref_sd_token, box_vis_level=box_vis_level,
                                                use_flat_vehicle_coordinates=use_flat_vehicle_coordinates)

        # Show boxes.
        if with_anns:
            for box in boxes:
                c = np.array(v2x_sim.colormap[box.name]) / 255.0
                box.render(ax, view=np.eye(4), colors=(c, c, c))

        # Limit visible range.
        ax.set_xlim(-axes_limit, axes_limit)
        ax.set_ylim(-axes_limit, axes_limit)

    elif sensor_modality == 'camera':
        # Load boxes and image.
        data_path, boxes, camera_intrinsic = v2x_sim.get_sample_data(sample_data_token,
                                                                        box_vis_level=box_vis_level)
        data = Image.open(data_path)

        # Init axes.
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(9, 16))

        # Show image.
        ax.imshow(data)

        # Show boxes.
        if with_anns:
            for box in boxes:
                c = np.array(v2x_sim.colormap[box.name]) / 255.0
                box.render(ax, view=camera_intrinsic, normalize=True, colors=(c, c, c))

        # Limit visible range.
        ax.set_xlim(0, data.size[0])
        ax.set_ylim(data.size[1], 0)

    else:
        raise ValueError("Error: Unknown sensor modality!")

    ax.axis('off')
    ax.set_title('{} {labels_type}'.format(
        sd_record['channel'], labels_type='(predictions)' if lidarseg_preds_bin_path else ''))
    ax.set_aspect('equal')

    if out_path is not None:
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=200)

    if verbose:
        plt.show()