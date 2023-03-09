import os
import json
import numpy as np
from pypcd import pypcd
import open3d as o3d
import random

def box2info(boxes):
    num_boxes = boxes.shape[0]
    # 预测box中心
    center = np.mean(boxes, axis=1)
    size = np.zeros((num_boxes, 3))
    size[:, 1] = (
        np.sum((boxes[:, 2, :] - boxes[:, 1, :]) ** 2, axis=1) ** 0.5
        + np.sum((boxes[:, 6, :] - boxes[:, 5, :]) ** 2, axis=1) ** 0.5
    ) / 2
    size[:, 0] = (
        np.sum((boxes[:, 4, :] - boxes[:, 0, :]) ** 2, axis=1) ** 0.5
        + np.sum((boxes[:, 6, :] - boxes[:, 2, :]) ** 2, axis=1) ** 0.5
    ) / 2
    size[:, 2] = (
        boxes[:, 1, :]
        + boxes[:, 2, :]
        + boxes[:, 5, :]
        + boxes[:, 6, :]
        - boxes[:, 0, :]
        - boxes[:, 3, :]
        - boxes[:, 4, :]
        - boxes[:, 7, :]
    )[:, 2] / 4
    # 长方体中心
    # center = boxes[:,7] + (size/2)
    return center, size


def read_pcd(path_pcd):
    return pypcd.PointCloud.from_path(path_pcd)


def concatenate_pcd2bin(pc1, pc2, path_save):
    np_x1 = (np.array(pc1.pc_data["x"], dtype=np.float32)).astype(np.float32)
    np_y1 = (np.array(pc1.pc_data["y"], dtype=np.float32)).astype(np.float32)
    np_z1 = (np.array(pc1.pc_data["z"], dtype=np.float32)).astype(np.float32)
    np_i1 = (np.array(pc1.pc_data["intensity"], dtype=np.float32)).astype(np.float32) / 255

    np_x2 = (np.array(pc2.pc_data["x"], dtype=np.float32)).astype(np.float32)
    np_y2 = (np.array(pc2.pc_data["y"], dtype=np.float32)).astype(np.float32)
    np_z2 = (np.array(pc2.pc_data["z"], dtype=np.float32)).astype(np.float32)
    np_i2 = (np.array(pc2.pc_data["intensity"], dtype=np.float32)).astype(np.float32)

    np_x = np.append(np_x1, np_x2)
    np_y = np.append(np_y1, np_y2)
    np_z = np.append(np_z1, np_z2)
    np_i = np.append(np_i1, np_i2)

    points_32 = np.transpose(np.vstack((np_x, np_y, np_z, np_i)))
    list_pcd = []
    for i in range(len(points_32)):

        x, y, z, intensity = points_32[i][0], points_32[i][1], points_32[i][2], points_32[i][3]
        list_pcd.append((x, y, z, intensity))
    dt = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4"), ("intensity", "f4")])
    np_pcd = np.array(list_pcd, dtype=dt)
    new_metadata = {}
    new_metadata["version"] = "0.7"
    new_metadata["fields"] = ["x", "y", "z", "intensity"]
    new_metadata["size"] = [4, 4, 4, 4]
    new_metadata["type"] = ["F", "F", "F", "F"]
    new_metadata["count"] = [1, 1, 1, 1]
    new_metadata["width"] = len(np_pcd)
    new_metadata["height"] = 1
    new_metadata["viewpoint"] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    new_metadata["points"] = len(np_pcd)
    new_metadata["data"] = "binary"
    pc_save = pypcd.PointCloud(new_metadata, np_pcd)
    pc_save.save_pcd(path_save, compression="binary_compressed")
    
    # pcd_array = pc_save.pc_data.view(np.float32).reshape(pc_save.pc_data.shape+(-1,))
    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(pcd_array[:,:3])
    # o3d.io.write_point_cloud('/workspace/fusion.pcd',point_cloud)

def get_filted_index(pointclouds, boxes, scores):
    # pointclouds: [N,4]
    # boxes: [M,8,3]
    # scores : [N]
    # retain_index = scores > 0.4
    # boxes = boxes[retain_index]
    # scores = scores[retain_index]

    center,size = box2info(boxes) # N 3
    centers = np.repeat(center.reshape(-1,1,3),8,axis=1)
    # for i in range(len(boxes)):
    #     trans = [boxes[i][7][0],boxes[i][7][1],boxes[i][7][2]]
    #     draw_mesh(size[i][0],size[i][1],size[i][2],trans,f'box_{i}.obj')
    boxes_dis = np.max(np.sum((centers-boxes)**2,axis=2)**0.5,axis=1) # N
    points = pointclouds[:,:3] # M 3
    retain_index = np.full(len(points), False, dtype=bool)
    for i in range(len(center)):
        points_dis = np.sum((points-center[i])**2,axis=1)**0.5
        retain_ind = points_dis < boxes_dis[i]
        retain_index += retain_ind
    return retain_index
    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(points[retain_index])
    # # print(np.asarray(point_cloud.points)[:10])
    # o3d.io.write_point_cloud('./retain_final.pcd',point_cloud)

def get_filted_index2(pointclouds, boxes,scores,n):
    # pointclouds: [M,4]
    # boxes: [N,8,3]
    # scores : [N]
    # retain_index = scores > 0.4
    # boxes = boxes[retain_index]
    # scores = scores[retain_index]

    center,size = box2info(boxes) # N 3
    # k = (1-0.5) / (np.max(scores)-np.min(scores))
    # weights = 0.5 + k*(scores-np.min(scores))
    # weights = np.repeat(weights.reshape(-1,1),3,axis=1)
    # centers = np.repeat(center.reshape(-1,1,3),8,axis=1)
    # boxes_dis = np.sum((centers-boxes)**2,axis=2)**0.5
    # boxes_dis = np.max(np.sum((centers-boxes)**2,axis=2)**0.5,axis=1) # N
    filt_range = (size/2)*n
    # for i in range(len(boxes)):
    #     trans = [boxes[i][7][0],boxes[i][7][1],boxes[i][7][2]]
    #     draw_mesh(size[i][0],size[i][1],size[i][2],trans,f'box_{i}.obj')
    
    points = pointclouds[:,:3] # M 3
    retain_index = np.full(len(points), False, dtype=bool)
    for i in range(len(center)):
        dim_dis = abs(points-center[i]) # M 3
        retain_ind = (dim_dis[:,0] < filt_range[i][0])*(dim_dis[:,1] < filt_range[i][1])*(dim_dis[:,2] < filt_range[i][2])
        retain_index += retain_ind
    return retain_index
    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(points[retain_index])
    # # print(np.asarray(point_cloud.points)[:10])
    # o3d.io.write_point_cloud('./retain_final.pcd',point_cloud)

def draw_mesh(size,translate,filename):
  
    mesh_box = o3d.geometry.TriangleMesh.create_box(width=size[0],
                                                    height=size[1],
                                                    depth=size[2])
    mesh_box.compute_vertex_normals()
    # mesh_box.paint_uniform_color([0.9, 0.1, 0.1])
    mesh_box.translate(translate)
    #o3d.visualization.draw_geometries([mesh_box])
    o3d.io.write_triangle_mesh(filename,mesh_box)

def save_mesh(pointclouds, boxes, folder_path):
    # pointclouds: [N,4]
    # boxes: [M,8,3]
    center,size = box2info(boxes) # N 3
    for i in range(len(boxes)):
        trans = [boxes[i][7][0],boxes[i][7][1],boxes[i][7][2]]
        draw_mesh(size[i],trans,folder_path+f'/box_{i}.obj')
    
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pointclouds[:,:3])
    # print(np.asarray(point_cloud.points)[:10])
    o3d.io.write_point_cloud(folder_path+f'/filted_points.pcd',point_cloud)

def filt_point_by_boxes(pcd, pred, cla_id,n):
    folder_path = '/workspace'
    pcd_array = pcd.pc_data.view(np.float32).reshape(pcd.pc_data.shape+(-1,))
    cla_index = (np.array(pred['labels_3d']) == cla_id)
    boxes = np.array(pred['boxes_3d'])[cla_index]
    scores = np.array(pred['scores_3d'])[cla_index]
    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(pcd_array[:,:3])
    # o3d.io.write_point_cloud(folder_path+f'/source_points.pcd',point_cloud)

    retain_ind = get_filted_index2(pcd_array,boxes,scores,n)
    pcd.pc_data = pcd.pc_data[retain_ind]
    # ratio = 0.5
    # seq = np.arange(len(pcd.pc_data)).tolist()
    # keep = random.sample(seq,int(len(seq)*ratio))
    # pcd.pc_data = pcd.pc_data[keep]
    # filted_pcd_array = pcd.pc_data.view(np.float32).reshape(pcd.pc_data.shape+(-1,))
    # save_mesh(filted_pcd_array, boxes, folder_path)
    return pcd