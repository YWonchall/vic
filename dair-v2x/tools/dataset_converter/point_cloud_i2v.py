import os
import json
import argparse
import numpy as np
from pypcd import pypcd
import open3d as o3d
from tqdm import tqdm
import errno

from concurrent import futures as futures


def read_json(path_json):
    with open(path_json, "r") as load_f:
        my_json = json.load(load_f)
    return my_json


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def get_virtuallidar2world(path_virtuallidar2world):
    virtuallidar2world = read_json(path_virtuallidar2world)
    rotation = virtuallidar2world["rotation"]
    translation = virtuallidar2world["translation"]
    delta_x = virtuallidar2world["relative_error"]["delta_x"]
    delta_y = virtuallidar2world["relative_error"]["delta_y"]
    return rotation, translation, delta_x, delta_y


def get_novatel2world(path_novatel2world):
    novatel2world = read_json(path_novatel2world)
    rotation = novatel2world["rotation"]
    translation = novatel2world["translation"]
    return rotation, translation


def get_lidar2novatel(path_lidar2novatel):
    lidar2novatel = read_json(path_lidar2novatel)
    rotation = lidar2novatel["transform"]["rotation"]
    translation = lidar2novatel["transform"]["translation"]
    return rotation, translation


def get_data(data_info, path_pcd):
    for data in data_info:
        name1 = os.path.split(path_pcd)[-1]
        name2 = os.path.split(data["pointcloud_path"])[-1]
        if name1 == name2:
            return data


def trans(input_point, translation, rotation):
    input_point = np.array(input_point).reshape(3, -1)
    translation = np.array(translation).reshape(3, 1)
    rotation = np.array(rotation).reshape(3, 3)
    output_point = np.dot(rotation, input_point).reshape(3, -1) + np.array(translation).reshape(3, 1)
    return output_point


def rev_matrix(R):
    R = np.matrix(R)
    rev_R = R.I
    rev_R = np.array(rev_R)
    return rev_R


def trans_point_i2v(input_point, path_virtuallidar2world, path_novatel2world, path_lidar2novatel):
    # print('0:', input_point)

    # virtuallidar to world
    rotation, translation, delta_x, delta_y = get_virtuallidar2world(path_virtuallidar2world)
    point = trans(input_point, translation, rotation) + np.array([delta_x, delta_y, 0]).reshape(3, 1)
    """
    print('rotation, translation, delta_x, delta_y', rotation, translation, delta_x, delta_y)
    print('1:', point)
    """

    # world to novatel
    rotation, translation = get_novatel2world(path_novatel2world)
    new_rotation = rev_matrix(rotation)
    new_translation = -np.dot(new_rotation, translation)
    point = trans(point, new_translation, new_rotation)
    """
    print('rotation, translation:', rotation, translation)
    print('new_translation, new_rotation:', new_translation, new_rotation)
    print('2:', point)
    """

    # novatel to lidar
    rotation, translation = get_lidar2novatel(path_lidar2novatel)
    new_rotation = rev_matrix(rotation)
    new_translation = -np.dot(new_rotation, translation)
    point = trans(point, new_translation, new_rotation)
    """
    print('rotation, translation:', rotation, translation)
    print('new_translation, new_rotation:', new_translation, new_rotation)
    print('3:', point)
    """
    point = point.T

    return point


def read_pcd(path_pcd):
    pointpillar = o3d.io.read_point_cloud(path_pcd)
    points = np.asarray(pointpillar.points)
    return points


def show_pcd(path_pcd):
    pcd = read_pcd(path_pcd)
    o3d.visualization.draw_geometries([pcd])


def write_pcd(path_pcd, new_points, path_save, retain_index):
    pc = pypcd.PointCloud.from_path(path_pcd)
    # 2
    pc.pc_data = pc.pc_data[retain_index]
    #pc.pc_data = pc.pc_data[retain_index]
    pc.pc_data["x"] = new_points[:, 0]
    pc.pc_data["y"] = new_points[:, 1]
    pc.pc_data["z"] = new_points[:, 2]
    pc.width=len(pc.pc_data)
    pc.points=len(pc.pc_data)
    pc.save_pcd(path_save, compression="binary_compressed")
    
    # pcd_array = pc.pc_data.view(np.float32).reshape(pc.pc_data.shape+(-1,))
    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(pcd_array[:,:3])
    # o3d.io.write_point_cloud(path_save, point_cloud)



def trans_pcd_i2v(path_pcd, path_virtuallidar2world, path_novatel2world, path_lidar2novatel, path_save, retain_labels):
    # (n, 3)
    points = read_pcd(path_pcd)
    # 1
    retain_index = get_filted_index(points, retain_labels)
    points = points[retain_index]
    # (n, 3)
    new_points = trans_point_i2v(points.T, path_virtuallidar2world, path_novatel2world, path_lidar2novatel)
    write_pcd(path_pcd, new_points, path_save, retain_index)

    
def map_func(data, path_c, path_dest, i_data_info, v_data_info):
    filt_classes = ['Car','Truck','Van','Bus']
    path_pcd_i = os.path.join(path_c, data["infrastructure_pointcloud_path"])
    path_pcd_v = os.path.join(path_c, data["vehicle_pointcloud_path"])
    i_data = get_data(i_data_info, path_pcd_i)
    v_data = get_data(v_data_info, path_pcd_v)
    path_virtuallidar2world = os.path.join(
        path_c, "infrastructure-side", i_data["calib_virtuallidar_to_world_path"]
    )
    path_novatel2world = os.path.join(path_c, "vehicle-side", v_data["calib_novatel_to_world_path"])
    path_lidar2novatel = os.path.join(path_c, "vehicle-side", v_data["calib_lidar_to_novatel_path"])
    name = os.path.split(path_pcd_i)[-1]
    path_save = os.path.join(path_dest, name)

    # inf_pointcloud_path = os.path.join(path_c, data_info['infrastructure_pointcloud_path'])
    # # 先采用路端标注
    idx = path_pcd_i[-10:-4]
    inf_label_path = os.path.join(path_c,f"infrastructure-side/label/virtuallidar/{idx}.json")
    # pointclouds = read_pcd(inf_pointcloud_path)
    inf_labels = read_json(inf_label_path)
    retain_labels = []
    for label in inf_labels:
        if label['type'] in filt_classes:
            box_dic = {}
            box_dic.update(label['3d_location'])
            box_dic.update(label['3d_dimensions'])
            retain_labels.append(box_dic)


    trans_pcd_i2v(path_pcd_i, path_virtuallidar2world, path_novatel2world, path_lidar2novatel, path_save, retain_labels)

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
    # center,size = box2info(boxes) # N 3
    for i in range(len(boxes)):
        size = [boxes[i]['l'],boxes[i]['w'],boxes[i]['h']]
        trans = [boxes[i]['x']-size[0]/2,boxes[i]['y']-size[1]/2,boxes[i]['z']-size[2]/2]
        draw_mesh(size,trans,folder_path+f'/box_{i}.obj')
    
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pointclouds)
    # print(np.asarray(point_cloud.points)[:10])
    o3d.io.write_point_cloud(folder_path+f'/filted_points.pcd',point_cloud)


def get_filted_index(pointclouds, boxes):
    # pointclouds: [N,3]
    # boxes: {x,y,z,w,h,l}
    # boxes_dis = np.max(np.sum((centers-boxes)**2,axis=2)**0.5,axis=1) # N
    # points = pointclouds[:,:3] # M 3
    retain_index = np.full(len(pointclouds), False, dtype=bool)
    for i in range(len(boxes)):
        indx = abs(pointclouds[:,0]-boxes[i]['x'])<boxes[i]['l']/2
        indy = abs(pointclouds[:,1]-boxes[i]['y'])<boxes[i]['w']/2
        indz = abs(pointclouds[:,2]-boxes[i]['z'])<boxes[i]['h']/2
        #points_dis = np.sum((points-center[i])**2,axis=1)**0.5
        retain_ind = indx*indy*indz
        retain_index += retain_ind
    return retain_index  

# def filt_points(data_info,path_c,filt_classes):
#     file_folder = '/workspace/pointcloud-test'
#     inf_pointcloud_path = os.path.join(path_c, data_info['infrastructure_pointcloud_path'])
#     # 先采用路端标注
#     idx = inf_pointcloud_path[-10:-4]
#     inf_label_path = os.path.join(path_c,f"infrastructure-side/label/virtuallidar/{idx}.json")
#     pointclouds = read_pcd(inf_pointcloud_path)
#     inf_labels = read_json(inf_label_path)
#     retain_labels = []
#     for label in inf_labels:
#         if label['type'] in filt_classes:
#             box_dic = {}
#             box_dic.update(label['3d_location'])
#             box_dic.update(label['3d_dimensions'])
#             retain_labels.append(box_dic)
    
#     retain_index = get_filted_index(pointclouds, retain_labels)
#     # save_mesh(pointclouds[retain_index],retain_labels,file_folder)
#     # point_cloud = o3d.geometry.PointCloud()
#     # point_cloud.points = o3d.utility.Vector3dVector(pointclouds)
#     # # print(np.asarray(point_cloud.points)[:10])
#     # o3d.io.write_point_cloud(file_folder+f'/source_points.pcd',point_cloud)
#     # print(show_msg) 

def get_i2v(path_c, path_dest, num_worker):
    mkdir_p(path_dest)
    path_c_data_info = os.path.join(path_c, "cooperative/data_info.json")
    path_i_data_info = os.path.join(path_c, "infrastructure-side/data_info.json")
    path_v_data_info = os.path.join(path_c, "vehicle-side/data_info.json")
    c_data_info = read_json(path_c_data_info)
    i_data_info = read_json(path_i_data_info)
    v_data_info = read_json(path_v_data_info)
    
    total = len(c_data_info)
    with tqdm(total=total) as pbar:
        with futures.ProcessPoolExecutor(num_worker) as executor:
            # 优化语句
            # 联合label在世界坐标系
            # 路端点云在虚拟lidar(路端lidar)坐标系
            # 优化时，拿到联合label box后转化至虚拟lidar然后将box和点云可视化
            # 可用联合label过滤，个人感觉用路端的label更合理，因为过滤模型基于路端label训练
            # for data in c_data_info:
            #     filt_points(data,path_c,filt_classes)
            #     # print(data)
            #     return 0
            res = [executor.submit(map_func, data, path_c, path_dest, i_data_info, v_data_info) for data in c_data_info]
            for _ in futures.as_completed(res):
                pbar.update(1)

                
parser = argparse.ArgumentParser("Convert The Point Cloud from Infrastructure to Ego-vehicle")
parser.add_argument(
    "--source-root",
    type=str,
    default="./data/DAIR-V2X/cooperative-vehicle-infrastructure",
    help="Raw data root about DAIR-V2X-C.",
)
parser.add_argument(
    "--target-root",
    type=str,
    default="./data/DAIR-V2X/cooperative-vehicle-infrastructure/vic3d-early-fusion/velodyne/lidar_i2v",
    help="The data root where the data with ego-vehicle coordinate is generated",
)
parser.add_argument(
    "--num-worker",
    type=int,
    default=8,
    help="Number of workers for multi-processing",
)

if __name__ == "__main__":
    args = parser.parse_args()
    source_root = args.source_root
    target_root = args.target_root
    num_worker = args.num_worker

    get_i2v(source_root, target_root, num_worker)
