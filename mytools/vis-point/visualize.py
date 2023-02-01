import open3d as o3d
import numpy as np
import json


def read_obj(objFilePath):
    #objFilePath = '/workspace/vic-competition/mmdetection3d/work-dirs/exam-c/vis_dataset/000086/000086_gt.obj'
    with open(objFilePath) as file:
        points = []
        while 1:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "v":
                points.append((float(strs[1]), float(strs[2]), float(strs[3])))
            if strs[0] == "vt":
                break       
    points = np.array(points)
    return points

def draw_line():
    points = np.array([[23.691616, -25.515982, 4.530614]])

    triangle_points = np.array([[93.27787780761719, -8.92260456085205, -1.5294616222381592], [93.27787780761719, -8.92260456085205, -0.03455829620361328], [93.13224792480469, -10.765183448791504, -0.03455829620361328], [93.13224792480469, -10.765183448791504, -1.5294616222381592], [89.11761474609375, -8.593794822692871, -1.5294616222381592], [89.11761474609375, -8.593794822692871, -0.03455829620361328], [88.97198486328125, -10.436373710632324, -0.03455829620361328], [88.97198486328125, -10.436373710632324, -1.5294616222381592]], dtype=np.float32)
    lines = [[0, 1], [0, 3], [0, 4],[1,2],[1,5],[2,3],[2,6],[3,7],[4,5],[4,7],[5,6],[6,7]]  # Right leg
    colors = [[0, 0, 1] for i in range(len(lines))]  # Default blue

    source_data = points#np.load('curtain_0088.npy')[:,0:3]  #10000x3
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(source_data)
    point_cloud.paint_uniform_color([1, 0, 0])

    lines_pcd = o3d.geometry.LineSet()
    lines_pcd.lines = o3d.utility.Vector2iVector(lines)
    lines_pcd.colors = o3d.utility.Vector3dVector(colors) #线条颜色
    lines_pcd.points = o3d.utility.Vector3dVector(triangle_points)


    o3d.visualization.draw_geometries([point_cloud])

def save_mesh(w,h,d,translate,filename):
  
    mesh_box = o3d.geometry.TriangleMesh.create_box(width=w,
                                                    height=h,
                                                    depth=d)
    mesh_box.compute_vertex_normals()
    mesh_box.paint_uniform_color([0.9, 0.1, 0.1])
    mesh_box.translate(translate)
    #o3d.visualization.draw_geometries([mesh_box])
    o3d.io.write_triangle_mesh(filename,mesh_box)


def box2info(boxes):
    num_boxes = boxes.shape[0]
    center = np.mean(boxes, axis=1)
    size = np.zeros((num_boxes, 3))
    size[:, 0] = (
        np.sum((boxes[:, 2, :] - boxes[:, 1, :]) ** 2, axis=1) ** 0.5
        + np.sum((boxes[:, 6, :] - boxes[:, 5, :]) ** 2, axis=1) ** 0.5
    ) / 2
    size[:, 1] = (
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
    return center, size


if __name__ == '__main__':
    
    with open("/workspace/boxes.json",'r') as load_f:
        boxes = json.load(load_f)
    boxes_3d = np.array(boxes["boxes_3d"])
    center,size = box2info(boxes_3d)
    #
    #print(np.mean(np.mean(boxes_3d,axis=0),axis=0))
    # for i in range(2,len(boxes_3d)):
    #     trans = [boxes_3d[i][4][0],-boxes_3d[i][4][1],boxes_3d[i][4][2]]
    #     save_mesh(size[i][1],size[i][0],size[i][2],trans,f'./{i}.obj')

    # points = np.load('/workspace/pointcloud.npy')[:,0:3]

    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(points)
    # print(np.asarray(point_cloud.points)[:10])
    # o3d.io.write_point_cloud('/workspace/points.pcd',point_cloud)

    # box_8 = np.array([ [
    #             [
    #                 52.075233459472656,
    #                 -1.8995122909545898,
    #                 -1.7324018478393555
    #             ],
    #             [
    #                 52.075233459472656,
    #                 -1.8995122909545898,
    #                 -0.0830308198928833
    #             ],
    #             [
    #                 52.08591079711914,
    #                 -3.943270444869995,
    #                 -0.0830308198928833
    #             ],
    #             [
    #                 52.08591079711914,
    #                 -3.943270444869995,
    #                 -1.7324018478393555
    #             ],
    #             [
    #                 47.88676071166992,
    #                 -1.9213945865631104,
    #                 -1.7324018478393555
    #             ],
    #             [
    #                 47.88676071166992,
    #                 -1.9213945865631104,
    #                 -0.0830308198928833
    #             ],
    #             [
    #                 47.897438049316406,
    #                 -3.9651527404785156,
    #                 -0.0830308198928833
    #             ],
    #             [
    #                 47.897438049316406,
    #                 -3.9651527404785156,
    #                 -1.7324018478393555
    #             ]
    #         ]])
    # center,size = box2info(box_8)
    # trans = [-box_8[0][4][1],box_8[0][4][0],box_8[0][4][2]]
    # draw_mesh(size[0][0],size[0][1],size[0][2],trans)



