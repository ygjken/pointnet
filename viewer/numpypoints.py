import open3d as o3d


def numpy_point_viewer(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])
    print("size:", points.shape)
