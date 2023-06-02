import open3d as o3d
import numpy as np
import sys

if __name__ == "__main__":
    
    # Loading point cloud
    print("Loading point cloud")
    ptCloud = o3d.io.read_point_cloud(sys.argv[1] + ".ply")

    # confirmation
    print(ptCloud)
    print(np.asarray(ptCloud.points))
    
    # Visualization in window
    o3d.visualization.draw_geometries([ptCloud])
    
    # Saving point cloud
    #o3d.io.write_point_cloud("output.ply", ptCloud)