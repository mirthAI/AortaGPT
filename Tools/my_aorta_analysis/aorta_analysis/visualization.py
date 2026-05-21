import matplotlib.pyplot as plt
import numpy as np


def visualize_3d_skeleton(skeleton):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    z, y, x = np.where(skeleton > 0)
    ax.scatter(x, y, z, c='red', marker='o', alpha=0.8, s=5)
    plt.show()

#for left and right aorta
def plot_centerline_and_fitted_curve(original_path, ascending_curve, descending_curve, 
                                   ascending_tangents, descending_tangents, special_index=200):
    plt.ioff()

    x = original_path[:, 0]
    y = original_path[:, 1]
    z = original_path[:, 2]

    fig = plt.figure(figsize=(12, 16))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c='gray', marker='.', alpha=0.3)
    ax.plot(ascending_curve[:, 0], ascending_curve[:, 1], ascending_curve[:, 2], 
            color='green', linewidth=2)
    ax.plot(descending_curve[:, 0], descending_curve[:, 1], descending_curve[:, 2], 
            color='red', linewidth=2)

    step = len(ascending_curve) // 10
    scale = 20
    for curve, tangents, color in [(ascending_curve, ascending_tangents, 'green'),
                                   (descending_curve, descending_tangents, 'red')]:
        for i in range(0, len(curve), step):
            point = curve[i]
            tangent = tangents[i]
            ax.quiver(point[0], point[1], point[2],
                      tangent[0]*scale, tangent[1]*scale, tangent[2]*scale,
                      color=color, alpha=0.6)

    total_points = len(ascending_curve) + len(descending_curve)
    if special_index < total_points:
        if special_index < len(ascending_curve):
            point = ascending_curve[special_index]
            tangent = ascending_tangents[special_index]
            color = 'blue'
        else:
            point = descending_curve[special_index - len(ascending_curve)]
            tangent = descending_tangents[special_index - len(ascending_curve)]
            color = 'purple'

        ax.scatter([point[0]], [point[1]], [point[2]], color=color, s=100)
        ax.quiver(point[0], point[1], point[2],
                  tangent[0]*scale, tangent[1]*scale, tangent[2]*scale,
                  color=color, linewidth=2)

    ax.view_init(elev=90, azim=180)
    ax.set_box_aspect([np.ptp(a) for a in [x, y, z]])
    ax.invert_yaxis()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    plt.show()

    try:
        fig.savefig('centerline_visualization.png')
        print("Successfully saved the figure")
    except Exception as e:
        print(f"Warning: Failed to save figure: {e}")
    finally:
        plt.close(fig)

# for main aorta       
def plot_centerline_and_fitted_curve_aorta(original_path, ascending_curve, descending_curve, 
                                   ascending_tangents, descending_tangents, special_index=200):
    import matplotlib
    import matplotlib.pyplot as plt
    plt.ioff()  # 关闭交互模式
    
    x = original_path[:, 0]
    y = original_path[:, 1]
    z = original_path[:, 2]
    
    fig = plt.figure(figsize=(12, 16))
    ax = fig.add_subplot(111, projection='3d', label='')
    
    ax.scatter(x, y, z, c='gray', marker='.', alpha=0.3)
    ax.plot(ascending_curve[:, 0], ascending_curve[:, 1], ascending_curve[:, 2], 
            color='green', linewidth=2)
    ax.plot(descending_curve[:, 0], descending_curve[:, 1], descending_curve[:, 2], 
            color='red', linewidth=2)
    
    step = len(ascending_curve) // 10
    scale = 20
    for curve, tangents, color in [(ascending_curve, ascending_tangents, 'green'),
                                  (descending_curve, descending_tangents, 'red')]:
        for i in range(0, len(curve), step):
            point = curve[i]
            tangent = tangents[i]
            ax.quiver(point[0], point[1], point[2],
                     tangent[0]*scale, tangent[1]*scale, tangent[2]*scale,
                     color=color, alpha=0.6)
    
    total_points = len(ascending_curve) + len(descending_curve)
    if special_index < total_points:
        if special_index < len(ascending_curve):
            point = ascending_curve[special_index]
            tangent = ascending_tangents[special_index]
            color = 'blue'
        else:
            point = descending_curve[special_index - len(ascending_curve)]
            tangent = descending_tangents[special_index - len(ascending_curve)]
            color = 'purple'
        
        ax.scatter([point[0]], [point[1]], [point[2]], 
                  color=color, s=100)
        ax.quiver(point[0], point[1], point[2],
                 tangent[0]*scale, tangent[1]*scale, tangent[2]*scale,
                 color=color, linewidth=2)
    
    ax.view_init(elev=90, azim=180)
    ax.set_box_aspect([np.ptp(a) for a in [x, y, z]])
    ax.invert_yaxis()
    
    # 移除标签
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    
    # 在Jupyter中显示图像
    plt.show()
    
    try:
        # 同时保存图像
        fig.savefig('centerline_visualization.png')
        print("Successfully saved the figure")
    except Exception as e:
        print(f"Warning: Failed to save figure: {e}")
    finally:
        plt.close(fig)