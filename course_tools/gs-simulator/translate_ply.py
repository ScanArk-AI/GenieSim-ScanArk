#!/usr/bin/env python3
#
# 读取3DGS PLY文件，将所有点沿指定轴平移

import numpy as np
import argparse
from plyfile import PlyData, PlyElement


def translate_ply(ply_path, output_path, axis='z', translation=1.0):
    """
    读取PLY文件，将所有点沿指定轴平移，然后保存
    
    Args:
        ply_path: 输入PLY文件路径
        output_path: 输出PLY文件路径
        axis: 要平移的轴 ('x', 'y', 'z')，默认'z'
        translation: 平移量（米），默认1.0
    """
    print(f"Reading PLY file: {ply_path}")
    plydata = PlyData.read(ply_path)
    
    # 获取vertex元素
    vertices = plydata['vertex']
    
    # 获取所有属性名称
    property_names = list(vertices.data.dtype.names)
    print(f"Found {len(property_names)} properties: {property_names[:10]}...")
    
    # 获取点的数量
    num_points = len(vertices)
    print(f"Total points: {num_points}")
    
    # 读取xyz坐标
    x = np.asarray(vertices['x'])
    y = np.asarray(vertices['y'])
    z = np.asarray(vertices['z'])
    
    # 根据指定的轴进行平移
    if axis == 'x':
        coord_before = x
        coord_after = x + translation
        coord_name = 'X'
    elif axis == 'y':
        coord_before = y
        coord_after = y + translation
        coord_name = 'Y'
    elif axis == 'z':
        coord_before = z
        coord_after = z + translation
        coord_name = 'Z'
    else:
        raise ValueError(f"Invalid axis: {axis}. Must be 'x', 'y', or 'z'")
    
    # 显示平移前的坐标范围
    print(f"{coord_name} coordinate range before translation: [{coord_before.min():.3f}, {coord_before.max():.3f}]")
    
    # 应用平移
    print(f"{coord_name} coordinate range after translation: [{coord_after.min():.3f}, {coord_after.max():.3f}]")
    print(f"Translation applied: {translation:.3f}m along {coord_name.upper()}-axis")
    
    # 创建新的数据数组，保留所有原始属性
    # 首先构建dtype
    dtype_list = []
    for prop_name in property_names:
        dtype_list.append((prop_name, vertices[prop_name].dtype))
    
    # 创建新的数据数组
    new_data = np.empty(num_points, dtype=dtype_list)
    
    # 复制所有属性
    for prop_name in property_names:
        if prop_name == axis:
            new_data[prop_name] = coord_after
        else:
            new_data[prop_name] = vertices[prop_name]
    
    # 创建新的PlyElement
    new_vertex = PlyElement.describe(new_data, 'vertex')
    
    # 创建新的PlyData（保留其他元素，如果有的话）
    new_elements = [new_vertex]
    for element in plydata.elements:
        if element.name != 'vertex':
            new_elements.append(element)
    
    new_plydata = PlyData(new_elements, text=plydata.text)
    
    # 保存文件
    print(f"Saving translated PLY file to: {output_path}")
    new_plydata.write(output_path)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Translate PLY file along specified axis")
    parser.add_argument("--input", "-i", type=str, required=True,
                       help="Input PLY file path")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output PLY file path (default: input file with '_translated' suffix)")
    parser.add_argument("--axis", "-a", type=str, choices=['x', 'y', 'z'], default='z',
                       help="Axis to translate along (default: 'z')")
    parser.add_argument("--translation", "-t", type=float, default=1.0,
                       help="Translation amount in meters (default: 1.0)")
    
    args = parser.parse_args()
    
    # 如果没有指定输出路径，自动生成
    if args.output is None:
        import os
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_translated{ext}"
    
    translate_ply(args.input, args.output, args.axis, args.translation)


if __name__ == "__main__":
    main()

