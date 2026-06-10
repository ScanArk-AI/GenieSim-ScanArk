from pxr import Usd, UsdPhysics
stage = Usd.Stage.Open("/geniesim/main/source/geniesim/assets/robot/G2_omnipicker/robot.usd")
for prim in stage.Traverse():
    path_str = prim.GetPath().pathString.lower()
    if "chassis" in path_str and "joint" in path_str:
        print(f"Path: {prim.GetPath()}, Type: {prim.GetTypeName()}")
        for child in prim.GetChildren():
            print(f"  Child: {child.GetPath()}, Type: {child.GetTypeName()}")
            for attr in child.GetAttributes():
                if attr.Get() is not None:
                    print(f"    {attr.GetName()} = {attr.Get()}")
