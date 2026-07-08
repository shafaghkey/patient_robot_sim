import xml.etree.ElementTree as ET
from mjcf_urdf_simple_converter import convert
import os
from pybullet_utils import bullet_client as bc
import pybullet_data as pd
import pybullet_utils.urdfEditor as ed

file_path = os.path.dirname(os.path.abspath(__file__))

def sanitize_mjcf(input_path, output_path):
    """Remove MuJoCo-specific attributes before conversion"""
    tree = ET.parse(input_path)
    root = tree.getroot()
    
    # Remove problematic attributes from all <position> elements
    for elem in root.iter('position'):
        for attr in ['dampratio', 'ctrlrange', 'kp']:
            if attr in elem.attrib:
                del elem.attrib[attr]
    
    # Remove unsupported MuJoCo elements
    for elem in root.findall('.//default'):
        # root.remove(elem)
        elem.clear()
    
    tree.write(output_path, encoding='UTF-8', xml_declaration=True)

def clean_convert(input_path, output_path):
    # Remove MuJoCo-specific attributes pre-conversion
    with open(input_path) as f:
        content = f.read()
    
    # Remove problematic attributes
    content = content.replace(' dampratio="', ' mujoco_dampratio="')
    content = content.replace(' ctrlrange="', ' mujoco_ctrlrange="')
    
    with open("_temp.xml", "w") as f:
        f.write(content)
    
    # Perform conversion
    convert("_temp.xml", output_path, asset_file_prefix="package://umi_gripper/assets/")


###################################################################
model = "umi_gripper"

# Method 1: Using sanitize_mjcf and convert
# # Sanitize the MJCF file
# sanitize_mjcf(
#     os.path.join(file_path, f"{model}.xml"),
#     os.path.join(file_path, f"{model}_sanitized.xml")
# )
# # Convert the sanitized MJCF file to URDF
# convert(
#     os.path.join(file_path, f"{model}_sanitized.xml"),
#     os.path.join(file_path, f"{model}.urdf"),
#     asset_file_prefix="package://umi_gripper/assets/"
# )

# Method 2: Directly clean and convert
# clean_convert(
#     os.path.join(file_path, f"{model}.xml"),
#     os.path.join(file_path, f"{model}.urdf"),
#     # asset_file_prefix="package://umi_gripper/assets/"
# )

file_path = os.path.join(file_path, f"{model}.xml")
p = bc.BulletClient()
p.setAdditionalSearchPath(pd.getDataPath())
objs = p.loadMJCF(file_path, flags=p.URDF_USE_IMPLICIT_CYLINDER)

for o in objs:
    humanoid = objs[o]
    ed0 = ed.UrdfEditor()
    ed0.initializeFromBulletBody(humanoid, p._client)
    # robotName = str(p.getBodyInfo(o), 'utf-8')
    body_info = p.getBodyInfo(o)
    robotName = str(body_info[0], 'utf-8')  # [0] gets body name, [1] gets path
    ed0.saveUrdf(robotName + ".urdf", saveVisuals=False)
