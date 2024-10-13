import requests
import json
import re
import argparse
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch
from tqdm import tqdm
import imageio
import glob
from IPython.display import display, Video
token = '' # openai token

def ask_gpt(input_prompt, llm_name, cached_response_path, use_response_cache):
    url = "https://api.openai.com/v1/chat/completions"
    
    # Load existing cache
    if os.path.exists(cached_response_path):
        with open(cached_response_path, 'r', encoding='utf-8') as cache_file:
            cache = json.load(cache_file)
    else:
        cache = {}
    
    if llm_name not in cache:
        cache[llm_name] = {}
    
    if use_response_cache and input_prompt in cache[llm_name]:
        print(f'### Loading response from {cached_response_path}')
        response = cache[llm_name][input_prompt]
    else:
        print(f'### response cache not found, asking GPT')
        payload = json.dumps({
        "model": llm_name, # "gpt-4-1106-preview",
        "messages": [
            {
                "role": "user",
                "content": input_prompt
            }
        ]
        })
        headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {token}',
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'
        }
        print('### Waiting for GPT response')
        for i in tqdm(range(1)): # just to show the progress bar when waiting for GPT response
            response = requests.request("POST", url, headers=headers, data=payload)
        response=response.json()
    
        # Save the response to cache
        cache[llm_name][input_prompt] = response
        print(f'### Saving response to {cached_response_path}')
        with open(cached_response_path, 'w', encoding='utf-8') as cache_file:
            json.dump(cache, cache_file, indent=4, ensure_ascii=False) # ensure_ascii=False to save in utf-8
    
    try:
        output=response['choices'][0]['message']['content']
    except Exception as e:
        print("###################")
        print("# Exception caught")
        print(f"# {e}")
        print(f"# response # {response}")
        print("###################")
        raise
    return {'content':output, 'response': response}

def print_msg(msg):
    print(len(msg) * '#')
    print(msg)
    print(len(msg) * '#')

def draw_bounding_box(ax, dimensions, position, color='blue'):
    # Create a rectangular prism
    dx, dy, dz = dimensions['x'], dimensions['y'], dimensions['z']
    px, py, pz = position['x'] - dx/2, position['y'] - dy/2, position['z'] - dz/2

    # Define the corners of the bounding box
    corners = [
        (px, py, pz),
        (px+dx, py, pz),
        (px+dx, py+dy, pz),
        (px, py+dy, pz),
        (px, py, pz+dz),
        (px+dx, py, pz+dz),
        (px+dx, py+dy, pz+dz),
        (px, py+dy, pz+dz)
    ]

    # Define the edges of the bounding box
    edges = [
        (corners[0], corners[1]),
        (corners[1], corners[2]),
        (corners[2], corners[3]),
        (corners[3], corners[0]),
        (corners[4], corners[5]),
        (corners[5], corners[6]),
        (corners[6], corners[7]),
        (corners[7], corners[4]),
        (corners[0], corners[4]),
        (corners[1], corners[5]),
        (corners[2], corners[6]),
        (corners[3], corners[7])
    ]

    # Draw the edges of the bounding box
    kwargs = {'alpha': 0.5, 'color': color}
    for edge in edges:
        x = [edge[0][0], edge[1][0]]
        y = [edge[0][1], edge[1][1]]
        z = [edge[0][2], edge[1][2]]
        ax.plot3D(x, y, z, **kwargs)
        
        
    min_coord_xy = min([px, py]) # NOTE no pz
    max_coord_xy = max([px+dx, py+dy])
    return min_coord_xy, max_coord_xy

def save_3d_bbox_as_png(json_obj, cache_dir, normalize, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_title(f'{"Normalized" if normalize else ""} 3D Bounding Boxes Visualization')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    # Generate a color palette with the same number of colors as models
    palette = sns.color_palette("hsv", len(json_obj['scene_models']))

    legend_handles = []
    min_coord_xy_list = []
    max_coord_xy_list = []
    for i, obj in enumerate(json_obj['scene_models']):
        min_coord_xy, max_coord_xy = draw_bounding_box(ax, obj['dimensions'], obj['position'], color=palette[i])
        min_coord_xy_list.append(min_coord_xy)
        max_coord_xy_list.append(max_coord_xy)
        legend_handles.append(Patch(facecolor=palette[i], edgecolor='k', label=obj['name']))
    min_coord_xy = min(min_coord_xy_list)
    max_coord_xy = max(max_coord_xy_list)
    if normalize:
        ax.set_xlim(-1.00, 1.00)
        ax.set_ylim(-1.00, 1.00)
        ax.set_zlim(-1.00, 1.00)
    else:
        ax.set_xlim(min_coord_xy * 1.05, max_coord_xy * 1.05)
        ax.set_ylim(min_coord_xy * 1.05, max_coord_xy * 1.05)

    ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(0.95, 0.95))
    plt.savefig(f'{cache_dir}/{filename}')
    
def plot_bboxes(bbox_min: np.ndarray, bbox_max: np.ndarray, colors=None, output_path=None, image_title=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_title(image_title)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    assert bbox_min.shape[1] == 3 and bbox_max.shape[1] == 3
    
    num_groups = bbox_min.shape[0]
    

    if colors is None:
        import seaborn as sns
        colors = sns.color_palette("husl", n_colors=num_groups)
    else:
        assert len(colors) == num_groups
    
    for i in range(num_groups):
        min_corner = bbox_min[i]
        max_corner = bbox_max[i]
        
        corners = np.array([
            [min_corner[0], min_corner[1], min_corner[2]],
            [min_corner[0], min_corner[1], max_corner[2]],
            [min_corner[0], max_corner[1], min_corner[2]],
            [min_corner[0], max_corner[1], max_corner[2]],
            [max_corner[0], min_corner[1], min_corner[2]],
            [max_corner[0], min_corner[1], max_corner[2]],
            [max_corner[0], max_corner[1], min_corner[2]],
            [max_corner[0], max_corner[1], max_corner[2]],
        ])
        

        for start, end in [
            (0, 1), (1, 3), (3, 2), (2, 0),
            (4, 5), (5, 7), (7, 6), (6, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]:
            ax.plot(*zip(corners[start], corners[end]), color=colors[i])
    

    if output_path:
        plt.savefig(output_path)
        plt.close()

def convert_parts_to_scene_models(parts):
    """
    Convert a list of parts into a scene_models format similar to json_obj['scene_models'].
    This is for visualization of all parts in one image and video.
    
    Args:
    parts (list): List of part dictionaries.
    
    Returns:
    dict: A dictionary with 'scene_models' key containing the converted parts.
    """
    fake_json_obj = {'scene_models': []}
    for i, part in enumerate(parts):
        max_corner = np.array(part['max_corner'])
        min_corner = np.array(part['min_corner'])
        dimensions = max_corner - min_corner
        position = min_corner + dimensions / 2
        fake_obj = {
            'name': part['prompt'],
            'dimensions': {
                'x': dimensions[0],
                'y': dimensions[1],
                'z': dimensions[2]
            },
            'position': {
                'x': position[0],
                'y': position[1],
                'z': position[2]
            }
        }
        fake_json_obj['scene_models'].append(fake_obj)
    
    return fake_json_obj

##########################################
### visualize 3d bounding box as video ###
##########################################

def save_video_with_imageio(array_list, output_path, fps=10):
    if not array_list:
        raise ValueError("The array list is empty.")

    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8, pixelformat='yuv420p')
    
    for frame in array_list:
        if frame.shape[2] == 3:
            writer.append_data(frame)
        else:
            raise ValueError("Each frame must have 3 channels (RGB).")
    
    writer.close()

def generate_frame(ax, json_obj, elev, azim, normalize):

    ax.set_title(f'{"Normalized" if normalize else ""} 3D Bounding Boxes Visualization')

    # Generate a color palette with the same number of colors as models
    palette = sns.color_palette("hsv", len(json_obj['scene_models']))

    legend_handles = []
    min_coord_xy_list = []
    max_coord_xy_list = []
    for i, obj in enumerate(json_obj['scene_models']):
        min_coord_xy, max_coord_xy = draw_bounding_box(ax, obj['dimensions'], obj['position'], color=palette[i])
        min_coord_xy_list.append(min_coord_xy)
        max_coord_xy_list.append(max_coord_xy)
        legend_handles.append(Patch(facecolor=palette[i], edgecolor='k', label=obj['name']))
    min_coord_xy = min(min_coord_xy_list)
    max_coord_xy = max(max_coord_xy_list)
    if normalize:
        ax.set_xlim(-1.00, 1.00)
        ax.set_ylim(-1.00, 1.00)
        ax.set_zlim(-1.00, 1.00)
    else:
        ax.set_xlim(min_coord_xy * 1.05, max_coord_xy * 1.05)
        ax.set_ylim(min_coord_xy * 1.05, max_coord_xy * 1.05)

    ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(0.95, 0.95))
    
    ax.view_init(elev, azim)
    
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    # ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    # ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    # ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    
    ax.xaxis.line.set_visible(False)
    ax.yaxis.line.set_visible(False)
    ax.zaxis.line.set_visible(False)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    # Save to buffer
    ax.figure.canvas.draw()
    image = np.frombuffer(ax.figure.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(ax.figure.canvas.get_width_height()[::-1] + (3,))
    return image  # numpy.ndarray, shape (480, 640, 3)

def prepare_figure():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlabel(' ') # ax.set_xlabel('') will cause error
    # ax.set_ylabel(' ')
    # ax.set_zlabel(' ')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    return fig, ax

def save_3d_bbox_as_video(json_obj, elev, n_frames, 
                          n_circles=1, fps=10, normalize=False,
                          output_path='tmp_output.mp4'
                          ):
    fig, ax = prepare_figure()
    frames = []
    azimuth = np.linspace(0, 360 * n_circles, n_frames)
    for az in tqdm(azimuth):
        frame = generate_frame(ax, json_obj, elev, az, normalize)
        frames.append(frame)
    
    save_video_with_imageio(frames, output_path=output_path, fps=fps)
    plt.close(fig)  # Close the figure after processing all frames

def calculate_bboxes(data, bbox_min=np.array([0.0, 0.0, 0.0], dtype=float), bbox_max=np.array([1.0, 1.0, 1.0], dtype=float)):
    result = []
    bbox_min = np.array(bbox_min, dtype=float)
    bbox_max = np.array(bbox_max, dtype=float)
    
    def process_split(split_data, current_min, current_max, axis):
        total_size = np.sum([float(item['size']) for item in split_data])
        current_pos = current_min[axis]
        
        for item in split_data:
            size_ratio = float(item['size']) / total_size
            next_pos = current_pos + size_ratio * (current_max[axis] - current_min[axis])
            
            new_min = current_min.copy()
            new_max = current_max.copy()
            new_min[axis] = current_pos
            new_max[axis] = next_pos
            
            if 'prompt' in item:
                result.append({
                    'prompt': item['prompt'],
                    'min_corner': new_min.tolist(),
                    'max_corner': new_max.tolist()
                })
            elif 'horizontal split' in item:
                process_split(item['horizontal split'], new_min, new_max, 1)  # y-axis
            elif 'vertical split' in item:
                process_split(item['vertical split'], new_min, new_max, 2)  # z-axis
            
            current_pos = next_pos

    process_split(data['depth split'], bbox_min, bbox_max, 0)  # x-axis
    return result

#############################
### Main, parse arguments ###
#############################

parser = argparse.ArgumentParser()
parser.add_argument('--user_prompt', type=str, help='input user prompt', default='A corgi is beside a house.')
parser.add_argument('--batch_prompt_file', type=str, help='path to batch prompt file', default=None)
parser.add_argument('--llm_name', 
                    type=str, 
                    default='gpt-4-32k',
                    # when number of models in user_prompt is large, gpt-4-32k is needed
                    # when "NameError: name 'scene_json' is not defined" or "KeyError: 'dimensions'" try gpt-4-32k
                    choices=['o1-mini', 'o1-preview', 'gpt-4o-2024-05-13', 'gpt-4-1106-preview', 'gpt-3.5-turbo', 'gpt-4-32k']
                    )
parser.add_argument('--cache_dir', type=str, default='./layouts_cache')
parser.add_argument('--disable_response_cache', action='store_true', default=False)
parser.add_argument('--template_version', type=str, default='v0.14_PAL_updated_incontext')
args = parser.parse_args()
user_prompt = args.user_prompt
batch_prompt_file = args.batch_prompt_file
template_version = args.template_version
cached_response_path = f'./layouts_cache/cached_response.json' # NOTE: cache response for all prompts
if not os.path.exists('./layouts_cache'):
    os.makedirs('./layouts_cache')
if not os.path.exists(f'{args.cache_dir}/{template_version}'):
    os.makedirs(f'{args.cache_dir}/{template_version}')

def generate_layout_once(user_prompt):
    template_path = f'./templates/{template_version}.txt'
    with open(template_path, 'r', encoding='utf-8') as file:
        instruct_prompt = file.read()
        instruct_prompt = instruct_prompt.replace('{user_prompt}', user_prompt)
    
    ###############################
    ### ask gpt and save answer ###
    ###############################
    answer = ask_gpt(instruct_prompt, args.llm_name, cached_response_path, 
                    use_response_cache = not args.disable_response_cache,
                    )
    cache_dir =  f'{args.cache_dir}/{template_version}/{args.llm_name}/{user_prompt.replace(" ", "_")[:108]}'
    
    ### cache_dir = base cache_dir path + number appended
    if not os.path.exists(cache_dir):
        # If the directory does not exist, create the first version of this cache directory
        cache_dir = cache_dir + '/1'
        os.makedirs(cache_dir)
    else:
        # If the directory already exists, find the next available numbered subdirectory to create
        # We use glob to find all folders that match the base cache_dir path with any number appended
        pattern = f"{cache_dir}/*"
        existing_dirs = glob.glob(pattern)
        # Finding the maximum existing directory number
        max_number = 1
        for dir in existing_dirs:
            # Extract the number from directory name assuming the structure 'cache_dir/N'
            try:
                number = int(os.path.basename(dir))
                if number > max_number:
                    max_number = number
            except ValueError:
                continue
        # Create a new directory with the next available number
        new_dir_number = max_number + 1
        cache_dir = f"{cache_dir}/{new_dir_number}"
        os.makedirs(cache_dir)
    
    
    with open(f'{cache_dir}/scene_divid_response.txt', 'w') as file:
        json.dump(answer['response'], file, indent=4)
    with open(f'{cache_dir}/scene_divid_answer.txt', 'w') as file:
        file.write(answer['content'])

    ################################################
    ### parse user prompt to object-level layout ###
    ################################################
    print_msg("Parsing user prompt to object-level layout")
    python_matches = re.findall(r'```python\n(.+?)\n```', answer['content'], re.DOTALL)
    if not python_matches:
        raise Exception("No python code blocks found in the answer.")
    local_vars = {}
    combined_python_code = '\n\n'.join(python_matches)
    exec(combined_python_code, None, local_vars)
    
    if (template_version == 'v0.13_PAL_not_all_in_json' 
        or template_version == 'v0.14_PAL_updated_incontext'):
        models = local_vars.get('models')
        scene_models = []
        for name, attributes in models.items():
            scene_model = {
                "name": name,
                "stable diffusion prompt": attributes["object description"],
                "dimensions": attributes["dimension"],
                "position": attributes["position"]
            }
            scene_models.append(scene_model)

        scene_json = json.dumps({"scene_models": scene_models}, indent=4)
    else:
        scene_json = local_vars.get('scene_json')

    if scene_json is None:
        raise ValueError("scene_json is None")
    json_obj = json.loads(scene_json) # scene_json is defined in the gpt answer
    
    json_obj['user_prompt'] = user_prompt.strip(' .')

    #####################################
    ### Recaption object descriptions ###
    #####################################
    print_msg("Recaption object descriptions")
    with open(f'templates/recaption/v0.2.txt', 'r') as file:
        recaption_template = file.read()
    
    # Prepare the recaption_obj with only name and stable diffusion prompt
    recaption_obj = {
        "scene_prompt": json_obj['user_prompt'],
        "objects": []
    }
    for model in json_obj['scene_models']:
        recaption_obj["objects"].append({
            "name": model["name"],
            "stable diffusion prompt": model["stable diffusion prompt"]
        })
    recaption_prompt = recaption_template.replace('{original_json}', json.dumps(recaption_obj, indent=4))
    with open(f'{cache_dir}/recaption_prompt.txt', 'w') as file:
        file.write(recaption_prompt)
    recaption_answer = ask_gpt(recaption_prompt, 
                                # We use gpt-4o-2024-05-13 for recaption and summarization.
                                "gpt-4o-2024-05-13", 
                                cached_response_path,
                                use_response_cache = not args.disable_response_cache,
                                )
    with open(f'{cache_dir}/recaption_answer.txt', 'w') as file:
        file.write(recaption_answer['content'])
    # Extract JSON content from the code block
    json_match = re.search(r'```json\n(.*?)\n```', recaption_answer['content'], re.DOTALL)
    if json_match:
        json_content = json_match.group(1)
        try:
            recaption_json = json.loads(json_content)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from recaption answer: {e}")
    else:
        print("No JSON content found in recaption answer")
        recaption_json = None
    for obj, recaption_obj in zip(json_obj['scene_models'], recaption_json['objects']):
        obj['stable diffusion prompt'] = recaption_obj['stable diffusion prompt']
    
    ################################################
    ### parse each object into part-level layouts ###
    ################################################
    obj_divid_template_path = f'./templates/divid_object/v0.0.txt'
    with open(obj_divid_template_path, 'r', encoding='utf-8') as file:
        obj_divid_template_prompt = file.read()
    
    print_msg("Parsing each object into part-level layouts")
    all_parts = []
    for id, obj in tqdm(enumerate(json_obj['scene_models'])):
        #########################################
        ### get parts_in_complementary_format ###
        #########################################
        obj_divid_instruct_prompt = obj_divid_template_prompt.replace('{user_prompt}', obj['stable diffusion prompt'])
        obj_divid_answer = ask_gpt(obj_divid_instruct_prompt, args.llm_name, cached_response_path, 
                    use_response_cache = not args.disable_response_cache,
                    )
        with open(f'{cache_dir}/obj_divid_answer{id}.txt', 'w') as file:
            json.dump(obj_divid_answer['response'], file, indent=4)
        with open(f'{cache_dir}/obj_divid_answer{id}.txt', 'w') as file:
            file.write(obj_divid_answer['content'])
        
        # Extract JSON content from the code block
        json_match = re.search(r'```json\n(.*?)\n```', obj_divid_answer['content'], re.DOTALL)
        if json_match:
            json_content = json_match.group(1)
            try:
                obj_divid_json = json.loads(json_content)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON for object {obj['name']}: {e}")
        else:
            print(f"No JSON content found for object {obj['name']}")
        
        # NOTE Reverse the order of the "depth split" list in obj_divid_json, 
        # because in the gpt answer, the x-axis is towards into the screen,
        # but we want the x-axis towards outside the screen, so we reverse the order of the "depth split" list
        if "depth split" in obj_divid_json:
            obj_divid_json["depth split"] = obj_divid_json["depth split"][::-1]
        
        obj['parts_in_complementary_format'] = obj_divid_json
        
        ############################################
        ### get parts_in_local and global coords ###
        ############################################
        
        obj['parts_in_local_01_coords'] = calculate_bboxes(obj_divid_json, 
                                        bbox_min=np.array([0, 0, 0]), 
                                        bbox_max=np.array([1, 1, 1]),
                                        )
        
        position = np.array([obj['position']['x'], obj['position']['y'], obj['position']['z']])
        dimension = np.array([obj['dimensions']['x'], obj['dimensions']['y'], obj['dimensions']['z']])
        min_corners = position - dimension / 2
        max_corners = position + dimension / 2
        obj['parts_in_global_coords'] = calculate_bboxes(obj_divid_json, 
                                        bbox_min=min_corners, 
                                        bbox_max=max_corners,
                                        )
        all_parts.extend(obj['parts_in_global_coords'])
        
        
    
    ###############################################
    ### save raw.json and raw bounding box .png ###
    ###############################################
    cache_path = f'{cache_dir}/raw.json'

    with open(cache_path, 'w') as json_file:
        json.dump(json_obj, json_file, indent=4)
        
    save_3d_bbox_as_png(json_obj, cache_dir, normalize=False, filename='raw_objects.png')
    save_3d_bbox_as_video(json_obj, elev=20, n_frames=12, 
                        n_circles=1 / 4, fps=10, normalize=False,
                        output_path=f'{cache_dir}/raw_objects.mp4',
                        )
    
    fake_json_obj4vis = convert_parts_to_scene_models(all_parts)
    save_3d_bbox_as_png(fake_json_obj4vis, cache_dir, normalize=False, filename='raw_parts.png')
    save_3d_bbox_as_video(fake_json_obj4vis, elev=20, n_frames=12, 
                          n_circles=1/4, fps=10, normalize=False,
                          output_path=f'{cache_dir}/raw_parts.mp4')

    #################
    ### normalize ###
    #################
    positions = np.array([[obj["position"]["x"], obj["position"]["y"], obj["position"]["z"]] 
                        for obj in json_obj["scene_models"]], dtype=np.float64)
    dimensions = np.array([[obj["dimensions"]["x"], obj["dimensions"]["y"], obj["dimensions"]["z"]] 
                        for obj in json_obj["scene_models"]], dtype=np.float64)


    all_corners = []

    for i in range(len(json_obj["scene_models"])):
        dx, dy, dz = dimensions[i]
        px, py, pz = positions[i]
        
        corners = [
            (px - dx/2, py - dy/2, pz - dz/2),
            (px + dx/2, py - dy/2, pz - dz/2),
            (px + dx/2, py + dy/2, pz - dz/2),
            (px - dx/2, py + dy/2, pz - dz/2),
            (px - dx/2, py - dy/2, pz + dz/2),
            (px + dx/2, py - dy/2, pz + dz/2),
            (px + dx/2, py + dy/2, pz + dz/2),
            (px - dx/2, py + dy/2, pz + dz/2)
        ]
        
        all_corners.extend(corners)
    # convert all corners to numpy array
    all_corners = np.array(all_corners, dtype=np.float64)

    positions -= all_corners.mean(axis=0, keepdims=True)
    max_distance = np.linalg.norm(all_corners, axis=-1).max()
    positions = positions / (max_distance + 1e-5) # -1 ~ 1?
    dimensions = dimensions / (max_distance + 1e-5) # 0 ~ 1?
    scales = np.linalg.norm(dimensions, axis=-1)
    # scales = scales / (max_distance + 1e-5)

    for i, obj in enumerate(json_obj["scene_models"]):
        obj["position"]["x"], obj["position"]["y"], obj["position"]["z"] = positions[i].tolist()
        obj["dimensions"]["x"], obj["dimensions"]["y"], obj["dimensions"]["z"] = dimensions[i].tolist()
        obj["scale"] = scales[i]
        
    ### normalize parts_in_global_coords ###
    all_parts = []
    for id, obj in tqdm(enumerate(json_obj['scene_models'])):
        obj_divid_json = obj['parts_in_complementary_format']
        
        position = np.array([obj['position']['x'], obj['position']['y'], obj['position']['z']])
        dimension = np.array([obj['dimensions']['x'], obj['dimensions']['y'], obj['dimensions']['z']])
        min_corners = position - dimension / 2
        max_corners = position + dimension / 2
        obj['parts_in_global_coords'] = calculate_bboxes(obj_divid_json, 
                                        bbox_min=min_corners, 
                                        bbox_max=max_corners,
                                        )
        all_parts.extend(obj['parts_in_global_coords'])
        
    ###############################################
    ### save normalized.json and normalized.png ###
    ###############################################
    cache_path = f'{cache_dir}/normalized.json'
    with open(cache_path, 'w') as json_file:
        json.dump(json_obj, json_file, indent=4)
        
        
    save_3d_bbox_as_png(json_obj, cache_dir, normalize=True, filename='normalized_objects.png')
    save_3d_bbox_as_video(json_obj, elev=20, n_frames=12, 
                        n_circles=1 / 4, fps=10, normalize=True,
                        output_path=f'{cache_dir}/normalized_objects.mp4',
                        )
    
    fake_json_obj4vis = convert_parts_to_scene_models(all_parts)
    save_3d_bbox_as_png(fake_json_obj4vis, cache_dir, normalize=True, filename='normalized_parts.png')
    save_3d_bbox_as_video(fake_json_obj4vis, elev=20, n_frames=12, 
                          n_circles=1/4, fps=10, normalize=True,
                          output_path=f'{cache_dir}/normalized_parts.mp4')

    
    
if batch_prompt_file is not None:
    if not os.path.isfile(batch_prompt_file):
        raise FileNotFoundError(f"The file {batch_prompt_file} does not exist.")
    
    msg = f"### Overwriting user_prompt with prompts from {batch_prompt_file} ###"
    print_msg(msg)
    
    with open(batch_prompt_file, 'r') as file:  
        user_prompts = [line.strip() for line in file if line.strip()]
    
    success_log_path = f'{args.cache_dir}/success_log.txt'
    failed_prompts_path = f'{args.cache_dir}/failed_prompts.txt'
    if os.path.exists(success_log_path):
        os.remove(success_log_path)
    if os.path.exists(failed_prompts_path):
        os.remove(failed_prompts_path)
    for id, user_prompt in tqdm(enumerate(user_prompts), desc="Processing prompts"):
        try:
            generate_layout_once(user_prompt)
            msg = f"{id + 1}. succeed ### {user_prompt} ###"
            print_msg(msg)
            with open(success_log_path, 'a') as file:
                file.write(f"{msg}\n")
        except Exception as e:
            msg = f"{id + 1}. fail ### {user_prompt}"
            print_msg(msg)
            print(f"###########################")
            print(f"### {e}")
            print(f"###########################")
            
            
            with open(success_log_path, 'a') as file:
                file.write(f"{msg}\n")
                file.write(f"###########################\n")
                file.write(f"{e}\n")
                file.write(f"###########################\n")
            
            with open(failed_prompts_path, 'a') as file:
                file.write(f"{user_prompt}\n")
else:
    msg = f"### Single user_prompt: {user_prompt} ###"
    print_msg(msg)
    generate_layout_once(user_prompt)