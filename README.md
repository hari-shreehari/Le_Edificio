# BLUEPRINT_TO_BLENDER

**BLUEPRINT_TO_BLENDER** is a Python tool that converts a 2D blueprint into a Blender file with textures applied for walls and floors.

## Requirements

This tool requires **Python 3.11** and the following Python packages:

- `opencv-python`
- `numpy`
- `pdf2image`
- `wand`
- `Pillow`

It also requires the following system dependencies:

- `poppler-utils` (for `pdf2image`)
- `imagemagick` (for `wand`)
- Blender installation (for the `bpy` module)

### Install System Dependencies

On Ubuntu/Debian-based systems, you can install the system dependencies using:

```bash
sudo apt-get install poppler-utils imagemagick
```

Ensure that Blender is installed and accessible via its Python environment.

### Install Python Dependencies

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

### For Conda 

On linux:
```bash
conda env create -f linux-environment.yml
```
On windows:
```bash
conda env create -f windows-environment.yml
```
Install lock using:  
```bash
conda-lock install --name YOURENV windows-conda-lock.yml
```  

Then activate the environment with the following command:
```bash
conda activate twoD2threeD
```


## Usage

Run the script with the following command:

### For conversion:  

#### Syntax:

```bash
python main.py --convert "<Path_To_Blueprint>" --wall_texture "<Wall_Texture>" --floor_texture "<Floor_Texture>" --scale_factor 0.1 --wall_height 10 --wall_thickness 0.5 --output_directory "<Path_To_Write_Results>"
```

#### Sample :

```
python main.py --convert "./Blueprints/example-blueprint1.png" --wall_texture "brown-brick.jpg" --floor_texture "plank-flooring.jpg" --scale_factor 0.1 --wall_height 10 --wall_thickness 0.5 --output_directory "./results"
```

### For stacking:

#### Syntax:

```bash
python main.py --stack --output_directory "<Directory with glb files of each floor>"
```

#### Sample :

```bash
python main.py --stack --output_directory "./results/floors"
```

**Note** : The folder passed for stacking must contain glb named as "floor(x)" as in "floor0", "floor1"...

## Parameters:
    
--convert : Flag to denote conversion process  
path_to_blueprint: Path to the 2D blueprint file.  
--wall_texture: Name of the wall texture file located in the assets folder.  
--floor_texture: Name of the floor texture file located in the assets folder.  
--scale_factor: Scale factor for the building (default: 0.1).  
--wall_height: Height of the walls in the building (default: 10).  
--wall_thickness: Thickness of walls in the building (default: 0.5).  
--output_directory: Folder where the output would be written.  
  
--stack : Flag to denote stacking process  
--output_directory: Folder with glb of each floor where result will be written too.  


## Results

The program outputs: 
  
**For Conversion :**    
    A Blender file saved to the results folder as output.blend.
    Exported models in .glb formats in the same results folder.

**For stacking :**  
    A glb file will be saved at the same output directory passed as argument.


## Pyinstaller

To create a standalone executable, use the following command:

```bash
pyinstaller --onefile main.py --collect-all "bpy" --add-data "assets:assets"
```

This will create a standalone executable in the dist folder.

for a single folder executable

```bash
pyinstaller --onedir main.py --collect-all "bpy" --add-data "assets:assets"
```

To execute this, you can run the following command in the terminal:

```bash
cd dist
```

Then, you can run the executable with the following command:

```bash
.\main --convert "..\Blueprints\example-blueprint.png" --wall_texture "glass.jpg" --floor_texture "plank-flooring.jpg" --scale_factor 0.1 --wall_height 10 --wall_thickness 0.5 --output_directory "..\results"
```

```bash
.\main --stack --output_directory "./results/floors"
```
