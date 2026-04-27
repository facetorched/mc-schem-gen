import zipfile
import json
import numpy as np
from PIL import Image
import os
import csv

# Change this to your local Minecraft client JAR location
JAR_LOCATION = r"C:\Users\kevinkt2\PrismLauncher-Windows-MSVC-Portable-8.0\libraries\com\mojang\minecraft\1.21\minecraft-1.21-client.jar"

SAVE_DIR = "src/mcschematic_plus/data/block_colormaps/"

EXCLUDE_EXACT = ["redstone_lamp_on", "budding_amethyst"]
EXCLUDE_CONTAINS = ["template_", "lit", "powered", "lit_powered", "structure_block",]
REMOVE_STRINGS = ["_inventory"]
EXCLUDE_EXACT_LIGHT_SOURCES = ["glowstone", "froglight", "sea_lantern", "shroomlight", "crying_obsidian", "redstone_ore", "deepslate_redstone_ore", "magma_block"]
EXCLUDE_EXACT_GRAVITY = ["sand", "red_sand", "gravel", "suspicious_sand", "suspicious_gravel"]
EXCLUDE_CONTAINS_GRAVITY = ["concrete_powder"]

def extract_uniform_blocks(jar_path, exclude_exact=None, exclude_contains=None, remove_strings=None, alpha_threshold=None, variance_threshold=None, cube_all=True):
    """
    Extracts average colors for blocks that use the 'cube_all' model (same texture on all 6 sides).
    """
    block_colors = {}
    print(f"Extracting block colors from {jar_path}...")

    with zipfile.ZipFile(jar_path, 'r') as z:
        # sort files to ensure consistent ordering
        all_files = sorted(set(z.namelist()))
        
        for file_path in all_files:
            if not file_path.startswith("assets/minecraft/models/block/") or not file_path.endswith(".json"):
                continue
                
            block_name = os.path.basename(file_path).replace(".json", "")
            
            # Apply filters
            for rem_str in (remove_strings or []):
                block_name = block_name.replace(rem_str, "")
            if any(s in block_name for s in (exclude_contains or [])):
                continue
            if block_name in (exclude_exact or []):
                continue
            
            # These look identical to their non-waxed counterparts, but don't change over time.
            if "copper" in block_name and not ("raw" in block_name or "ore" in block_name):
                block_name = "waxed_" + block_name

            try:
                with z.open(file_path) as f:
                    data = json.load(f)
            except Exception:
                continue

            # Check if it inherits from 'cube_all'
            parent = data.get("parent", "")
            if cube_all and "cube_all" not in parent:
                continue
            
            textures = data.get("textures", {})
            texture_path_raw = textures.get("all")
            
            if not texture_path_raw:
                continue

            # Resolve texture path (e.g., "minecraft:block/dirt" -> "assets/minecraft/textures/block/dirt.png")
            if ":" in texture_path_raw:
                namespace, path = texture_path_raw.split(":")
            else:
                namespace, path = "minecraft", texture_path_raw
            
            # Construct the path inside the JAR
            final_texture_path = f"assets/{namespace}/textures/{path}.png"
            
            if final_texture_path not in all_files:
                continue

            # Load Texture and Calculate Average
            try:
                with z.open(final_texture_path) as img_file:
                    with Image.open(img_file) as img:
                        img = img.convert("RGBA")
                        arr = np.array(img)
                        
                        rgb = arr[:, :, :3]
                        alpha = arr[:, :, 3]
                        
                        valid_pixels = alpha > 0
                        
                        if not np.any(valid_pixels):
                            continue # Skip invisible blocks
                            
                        avg_rgb = np.mean(rgb[valid_pixels], axis=0)
                        if variance_threshold is not None:
                            var_rgb = np.var(rgb[valid_pixels], axis=0)
                            if np.any(var_rgb > variance_threshold):
                                continue
                        
                        avg_alpha = np.mean(alpha)
                        if alpha_threshold is not None and not avg_alpha >= alpha_threshold:
                            continue
                        
                        # Add to block colors dictionary
                        block_colors["minecraft:" + block_name] = (
                            int(avg_rgb[0]), 
                            int(avg_rgb[1]), 
                            int(avg_rgb[2]), 
                            int(avg_alpha)
                        )
            except Exception as e:
                print(f"Error processing {block_name}: {e}")
                continue

    print(f"Finished! Found {len(block_colors)} uniform blocks.")
    return block_colors

def write_colors_to_csv(colors, output_file):
    with open(output_file, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["block_state", "r", "g", "b", "a"])
        for block_name, (r, g, b, a) in colors.items():
            writer.writerow([block_name, r, g, b, a])

if __name__ == "__main__":
    colors = extract_uniform_blocks(JAR_LOCATION, exclude_exact=EXCLUDE_EXACT, exclude_contains=EXCLUDE_CONTAINS, remove_strings=REMOVE_STRINGS) # Remove only what is necessary
    write_colors_to_csv(colors, SAVE_DIR + "all.csv")

    colors = extract_uniform_blocks(JAR_LOCATION, exclude_exact=EXCLUDE_EXACT + EXCLUDE_EXACT_LIGHT_SOURCES + EXCLUDE_EXACT_GRAVITY, exclude_contains=EXCLUDE_CONTAINS + EXCLUDE_CONTAINS_GRAVITY, remove_strings=REMOVE_STRINGS, alpha_threshold=255)
    write_colors_to_csv(colors, SAVE_DIR + "standard.csv")

    colors = extract_uniform_blocks(JAR_LOCATION, exclude_exact=EXCLUDE_EXACT + EXCLUDE_EXACT_LIGHT_SOURCES + EXCLUDE_EXACT_GRAVITY, exclude_contains=EXCLUDE_CONTAINS + EXCLUDE_CONTAINS_GRAVITY, remove_strings=REMOVE_STRINGS, alpha_threshold=255, variance_threshold=500)
    write_colors_to_csv(colors, SAVE_DIR + "smooth.csv")