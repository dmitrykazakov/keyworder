import warnings
warnings.filterwarnings("ignore")

import os
import sys
import base64
import json
import csv
import logging
import shutil
import argparse
import io
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

from PIL import Image, ExifTags
from openai import OpenAI
from tqdm import tqdm
from pydantic import BaseModel, ValidationError


def get_application_path():
    if getattr(sys, 'frozen', False):
        application_path = os.path.dirname(sys.executable)
    else:
        application_path = os.path.dirname(os.path.abspath(__file__))
    return application_path


SCRIPT_DIR = get_application_path()
SETTINGS_FILE_PATH = os.path.join(get_application_path(), "settings.json")


def load_app_settings(settings_path):
    try:
        with open(settings_path, 'r') as f:
            settings_data = json.load(f)

        if "OPENAI_API_KEY" not in settings_data:
            logging.critical("OPENAI_API_KEY not found in settings.json. Please add it.")
            sys.exit(1)

        if "LLM_GENERATION_PROMPT" not in settings_data or not settings_data["LLM_GENERATION_PROMPT"]:
            logging.critical("LLM_GENERATION_PROMPT not found or is empty in settings.json. This is required.")
            sys.exit(1)

        settings_data.setdefault("MAX_WORKERS", 10)
        settings_data.setdefault("LOG_LEVEL", "INFO")
        settings_data.setdefault("DEFAULT_LOG_DIR", None)
        settings_data.setdefault("IMAGE_EXTENSIONS", ['.jpg', '.jpeg'])
        settings_data.setdefault("OPENAI_MODEL", "gpt-4o-mini")
        return settings_data
    except FileNotFoundError:
        logging.critical(f"Settings file not found: {settings_path}. Please create it.")
        sys.exit(1)
    except json.JSONDecodeError:
        logging.critical(f"Error decoding JSON from {settings_path}. Please check its format.")
        sys.exit(1)
    except Exception as e:
        logging.critical(f"An unexpected error occurred while loading settings: {e}", exc_info=True)
        sys.exit(1)


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s (bootstrap)')

APP_SETTINGS = None
client = None


class Description(BaseModel):
    title: str
    keywords: list[str]


def setup_logging(log_file_path, level="INFO"):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    logging.basicConfig(
        filename=log_file_path,
        filemode='a',
        format='%(asctime)s [%(levelname)s] %(message)s',
        level=numeric_level
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)


def fix_image_orientation(img):
    try:
        for orientation_tag_key in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation_tag_key] == 'Orientation':
                break
        exif = img._getexif()
        if exif is not None:
            orientation_val = exif.get(orientation_tag_key)
            if orientation_val == 3:
                img = img.rotate(180, expand=True)
            elif orientation_val == 6:
                img = img.rotate(270, expand=True)
            elif orientation_val == 8:
                img = img.rotate(90, expand=True)
    except Exception:
        pass
    return img


def resize_image_to_longest_side_and_get_base64(image_path, target_longest_side=512):
    try:
        with Image.open(image_path) as img:
            img = fix_image_orientation(img)
            img = img.convert("RGB")
            w, h = img.size
            if max(w, h) > target_longest_side:
                if w > h:
                    new_w = target_longest_side
                    new_h = int(h * target_longest_side / w)
                else:
                    new_h = target_longest_side
                    new_w = int(w * target_longest_side / h)
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        logging.error(f"Failed to open, resize, or encode image {image_path}: {e}", exc_info=True)
        return None


def generate_metadata(image_base64, llm_prompt_text):
    try:
        model = APP_SETTINGS.get("OPENAI_MODEL")
        if not llm_prompt_text or not image_base64:
            return None, None

        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": llm_prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}", "detail": "low"}},
                ],
            }],
            response_format={"type": "json_object"},
        )
        content_str = response.choices[0].message.content
        parsed = Description(**json.loads(content_str))
        title = parsed.title.strip()
        keywords = ",".join([kw.strip().lower() for kw in parsed.keywords if kw.strip()])
        return title, keywords
    except (ValidationError, json.JSONDecodeError) as e:
        logging.error(f"Failed to parse GPT response: {e}")
    except Exception as e:
        logging.error(f"OpenAI API request failed: {e}", exc_info=True)
    return None, None


def write_csv(file_path, headers, data):
    try:
        with open(file_path, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            for row in data:
                writer.writerow(row)
        logging.info(f"CSV created: {file_path}")
    except Exception as e:
        logging.error(f"Failed to write CSV {file_path}: {e}", exc_info=True)


def process_file(file_info, image_extensions, input_root_dir):
    file_path, relative_file_path = file_info
    ext = os.path.splitext(file_path)[1].lower()
    filename = os.path.basename(file_path)
    base_llm_prompt = APP_SETTINGS.get("LLM_GENERATION_PROMPT")

    if ext in image_extensions:
        logging.info(f"Processing image: {filename}")
        image_base64 = resize_image_to_longest_side_and_get_base64(file_path, 512)
        if not image_base64:
            errors_folder = os.path.join(input_root_dir, "errors")
            os.makedirs(os.path.dirname(os.path.join(errors_folder, relative_file_path)), exist_ok=True)
            shutil.move(file_path, os.path.join(errors_folder, relative_file_path))
            return None

        title, keywords = generate_metadata(image_base64, base_llm_prompt)
        if not title or not keywords:
            errors_folder = os.path.join(input_root_dir, "errors")
            os.makedirs(os.path.dirname(os.path.join(errors_folder, relative_file_path)), exist_ok=True)
            shutil.move(file_path, os.path.join(errors_folder, relative_file_path))
            return None
        return filename, title, keywords
    else:
        return None


def main():
    global APP_SETTINGS, client
    APP_SETTINGS = load_app_settings(SETTINGS_FILE_PATH)
    client = OpenAI(api_key=APP_SETTINGS["OPENAI_API_KEY"])

    parser = argparse.ArgumentParser(description="Process JPG images to generate metadata and CSV files.")
    default_input_dir = os.path.join(SCRIPT_DIR, "input")
    default_output_dir = os.path.join(SCRIPT_DIR, "output")
    log_dir_from_settings = APP_SETTINGS.get("DEFAULT_LOG_DIR")

    if log_dir_from_settings and os.path.isabs(log_dir_from_settings):
        effective_default_log_dir = log_dir_from_settings
    elif log_dir_from_settings:
        effective_default_log_dir = os.path.join(SCRIPT_DIR, log_dir_from_settings)
    else:
        effective_default_log_dir = os.path.join(SCRIPT_DIR, "logs")

    parser.add_argument("--input_dir", type=str, default=default_input_dir)
    parser.add_argument("--output_dir", type=str, default=default_output_dir)
    parser.add_argument("--log_dir", type=str, default=effective_default_log_dir)
    args = parser.parse_args()

    input_directory = os.path.abspath(args.input_dir)
    output_directory = os.path.abspath(args.output_dir)
    log_directory = os.path.abspath(args.log_dir)

    os.makedirs(log_directory, exist_ok=True)
    os.makedirs(output_directory, exist_ok=True)

    log_file_path = os.path.join(log_directory, "processing.log")
    setup_logging(log_file_path, level=APP_SETTINGS.get("LOG_LEVEL", "INFO"))

    image_extensions = APP_SETTINGS.get("IMAGE_EXTENSIONS")

    all_files_to_process = []
    for root, dirs, files in os.walk(input_directory):
        files = [f for f in files if not f.startswith('.')]
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        if os.path.basename(root) == "errors" and os.path.dirname(root) == input_directory:
            continue
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, input_directory)
                all_files_to_process.append((full_path, relative_path))

    if not all_files_to_process:
        logging.info(f"No JPG files found in '{input_directory}'. Exiting.")
        sys.exit(0)

    adobestock_data = []
    stocksubmitter_data = []
    max_workers = int(APP_SETTINGS.get("MAX_WORKERS"))
    description_ext_setting = APP_SETTINGS.get("DESCRIPTION_EXT", "")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file, file_info, image_extensions, input_directory)
                   for file_info in all_files_to_process]
        for future in tqdm(futures, desc="Processing files", total=len(all_files_to_process)):
            try:
                result = future.result()
                if result:
                    filename, title, keywords = result
                    adobestock_data.append([filename, title, keywords, "", ""])
                    stocksubmitter_data.append([filename, title, title + description_ext_setting, keywords, ""])
                    stocksubmitter_data[-1] = [col.replace("..", ".") for col in stocksubmitter_data[-1]]
            except Exception as e:
                logging.error(f"Unhandled error processing a file: {e}", exc_info=True)

    write_csv(os.path.join(output_directory, "adobestock.csv"),
              ["Filename", "Title", "Keywords", "Category", "Releases"], adobestock_data)
    write_csv(os.path.join(output_directory, "stocksubmitter.csv"),
              ["Filename", "Title", "Description", "Keywords", "Release name"], stocksubmitter_data)

    logging.info(f"Processing complete. CSV files generated in '{output_directory}'.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
