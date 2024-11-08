import time

import pypdfium2 # Needs to be at the top to avoid warnings
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" # For some reason, transformers decided to use .isin for a simple op, which is not supported on MPS

from marker.convert import convert_single_pdf
from marker.logger import configure_logging
from marker.models import load_all_models
from marker.output import save_markdown

configure_logging()

class MarkerArgs:
    filename = "/Users/luwi/Documents/Frequently_Used/GraphRAG_Paper.pdf"
    output = "output/"
    max_pages = None
    start_page = None
    langs = None
    batch_multiplier = 2

def parse_pdf(args: MarkerArgs, save: bool = True):
    langs = args.langs.split(",") if args.langs else None
    fname = args.filename

    model_lst = load_all_models()

    start = time.time()
    full_text, images, out_meta = convert_single_pdf(fname, model_lst, max_pages=args.max_pages, langs=langs, batch_multiplier=args.batch_multiplier)

    if save:
        fname = os.path.basename(fname)
        subfolder_path = save_markdown(args.output, fname, full_text, images, out_meta)
        print(f"Saved markdown to the {subfolder_path} folder")

    print(f"Total time: {time.time() - start}")
    return full_text


if __name__ == "__main__":
    args = MarkerArgs()
    os.makedirs(args.output, exist_ok=True)
    parse_pdf(args)