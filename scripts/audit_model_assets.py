"""Report model assets that are referenced, missing, or candidates for archive."""

from pathlib import Path
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = PROJECT_ROOT / "models"


def mesh_references(model_path):
    root = ET.parse(model_path).getroot()
    references = set()
    for mesh in root.findall(".//mesh"):
        value = mesh.get("file") or mesh.get("filename")
        if value:
            references.add(Path(value).name)
    return references


def main():
    asset_dir = MODEL_ROOT / "assets"
    assets = {path.name: path for path in asset_dir.iterdir() if path.is_file()}
    model_paths = (MODEL_ROOT / "g1_29dof.xml", MODEL_ROOT / "g1_29dof.urdf")
    references = set().union(*(mesh_references(path) for path in model_paths))
    missing = sorted(references - assets.keys())
    unused = sorted(assets.keys() - references)
    unused_bytes = sum(assets[name].stat().st_size for name in unused)
    print(f"assets={len(assets)} referenced={len(references)}")
    print(f"missing={len(missing)} archive_candidates={len(unused)}")
    print(f"archive_candidate_size_mb={unused_bytes / 1024 / 1024:.1f}")
    for name in missing:
        print(f"MISSING {name}")
    for name in unused:
        print(f"UNREFERENCED {name}")


if __name__ == "__main__":
    main()
