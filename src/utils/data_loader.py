from pathlib import Path
import json
import os

# Convert DataFrame annotations to YOLO format and save as .txt files.

def convert_bdd_to_yolo(df, class_mapping, output_dir, img_width=1280, img_height=720):
    for file, group in df.groupby('file'):
        txt_file = os.path.join(output_dir, os.path.splitext(file)[0] + '.txt')
        os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn’t exist
        with open(txt_file, 'w') as f:
            for _, row in group.iterrows():
                class_id = class_mapping.get(row['type'], -1)
                if class_id == -1:
                    continue  # Skip unknown classes
                xmin, ymin, xmax, ymax = row['bbox_xmin'], row['bbox_ymin'], row['bbox_xmax'], row['bbox_ymax']
                x_center = (xmin + xmax) / 2 / img_width
                y_center = (ymin + ymax) / 2 / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height
                f.write(f'{class_id} {x_center} {y_center} {width} {height}\n')                                
                
                
# Convert only those KITTI JSONs whose size = (target_w x target_h) to YOLO-format .txt files.

def convert_kitti_to_yolo(
    ann_dir,
    output_dir,
    kitti_class_mapping,
    target_w=1242,
    target_h=375
):
    ann_dir   = Path(ann_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for jpath in ann_dir.glob('*.json'):
        data = json.loads(jpath.read_text())

        # 1) Resolution check
        size = data.get('size', {})
        if size.get('width') != target_w or size.get('height') != target_h:
            print(f'Skipping {jpath.name}: {size.get("width")}×{size.get("height")} != {target_w}×{target_h}')
            continue

        # 2) Build YOLO lines
        lines = []
        for obj in data.get('objects', []):
            if obj.get('geometryType') != 'rectangle':
                continue

            cid = kitti_class_mapping.get(obj.get('classTitle', '').lower(), -1)
            if cid < 0:
                continue

            pts = obj.get('points', {}).get('exterior', [])
            if len(pts) != 2:
                continue

            (x1, y1), (x2, y2) = pts
            xmin, xmax = sorted([x1, x2])
            ymin, ymax = sorted([y1, y2])

            x_c = (xmin + xmax) / 2.0 / target_w
            y_c = (ymin + ymax) / 2.0 / target_h
            w_n = (xmax  - xmin) / target_w
            h_n = (ymax  - ymin) / target_h

            lines.append(f'{cid} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}')

        # 3) Write out if any boxes found
        if lines:
            stem = jpath.stem.split('.')[0]
            out_path = output_dir / f'{stem}.txt'
            out_path.write_text('\n'.join(lines) + '\n')
            
            
def filter_images_by_labels(val_img_dir, val_lbl_dir):
    img_dir = Path(val_img_dir)
    lbl_dir = Path(val_lbl_dir)

    # Collect stems
    img_stems = {p.stem for p in img_dir.glob('*.png')}
    lbl_stems = {p.stem for p in lbl_dir.glob('*.txt')}

    # Find orphaned images (have no corresponding label)
    orphan_imgs = img_stems - lbl_stems
    print(f'Orphan images to remove: {len(orphan_imgs)}')

    # Delete only those images
    for stem in orphan_imgs:
        (img_dir / f'{stem}.png').unlink()

    # Re‑compute
    final_imgs = {p.stem for p in img_dir.glob('*.png')}
    final_lbls = {p.stem for p in lbl_dir.glob('*.txt')}

    # Final check
    assert final_imgs == final_lbls, (
        f'Mismatch after filtering: '
        f'{len(final_imgs)} images vs {len(final_lbls)} labels'
    )
    print('✅ All remaining images and labels match 1:1')