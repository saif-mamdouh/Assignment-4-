import re




def measure_fps_from_log(log_path: str):
"""Parse a log file to estimate FPS. This is heuristic and depends on the eval log format."""
txt = open(log_path, 'r', errors='ignore').read()
# Attempt to find lines like: "Elapsed time: 12.34s, frames: 600"
m = re.search(r'Elapsed\s+time[:=]\s*([0-9\.]+)s.*frames[:=]\s*([0-9]+)', txt, re.IGNORECASE | re.S)
if m:
elapsed = float(m.group(1))
frames = int(m.group(2))
return frames / elapsed if elapsed>0 else None
# fallback None
return None




def aggregate_metrics(metrics_dict: dict):
"""Convert metrics_by_checkpoint to CSV-friendly list of rows."""
rows = []
for ckpt, m in metrics_dict.items():
row = {
'checkpoint': ckpt,
'iou': m.get('iou'),
'precision': m.get('precision'),
'auc': m.get('auc') or m.get('AUC'),
'fps': m.get('fps'),
'eval_time_s': m.get('eval_time_s')
}
rows.append(row)
return rows