# src/vexto/utils/paths.py
import re
import logging
log = logging.getLogger(__name__)

def deep_get(obj, path: str, default=None):
    try:
        cur = obj
        for part in path.split('.'):
            if not part:
                continue
            if '[' in part and ']' in part:
                if ':' in part:  # Slice detected
                    log.debug(f"Rejecting slice notation in path: {part}")
                    return default
                base_name = part.split('[')[0]
                if base_name:
                    if not isinstance(cur, dict):
                        return default
                    cur = cur.get(base_name)
                    if cur is None:
                        return default
                for idx_str in re.findall(r'\[(\d+)\]', part):
                    if not isinstance(cur, list):
                        return default
                    i = int(idx_str)
                    if i < 0 or i >= len(cur):
                        return default
                    cur = cur[i]
            else:
                if isinstance(cur, dict):
                    cur = cur.get(part)
                elif isinstance(cur, list) and part.isdigit():
                    idx = int(part)
                    if 0 <= idx < len(cur):
                        cur = cur[idx]
                    else:
                        return default
                else:
                    return default
                if cur is None:
                    return default
        return cur
    except Exception as e:
        log.debug(f"Unexpected error in path traversal '{path}': {e}")
        return default
